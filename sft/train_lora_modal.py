"""SFT データ（sft/train.jsonl, sft/valid.jsonl）で Qwen3.5 に LoRA を学習し、元モデルに統合したものを Modal Volume に保存する。

統合済みモデルは /models/<run名>/merged に置かれ、そのまま評価に使える:
    modal run eval/run_eval_modal.py --run-name <run名> --model /models/<run名>/merged

使い方:
    modal run sft/train_lora_modal.py --run-name lora-0.8b --base-model Qwen/Qwen3.5-0.8B
    modal run sft/train_lora_modal.py --run-name lora-4b --base-model Qwen/Qwen3.5-4B --gpu L40S

全部の重みを学習する（フルファインチューニング）場合は --full。小さいモデル（0.8B）への蒸留用:
    modal run sft/train_lora_modal.py --run-name qwen08b-distill --base-model Qwen/Qwen3.5-0.8B --full --gpu H100 \
        --train-file distill_sft.jsonl --holdout 200 --lr 1e-5 --batch-size 2 --grad-accum 16 --max-len 2048 --drop-long --grad-ckpt
"""

import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
SFT_DIR = ROOT / "sft"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "peft==0.21.0",
        "accelerate==1.15.0",
        # Qwen3.5 の線形アテンション層（Gated DeltaNet）の高速カーネル。なくても動くが遅い
        "flash-linear-attention==0.5.2",
    )
    .env({"HF_HUB_CACHE": "/hf-cache"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
models = modal.Volume.from_name("pitcher-models", create_if_missing=True)
app = modal.App("pitcher-train", image=image)


def encode(tokenizer, messages: list[dict], max_len: int) -> dict:
    """推論時と同じ形（enable_thinking=False の生成プロンプト + 応答）にして、応答部分だけを学習対象にする。"""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    answer = messages[-1]["content"] + "<|im_end|>\n"
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


@app.function(gpu="L4", timeout=3 * 60 * 60, volumes={"/hf-cache": hf_cache, "/models": models})
def train(train_rows: list[dict], valid_rows: list[dict], base_model: str, run_name: str, hp: dict) -> dict:
    import math
    import os
    import shutil
    import time

    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    torch.manual_seed(hp["seed"])
    out_dir = f"/models/{run_name}"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    full = hp.get("full", False)
    # フルファインチューニングでは、小さい学習率の更新が bf16 の丸めで消えないよう、重みは fp32 で持つ（計算は bf16）
    model = AutoModelForImageTextToText.from_pretrained(base_model, dtype=torch.float32 if full else torch.bfloat16, device_map="cuda")

    if full:
        # 言語モデル側だけを学習し、画像エンコーダは固定する
        for name, prm in model.named_parameters():
            prm.requires_grad = "visual" not in name
        n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"フルファインチューニング：学習するパラメータ {n_train_p:,}")
    else:
        # LoRA は言語モデル側の全線形層に入れる（画像エンコーダと出力層は対象外）
        targets = sorted(
            {name.split(".")[-1] for name, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and "language_model" in name and "visual" not in name}
        )
        print("LoRA 対象:", targets)
        model = get_peft_model(model, LoraConfig(
            r=hp["lora_r"], lora_alpha=hp["lora_alpha"], lora_dropout=hp["lora_dropout"],
            target_modules=rf".*language_model.*\.({'|'.join(targets)})$", task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()

    if hp.get("drop_long"):
        # 上限を超える例は途中で切らずに除く（途中で切れた回答を学習させないため）
        n_before = len(train_rows)
        train_rows = [r for r in train_rows if len(encode(tokenizer, r["messages"], 10**9)["input_ids"]) <= hp["max_len"]]
        valid_rows = [r for r in valid_rows if len(encode(tokenizer, r["messages"], 10**9)["input_ids"]) <= hp["max_len"]]
        print(f"上限 {hp['max_len']} トークンを超える例を除外：{n_before} → {len(train_rows)} 件")
    train_ds = [encode(tokenizer, r["messages"], hp["max_len"]) for r in train_rows]
    valid_ds = [encode(tokenizer, r["messages"], hp["max_len"]) for r in valid_rows]
    n_tokens = sum(len(x["input_ids"]) for x in train_ds)
    print(f"学習 {len(train_ds)} 件 / 検証 {len(valid_ds)} 件 / 学習トークン {n_tokens:,}（1周あたり）")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(batch):
        n = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), n), pad_id)
        labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, x in enumerate(batch):
            L = len(x["input_ids"])
            ids[i, :L] = torch.tensor(x["input_ids"])
            labels[i, :L] = torch.tensor(x["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    history = []

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                history.append({"step": state.global_step, **logs})

    total_steps = math.ceil(len(train_ds) / (hp["batch_size"] * hp["grad_accum"])) * hp["epochs"]
    args = TrainingArguments(
        output_dir="/tmp/out",
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["grad_accum"],
        num_train_epochs=hp["epochs"],
        learning_rate=hp["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(total_steps * 0.03)),
        weight_decay=0.0,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=hp["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        # 長い応答を含むデータではメモリが足りなくなるので、計算をやり直す代わりにメモリを節約する
        gradient_checkpointing=hp.get("grad_ckpt", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
                      data_collator=collate, callbacks=[Log()])
    t0 = time.time()
    eval_before = trainer.evaluate()["eval_loss"]
    trainer.train()
    train_sec = time.time() - t0
    eval_after = trainer.evaluate()["eval_loss"]

    if full:
        merged = model.to(torch.bfloat16)
    else:
        model.save_pretrained(f"{out_dir}/adapter")
        merged = model.merge_and_unload()
    merged.save_pretrained(f"{out_dir}/merged", safe_serialization=True)
    tokenizer.save_pretrained(f"{out_dir}/merged")
    # vLLM が読むための前処理設定・チャットテンプレートなど、重み以外のファイルを元モデルからそろえる。
    # キャッシュのフォルダには元モデルの重みも入っているので、重みと index は絶対にコピーしない
    # （vLLM はフォルダ内の safetensors を全部読むため、元の重みで学習結果が上書きされる）
    snap = snapshot_download(base_model, allow_patterns=["*.json", "*.jinja", "*.txt"])
    for f in os.listdir(snap):
        dst = f"{out_dir}/merged/{f}"
        if f.endswith((".json", ".jinja", ".txt")) and not f.endswith("index.json") and not os.path.exists(dst):
            shutil.copy(f"{snap}/{f}", dst)
    models.commit()

    result = {"run": run_name, "base_model": base_model, "hp": hp, "n_train": len(train_ds), "n_valid": len(valid_ds),
              "train_tokens_per_epoch": n_tokens, "eval_loss_before": eval_before, "eval_loss_after": eval_after,
              "train_seconds": train_sec, "gpu": torch.cuda.get_device_name(), "history": history}
    with open(f"{out_dir}/train_result.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    models.commit()
    return result


DEFAULT_HP = dict(lora_r=16, lora_alpha=32, lora_dropout=0.05, lr=2e-4, epochs=2, batch_size=16, grad_accum=1,
                  max_len=1024, seed=0)


@app.local_entrypoint()
def main(run_name: str, base_model: str = "Qwen/Qwen3.5-0.8B", gpu: str = "L4", lr: float = 2e-4, epochs: int = 2,
         train_file: str = "train.jsonl", valid_file: str = "valid.jsonl", batch_size: int = 16, grad_accum: int = 1,
         max_len: int = 1024, grad_ckpt: bool = False, full: bool = False, holdout: int = 0, drop_long: bool = False):
    import random

    load = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    train_rows = load(SFT_DIR / train_file)
    if holdout:
        # 検証用に、学習データから holdout 件を取り分ける（同じ種類のデータで検証するため）
        random.Random(0).shuffle(train_rows)
        train_rows, valid_rows = train_rows[holdout:], train_rows[:holdout]
        valid_file = f"{train_file} から {holdout} 件"
    else:
        valid_rows = load(SFT_DIR / valid_file)
    hp = DEFAULT_HP | {"lr": lr, "epochs": epochs, "train_file": train_file, "valid_file": valid_file, "batch_size": batch_size,
                       "grad_accum": grad_accum, "max_len": max_len, "grad_ckpt": grad_ckpt, "full": full, "drop_long": drop_long}
    result = train.with_options(gpu=gpu).remote(train_rows, valid_rows, base_model, run_name, hp)
    out = SFT_DIR / "runs" / f"{run_name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"検証 loss: {result['eval_loss_before']:.3f} → {result['eval_loss_after']:.3f}"
          f"（学習 {result['train_seconds'] / 60:.1f} 分, {result['gpu']}）")
    print(f"{out} に保存しました")
