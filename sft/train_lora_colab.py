"""Colab のランタイム上で LoRA を学習し、元モデルに統合して保存する（Spark-X2.5 など transformers 4.57 が必要なモデル用）。

Colab CLI（colab upload / colab exec）でこのファイルとデータを VM に送り、VM 上で次のように実行する:
    python train_lora_colab.py --run-name spark-lora --base-model XHToken/Spark-X2.5-4B \
        --train /content/data/train.jsonl --valid /content/data/valid.jsonl --out /content/out

出力:
    <out>/<run名>/adapter/         LoRA 本体（小さいので手元に持ち帰る）
    <out>/<run名>/merged/          元モデルに統合したもの（評価用）
    <out>/<run名>/train_result.json
"""

import argparse
import json
import math
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments


def answer_suffix(tokenizer) -> str:
    """応答の後ろに付く「発言の終わり」の記号を、チャットテンプレートから取り出す。"""
    probe = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "XYZ"}]
    full = tokenizer.apply_chat_template(probe, tokenize=False, enable_thinking=False)
    return full[full.rindex("XYZ") + 3 :]


def encode(tokenizer, messages: list[dict], suffix: str, max_len: int) -> dict:
    """推論時と同じ形（thinking なしの生成プロンプト + 応答 + 終わりの記号）にして、応答部分だけを学習対象にする。"""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(messages[-1]["content"] + suffix, add_special_tokens=False)["input_ids"]
    return {"input_ids": (p_ids + a_ids)[:max_len], "labels": ([-100] * len(p_ids) + a_ids)[:max_len]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", default="/content/out")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out_dir = f"{a.out}/{a.run_name}"
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(a.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.base_model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")

    # LoRA は全線形層に入れる（出力層 lm_head は除く）
    targets = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not n.endswith("lm_head")})
    print("LoRA 対象:", targets, flush=True)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    if a.grad_ckpt:
        model.enable_input_require_grads()

    suffix = answer_suffix(tokenizer)
    print("応答の終わりの記号:", repr(suffix), flush=True)
    load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    train_ds = [encode(tokenizer, r["messages"], suffix, a.max_len) for r in load(a.train)]
    valid_ds = [encode(tokenizer, r["messages"], suffix, a.max_len) for r in load(a.valid)]
    n_tokens = sum(len(x["input_ids"]) for x in train_ds)
    print(f"学習 {len(train_ds)} 件 / 検証 {len(valid_ds)} 件 / 学習トークン {n_tokens:,}（1周あたり）", flush=True)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(batch):
        n = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), n), pad_id)
        labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, x in enumerate(batch):
            L = len(x["input_ids"])
            ids[i, :L], labels[i, :L], mask[i, :L] = torch.tensor(x["input_ids"]), torch.tensor(x["labels"]), 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    history = []

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                history.append({"step": state.global_step, **logs})
                print(json.dumps({"step": state.global_step, **logs}), flush=True)

    total_steps = math.ceil(len(train_ds) / (a.batch_size * a.grad_accum)) * a.epochs
    args = TrainingArguments(
        output_dir="/tmp/trainer", per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(total_steps * 0.03)), weight_decay=0.0, bf16=True, logging_steps=10, eval_strategy="epoch",
        save_strategy="no", report_to=[], seed=a.seed, remove_unused_columns=False, dataloader_num_workers=0,
        gradient_checkpointing=a.grad_ckpt, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds, data_collator=collate, callbacks=[Log()])
    eval_before = trainer.evaluate()["eval_loss"]
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    eval_after = trainer.evaluate()["eval_loss"]

    model.save_pretrained(f"{out_dir}/adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(f"{out_dir}/merged", safe_serialization=True)
    tokenizer.save_pretrained(f"{out_dir}/merged")
    # 独自構造のモデルは、同梱コード（modeling_*.py / configuration_*.py）も統合モデルの隣に置く必要がある
    from huggingface_hub import snapshot_download

    # 元モデルが手元のフォルダ（第1段階で統合したモデルなど）のときは、そこから取る
    snap = a.base_model if os.path.isdir(a.base_model) else snapshot_download(a.base_model, allow_patterns=["*.py", "*.jinja", "generation_config.json"])
    for f in os.listdir(snap):
        if f.endswith((".py", ".jinja")) or f == "generation_config.json":
            dst = f"{out_dir}/merged/{f}"
            if not os.path.exists(dst):
                os.system(f"cp '{snap}/{f}' '{dst}'")

    result = {"run": a.run_name, "base_model": a.base_model, "hp": vars(a), "n_train": len(train_ds), "n_valid": len(valid_ds),
              "train_tokens_per_epoch": n_tokens, "eval_loss_before": eval_before, "eval_loss_after": eval_after,
              "train_seconds": train_sec, "gpu": torch.cuda.get_device_name(), "history": history}
    json.dump(result, open(f"{out_dir}/train_result.json", "w"), ensure_ascii=False, indent=1)
    print(f"検証 loss: {eval_before:.3f} → {eval_after:.3f}（学習 {train_sec / 60:.1f} 分, {result['gpu']}）", flush=True)


if __name__ == "__main__":
    main()
