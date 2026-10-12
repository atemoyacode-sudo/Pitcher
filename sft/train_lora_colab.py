"""Colab のランタイム上で LoRA を学習し、元モデルに統合して保存する（Spark-X2.5 など transformers 4.57 が必要なモデル用）。

Colab CLI（colab upload / colab exec）でこのファイルとデータを VM に送り、VM 上で次のように実行する:
    python train_lora_colab.py --run-name spark-lora --base-model XHToken/Spark-X2.5-4B \
        --train /content/data/train.jsonl --valid /content/data/valid.jsonl --out /content/out

Qwen3.5-0.8B 系（画像入力つきの形で保存されたモデル）は --image-text を付ける。言語モデル側だけに LoRA を入れる。
部品として使うだけなら --no-merge で統合モデルを作らない（llama.cpp 用には adapter を convert_lora_to_gguf.py で変換）:
    pip install flash-linear-attention   # Qwen3.5 の線形アテンション層の高速カーネル（なくても動くが遅い）
    python train_lora_colab.py --run-name qwen08b-think-verify-lora --base-model /content/qwen08b-think \
        --train /content/data/train.jsonl --valid /content/data/valid.jsonl --image-text --no-merge \
        --epochs 1 --batch-size 4 --grad-accum 4 --max-len 2048 --drop-long --grad-ckpt

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
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments


def answer_suffix(tokenizer) -> str:
    """応答の後ろに付く「発言の終わり」の記号を、チャットテンプレートから取り出す。"""
    probe = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "XYZ"}]
    full = tokenizer.apply_chat_template(probe, tokenize=False, enable_thinking=False)
    return full[full.rindex("XYZ") + 3 :]


def encode(tokenizer, messages: list[dict], suffix: str, max_len: int, thinking: str | None = None) -> dict:
    """推論時と同じ形（thinking なしの生成プロンプト + 応答 + 終わりの記号）にして、応答部分だけを学習対象にする。

    thinking（考える部分）があるときは思考ありの形（生成プロンプトが「<think>\\n」で終わり、応答は「考える部分\\n</think>\\n\\n答え」）にする。
    """
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    answer = messages[-1]["content"] + suffix
    if thinking is not None:
        assert prompt.endswith("<think>\n\n</think>\n\n"), prompt[-40:]
        prompt = prompt.removesuffix("\n</think>\n\n")
        answer = thinking.strip() + "\n</think>\n\n" + answer
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
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
    ap.add_argument("--drop-long", action="store_true", help="上限（--max-len）を超える例は途中で切らずに除く")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-text", action="store_true", help="画像入力つきの形のモデル（Qwen3.5-0.8B など）。言語モデル側だけを学習する")
    ap.add_argument("--no-merge", action="store_true", help="LoRA だけを保存し、元モデルとの統合はしない")
    ap.add_argument("--text", action="store_true", help="行の \"text\" をそのまま読ませる（会話の形にしない追加の事前学習）")
    ap.add_argument("--fp32", action="store_true", help="bf16 を使わず fp32 で持って計算する（bf16 のない T4 など。Kaggle 用）")
    ap.add_argument("--chunked-loss", action="store_true", help="出力層と損失を 1,024 トークンずつ分けて計算する（語彙25万の出力でメモリが足りない T4 用）")
    ap.add_argument("--optim", default="adamw_torch", help="最適化（T4 の全部の重みの学習では adamw_bnb_8bit）")
    ap.add_argument("--full", action="store_true", help="LoRA ではなく全部の重みを学習する（0.8B のベースの作り直し用。sft/train_lora_modal.py の --full と同じ）")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out_dir = f"{a.out}/{a.run_name}"
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(a.base_model, trust_remote_code=True)
    cls = AutoModelForImageTextToText if a.image_text else AutoModelForCausalLM
    # 全部の重みを学習するときは、小さい学習率の更新が bf16 の丸めで消えないよう fp32 で持つ（計算は bf16）
    model = cls.from_pretrained(a.base_model, trust_remote_code=True, dtype=torch.float32 if a.full or a.fp32 else torch.bfloat16, device_map="cuda")
    if a.full:
        for n, p in model.named_parameters():
            p.requires_grad = "visual" not in n  # 画像エンコーダは固定
        print(f"全部の重みを学習：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    # LoRA は全線形層に入れる（出力層 lm_head は除く。画像入力つきの形では画像エンコーダも除く）
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not n.endswith("lm_head")
             and (not a.image_text or ("language_model" in n and "visual" not in n))]
    targets = sorted({n.split(".")[-1] for n in names})
    if not a.full:
        print("LoRA 対象:", targets, flush=True)
        target_modules = rf".*language_model.*\.({'|'.join(targets)})$" if a.image_text else targets
        model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05,
                                                 target_modules=target_modules, task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    if a.grad_ckpt:
        model.enable_input_require_grads()

    suffix = answer_suffix(tokenizer)
    print("応答の終わりの記号:", repr(suffix), flush=True)
    load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    if a.text:
        # 文章をそのまま読ませる（論文などの追加の事前学習）。全部つなげて max_len ずつに区切り、全トークンを学習する
        def blocks(rows):
            ids = []
            for r in rows:
                ids += tokenizer(r["text"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id or 0]
            return [{"input_ids": ids[i:i + a.max_len], "labels": ids[i:i + a.max_len]} for i in range(0, len(ids) - 64, a.max_len)]
        train_ds, valid_ds = blocks(load(a.train)), blocks(load(a.valid))
    elif a.drop_long:
        # 途中で切った例を学習すると「文の途中で終わる」ことを覚えるので、長すぎる例は使わない
        full = [encode(tokenizer, r["messages"], suffix, 10**9, r.get("thinking")) for r in load(a.train)]
        train_ds = [x for x in full if len(x["input_ids"]) <= a.max_len]
        print(f"上限 {a.max_len} トークンを超える {len(full) - len(train_ds)} 件を除外", flush=True)
    else:
        train_ds = [encode(tokenizer, r["messages"], suffix, a.max_len, r.get("thinking")) for r in load(a.train)]
    if not a.text:
        valid_ds = [encode(tokenizer, r["messages"], suffix, a.max_len, r.get("thinking")) for r in load(a.valid)]
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
        warmup_steps=max(1, int(total_steps * 0.03)), weight_decay=0.0, bf16=not a.fp32, logging_steps=10, eval_strategy="epoch",
        save_strategy="no", report_to=[], seed=a.seed, remove_unused_columns=False, dataloader_num_workers=0,
        gradient_checkpointing=a.grad_ckpt, gradient_checkpointing_kwargs={"use_reentrant": False}, optim=a.optim,
        prediction_loss_only=a.chunked_loss,
    )

    class ChunkedTrainer(Trainer):
        # 出力層（語彙25万）の値を一度に全部作ると T4 のメモリに載らないので、1,024 トークンずつ計算して足す
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.model_accepts_loss_kwargs = True  # 下で全体のトークン数で割るので、勾配の積み重ねの回数ではもう割らない

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            from torch.utils.checkpoint import checkpoint
            base = model.base_model.model if hasattr(model, "base_model") and not a.full else model
            h = base.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state[:, :-1]
            y = inputs["labels"][:, 1:]
            w = base.lm_head.weight

            def part(hc, yc):
                return torch.nn.functional.cross_entropy((hc @ w.t()).float().flatten(0, 1), yc.flatten(), ignore_index=-100, reduction="sum")

            total = h.new_zeros((), dtype=torch.float32)
            for i in range(0, h.size(1), 1024):
                total = total + checkpoint(part, h[:, i:i + 1024], y[:, i:i + 1024], use_reentrant=False)
            n = num_items_in_batch if num_items_in_batch is not None else (y != -100).sum()
            loss = total / n
            return (loss, {"loss": loss}) if return_outputs else loss  # 評価では損失だけを使う（prediction_loss_only）

    trainer = (ChunkedTrainer if a.chunked_loss else Trainer)(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds, data_collator=collate, callbacks=[Log()])
    eval_before = trainer.evaluate()["eval_loss"]
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    eval_after = trainer.evaluate()["eval_loss"]

    if not a.full:
        model.save_pretrained(f"{out_dir}/adapter")
    result = {"run": a.run_name, "base_model": a.base_model, "hp": vars(a), "n_train": len(train_ds), "n_valid": len(valid_ds),
              "train_tokens_per_epoch": n_tokens, "eval_loss_before": eval_before, "eval_loss_after": eval_after,
              "train_seconds": train_sec, "gpu": torch.cuda.get_device_name(), "history": history}
    json.dump(result, open(f"{out_dir}/train_result.json", "w"), ensure_ascii=False, indent=1)
    print(f"検証 loss: {eval_before:.3f} → {eval_after:.3f}（学習 {train_sec / 60:.1f} 分, {result['gpu']}）", flush=True)
    if a.no_merge:
        return
    merged = model.to(torch.bfloat16) if a.full else model.merge_and_unload().to(torch.bfloat16)
    merged.save_pretrained(f"{out_dir}/merged", safe_serialization=True)
    tokenizer.save_pretrained(f"{out_dir}/merged")
    # 独自構造のモデルは、同梱コード（modeling_*.py / configuration_*.py）も統合モデルの隣に置く必要がある
    from huggingface_hub import snapshot_download

    # 元モデルが手元のフォルダ（第1段階で統合したモデルなど）のときは、そこから取る
    snap = a.base_model if os.path.isdir(a.base_model) else snapshot_download(a.base_model, allow_patterns=["*.py", "*.jinja", "*.json", "*.txt"])
    for f in os.listdir(snap):
        # 前処理の設定なども元モデルからそろえる（重みの一覧 index.json は写さない）
        if f.endswith((".py", ".jinja", ".json", ".txt")) and not f.endswith("index.json"):
            dst = f"{out_dir}/merged/{f}"
            if not os.path.exists(dst):
                os.system(f"cp '{snap}/{f}' '{dst}'")


if __name__ == "__main__":
    main()
