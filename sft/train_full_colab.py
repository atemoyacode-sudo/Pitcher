"""Qwen3.5 の全部の重みを学習する（フルファインチューニング）Colab 版。Modal 版（train_lora_modal.py --full）と同じ学習の形。

応答部分だけを学習対象にし、画像エンコーダは固定する。重みは fp32 で持ち、計算は bf16。

    python train_full_colab.py --base-model /content/ckpt --train tengentoppa_train.jsonl --valid tengentoppa_valid.jsonl \
        --out /content/out/qwen08b-tengen --lr 1e-5 --epochs 2 --batch-size 2 --grad-accum 16 --max-len 2048
"""

import argparse
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def encode(tokenizer, messages: list[dict], max_len: int) -> dict:
    """推論時と同じ形（enable_thinking=False の生成プロンプト + 応答）にして、応答部分だけを学習対象にする。"""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    answer = messages[-1]["content"] + "<|im_end|>\n"
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    torch.manual_seed(a.seed)
    # cuDNN の注意計算は入力の形ごとに追加のメモリを使い、Modal でメモリ不足になったので使わない
    torch.backends.cuda.enable_cudnn_sdp(False)
    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    model = AutoModelForImageTextToText.from_pretrained(a.base_model, dtype=torch.float32, device_map="cuda")
    for name, prm in model.named_parameters():
        prm.requires_grad = "visual" not in name

    load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    train_rows, valid_rows = load(a.train), load(a.valid)
    train_ds = [x for x in (encode(tokenizer, r["messages"], 10**9) for r in train_rows) if len(x["input_ids"]) <= a.max_len]
    valid_ds = [x for x in (encode(tokenizer, r["messages"], 10**9) for r in valid_rows) if len(x["input_ids"]) <= a.max_len]
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
            ids[i, :L] = torch.tensor(x["input_ids"])
            labels[i, :L] = torch.tensor(x["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    history = []

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                history.append({"step": state.global_step, **logs})
                print(json.dumps({"step": state.global_step, **logs}), flush=True)

    total_steps = math.ceil(len(train_ds) / (a.batch_size * a.grad_accum)) * a.epochs
    args = TrainingArguments(
        output_dir="/tmp/out", per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(total_steps * 0.03)), weight_decay=0.0, bf16=True,
        logging_steps=10, eval_strategy="epoch", save_strategy="no", report_to=[], seed=a.seed,
        remove_unused_columns=False, dataloader_num_workers=0,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds, data_collator=collate, callbacks=[Log()])
    t0 = time.time()
    eval_before = trainer.evaluate()["eval_loss"]
    trainer.train()
    train_sec = time.time() - t0
    eval_after = trainer.evaluate()["eval_loss"]

    model.to(torch.bfloat16).save_pretrained(f"{a.out}/merged", safe_serialization=True)
    tokenizer.save_pretrained(f"{a.out}/merged")
    # 前処理設定・チャットテンプレートなど、重み以外のファイルを元モデルからそろえる（重みと index はコピーしない）
    if os.path.isdir(a.base_model):
        import shutil

        for f in os.listdir(a.base_model):
            dst = f"{a.out}/merged/{f}"
            if f.endswith((".json", ".jinja", ".txt")) and not f.endswith("index.json") and not os.path.exists(dst):
                shutil.copy(f"{a.base_model}/{f}", dst)
    result = {"base_model": a.base_model, "hp": vars(a), "n_train": len(train_ds), "n_valid": len(valid_ds),
              "train_tokens_per_epoch": n_tokens, "eval_loss_before": eval_before, "eval_loss_after": eval_after,
              "train_seconds": train_sec, "gpu": torch.cuda.get_device_name(), "history": history}
    json.dump(result, open(f"{a.out}/train_result.json", "w"), ensure_ascii=False, indent=1)
    print(f"検証 loss: {eval_before:.3f} → {eval_after:.3f}（学習 {train_sec / 60:.1f} 分, {result['gpu']}）", flush=True)


if __name__ == "__main__":
    main()
