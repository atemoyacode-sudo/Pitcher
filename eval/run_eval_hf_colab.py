"""Colab のランタイム上で transformers を使って回答を生成する（run_eval_hf_modal.py の Colab 版。生成方法と出力形式は同じ）。

    python run_eval_hf_colab.py eval140 --model /content/out/spark-lora/merged --run-name spark-lora \
        --prompts /content/data/prompts.json --out /content/results
    python run_eval_hf_colab.py replay --model XHToken/Spark-X2.5-4B \
        --prompts /content/data/replay_prompts.jsonl --out /content/results
"""

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SAMPLED = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=0, max_new_tokens=512)  # Spark-X2.5 の推奨値
GREEDY = dict(do_sample=False, max_new_tokens=256)
N_SAMPLES = {"emotion": 5, "general": 3, "knowledge": 1, "knowledge_en": 1}


def generate(model_id: str, convs: list[list[dict]], cfgs: list[dict], seed: int = 0, batch_size: int = 48) -> list[dict]:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False) for c in convs]
    order = sorted(range(len(texts)), key=lambda i: (json.dumps(cfgs[i], sort_keys=True), len(texts[i])))
    results = [None] * len(texts)
    for b in range(0, len(order), batch_size):
        groups = {}
        for i in order[b : b + batch_size]:
            groups.setdefault(json.dumps(cfgs[i], sort_keys=True), []).append(i)
        for cfg_json, ids in groups.items():
            torch.manual_seed(seed + b)
            enc = tok([texts[i] for i in ids], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, **json.loads(cfg_json), pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            for k, i in enumerate(ids):
                gen = out[k, enc["input_ids"].shape[1]:]
                text = tok.decode(gen, skip_special_tokens=True)
                results[i] = {"output": re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip(),
                              "finish_reason": "stop" if (gen == tok.eos_token_id).any().item() else "length"}
        print(f"{min(b + batch_size, len(order))}/{len(order)}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval140", "replay"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-name", default="")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="/content/results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.mode == "eval140":
        items = json.load(open(a.prompts))["items"]
        convs, cfgs, meta = [], [], []
        for it in items:
            msgs = ([{"role": "system", "content": it["system"]}] if it["system"] else []) + [{"role": "user", "content": it["prompt"]}]
            for k in range(N_SAMPLES[it["category"]]):
                convs.append(msgs)
                cfgs.append(GREEDY if it["category"].startswith("knowledge") else SAMPLED)
                meta.append((it, k))
        outs = generate(a.model, convs, cfgs, a.seed)
        rows = [{"run": a.run_name, "model": a.model, **it, "sample": k, **o} for (it, k), o in zip(meta, outs)]
        path = f"{a.out}/{a.run_name}.jsonl"
    else:
        # リプレイ：一般的な依頼文に元モデル自身が答えたもの（sft/gen_replay_modal.py と同じ。回答の上限は 1024 トークン）
        prompts = [json.loads(l) for l in open(a.prompts) if l.strip()]
        cfg = dict(SAMPLED, max_new_tokens=1024)
        outs = generate(a.model, [[{"role": "user", "content": p["prompt"]}] for p in prompts], [cfg] * len(prompts), a.seed, batch_size=32)
        rows = [{**p, "model": a.model, **o} for p, o in zip(prompts, outs)]
        path = f"{a.out}/replay_{a.model.rstrip('/').split('/')[-1]}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} 件を {path} に保存しました", flush=True)


if __name__ == "__main__":
    main()
