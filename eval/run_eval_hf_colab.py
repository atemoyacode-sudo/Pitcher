"""Colab のランタイム上で transformers を使って回答を生成する（run_eval_hf_modal.py の Colab 版。生成方法と出力形式は同じ）。

    python run_eval_hf_colab.py eval140 --model /content/out/spark-lora/merged --run-name spark-lora \
        --prompts /content/data/prompts.json --out /content/results
    python run_eval_hf_colab.py replay --model XHToken/Spark-X2.5-4B \
        --prompts /content/data/replay_prompts.jsonl --out /content/results
    python run_eval_hf_colab.py mcqa --model /content/out/spark-ja/merged --run-name spark-ja --out /content/results
    python run_eval_hf_colab.py math --model XHToken/Spark-X2.5-4B --run-name spark-x2.5-4b --out /content/results
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


MATH_TEMPLATES = {
    "ja": "次の問題を解いてください。最後に「答え：数字」の形で答えを書いてください。\n\n{q}",
    "en": "Solve the following problem. At the end, write the answer in the form \"Answer: number\".\n\n{q}",
}


def math_questions() -> list[dict]:
    import pandas as pd

    rows = []
    for lang in ("ja", "en"):
        df = pd.read_parquet(f"https://huggingface.co/api/datasets/juletxara/mgsm/parquet/{lang}/test/0.parquet")
        for i, r in enumerate(df.itertuples()):
            rows.append({"lang": lang, "id": f"mgsm-{i:03d}", "answer": float(r.answer_number), "prompt": MATH_TEMPLATES[lang].format(q=r.question)})
    return rows


def extract_number(text: str):
    """「答え：」「Answer:」の後の数字を優先し、なければ最後に出てくる数字を答えとみなす。"""
    t = text.translate(str.maketrans("０１２３４５６７８９，．－", "0123456789,.-")).replace(",", "")
    m = re.findall(r"(?:答え|Answer)\s*[:：は]?\s*\**\s*\$?\s*(-?\d+(?:\.\d+)?)", t, flags=re.I)
    nums = m or re.findall(r"-?\d+(?:\.\d+)?", t)
    return float(nums[-1]) if nums else None


def mcqa_questions() -> list[dict]:
    import pandas as pd

    ja_t = "次の質問に対して、最も適切な答えを選択肢から1つ選び、番号（1〜5）だけを答えてください。\n\n質問：{q}\n{choices}\n\n答え："
    en_t = "Choose the most appropriate answer to the following question from the options, and reply with only its number (1-5).\n\nQuestion: {q}\n{choices}\n\nAnswer:"
    rows = []
    for r in pd.read_parquet("https://huggingface.co/api/datasets/sbintuitions/JCommonsenseQA/parquet/default/validation/0.parquet").itertuples():
        rows.append({"lang": "ja", "id": str(r.q_id), "question": r.question, "choices": [getattr(r, f"choice{i}") for i in range(5)], "answer": int(r.label) + 1})
    for r in pd.read_parquet("https://huggingface.co/api/datasets/tau/commonsense_qa/parquet/default/validation/0.parquet").itertuples():
        labels, texts = list(r.choices["label"]), list(r.choices["text"])
        rows.append({"lang": "en", "id": r.id, "question": r.question, "choices": texts, "answer": labels.index(r.answerKey) + 1})
    for x in rows:
        t = ja_t if x["lang"] == "ja" else en_t
        x["prompt"] = t.format(q=x["question"], choices="\n".join(f"{i + 1}. {c}" for i, c in enumerate(x["choices"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval140", "replay", "mcqa", "math"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-name", default="")
    ap.add_argument("--prompts", default="")
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
    elif a.mode == "math":
        # 算数の文章題 MGSM（juletxara/mgsm、CC BY-SA 4.0）。同じ250問の英語版と日本語版を、思考なし・貪欲法で解かせる
        qs = math_questions()
        outs = generate(a.model, [[{"role": "user", "content": q["prompt"]}] for q in qs], [dict(do_sample=False, max_new_tokens=768)] * len(qs))
        rows = []
        for q, o in zip(qs, outs):
            pred = extract_number(o["output"])
            rows.append({"model": a.model, "lang": q["lang"], "id": q["id"], "answer": q["answer"], "pred": pred,
                         "correct": pred is not None and abs(pred - q["answer"]) < 1e-6, "finish_reason": o["finish_reason"], "output": o["output"]})
        for lang in ("ja", "en"):
            rs = [r for r in rows if r["lang"] == lang]
            print(f"{a.run_name} MGSM {lang}: 正答率 {sum(r['correct'] for r in rs) / len(rs):.1%}（{len(rs)}問）", flush=True)
        path = f"{a.out}/math_{a.run_name}.jsonl"
    elif a.mode == "mcqa":
        # 日本語（JCommonsenseQA）と英語（CommonsenseQA）の5択問題。bench_mcqa_modal.py と同じ問題・同じ書式
        qs = mcqa_questions()
        outs = generate(a.model, [[{"role": "user", "content": q["prompt"]}] for q in qs], [dict(do_sample=False, max_new_tokens=16)] * len(qs))
        rows = []
        for q, o in zip(qs, outs):
            m = re.search(r"[1-5]", o["output"].translate(str.maketrans("１２３４５", "12345")))
            pred = int(m.group()) if m else None
            rows.append({"model": a.model, "lang": q["lang"], "id": q["id"], "answer": q["answer"], "pred": pred,
                         "correct": pred == q["answer"], "output": o["output"]})
        for lang in ("ja", "en"):
            rs = [r for r in rows if r["lang"] == lang]
            print(f"{a.run_name} {lang}: 正答率 {sum(r['correct'] for r in rs) / len(rs):.1%}（{len(rs)}問）", flush=True)
        path = f"{a.out}/mcqa_{a.run_name}.jsonl"
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
