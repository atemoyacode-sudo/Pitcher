"""Colab のランタイム上で vLLM を使う：先生役のモデルに回答を作らせる（蒸留）／採点役に回答を採点させる。

vLLM は transformers 5 系が必要なので、Spark 用（transformers 4.57）とは別の環境（/content/vllm_env）で動かす。

    python vllm_colab.py distill --model Qwen/Qwen3.8-27B --prompts distill_prompts.jsonl --out distill_Qwen3.8-27B.jsonl
    python vllm_colab.py judge --runs spark-ja2-4b,spark-ja-4b --results /content/results
    python vllm_colab.py eval --model /content/out/qwen08b-tengen/merged --run-name qwen08b-tengen   # Qwen3.5 の評価（run_eval_modal.py と同じ）
    python vllm_colab.py eval ... --only emo-none,gen --max-tokens 2048 --temperature 0.7 --presence 0   # 生成設定を変えて比べる
    python vllm_colab.py mcqa --model /content/out/qwen08b-tengen/merged --run-name qwen08b-tengen
"""

import argparse
import json
import os
import re

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# Qwen3.8 のモデルカードの推奨値（thinking なし）
DISTILL_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, max_tokens=1536)


def distill(a):
    from vllm import LLM, SamplingParams

    prompts = [json.loads(l) for l in open(a.prompts) if l.strip()]
    if a.limit:
        prompts = prompts[: a.limit]
    llm = LLM(model=a.model, max_model_len=4096, limit_mm_per_prompt={"image": 0, "video": 0}, gpu_memory_utilization=0.92)
    params = [SamplingParams(**DISTILL_SAMPLING, seed=i) for i in range(len(prompts))]
    outs = llm.chat([[{"role": "user", "content": p["prompt"]}] for p in prompts], params, chat_template_kwargs={"enable_thinking": False})
    with open(a.out, "w") as f:
        for p, o in zip(prompts, outs):
            text = re.sub(r"<think>.*?</think>\s*", "", o.outputs[0].text, flags=re.S).strip()
            f.write(json.dumps({**p, "model": a.model, "output": text, "finish_reason": o.outputs[0].finish_reason}, ensure_ascii=False) + "\n")
    print(f"{len(prompts)} 件を {a.out} に保存しました", flush=True)


# run_eval_modal.py と同じ（Qwen3.5 の推奨値、thinking なし）
QWEN_SAMPLED = dict(temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0, max_tokens=512)
GREEDY = dict(temperature=0.0, max_tokens=256)
N_SAMPLES = {"emotion": 5, "general": 3, "knowledge": 1, "knowledge_en": 1}


def evaluate(a):
    from vllm import LLM, SamplingParams

    items = json.load(open(a.prompts_json))["items"]
    prefixes = [p for p in a.only.split(",") if p]
    if prefixes:
        items = [it for it in items if any(it["id"].startswith(p) for p in prefixes)]
    sampled = dict(QWEN_SAMPLED)
    if a.max_tokens:
        sampled["max_tokens"] = a.max_tokens
    if a.temperature is not None:
        sampled["temperature"] = a.temperature
    if a.presence is not None:
        sampled["presence_penalty"] = a.presence
    llm = LLM(model=a.model, max_model_len=max(4096, sampled["max_tokens"] + 1024), seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    convs, params, meta = [], [], []
    for it in items:
        msgs = ([{"role": "system", "content": it["system"]}] if it["system"] else []) + [{"role": "user", "content": it["prompt"]}]
        cfg = GREEDY if it["category"].startswith("knowledge") else sampled
        for k in range(N_SAMPLES[it["category"]]):
            convs.append(msgs)
            params.append(SamplingParams(**cfg, seed=k))
            meta.append((it, k))
    outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False})
    os.makedirs(a.results, exist_ok=True)
    with open(f"{a.results}/{a.run_name}.jsonl", "w") as f:
        for (it, k), o in zip(meta, outs):
            f.write(json.dumps({"run": a.run_name, "model": a.model, **it, "sample": k, "sampling": sampled,
                                "output": re.sub(r"<think>.*?</think>\s*", "", o.outputs[0].text, flags=re.S),
                                "finish_reason": o.outputs[0].finish_reason}, ensure_ascii=False) + "\n")
    print(f"{len(meta)} 件を {a.results}/{a.run_name}.jsonl に保存しました", flush=True)


def mcqa(a):
    """bench_mcqa_modal.py と同じ5択問題（JCommonsenseQA / CommonsenseQA）。"""
    import pandas as pd
    from vllm import LLM, SamplingParams

    ja_t = "次の質問に対して、最も適切な答えを選択肢から1つ選び、番号（1〜5）だけを答えてください。\n\n質問：{q}\n{choices}\n\n答え："
    en_t = "Choose the most appropriate answer to the following question from the options, and reply with only its number (1-5).\n\nQuestion: {q}\n{choices}\n\nAnswer:"
    qs = []
    for r in pd.read_parquet("https://huggingface.co/api/datasets/sbintuitions/JCommonsenseQA/parquet/default/validation/0.parquet").itertuples():
        qs.append({"lang": "ja", "id": str(r.q_id), "choices": [getattr(r, f"choice{i}") for i in range(5)], "answer": int(r.label) + 1, "q": r.question})
    for r in pd.read_parquet("https://huggingface.co/api/datasets/tau/commonsense_qa/parquet/default/validation/0.parquet").itertuples():
        labels = list(r.choices["label"])
        qs.append({"lang": "en", "id": r.id, "choices": list(r.choices["text"]), "answer": labels.index(r.answerKey) + 1, "q": r.question})
    llm = LLM(model=a.model, max_model_len=4096, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    convs = [[{"role": "user", "content": (ja_t if q["lang"] == "ja" else en_t).format(
        q=q["q"], choices="\n".join(f"{i + 1}. {c}" for i, c in enumerate(q["choices"])))}] for q in qs]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=16), chat_template_kwargs={"enable_thinking": False})
    rows = []
    for q, o in zip(qs, outs):
        text = re.sub(r"<think>.*?</think>\s*", "", o.outputs[0].text, flags=re.S)
        m = re.search(r"[1-5]", text.translate(str.maketrans("１２３４５", "12345")))
        pred = int(m.group()) if m else None
        rows.append({"model": a.model, "lang": q["lang"], "id": q["id"], "answer": q["answer"], "pred": pred, "correct": pred == q["answer"], "output": text})
    os.makedirs(a.results, exist_ok=True)
    with open(f"{a.results}/mcqa_{a.run_name}.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    for lang in ("ja", "en"):
        rs = [r for r in rows if r["lang"] == lang]
        print(f"{a.run_name} {lang}: 正答率 {sum(r['correct'] for r in rs) / len(rs):.1%}（{len(rs)}問）", flush=True)


def judge(a):
    from vllm import LLM, SamplingParams

    from judge_common import CALIBRATION, JUDGE_PROMPT, parse

    items = [dict(c, run="calibration", category="calibration", persona="-", sample=0) for c in CALIBRATION]
    for run in a.runs.split(","):
        for l in open(f"{a.results}/{run}.jsonl"):
            r = json.loads(l)
            if r["category"] in ("emotion", "general"):
                items.append({k: r[k] for k in ("id", "sample", "category", "persona", "system", "prompt", "output")} | {"run": run})
    print(f"{len(items)} 件を採点します", flush=True)
    # Modal 版と同じ採点役・同じ量子化（FP8）。A100 では重みだけ FP8 にして動く
    llm = LLM(model="google/gemma-4-31B-it", max_model_len=4096, quantization="fp8", gpu_memory_utilization=0.92,
              limit_mm_per_prompt={"image": 0})
    convs = [[{"role": "user", "content": JUDGE_PROMPT.format(system=it["system"] or "（なし）", prompt=it["prompt"],
                                                              output=it["output"].strip()[:2500])}] for it in items]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=200), chat_template_kwargs={"enable_thinking": False})
    by_run = {}
    for it, o in zip(items, outs):
        text = o.outputs[0].text
        by_run.setdefault(it["run"], []).append({k: it[k] for k in ("run", "id", "sample", "category", "persona")} | {"judge": parse(text), "raw": text})
    for c, row in zip(CALIBRATION, by_run["calibration"]):
        print(f"校正 {c['id']:22} 期待={c['expect']:12} 採点={row['judge']}", flush=True)
    for run, rows in by_run.items():
        if run != "calibration":
            with open(f"{a.results}/judge_{run}.jsonl", "w") as f:
                f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            print(f"{run}: {len(rows)} 件（採点の読み取り失敗 {sum(r['judge'] is None for r in rows)} 件）", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["distill", "judge", "eval", "mcqa"])
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--prompts", default="distill_prompts.jsonl")
    ap.add_argument("--out", default="distill_Qwen3.8-27B.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--runs", default="")
    ap.add_argument("--results", default="/content/results")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--prompts-json", default="prompts.json")
    ap.add_argument("--only", default="")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--presence", type=float, default=None)
    a = ap.parse_args()
    {"distill": distill, "judge": judge, "eval": evaluate, "mcqa": mcqa}[a.mode](a)


if __name__ == "__main__":
    main()
