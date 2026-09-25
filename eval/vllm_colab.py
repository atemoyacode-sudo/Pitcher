"""Colab のランタイム上で vLLM を使う：先生役のモデルに回答を作らせる（蒸留）／採点役に回答を採点させる。

vLLM は transformers 5 系が必要なので、Spark 用（transformers 4.57）とは別の環境（/content/vllm_env）で動かす。

    python vllm_colab.py distill --model Qwen/Qwen3.8-27B --prompts distill_prompts.jsonl --out distill_Qwen3.8-27B.jsonl
    python vllm_colab.py judge --runs spark-ja2-4b,spark-ja-4b --results /content/results
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
    ap.add_argument("mode", choices=["distill", "judge"])
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--prompts", default="distill_prompts.jsonl")
    ap.add_argument("--out", default="distill_Qwen3.8-27B.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--runs", default="")
    ap.add_argument("--results", default="/content/results")
    a = ap.parse_args()
    distill(a) if a.mode == "distill" else judge(a)


if __name__ == "__main__":
    main()
