"""回答の「前後の一貫性」「日本語の自然さ」「質問への適合」「キャラ設定への適合」を、別系統の大きなモデルに採点させる。

採点役は Gemma 4 31B（Qwen3.5 とも MiniCPM5 とも別系統。採点役が自分と同じ系統のモデルをひいきする偏りを避けるため）。
正規表現では測れない「話が前後でつながっているか」を数値にするのが目的。

最初に、答えがわかっている校正用の回答（一貫している・前後で矛盾する・文法が崩れている・質問と無関係）を採点させ、
採点役が期待どおりに区別できるかを確かめる。

使い方:
    modal run eval/judge_modal.py --runs base,minicpm5-1b,base-4b
    → eval/results/judge_<run>.jsonl と、集計の表示
"""

import json
import pathlib

import modal

from judge_common import CALIBRATION, JUDGE_PROMPT, parse

HERE = pathlib.Path(__file__).parent
JUDGE_MODEL = "google/gemma-4-31B-it"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0")
    .env({"HF_HUB_CACHE": "/hf-cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("judge_common")
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
app = modal.App("pitcher-judge", image=image)

@app.function(gpu="H100", timeout=2 * 60 * 60, volumes={"/hf-cache": hf_cache})
def judge(items: list[dict]) -> list[str]:
    from vllm import LLM, SamplingParams

    # bf16 のままだと重みで GPU メモリがほぼ埋まり、同時に1〜2件しか採点できない（A100 80GB で確認）。
    # 重みを FP8 にして KV キャッシュの空きを作る
    llm = LLM(model=JUDGE_MODEL, max_model_len=4096, quantization="fp8", gpu_memory_utilization=0.92,
              limit_mm_per_prompt={"image": 0})
    convs = [[{"role": "user", "content": JUDGE_PROMPT.format(system=it["system"] or "（なし）", prompt=it["prompt"],
                                                              output=it["output"].strip()[:2500])}] for it in items]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=200), chat_template_kwargs={"enable_thinking": False})
    return [o.outputs[0].text for o in outs]


@app.local_entrypoint()
def main(runs: str, categories: str = "emotion,general"):
    cats = categories.split(",")
    items = [dict(c, run="calibration", category="calibration", persona="-", sample=0) for c in CALIBRATION]
    for run in runs.split(","):
        for l in (HERE / "results" / f"{run}.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r["category"] in cats:
                items.append({"run": run, "id": r["id"], "sample": r["sample"], "category": r["category"],
                              "persona": r["persona"], "system": r["system"], "prompt": r["prompt"], "output": r["output"]})
    print(f"{len(items)} 件を採点します")
    outs = judge.remote(items)

    by_run = {}
    for it, text in zip(items, outs):
        by_run.setdefault(it["run"], []).append({k: it[k] for k in ("run", "id", "sample", "category", "persona")}
                                                | {"judge": parse(text), "raw": text})
    for c, row in zip(CALIBRATION, by_run["calibration"]):
        print(f"校正 {c['id']:22} 期待={c['expect']:12} 採点={row['judge']}")
    for run, rows in by_run.items():
        if run == "calibration":
            continue
        (HERE / "results" / f"judge_{run}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        bad = sum(r["judge"] is None for r in rows)
        print(f"{run}: {len(rows)} 件（採点の読み取り失敗 {bad} 件）")
