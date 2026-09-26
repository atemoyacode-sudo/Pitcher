"""学習前後で同じ評価プロンプトを Modal 上の vLLM で生成し、結果を results/<run>.jsonl に保存する。

使い方:
    modal run eval/run_eval_modal.py --run-name base
    modal run eval/run_eval_modal.py --run-name lora-v1 --model /models/lora-v1-merged
    modal run eval/run_eval_modal.py --run-name base --only emo-1p-   # 一部の質問だけ回して既存の結果に追加
    modal run eval/run_eval_modal.py --run-name base-4b-long --model Qwen/Qwen3.5-4B --only emo-none,gen --max-tokens 2048   # 上限を上げた再テスト
"""

import json
import pathlib
import re

import modal

MODEL_ID = "Qwen/Qwen3.5-0.8B"
HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0")
    # FlashInfer のサンプラーは実行時に nvcc でビルドしようとして落ちるので使わない
    .env({"HF_HUB_CACHE": "/hf-cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
models = modal.Volume.from_name("pitcher-models", create_if_missing=True)

app = modal.App("pitcher-eval", image=image)

# 各モデルのモデルカード推奨値（thinking なし）。学習したモデルは、パスに元モデルの系統名（minicpm / spark）を含めること
SAMPLED = dict(temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0, max_tokens=512)  # Qwen3.5
SAMPLED_MINICPM = dict(temperature=0.7, top_p=0.95, max_tokens=512)  # MiniCPM5
SAMPLED_SPARK = dict(temperature=1.0, top_p=0.95, max_tokens=512)  # Spark-X2.5
GREEDY = dict(temperature=0.0, max_tokens=256)
N_SAMPLES = {"emotion": 5, "general": 3, "knowledge": 1, "knowledge_en": 1}


@app.function(gpu="L4", timeout=30 * 60, volumes={"/hf-cache": hf_cache, "/models": models})
def generate(items: list[dict], model: str, seed: int, max_tokens: int = 0) -> list[dict]:
    from vllm import LLM, SamplingParams

    name = model.lower()
    if "minicpm" in name:
        kwargs, sampled = {}, SAMPLED_MINICPM
    elif "spark" in name:
        # Spark-X2.5 は独自構造で、モデルに同梱のコード（modeling_spark.py、中身は確認済み）を実行する必要がある
        kwargs, sampled = {"trust_remote_code": True}, SAMPLED_SPARK
    else:
        # Qwen3.5 は画像も扱うモデルなので、画像・動画の入力枠を 0 にしておく
        kwargs, sampled = {"limit_mm_per_prompt": {"image": 0, "video": 0}}, SAMPLED
    if max_tokens:  # 回答の長さの上限を変える（打ち切りの影響を調べる再テスト用）
        sampled = {**sampled, "max_tokens": max_tokens}
    llm = LLM(model=model, max_model_len=max(4096, max_tokens + 1024), seed=seed, **kwargs)

    convs, params, meta = [], [], []
    for it in items:
        msgs = ([{"role": "system", "content": it["system"]}] if it["system"] else []) + [
            {"role": "user", "content": it["prompt"]}
        ]
        cfg = GREEDY if it["category"].startswith("knowledge") else sampled
        for k in range(N_SAMPLES[it["category"]]):
            convs.append(msgs)
            params.append(SamplingParams(**cfg, seed=seed + k))
            meta.append((it, k))

    outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False})
    return [
        # thinking なしでも空の <think></think> を出すモデルがあるので取り除く
        {**it, "sample": k, "output": re.sub(r"<think>.*?</think>\s*", "", o.outputs[0].text, flags=re.S),
         "finish_reason": o.outputs[0].finish_reason}
        for (it, k), o in zip(meta, outs)
    ]


@app.local_entrypoint()
def main(run_name: str = "base", model: str = MODEL_ID, seed: int = 0, only: str = "", gpu: str = "L4", max_tokens: int = 0):
    items = json.loads((HERE / "prompts.json").read_text())["items"]
    prefixes = [p for p in only.split(",") if p]
    if prefixes:
        items = [it for it in items if any(it["id"].startswith(p) for p in prefixes)]
    rows = [{"run": run_name, "model": model, **r} for r in generate.with_options(gpu=gpu).remote(items, model, seed, max_tokens)]

    out = HERE / "results" / f"{run_name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    if prefixes and out.exists():
        # 既存の結果のうち、今回回した質問だけ差し替える
        new_keys = {(r["id"], r["sample"]) for r in rows}
        old = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        rows = [r for r in old if (r["id"], r["sample"]) not in new_keys] + rows
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} 件を {out} に保存しました")
