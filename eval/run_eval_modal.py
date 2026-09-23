"""学習前後で同じ評価プロンプトを Modal 上の vLLM で生成し、結果を results/<run>.jsonl に保存する。

使い方:
    modal run eval/run_eval_modal.py --run-name base
    modal run eval/run_eval_modal.py --run-name lora-v1 --model /models/lora-v1-merged
    modal run eval/run_eval_modal.py --run-name base --only emo-1p-   # 一部の質問だけ回して既存の結果に追加
"""

import json
import pathlib

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

# Qwen3.5 モデルカード推奨値（non-thinking, text）
SAMPLED = dict(temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0, max_tokens=512)
GREEDY = dict(temperature=0.0, max_tokens=256)
N_SAMPLES = {"emotion": 5, "general": 3, "knowledge": 1}


@app.function(gpu="L4", timeout=30 * 60, volumes={"/hf-cache": hf_cache, "/models": models})
def generate(items: list[dict], model: str, seed: int) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        max_model_len=4096,
        seed=seed,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )

    convs, params, meta = [], [], []
    for it in items:
        msgs = ([{"role": "system", "content": it["system"]}] if it["system"] else []) + [
            {"role": "user", "content": it["prompt"]}
        ]
        cfg = GREEDY if it["category"] == "knowledge" else SAMPLED
        for k in range(N_SAMPLES[it["category"]]):
            convs.append(msgs)
            params.append(SamplingParams(**cfg, seed=seed + k))
            meta.append((it, k))

    outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False})
    return [
        {**it, "sample": k, "output": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason}
        for (it, k), o in zip(meta, outs)
    ]


@app.local_entrypoint()
def main(run_name: str = "base", model: str = MODEL_ID, seed: int = 0, only: str = "", gpu: str = "L4"):
    items = json.loads((HERE / "prompts.json").read_text())["items"]
    prefixes = [p for p in only.split(",") if p]
    if prefixes:
        items = [it for it in items if any(it["id"].startswith(p) for p in prefixes)]
    rows = [{"run": run_name, "model": model, **r} for r in generate.with_options(gpu=gpu).remote(items, model, seed)]

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
