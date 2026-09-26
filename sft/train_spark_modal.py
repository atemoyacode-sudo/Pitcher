"""Spark-X2.5-4B 系の LoRA を Modal で学習する（Colab 用の train_lora_colab.py / merge_lora_colab.py をそのまま Modal の中で動かす）。

Spark は同梱コードが transformers 4.57 向けなので、Qwen 用の train_lora_modal.py（transformers 5 系）とは別のイメージを使う。

事前に、元にする LoRA と学習データを Volume に置く:
    modal volume put pitcher-models adapters/spark-ja-4b /adapters/spark-ja-4b
    modal volume put pitcher-models sft/distill_sft.jsonl /data/distill_sft.jsonl
    modal volume put pitcher-models sft/ja_general_valid.jsonl /data/ja_general_valid.jsonl

使い方（第1段階の日本語化 LoRA を統合したモデルに、蒸留データで追加学習する）:
    modal run sft/train_spark_modal.py --run-name spark-ja2-4b --base-adapter /models/adapters/spark-ja-4b \
        --train /models/data/distill_sft.jsonl --valid /models/data/ja_general_valid.jsonl
結果は Volume の /<run名>/adapter と /<run名>/merged に保存される。

すでに統合済みのモデル（Volume 上）を元にする場合は --base-model を使う（例：蒸留後のモデルにキャラクターを学習させる）:
    modal run sft/train_spark_modal.py --run-name spark-pitcher2-4b --base-model /models/spark-ja2-4b/merged \
        --train /models/data/stage2b_train.jsonl --valid /models/data/valid.jsonl --extra "..."
"""

import pathlib
import subprocess

import modal

HERE = pathlib.Path(__file__).resolve().parent
BASE = "XHToken/Spark-X2.5-4B"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.9.1", "transformers==4.57.1", "peft==0.17.1", "accelerate")
    .env({"HF_HUB_CACHE": "/hf-cache", "USE_TF": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(HERE / "train_lora_colab.py", "/root/train_lora_colab.py")
    .add_local_file(HERE / "merge_lora_colab.py", "/root/merge_lora_colab.py")
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
models = modal.Volume.from_name("pitcher-models", create_if_missing=True)
app = modal.App("pitcher-train-spark", image=image)


@app.function(gpu="H100", timeout=6 * 60 * 60, volumes={"/hf-cache": hf_cache, "/models": models})
def train(run_name: str, base_adapter: str, train_file: str, valid_file: str, extra: list[str], base_model: str = ""):
    import os

    base = base_model or BASE
    if base_adapter:
        # 前の段階の LoRA を元モデルに統合したものを、今回の学習の元にする
        name = base_adapter.rstrip("/").split("/")[-1]
        base = f"/models/{name}/merged"
        if not os.path.exists(f"{base}/config.json"):
            subprocess.run(["python", "/root/merge_lora_colab.py", "--base-model", BASE, "--adapter", base_adapter, "--out", base], check=True)
            models.commit()
    subprocess.run(["python", "/root/train_lora_colab.py", "--run-name", run_name, "--base-model", base,
                    "--train", train_file, "--valid", valid_file, "--out", "/models", *extra], check=True)
    models.commit()
    return open(f"/models/{run_name}/train_result.json").read()


@app.local_entrypoint()
def main(run_name: str, train: str, valid: str, base_adapter: str = "", base_model: str = "", extra: str = ""):
    result = train_fn(run_name, base_adapter, train, valid, extra.split(), base_model)
    out = HERE / "runs" / f"{run_name}.json"
    out.write_text(result)
    print(f"{out} に保存しました")


train_fn = train.remote
