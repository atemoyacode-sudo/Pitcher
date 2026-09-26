"""手元に保存した LoRA（adapters/<run名>）を元モデルに統合し直す。Colab の VM は止めると中身が消えるため、次の学習・評価の前に使う。

    python merge_lora_colab.py --base-model XHToken/Spark-X2.5-4B --adapter /content/adapters/spark-ja-4b --out /content/out/spark-ja-4b/merged
    python sft/merge_lora_colab.py --base-model XHToken/Spark-X2.5-4B --adapter adapters/spark-ja-4b --out tools/models/spark-ja-4b/merged --device cpu   # Mac
"""

import argparse
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", help="Mac では cpu")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.base_model, trust_remote_code=True, dtype=torch.bfloat16, device_map=a.device)
    merged = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    # 独自構造のモデルは、同梱コードも統合モデルの隣に置く
    snap = a.base_model if os.path.isdir(a.base_model) else snapshot_download(a.base_model, allow_patterns=["*.py", "*.jinja", "generation_config.json"])
    for f in os.listdir(snap):
        if (f.endswith((".py", ".jinja")) or f == "generation_config.json") and not os.path.exists(f"{a.out}/{f}"):
            shutil.copy(f"{snap}/{f}", f"{a.out}/{f}")
    print(f"{a.out} に保存しました", flush=True)


if __name__ == "__main__":
    main()
