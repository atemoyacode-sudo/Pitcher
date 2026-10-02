"""同じ流れで学習した2つのモデルの重みを、割合を決めて混ぜる（重みの平均。追加の学習はしない）。

0.8B v2 では、知識が強い版（Wikipedia を読ませて第2段階をやり直した qwen08b-wiki-sft）と、会話が自然で崩れにくい版
（手本の会話と DPO まで行った qwen08b-wiki-dpo）を混ぜ、両方の良さを残せるかを調べる。

    python sft/q08b_v2/merge_soup.py A のフォルダ B のフォルダ 出力のフォルダ --b-weight 0.5
"""

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--b-weight", type=float, default=0.5)
    x = ap.parse_args()
    wa, wb = load_file(f"{x.a}/model.safetensors"), load_file(f"{x.b}/model.safetensors")
    w = x.b_weight
    merged = {k: ((1 - w) * wa[k].float() + w * wb[k].float()).to(wa[k].dtype) for k in wa}
    Path(x.out).mkdir(parents=True, exist_ok=True)
    save_file(merged, f"{x.out}/model.safetensors", metadata={"format": "pt"})
    for f in Path(x.b).iterdir():
        if f.name != "model.safetensors" and f.is_file():
            shutil.copy(f, Path(x.out) / f.name)
    print(f"{x.out}：B の割合 {w}")


if __name__ == "__main__":
    main()
