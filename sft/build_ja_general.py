"""第1段階（日本語化）の学習データを作る：SFT-General-Japanese-60K から、中国語の混入を除き、話題の偏りを均して選ぶ。

元データ（Apache 2.0）:
    OysterCoreAI/SFT-General-Japanese-60K（revision fdab76b7d780806e98bd1006a06c7444bb8e936a）
    = llm-jp/magpie-sft-v1.0（質問は cyberagent/calm3-22b-chat、回答は Qwen/Qwen2.5-32B-Instruct が生成）から選んだもの

事前に data/ja_general/ に元データの JSONL（12ファイル）を置く:
    for i in $(seq -w 0 11); do curl -sL "https://huggingface.co/datasets/OysterCoreAI/SFT-General-Japanese-60K/resolve/fdab76b7d780806e98bd1006a06c7444bb8e936a/data/train-000${i}-of-00012.jsonl" -o "data/ja_general/train-000${i}-of-00012.jsonl"; done

使い方:
    python3 sft/build_ja_general.py   # sft/ja_general_train.jsonl と sft/ja_general_valid.jsonl を作る
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from score import ZH_CHAR  # noqa: E402  評価と同じ基準で中国語の混入を判定する

SRC_DIR = ROOT / "data" / "ja_general"
PER_DOMAIN_CAP = 3000  # 科学・言語・技術・教育などの大きい話題はここまで。小さい話題は全部使う
N_VALID = 500


def main():
    rows = [json.loads(l) for f in sorted(SRC_DIR.glob("train-*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    assert len(rows) == 60000, len(rows)
    kept, dropped = [], Counter()
    for r in rows:
        msgs = r["messages"]
        if [m["role"] for m in msgs] != ["user", "assistant"]:
            dropped["bad_roles"] += 1
            continue
        if any(ZH_CHAR.search(m["content"]) for m in msgs):
            dropped["chinese"] += 1
            continue
        kept.append(r)

    rng = random.Random(0)
    rng.shuffle(kept)
    by_domain = {}
    for r in kept:
        by_domain.setdefault(r["domain"], []).append(r)
    selected = []
    for dom, rs in sorted(by_domain.items()):
        selected += rs[:PER_DOMAIN_CAP]
    rng.shuffle(selected)
    valid, train = selected[:N_VALID], selected[N_VALID:]

    for name, rs in (("train", train), ("valid", valid)):
        out = ROOT / "sft" / f"ja_general_{name}.jsonl"
        with out.open("w") as f:
            for r in rs:
                f.write(json.dumps({"messages": r["messages"], "dataset": "ja_general", "trait": r["domain"],
                                    "source_row": r["id"]}, ensure_ascii=False) + "\n")
    print(f"元データ {len(rows)} 件 → 除外 {dict(dropped)} → 採用 {len(selected)} 件（学習 {len(train)} / 検証 {len(valid)}）")
    print(Counter(r["domain"] for r in selected).most_common())


if __name__ == "__main__":
    main()
