"""先生役（Qwen3.8-27B）の回答から、第1段階の追加学習（蒸留）用の学習データを作る。

除くもの: 途中で切れた回答、中国語が混ざった回答、仮名のない回答、極端に短い回答、AI の感情・正体に触れる回答
（キャラクターの学習の土台になるので、「私は AI なので」という自己認識は教えない）

    python build_distill_sft.py --src distill_Qwen3.8-27B.jsonl --out distill_sft.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "eval"):
    sys.path.insert(0, str(p))
from score import AI_SELF, DENY_FEELINGS, ZH_CHAR  # noqa: E402

KANA = re.compile(r"[぀-ヿ]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, dropped = [], Counter()
    for r in map(json.loads, open(a.src)):
        t = r["output"].strip()
        if r["finish_reason"] != "stop":
            dropped["truncated"] += 1
        elif ZH_CHAR.search(t):
            dropped["chinese"] += 1
        elif not KANA.search(t) or len(t) < 20:
            dropped["no_kana_or_short"] += 1
        elif AI_SELF.search(t) or DENY_FEELINGS.search(t):
            dropped["ai_self"] += 1
        else:
            rows.append({"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": t}],
                         "dataset": "distill", "trait": r["source"], "source_row": r["id"]})
    with open(a.out, "w") as f:
        f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"{len(rows)} 件を {a.out} に保存（除外 {dict(dropped)}）", flush=True)


if __name__ == "__main__":
    main()
