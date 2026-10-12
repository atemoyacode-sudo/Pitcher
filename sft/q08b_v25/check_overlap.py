"""v2.5 の学習データ（公開版）と評価の問題の文面が重なっていないかを確かめる。

- MMLU-Pro（test 全体、約1.2万問）：空白を除いた文字の 50-gram が1つでも一致したら「重なり」
  （問題を集めた段階でも同じ基準で除いている。sft/build_reasoning_problems.py）
- MGSM 日本語（250問）・JCommonsenseQA（validation 1,119問）：空白を除いた文字の 20-gram が1つでも一致したら「重なり」
学習データは、ユーザーの発言・思考・答えのすべてを調べる。

    python3 sft/q08b_v25/check_overlap.py   # → 重なりの件数と例（data/release/q08b_v25_data/overlap_check.json）
"""

import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "release" / "q08b_v25_data"


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def grams(s: str, n: int, step: int = 1) -> set:
    """記号だけの並び（「-----」の罫線など）は、別の文章でも一致してしまうので数えない（文字・数字が半分以上のものだけ）。"""
    s = norm(s)
    out = {s[i:i + n] for i in range(0, max(1, len(s) - n + 1), step)} if len(s) >= n else set()
    return {g for g in out if sum(c.isalnum() for c in g) * 2 >= n}


def main():
    evals = {}
    df = pd.read_parquet("https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro/parquet/default/test/0.parquet")
    evals["MMLU-Pro"] = (50, [q for q in df["question"]])
    df = pd.read_parquet(hf_hub_download("juletxara/mgsm", "ja/test-00000-of-00001.parquet", repo_type="dataset"))
    evals["MGSM-ja"] = (20, list(df["question"]))
    try:
        df = pd.read_parquet("https://huggingface.co/api/datasets/sbintuitions/JCommonsenseQA/parquet/default/validation/0.parquet")
        evals["JCommonsenseQA"] = (20, list(df["question"]))
    except Exception as e:  # 取れなかったら、その旨を残す
        evals["JCommonsenseQA"] = (20, [])
        print("JCommonsenseQA を取得できませんでした", e)
    rows = [json.loads(l) for f in sorted(DATA.glob("stage*.jsonl")) for l in f.read_text().splitlines()]
    texts = ["\n".join([m["content"] for m in r["messages"]] + [r.get("thinking", "")]) for r in rows]
    res = {}
    for name, (n, qs) in evals.items():
        g = set().union(*(grams(q, n) for q in qs)) if qs else set()
        hits = [i for i, t in enumerate(texts) if g & grams(t, n)]
        res[name] = {"評価の問題数": len(qs), "基準": f"空白を除いた文字の {n}-gram", "重なった学習データの件数": len(hits),
                     "例": [texts[i][:200] for i in hits[:3]]}
        print(name, len(qs), "問 → 重なり", len(hits), "件")
    (DATA / "overlap_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
