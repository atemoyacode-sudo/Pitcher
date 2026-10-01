"""「考える長さを短くする」学習（解き直し・強化学習）に使う、答えを機械的に確かめられる問題を集める。

- 数学：zwhe99/DeepMath-103K（MIT）。答えが整数の問題だけ（証明問題は除く）。難しさ（difficulty）の列で分けて選ぶ
- 理科：nvidia/OpenScienceReasoning-2（CC BY 4.0）。A〜J の記号で答える選択問題だけ

評価に使う問題（AIME 2026、MMLU-Pro）と文面が重なる問題は除く（空白を除いた文字の 50-gram が1つでも一致したら除外）。
元データは大きい（合わせて約5GB）ので、必要な列だけを Hugging Face から読む。

    SSL_CERT_FILE=$(python3 -m certifi) python3 sft/build_reasoning_problems.py
    → data/reasoning/problems_pool.jsonl（候補。git 管理外）
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "reasoning"
BASE = "https://huggingface.co/api/datasets/{repo}/parquet/default/{split}/{i}.parquet"
N_MATH, N_SCI = 4000, 1000
SCI_PROMPT = ("Answer the following multiple choice question. Think step by step, then put the letter of the correct option "
              "within \\boxed{{}}.\n\n{body}")
MATH_PROMPT = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
INT = re.compile(r"-?\d+")
PROOF = re.compile(r"\b(prove|show that|justify|disprove)\b", re.I)


def read(repo: str, split: str, n_files: int, columns: list[str]) -> list[dict]:
    import fsspec
    import pyarrow.parquet as pq

    fs = fsspec.filesystem("https")
    rows = []
    for i in range(n_files):
        with fs.open(BASE.format(repo=repo, split=split, i=i), "rb", block_size=2**20) as f:
            rows += pq.read_table(f, columns=columns).to_pylist()
    return rows


def grams(text: str, n: int = 50, step: int = 10) -> set[str]:
    """空白を除いた文字の n-gram。評価の問題は全位置（step=1）、候補は10字おきに取れば、50字以上の重なりは必ず見つかる。"""
    t = re.sub(r"\s+", "", text)
    return {t[i:i + n] for i in range(0, max(1, len(t) - n + 1), step)}


def eval_grams() -> set[str]:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    g = set()
    df = pd.read_parquet(hf_hub_download("MathArena/aime_2026", "data/train-00000-of-00001.parquet", repo_type="dataset"))
    for p in df["problem"]:
        g |= grams(p, step=1)
    for r in read("TIGER-Lab/MMLU-Pro", "test", 1, ["question", "category"]):
        if r["category"] in ("physics", "chemistry", "biology"):
            g |= grams(r["question"], step=1)
    return g


def main():
    rng = random.Random(0)
    bad = eval_grams()
    math = read("zwhe99/DeepMath-103K", "train", 10, ["question", "final_answer", "difficulty"])
    print(f"DeepMath {len(math)} 件、難しさの分布 {sorted(Counter(r['difficulty'] for r in math).items())}")
    keep, dropped, seen = [], Counter(), set()
    for r in math:
        q, a = r["question"].strip(), str(r["final_answer"]).strip()
        if not INT.fullmatch(a):
            dropped["答えが整数でない"] += 1
        elif PROOF.search(q):
            dropped["証明"] += 1
        elif q in seen:
            dropped["重複"] += 1
        elif grams(q) & bad:
            dropped["評価の問題と重なる"] += 1
        else:
            seen.add(q)
            keep.append({"task": "math", "source": "zwhe99/DeepMath-103K", "difficulty": r["difficulty"], "answer": a,
                         "prompt": MATH_PROMPT.format(q=q)})
    print(f"数学：残り {len(keep)} 件、除外 {dict(dropped)}")
    print(f"  難しさの分布 {sorted(Counter(r['difficulty'] for r in keep).items())}")
    rng.shuffle(keep)
    math_pick = keep[:N_MATH]

    sci = read("nvidia/OpenScienceReasoning-2", "train", 3, ["input", "expected_answer"])
    keep, dropped, seen = [], Counter(), set()
    for r in sci:
        body = re.sub(r"^Solve the following problem\..*?inside \\boxed\{\}\.\s*", "", r["input"].strip(), flags=re.S)
        a = (r["expected_answer"] or "").strip()
        if not re.fullmatch(r"[A-J]", a) or not re.search(r"\n\s*A[:.)]", body):
            dropped["記号で答える選択問題でない"] += 1
        elif body in seen:
            dropped["重複"] += 1
        elif grams(body) & bad:
            dropped["評価の問題と重なる"] += 1
        else:
            seen.add(body)
            keep.append({"task": "sci", "source": "nvidia/OpenScienceReasoning-2", "answer": a, "prompt": SCI_PROMPT.format(body=body)})
    print(f"理科：残り {len(keep)} 件、除外 {dict(dropped)}")
    rng.shuffle(keep)
    rows = math_pick + keep[:N_SCI]
    for i, r in enumerate(rows):
        r["id"] = f"{r['task']}-{i:05d}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "problems_pool.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"data/reasoning/problems_pool.jsonl：{len(rows)} 件（数学 {len(math_pick)} / 理科 {min(N_SCI, len(keep))}）")


if __name__ == "__main__":
    main()
