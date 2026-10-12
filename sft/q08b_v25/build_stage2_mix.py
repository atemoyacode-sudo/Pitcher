"""v2.5 の2段目の学習データをまとめる。1段目のモデルから続けて学習する。

1段目だけのモデルは、思考ありで理系以外の質問をすると、96% で答えを考える部分に書いたまま終わった（思考ありの手本が理系だけだったため）。
そこで、会話・コード・答え方の練習の質問と答えはそのままに、間に入る短い思考だけを先生役に書かせた 1,600件（gen_think_chat_kaggle.py）と、
1段目のデータからの復習 3,000件（理系 1,500件・それ以外 1,500件）を混ぜる。

    python3 sft/q08b_v25/build_stage2_mix.py \\
        --stage1 <1段目の学習データ> --think-chat <gen_think_chat_kaggle.py の出力> --out sft/v25_stage2_train.jsonl
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True, help="1段目の学習データ（公開データの stage1_train と同じもの）")
    ap.add_argument("--think-chat", required=True, help="gen_think_chat_kaggle.py の出力")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    base = jl(a.stage1)
    tc = jl(a.think_chat)
    add = [{"messages": r["messages"], "thinking": r["thinking"], "dataset": "chat_think_" + r["dataset"]}
           for r in tc if r.get("thinking") and "error" not in r and r["dataset"] != "self_q"]
    # 1段目の作り直しの案（v2 から全部を1回で学習）と同じ乱数の使い方にそろえる（公開データの stage2_train を再現するため）
    rng.shuffle(base + add)
    sci = [r for r in base if r["dataset"] in ("sci_think", "sci_nothink")]
    other = [r for r in base if r["dataset"] not in ("sci_think", "sci_nothink")]
    rows = add + rng.sample(sci, 1500) + rng.sample(other, 1500)
    rng.shuffle(rows)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{a.out}：{len(rows)} 件", dict(Counter(r["dataset"] for r in rows)))


if __name__ == "__main__":
    main()
