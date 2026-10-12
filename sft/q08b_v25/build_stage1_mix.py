"""v2.5 の1段目の学習データをまとめる。公開の v2 から全部の重みを学習する。

- 理系の手本（思考あり・思考なし）：sft/build_think_sft.py の出力（sft/think_mix_train.jsonl）から。
  先生役（Qwen3.8-27B）が問題を日本語に訳して短く解き、正解と一致したもの（sft/gen_sci_modal.py）
- 日本語のコードの説明・確認・修正：data/code/code_train.jsonl（公開版は Japanese-Code-Review-OSS）
- 会話：v2 の会話の手本と Tengentoppa（sft/chat_mix_train.jsonl から 4,500件）、答え方の練習（400件）

できたデータ 9,842件のうち 150件を検証用にし、残り 9,692件で学習した（公開データの stage1_train / stage1_valid。
このスクリプトで作り直すと中身の集合は一致する。検証用の分け方は公開データの方が実際に使ったもの）。

    python3 sft/q08b_v25/build_stage1_mix.py   # → sft/v25_stage1_mix.jsonl（git 管理外）
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sft" / "v25_stage1_mix.jsonl"


def jl(p):
    return [json.loads(l) for l in (ROOT / p).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rng = random.Random(0)
    rows = []
    mix = jl("sft/think_mix_train.jsonl")
    rows += [r for r in mix if r["dataset"] in ("sci_think", "sci_nothink")]
    code = jl("data/code/code_train.jsonl")
    rows += [{"messages": r["messages"], "dataset": "code_ja"} for r in code]
    chat = [r for r in jl("sft/chat_mix_train.jsonl") if r["dataset"] in ("chat_teacher", "tengentoppa")]
    rows += [{"messages": r["messages"], "dataset": r["dataset"]} for r in rng.sample(chat, 4500)]
    drill = jl("sft/format_drill.jsonl")
    rows += [{"messages": r["messages"], "dataset": "format_drill"} for r in rng.sample(drill, 400)]
    rng.shuffle(rows)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{a.out}：{len(rows)} 件", dict(Counter(r["dataset"] for r in rows)))


if __name__ == "__main__":
    main()
