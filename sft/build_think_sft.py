"""0.8B に思考モード（考えてから答える形）を教えるための学習データを作る。

先生役（Qwen3.8-27B）が日本語で短く解き、正解と一致したものだけを残した理系の手本（sft/gen_sci_modal.py の
data/reasoning/sci_teacher.jsonl）の「解き方」を考える部分に移し、思考ありの形にする。短い思考だけを学ばせ、
考えすぎて止まらなくなる（元の 0.8B は MMLU-Pro 理科の 59% が 32k トークンで切れた）のを防ぐ。

思考なしの会話が崩れないよう、思考なしのデータも混ぜる：
- 同じ理系の手本の一部を、思考なしの形（手順を見せてから答える）で
- v2 の会話の学び直しに使ったデータ（sft/chat_mix_train.jsonl）と、答え方の練習（sft/format_drill.jsonl）から一部

問いの形は3通りに分ける（指示がなければ「答え：」、\\boxed{} を求められたら \\boxed{} で答える練習）。

    python3 sft/build_think_sft.py   # → sft/think_mix_train.jsonl（git 管理外）
"""

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "reasoning" / "sci_teacher.jsonl"
OUT = ROOT / "sft" / "think_mix_train.jsonl"
N_NOTHINK_SCI, N_CHAT, N_DRILL = 1000, 2500, 150
JA_BOXED = "\n\n考えたあと、最後に答えを \\boxed{} の中に書いてください。"
EN_SCI = ("Answer the following multiple choice question. Think step by step, then put the letter of the correct option "
          "within \\boxed{{}}.\n\nQuestion: {q}")
EN_MATH = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def split(row: dict) -> tuple[str, str, str]:
    q = row["messages"][0]["content"]
    sol, ans = row["messages"][1]["content"].rsplit("\n\n答え：", 1)
    return q, sol.strip(), ans.strip()


def think_row(row: dict, style: int) -> dict:
    q, sol, ans = split(row)
    lines = sol.splitlines()
    # 解き方が1行だけのときは、最後の1文をまとめにする（考える部分をそのまま繰り返さないため）
    last = lines[-1].strip() if len(lines) > 1 else lines[0].rstrip("。").rsplit("。", 1)[-1].strip() + "。"
    # 答えの部分は、考えた結果のまとめ（解き方の最後の1行。長すぎるときは省く）と答え
    lead = last + "\n\n" if len(last) <= 200 else ""
    if style == 0:
        user, out = q, f"{lead}答え：{ans}"
    elif style == 1:
        user, out = q + JA_BOXED, f"{lead}答え：\\boxed{{{ans}}}"
    else:
        user = (EN_SCI if row["task"] == "sci" else EN_MATH).format(q=q)
        out = f"{lead}\\boxed{{{ans}}}"
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": out}], "thinking": sol,
            "dataset": "sci_think", "task": row["task"]}


def main():
    rng = random.Random(0)
    sci = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    rows = [think_row(r, i % 3) for i, r in enumerate(sci)]
    for r in rng.sample(sci, N_NOTHINK_SCI):
        rows.append({"messages": r["messages"], "dataset": "sci_nothink", "task": r["task"]})
    chat = [json.loads(l) for l in (ROOT / "sft" / "chat_mix_train.jsonl").read_text().splitlines() if l.strip()]
    rows += [{"messages": r["messages"], "dataset": r["dataset"]} for r in rng.sample(chat, N_CHAT)]
    drill = [json.loads(l) for l in (ROOT / "sft" / "format_drill.jsonl").read_text().splitlines() if l.strip()]
    rows += [{"messages": r["messages"], "dataset": "format_drill"} for r in rng.sample(drill, N_DRILL)]
    rng.shuffle(rows)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{OUT}：{len(rows)} 件", dict(Counter(r["dataset"] for r in rows)))


if __name__ == "__main__":
    main()
