"""崩れにくくする学習（DPO）のために、0.8B に答えさせる質問を選ぶ。どれも 0.8B の学習に使っていない質問。

- 蒸留用に用意したが学習（sft/distill_sft.jsonl）には使わなかった質問 1,000個（sft/distill_prompts.jsonl）
- 上と同じ質問のうち、物語・詩・作文を書かせるもの 最大300個（長く書くと崩れやすいため）
- 「〇〇の短い物語を書いて」のように直接書かせる頼み方を、型とテーマの組み合わせで 最大120個
- 先生役が作った手本（sft/q08b_v2/gen_chat_modal.py）から、一言のつぶやき 500個と、2往復目の返事を求める続きの会話 300個
- Gemma が採点したが学習には選ばれなかった Tengentoppa の行の質問 1,000個（sft/tengentoppa_clean_pool.jsonl のうち
  tengentoppa_clean_scored_train.jsonl に入っていないもの。崩れやすい長い回答を求める質問を 700個、短い回答の質問を 300個）

どちらも作るときに、評価用の質問に似たもの・AI の感情や正体を問うもの・連絡先を含むものを除いてある。

    python3 sft/q08b_v2/build_dpo_prompts.py   # data/dpo/prompts.jsonl
"""

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return [json.loads(l) for l in (ROOT / "sft" / name).read_text().splitlines() if l.strip()]


def main():
    rng = random.Random(0)
    used = {x["messages"][0]["content"] for x in load("distill_sft.jsonl")}
    distill = [x for x in load("distill_prompts.jsonl") if x["prompt"] not in used]
    rng.shuffle(distill)
    picked = [{"id": f"dpo-d{i:04d}", "prompt": x["prompt"], "from": "distill_prompts"} for i, x in enumerate(distill[:1000])]
    # 長く書くと同じ言葉の繰り返しや前後の食い違いが出やすいので、物語・詩・作文を書かせる質問を足す
    creative_re = re.compile(r"物語|小説|ストーリー|詩を|短歌|俳句|エッセイ|作文|童話|お話")
    creative = [x for x in distill[1000:] if creative_re.search(x["prompt"])]
    picked += [{"id": f"dpo-c{i:04d}", "prompt": x["prompt"], "from": "distill_prompts_creative"} for i, x in enumerate(creative[:300])]
    # 「短い物語を書いて」のように、直接書かせる頼み方も足す（評価用の質問に似たものは除く）
    sys.path.insert(0, str(ROOT / "sft"))
    from build_distill_prompts import bigrams

    ev = [bigrams(it["prompt"]) for it in json.loads((ROOT / "eval" / "prompts.json").read_text())["items"]]
    forms = ["短い物語を書いて", "{t}の短い物語を書いてください", "{t}をテーマにした詩を書いて", "{t}についての短いエッセイを書いてください",
             "{t}の日の日記を書いてみて", "{t}を題材に、子ども向けのお話を作って", "{t}が出てくる短い物語を考えて"]
    themes = ["雨", "夏祭り", "海", "星空", "電車", "図書館", "冬の朝", "友だち", "引っ越し", "ロボット", "パン屋", "公園", "台風",
              "桜", "紅葉", "雪", "灯台", "駄菓子屋", "花火", "旅"]
    made = []
    for f in forms:
        for t in (themes if "{t}" in f else [""]):
            q = f.format(t=t)
            if not any(len(bigrams(q) & e) / max(1, len(bigrams(q) | e)) > 0.3 for e in ev):
                made.append(q)
    rng.shuffle(made)
    picked += [{"id": f"dpo-w{i:04d}", "prompt": q, "from": "template_creative"} for i, q in enumerate(made[:120])]
    train = {x["source_row"] for x in load("tengentoppa_clean_scored_train.jsonl")}
    pool = [x for x in load("tengentoppa_clean_pool.jsonl") if x["source_row"] not in train]
    rng.shuffle(pool)
    for kind, n in (("long", 700), ("short", 300)):
        rows = [x for x in pool if x["kind"] == kind][:n]
        picked += [{"id": f"dpo-t{kind[0]}{i:04d}", "prompt": x["messages"][0]["content"], "from": f"tengentoppa_{kind}"} for i, x in enumerate(rows)]
    # 一言のつぶやきと、続きの会話（sft/q08b_v2/gen_chat_modal.py で先生役が作った手本から。前のやり取りは先生役の返事を使う）
    chat_path = ROOT / "data" / "dpo" / "chat_teacher_clean.jsonl"
    if chat_path.exists():
        chat = [json.loads(l) for l in chat_path.read_text().splitlines()]
        rng.shuffle(chat)
        singles = [c for c in chat if len(c["messages"]) == 2][:500]
        picked += [{"id": f"dpo-s{i:04d}", "prompt": c["messages"][0]["content"], "from": "chat_single"} for i, c in enumerate(singles)]
        multis = [c for c in chat if len(c["messages"]) >= 4][:300]
        for i, c in enumerate(multis):
            ctx = c["messages"][:3]
            text = "\n".join(("ユーザー：" if m["role"] == "user" else "AI：") + m["content"] for m in ctx)
            picked.append({"id": f"dpo-m{i:04d}", "prompt": f"（次の会話の続き）\n{text}", "messages": ctx, "from": "chat_multi"})
    out = ROOT / "data" / "dpo" / "prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in picked))
    print(f"{out.relative_to(ROOT)}：{len(picked)} 個")


if __name__ == "__main__":
    main()
