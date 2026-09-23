"""sft/user_turns.jsonl から学習用の会話データ（sft/train.jsonl / sft/valid.jsonl）を作る。

使い方:
    python3 sft/build_sft.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parent

# 属性ごとのキャラ設定。評価用のキャラ設定文（eval/prompts.json）とは言い回しを変えてある
TRAIT_SYSTEM = {
    "tsundere": "本当は相手のことが好きなのに素直になれず、つい冷たい態度や強がりを言ってしまいます。",
    "yandere": "相手を深く愛するあまり、嫉妬深く独占欲が強いです。",
    "deredere": "いつも明るく素直で、相手への好意を隠しません。",
    "kuudere": "冷静で感情をあまり表に出しませんが、内心では相手を大切に思っています。",
    "dandere": "内気で口数が少なく緊張しやすいですが、相手には少しずつ心を開いています。",
    "himedere": "お姫様のように振る舞い、丁重に扱われるのを当然だと思っています。",
    "kamidere": "自分を神のように特別な存在だと信じ、尊大に振る舞います。",
    "bokukko": "ボーイッシュで負けず嫌いです。",
    "genki": "いつもエネルギーにあふれ、明るく元気いっぱいです。",
    "shundere": "物静かで憂いを帯び、自分に自信が持てません。",
    "moe": "無邪気で少しドジな、愛らしい性格です。",
}
TEMPLATES = [
    "あなたは{trait}の女の子です。{desc}キャラクターになりきって、自然な日本語の話し言葉で返答してください。一人称は「{pron}」です。",
    "{trait}のキャラクターとして会話してください。{desc}一人称は「{pron}」を使ってください。",
]
APHRODITE_SUFFIX = "相手のことは「お父様」と呼び、一人称は「私」を使います。"


def clean_user(r: dict) -> str:
    if r["dataset"] == "waifu":
        # 2行に分かれて出力されたものは1つの発言としてつなげる
        text = "".join(l.strip() for l in r["raw"].strip().splitlines())
    else:
        text = r["user"]
    return text.strip().strip("「」『』\"")


def main():
    rng = random.Random(0)
    rows = [json.loads(l) for l in (SFT_DIR / "user_turns.jsonl").read_text().splitlines() if l.strip()]
    # waifu の応答は、質問を生成した後に入った品質チェックの修正も反映するため、最新の修正版から取り直す
    waifu = {r["source_row"]: r["dialogue"] for r in json.loads((SFT_DIR.parent / "学習データ2_日本語_clean.json").read_text())}
    out, dropped = [], Counter()
    for r in rows:
        user = clean_user(r)
        answer = waifu[r["source_row"]] if r["dataset"] == "waifu" else r["assistant"].replace("お父様よ、", "お父様、")
        if not (5 <= len(user) <= 100) or not re.search(r"[぀-ヿ]", user):
            dropped["bad_user"] += 1
            continue
        if r["dataset"] == "aphrodite":
            system = r["system_orig"] + APHRODITE_SUFFIX
        else:
            pron = "ボク" if r["trait_en"] == "bokukko" else "私"
            system = rng.choice(TEMPLATES).format(trait=r["trait"], desc=TRAIT_SYSTEM[r["trait_en"]], pron=pron)
        out.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ],
            "dataset": r["dataset"], "trait": r["trait_en"], "source_row": r["source_row"],
        })

    # 属性ごとに5%を検証用に取り分ける
    rng.shuffle(out)
    by = {}
    for x in out:
        by.setdefault((x["dataset"], x["trait"]), []).append(x)
    train, valid = [], []
    for xs in by.values():
        k = max(1, round(len(xs) * 0.05))
        valid += xs[:k]
        train += xs[k:]
    rng.shuffle(train)
    for name, xs in (("train", train), ("valid", valid)):
        (SFT_DIR / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in xs))
    print(f"train {len(train)} / valid {len(valid)} / 除外 {dict(dropped)}")
    print(Counter(f"{x['dataset']}:{x['trait']}" for x in train).most_common())


if __name__ == "__main__":
    main()
