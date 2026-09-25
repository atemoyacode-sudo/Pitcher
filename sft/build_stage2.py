"""第2段階（日本語化したモデルにキャラクターを学習させる）の学習データを作る。

混ぜるもの:
1. キャラクターのセリフ（sft/train.jsonl、2,470件）：ヤンデレ・ツンデレなど属性ごとの人格
2. Cute_Synthetic_smoltalk_jp_sft（3,000件）：「やわらかくかわいい口調の女の子」の設定付き。
   キャラの口調のまま質問にきちんと長く答える例。セリフだけで学習すると何を聞かれても短く返すようになった問題への対策
3. SFT-General-Japanese-60K のうち第1段階で使わなかったもの（1,500件）：キャラ設定なし。
   設定がないときは普通のアシスタントのまま答えるための例（前回のリプレイの代わり。回答に中国語が混ざらない）

元データ（どちらも Apache 2.0）:
    RikkaBotan/Cute_Synthetic_smoltalk_jp_sft（revision 77a5e7a0439297d7841de3aac61d1f9fc43c3e0b、
        質問は llm-jp/magpie-sft-v1.0、回答は DeepSeek が生成。モデルのバージョンは記載なし）
    OysterCoreAI/SFT-General-Japanese-60K（sft/build_ja_general.py を参照）

事前に data/cute/data_00001.parquet を置く:
    curl -sL https://huggingface.co/datasets/RikkaBotan/Cute_Synthetic_smoltalk_jp_sft/resolve/77a5e7a0439297d7841de3aac61d1f9fc43c3e0b/data_00001.parquet -o data/cute/data_00001.parquet

使い方:
    python3 sft/build_stage2.py   # sft/stage2_train.jsonl を作る（検証は sft/valid.jsonl をそのまま使う）
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from score import AI_SELF, DENY_FEELINGS, ZH_CHAR  # noqa: E402  評価と同じ基準で除外する

N_CUTE = 3000
N_GENERAL = 1500
CUTE_SYSTEMS = [
    "あなたは、やわらかくてかわいらしい口調で話す、やさしい女の子です。相手の質問にはきちんと答えつつ、親しみやすい話し言葉で返答してください。一人称は「私」です。",
    "かわいくて穏やかな女の子として会話してください。ていねいに、でも堅くなりすぎない、ふんわりした話し言葉で答えてください。一人称は「私」を使ってください。",
]
KANA = re.compile(r"[぀-ヿ]")


def main():
    rng = random.Random(0)

    # 1. キャラクターのセリフ
    persona = [json.loads(l) for l in (ROOT / "sft" / "train.jsonl").read_text().splitlines() if l.strip()]

    # 2. Cute
    msgs = pd.read_parquet(ROOT / "data" / "cute" / "data_00001.parquet")["messages"].tolist()
    dropped = Counter()
    cute_ok = []
    for i, ms in enumerate(msgs):
        ms = [{"role": m["role"], "content": m["content"]} for m in ms]
        if [m["role"] for m in ms] != ["user", "assistant"]:
            dropped["bad_roles"] += 1
            continue
        a = ms[-1]["content"]
        if any(ZH_CHAR.search(m["content"]) for m in ms):
            dropped["chinese"] += 1
        elif not KANA.search(a):
            dropped["no_kana"] += 1
        elif re.search(r"俺|僕", a):
            dropped["ore_boku"] += 1  # 女の子の設定なので、一人称がずれるものは使わない
        elif AI_SELF.search(a) or DENY_FEELINGS.search(a):
            dropped["ai_self"] += 1  # キャラの学習で「私は AI」という自己認識を教えないため
        else:
            cute_ok.append((i, ms))
    rng.shuffle(cute_ok)
    cute = [{"messages": [{"role": "system", "content": rng.choice(CUTE_SYSTEMS)}] + ms,
             "dataset": "cute", "trait": "cute", "source_row": i} for i, ms in cute_ok[:N_CUTE]]

    # 3. 一般の日本語（第1段階の学習・検証に使っていないもの）
    used = {json.loads(l)["source_row"] for f in ("ja_general_train.jsonl", "ja_general_valid.jsonl")
            for l in (ROOT / "sft" / f).read_text().splitlines() if l.strip()}
    general_rows = [json.loads(l) for f in sorted((ROOT / "data" / "ja_general").glob("train-*.jsonl"))
                    for l in f.read_text().splitlines() if l.strip()]
    general_ok = [r for r in general_rows if r["id"] not in used
                  and not any(ZH_CHAR.search(m["content"]) for m in r["messages"])]
    rng.shuffle(general_ok)
    general = [{"messages": r["messages"], "dataset": "ja_general", "trait": r["domain"], "source_row": r["id"]}
               for r in general_ok[:N_GENERAL]]

    rows = persona + cute + general
    rng.shuffle(rows)
    out = ROOT / "sft" / "stage2_train.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"キャラ {len(persona)} + Cute {len(cute)}（候補 {len(cute_ok)}、除外 {dict(dropped)}）+ 一般 {len(general)} = {len(rows)} 件 → {out.name}")


if __name__ == "__main__":
    main()
