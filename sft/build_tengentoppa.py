"""Tengentoppa-sft-v1.0 から、長い回答 9,000件と短い回答 3,000件を取り出して学習データを作る。

蒸留後の Qwen3.5-0.8B（qwen08b-distill）は、学習データがほぼすべて長文（中央値 約1,900字）だったため、
どんな質問にも長く書き、長く書くほど話が破綻していた。質問に合った長さで答えることを教えるため、
短い回答を混ぜる（長文 3 : 短文 1）。

元データ：DeL-TaiseiOzaki/Tengentoppa-sft-v1.0（CC BY 4.0、revision 781a718ac17e070fb0135c5dbc3ac9ae7790758a）。
日本語の指示データ16〜17個を統合したもので、行ごとの出どころの列はない。

除くもの:
- 続きの文脈（input）がある行、指示が長すぎる行
- 評価用の質問に似た指示（文字の2-gram の重なり）、AI の感情・正体を問う指示
- 回答に中国語（日本語では使わない簡体字）、AI としての自己認識・感情の否定、電話番号・住所・外部 URL・メールアドレスを含む行
- 「申し訳ありませんが」で始まる回答（できないことの説明が中心で、会話の受け答えの例にならない）
- 同じ指示の重複

使い方:
    python3 sft/build_tengentoppa.py   # sft/tengentoppa_train.jsonl（12,000件）と sft/tengentoppa_valid.jsonl（200件）
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "sft"))
from build_distill_prompts import EXCLUDE, bigrams  # noqa: E402
from build_distill_release import reason  # noqa: E402
from score import AI_SELF, DENY_FEELINGS, ZH_CHAR  # noqa: E402

SRC = ROOT / "data" / "tengentoppa" / "processed_dataset.json"
N_LONG, N_SHORT = 9000, 3000
N_VALID_LONG, N_VALID_SHORT = 150, 50
LONG = (400, 2000)   # 回答の文字数
SHORT = (20, 200)
KANA = re.compile(r"[ぁ-んァ-ン]")
END = re.compile(r"[。！？!?」』）)♪～〜…w]$")


def main():
    rng = random.Random(0)
    rows = json.loads(SRC.read_text())
    eval_bg = [bigrams(it["prompt"]) for it in json.loads((ROOT / "eval" / "prompts.json").read_text())["items"]]
    order = list(range(len(rows)))
    rng.shuffle(order)

    need = {"long": N_LONG + N_VALID_LONG, "short": N_SHORT + N_VALID_SHORT}
    picked = {"long": [], "short": []}
    seen, dropped = set(), Counter()
    for i in order:
        if all(len(picked[k]) >= need[k] for k in need):
            break
        r = rows[i]
        ins, out = (r.get("instruction") or "").strip(), (r.get("output") or "").strip()
        n = len(out)
        kind = "long" if LONG[0] <= n <= LONG[1] else "short" if SHORT[0] <= n <= SHORT[1] else None
        if kind is None or len(picked[kind]) >= need[kind]:
            continue
        if (r.get("input") or "").strip():
            dropped["input"] += 1
        elif not (5 <= len(ins) <= 800) or ins in seen:
            dropped["instruction"] += 1
        elif EXCLUDE.search(ins):
            dropped["ai_question"] += 1
        elif not KANA.search(out) or ZH_CHAR.search(ins + out):
            dropped["language"] += 1
        elif AI_SELF.search(out) or DENY_FEELINGS.search(out):
            dropped["ai_self"] += 1
        elif out.startswith("申し訳"):
            dropped["apology"] += 1
        elif reason(ins + "\n" + out):
            dropped["contact_info"] += 1
        elif kind == "short" and not END.search(out):
            dropped["short_unfinished"] += 1
        elif any(len(bigrams(ins) & e) / max(1, len(bigrams(ins) | e)) > 0.3 for e in eval_bg):
            dropped["like_eval"] += 1
        else:
            seen.add(ins)
            picked[kind].append({"messages": [{"role": "user", "content": ins}, {"role": "assistant", "content": out}],
                                 "dataset": "tengentoppa", "kind": kind, "source_row": i})

    train = picked["long"][N_VALID_LONG:] + picked["short"][N_VALID_SHORT:]
    valid = picked["long"][:N_VALID_LONG] + picked["short"][:N_VALID_SHORT]
    rng.shuffle(train)
    for name, data in (("train", train), ("valid", valid)):
        (ROOT / "sft" / f"tengentoppa_{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data))
    lens = {k: sorted(len(x["messages"][1]["content"]) for x in v) for k, v in picked.items()}
    print(f"学習 {len(train)} 件（長文 {N_LONG} / 短文 {N_SHORT}）、検証 {len(valid)} 件")
    print(f"回答の文字数の中央値：長文 {lens['long'][len(lens['long']) // 2]}、短文 {lens['short'][len(lens['short']) // 2]}")
    print(f"除外 {dict(dropped)}")


if __name__ == "__main__":
    main()
