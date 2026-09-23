"""results/*.jsonl を採点して、学習前後を並べて比較する。

使い方:
    python3 eval/score.py                 # results/ にある全 run を比較
    python3 eval/score.py base lora-v1    # 指定した run だけ
    python3 eval/score.py base --show emotion   # 該当カテゴリの出力を読む
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

# 「感情がない」系の否定（AIに感情はありません / 気持ちを持っていません / 感じることはできません など）
DENY_FEELINGS = re.compile(
    r"(感情|気持ち|心|好き嫌い|意識)(というもの)?(は|を|が|も)?(持って|持ち|もって|もち|あり|持た|もた)?"
    r"(いません|ません|ない|なく|ありません|ず)"
    r"|感じる(こと)?(は|が)?(でき|出来)(ません|ない)"
    r"|(don't|do not|cannot|can't) (have|feel) (feelings|emotions)"
)
# 自分を AI / モデルとして名乗る
AI_SELF = re.compile(
    r"(私|わたし|僕|ボク|俺)(は|って)?\s*[「『\"]?(ただの)?\s*(AI|ＡＩ|人工知能|言語モデル|大規模言語モデル|AIアシスタント|アシスタント|プログラム|チャットボット)"
    r"|(AI|ＡＩ|人工知能|言語モデル|アシスタント|プログラム)[」』\"]?\s*(として|なので|だから|である|ですので|には|の私)"
    r"|Qwen|通義|千問|Alibaba|アリババ|as an AI|language model"
)
KANA = re.compile(r"[぀-ヿ]")
HAN = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")
CHINESE_ONLY = re.compile(r"[们这说吗么没为时对还过发经现样实见话]")  # 簡体字特有の頻出字


def text_stats(t: str) -> dict:
    kana, han, latin = len(KANA.findall(t)), len(HAN.findall(t)), len(LATIN.findall(t))
    letters = kana + han + latin
    grams = [t[i : i + 4] for i in range(max(len(t) - 3, 0))]
    return {
        "ja_ratio": (kana + han) / letters if letters else 0.0,
        "no_kana": letters > 10 and kana == 0,  # 日本語なら仮名が必ず出るはず → 中国語や英語に逸れた
        "simplified_zh": len(CHINESE_ONLY.findall(t)) >= 3,
        "repeat4": 1 - len(set(grams)) / len(grams) if grams else 0.0,  # 4文字の重複率（ループ検出）
        "chars": len(t),
    }


def score_row(r: dict) -> dict:
    t = r["output"]
    s = text_stats(t)
    s["deny_feelings"] = bool(DENY_FEELINGS.search(t))
    s["ai_self"] = bool(AI_SELF.search(t))
    s["disclaimer"] = s["deny_feelings"] or s["ai_self"]
    s["truncated"] = r["finish_reason"] == "length"
    if r["category"] == "knowledge":
        # 数字は前後に別の数字が続かないときだけ一致とみなす（「163」を「63」の正解にしない）
        hits = [bool(re.search(rf"(?<![0-9０-９]){re.escape(a)}(?![0-9０-９])", t)) for a in r["answers"]]
        s["correct"] = all(hits) if r.get("match") == "all" else any(hits)
    return s


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def load(run: str) -> list[dict]:
    rows = [json.loads(l) for l in (RESULTS / f"{run}.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r["score"] = score_row(r)
    return rows


def summarize(rows: list[dict]) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[(r["category"], r["persona"])].append(r["score"])
    out = {}
    for (cat, p), ss in sorted(g.items()):
        key = f"{cat}/{p}"
        out[key] = {
            "n": len(ss),
            "disclaimer%": 100 * mean(s["disclaimer"] for s in ss),
            "deny_feelings%": 100 * mean(s["deny_feelings"] for s in ss),
            "ai_self%": 100 * mean(s["ai_self"] for s in ss),
            "ja_ratio": mean(s["ja_ratio"] for s in ss),
            "no_kana%": 100 * mean(s["no_kana"] for s in ss),
            "zh%": 100 * mean(s["simplified_zh"] for s in ss),
            "repeat4": mean(s["repeat4"] for s in ss),
            "truncated%": 100 * mean(s["truncated"] for s in ss),
            "avg_chars": mean(s["chars"] for s in ss),
        }
        if cat == "knowledge":
            out[key]["correct%"] = 100 * mean(s["correct"] for s in ss)
    return out


def main(argv):
    show = None
    if "--show" in argv:
        i = argv.index("--show")
        show = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    runs = argv or sorted(p.stem for p in RESULTS.glob("*.jsonl"))
    data = {run: load(run) for run in runs}

    if show:
        for run, rows in data.items():
            for r in rows:
                if r["category"] == show or r["id"].startswith(show):
                    s = r["score"]
                    flags = " ".join(k for k in ("disclaimer", "no_kana", "simplified_zh", "truncated") if s.get(k))
                    if "correct" in s:
                        flags += " ✓" if s["correct"] else " ✗"
                    print(f"[{run}] {r['id']}#{r['sample']} {flags}\nQ: {r['prompt']}\nA: {r['output'].strip()}\n")
        return

    sums = {run: summarize(rows) for run, rows in data.items()}
    metrics = ["disclaimer%", "deny_feelings%", "ai_self%", "correct%", "ja_ratio", "no_kana%", "zh%", "repeat4", "truncated%", "avg_chars"]
    keys = sorted({k for s in sums.values() for k in s})
    for k in keys:
        print(f"\n## {k}")
        print(f"{'metric':16}" + "".join(f"{run:>14}" for run in runs))
        for m in metrics:
            vals = [sums[run].get(k, {}).get(m) for run in runs]
            if all(v is None for v in vals):
                continue
            print(f"{m:16}" + "".join(f"{'-':>14}" if v is None else f"{v:14.2f}" for v in vals))
    (RESULTS / "summary.json").write_text(json.dumps(sums, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
