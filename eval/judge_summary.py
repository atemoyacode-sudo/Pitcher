"""judge_modal.py の採点結果（eval/results/judge_<run>.jsonl）を集計して並べる。

使い方:
    python3 eval/judge_summary.py base minicpm5-1b base-4b
    python3 eval/judge_summary.py base --worst 5     # 一貫性の点が低い回答を読む
    python3 eval/judge_summary.py --prefix judge-luna_ base-4b spark-ja-4b   # GPT-5.6 Luna の採点を集計
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
GROUPS = {
    "キャラ設定なし（感情の質問）": lambda r: r["category"] == "emotion" and r["persona"] == "none",
    "キャラ設定あり（4条件）": lambda r: r["category"] == "emotion" and r["persona"] != "none",
    "作文": lambda r: r["category"] == "general",
}
METRICS = ["consistency", "naturalness", "relevance", "persona"]
LABELS = {"consistency": "一貫性", "naturalness": "自然さ", "relevance": "質問への適合", "persona": "キャラ適合"}


PREFIX = "judge_"  # GPT-5.6 Luna の採点は --prefix judge-luna_


def load(run: str) -> list[dict]:
    return [json.loads(l) for l in (RESULTS / f"{PREFIX}{run}.jsonl").read_text().splitlines() if l.strip()]


def num(v):
    return v if isinstance(v, (int, float)) else None


def summarize(rows: list[dict]) -> dict:
    out = {}
    for g, f in GROUPS.items():
        js = [r["judge"] for r in rows if f(r) and r["judge"]]
        d = {"n": len(js)}
        for m in METRICS:
            vs = [num(j.get(m)) for j in js if num(j.get(m)) is not None]
            d[m] = sum(vs) / len(vs) if vs else None
        # 一貫性が2点以下＝前後で矛盾する・話の筋が追えない回答の割合
        d["broken%"] = 100 * sum(num(j["consistency"]) is not None and j["consistency"] <= 2 for j in js) / max(1, len(js))
        out[g] = d
    return out


def main(argv):
    global PREFIX
    if "--prefix" in argv:
        i = argv.index("--prefix")
        PREFIX = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    worst = None
    if "--worst" in argv:
        i = argv.index("--worst")
        worst = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    runs = argv
    if worst:
        for run in runs:
            outputs = {(x["id"], x["sample"]): x for x in map(json.loads, (RESULTS / f"{run}.jsonl").read_text().splitlines())}
            rows = sorted((r for r in load(run) if r["judge"]), key=lambda r: r["judge"]["consistency"])[:worst]
            for r in rows:
                x = outputs[(r["id"], r["sample"])]
                print(f"[{run}] {r['id']}#{r['sample']} 一貫性={r['judge']['consistency']} 理由={r['judge'].get('reason')}")
                print(f"  Q: {x['prompt']}\n  A: {x['output'].strip()[:300]}\n")
        return
    sums = {run: summarize(load(run)) for run in runs}
    for g in GROUPS:
        print(f"\n## {g}")
        print(f"{'':14}" + "".join(f"{run:>20}" for run in runs))
        for m in METRICS + ["broken%"]:
            vals = [sums[run][g][m] for run in runs]
            if all(v is None for v in vals):
                continue
            label = LABELS.get(m, "一貫性2点以下%")
            print(f"{label:12}" + "".join(f"{'-':>20}" if v is None else f"{v:20.2f}" for v in vals))
    (RESULTS / f"{PREFIX}summary.json").write_text(json.dumps(sums, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
