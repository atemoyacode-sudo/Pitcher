"""採点役を外部のモデル（GPT-5.6 Luna）に頼むための書き出しと取り込み。採点の文面は judge_common.py と同じ。

書き出し（キャラ設定なしの感情の質問と作文だけ。1モデル130件＋校正用6件）:
    python3 eval/luna_judge.py export --runs base-4b,spark-x2.5-4b,spark-ja-4b,spark-ja2-4b
    → eval/luna/judge_prompts.jsonl（1行1件：{"id": ..., "prompt": ...}）

採点役には、各行の prompt をそのまま渡し、返ってきた文章をそのまま {"id": ..., "output": ...} の形で
eval/luna/judge_outputs.jsonl に1行ずつ保存してもらう。

取り込み:
    python3 eval/luna_judge.py import
    → eval/results/judge-luna_<run>.jsonl（judge_summary.py で judge-luna_ を指定して集計できる）
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge_common import CALIBRATION, JUDGE_PROMPT, parse  # noqa: E402

LUNA_DIR = HERE / "luna"


def export(runs: list[str], full: bool):
    LUNA_DIR.mkdir(exist_ok=True)
    rows = [{"id": f"calibration|{c['id']}|0", "prompt": JUDGE_PROMPT.format(system=c["system"] or "（なし）", prompt=c["prompt"], output=c["output"])}
            for c in CALIBRATION]
    for run in runs:
        for l in (HERE / "results" / f"{run}.jsonl").read_text().splitlines():
            r = json.loads(l)
            keep = r["category"] == "general" or (r["category"] == "emotion" and (full or r["persona"] == "none"))
            if keep:
                rows.append({"id": f"{run}|{r['id']}|{r['sample']}",
                             "prompt": JUDGE_PROMPT.format(system=r["system"] or "（なし）", prompt=r["prompt"], output=r["output"].strip()[:2500])})
    out = LUNA_DIR / "judge_prompts.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"{len(rows)} 件を {out} に書き出しました")


def import_():
    prompts = {json.loads(l)["id"] for l in (LUNA_DIR / "judge_prompts.jsonl").read_text().splitlines() if l.strip()}
    outs = {}
    for l in (LUNA_DIR / "judge_outputs.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            outs[r["id"]] = r["output"]
    print(f"書き出し {len(prompts)} 件 / 回答 {len(outs)} 件（足りない {len(prompts - set(outs))} 件）")
    meta = {}
    for run in {i.split("|")[0] for i in prompts} - {"calibration"}:
        for l in (HERE / "results" / f"{run}.jsonl").read_text().splitlines():
            r = json.loads(l)
            meta[f"{run}|{r['id']}|{r['sample']}"] = r
    by_run = {}
    for i, text in outs.items():
        run, item_id, sample = i.split("|")
        if run == "calibration":
            c = next(c for c in CALIBRATION if c["id"] == item_id)
            print(f"校正 {item_id:22} 期待={c['expect']:12} 採点={parse(text)}")
            continue
        r = meta[i]
        by_run.setdefault(run, []).append({"run": run, "id": item_id, "sample": int(sample), "category": r["category"],
                                           "persona": r["persona"], "judge": parse(text), "raw": text})
    for run, rows in by_run.items():
        (HERE / "results" / f"judge-luna_{run}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        print(f"{run}: {len(rows)} 件（読み取り失敗 {sum(x['judge'] is None for x in rows)} 件）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "import"])
    ap.add_argument("--runs", default="")
    ap.add_argument("--full", action="store_true", help="キャラ設定ありの回答も含める（1モデル530件）")
    a = ap.parse_args()
    export(a.runs.split(","), a.full) if a.mode == "export" else import_()


if __name__ == "__main__":
    main()
