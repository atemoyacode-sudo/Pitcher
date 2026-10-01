"""採点役を外部のモデル（GPT-5.6 Luna）に頼むための書き出しと取り込み。採点の文面は judge_common.py と同じ。

書き出し（キャラ設定なしの感情の質問と作文だけ。1モデル130件＋校正用6件）:
    python3 eval/luna_judge.py export --runs base-4b,spark-x2.5-4b,spark-ja-4b,spark-ja2-4b
    python3 eval/luna_judge.py export --runs base-4b-long,... --batch 3 --max-chars 8000   # 上限 2,048 トークンの再テスト
    → eval/luna/judge_prompts_<batch>.jsonl（1行1件：{"id": ..., "prompt": ...}）

採点役には、各行の prompt をそのまま渡し、返ってきた文章をそのまま {"id": ..., "output": ...} の形で
eval/luna/judge_outputs_<batch>.jsonl に1行ずつ保存してもらう。

取り込み:
    python3 eval/luna_judge.py import
    python3 eval/luna_judge.py import --batch 8   # 8回目だけ（同じ回で比べるとき）→ eval/results/judge-luna-b8_<run>.jsonl
    → eval/results/judge-luna_<run>.jsonl（judge_summary.py で judge-luna_ を指定して集計できる）
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge_common import CALIBRATION, JUDGE_PROMPT, parse  # noqa: E402

LUNA_DIR = HERE / "luna"


def export(runs: list[str], full: bool, batch: str, max_chars: int = 2500):
    LUNA_DIR.mkdir(exist_ok=True)
    rows = [{"id": f"calibration|{c['id']}|0", "prompt": JUDGE_PROMPT.format(system=c["system"] or "（なし）", prompt=c["prompt"], output=c["output"])}
            for c in CALIBRATION]
    for run in runs:
        for l in (HERE / "results" / f"{run}.jsonl").read_text().splitlines():
            r = json.loads(l)
            keep = r["category"] == "general" or (r["category"] == "emotion" and (full or r["persona"] == "none"))
            if keep:
                rows.append({"id": f"{run}|{r['id']}|{r['sample']}",
                             "prompt": JUDGE_PROMPT.format(system=r["system"] or "（なし）", prompt=r["prompt"], output=r["output"].strip()[:max_chars])})
    out = LUNA_DIR / f"judge_prompts_{batch}.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"{len(rows)} 件を {out} に書き出しました")


# 回をまたいで取り込む（judge-luna_<run>.jsonl を作る）のは7回目まで。8回目からは、比べるモデルを毎回同じ回に入れて
# --batch で回ごとに取り込む（judge-luna-b<回>_<run>.jsonl）。同じ回答が後の回で採点し直されても、報告済みの数字が変わらないようにするため
LAST_MERGED_BATCH = 7


def batch_no(f: Path) -> int:
    return int(re.sub(r"\D", "", f.stem.split("_")[-1]) or 0)


def batch_files(kind: str, batch: str | None) -> list[Path]:
    """回の番号の順に並べた judge_<kind>_<batch>.jsonl（batch を指定したときはその回だけ）。"""
    if batch:
        return [LUNA_DIR / f"judge_{kind}_{batch}.jsonl"]
    return sorted((f for f in LUNA_DIR.glob(f"judge_{kind}_*.jsonl") if batch_no(f) <= LAST_MERGED_BATCH), key=batch_no)


def import_(batch: str | None = None):
    # 書き出しは何回かに分けてよい（judge_prompts_<batch>.jsonl）。回答も judge_outputs_<batch>.jsonl に分けて置く。
    # 同じ回答が複数の回で採点されているときは、後の回の採点を使う
    prompts = {json.loads(l)["id"] for f in batch_files("prompts", batch) for l in f.read_text().splitlines() if l.strip()}
    outs = {}
    for f in batch_files("outputs", batch):
        for l in f.read_text().splitlines():
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
        (HERE / "results" / f"judge-luna{'-b' + batch if batch else ''}_{run}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        print(f"{run}: {len(rows)} 件（読み取り失敗 {sum(x['judge'] is None for x in rows)} 件）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "import"])
    ap.add_argument("--runs", default="")
    ap.add_argument("--full", action="store_true", help="キャラ設定ありの回答も含める（1モデル530件）")
    ap.add_argument("--batch", default=None, help="回の名前（書き出し：judge_prompts_<batch>.jsonl、既定は 1。取り込み：その回だけを取り込む）")
    ap.add_argument("--max-chars", type=int, default=2500, help="採点役に渡す回答の最大文字数（上限を上げた再テストでは大きくする）")
    a = ap.parse_args()
    export(a.runs.split(","), a.full, a.batch or "1", a.max_chars) if a.mode == "export" else import_(a.batch)


if __name__ == "__main__":
    main()
