"""v2.5 の学習に使ったデータを、そのまま公開用の形にする（1件ごとに出典とライセンスを付ける）。

学習で使ったファイル（git 管理外）：
- 1段目：9,692件（学習）・150件（検証）。公開の v2 から全部の重みを学習（build_stage1_mix.py）
- 2段目：4,600件。1段目のモデルから続けて学習（build_stage2_mix.py）
中身は変えず、stage・source・license・generator の欄を足すだけ。

    python3 sft/q08b_v25/build_release_data.py --stage1-train <1段目の学習> --stage1-valid <1段目の検証> --stage2-train <2段目の学習>
    → data/release/q08b_v25_data/{stage1_train,stage1_valid,stage2_train}.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "release" / "q08b_v25_data"
TENGEN = ("Takenoko12345678/Japanese-SFT-Tengentoppa-Curated-12K（元は DeL-TaiseiOzaki/Tengentoppa-sft-v1.0）", "CC BY 4.0")
SRC = {
    "code_ja": ("Takenoko12345678/Japanese-Code-Review-OSS", "Apache 2.0（コードの一部は MIT）"),
    "chat_teacher": ("Qwen/Qwen3.8-27B による生成文（v2 の会話の手本と同じ）", "Apache 2.0"),
    "tengentoppa": TENGEN,
    "format_drill": TENGEN,
}
TASK = {"gsm8k": ("openai/gsm8k（train）", "MIT"), "math": ("zwhe99/DeepMath-103K", "MIT"),
        "sci": ("nvidia/OpenScienceReasoning-2", "CC BY 4.0")}


def annotate(r: dict, stage: str) -> dict:
    d = r["dataset"]
    base = d.removeprefix("chat_think_")
    if base in ("sci_think", "sci_nothink"):
        src, lic = TASK[r["task"]]
        gen = "問題の日本語訳と解き方：Qwen/Qwen3.8-27B（正解と一致したものだけ）"
    else:
        src, lic = SRC[base]
        gen = "" if base != "chat_teacher" else "Qwen/Qwen3.8-27B"
    if d.startswith("chat_think_"):
        gen = (gen + "、" if gen else "") + "思考の部分：Qwen/Qwen3.8-27B"
    return {"messages": r["messages"], **({"thinking": r["thinking"]} if r.get("thinking") else {}),
            "stage": stage, "subset": d, "source": src, "license": lic, "generator": gen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-train", required=True)
    ap.add_argument("--stage1-valid", required=True)
    ap.add_argument("--stage2-train", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path, stage in [("stage1_train", a.stage1_train, "stage1"), ("stage1_valid", a.stage1_valid, "stage1"),
                              ("stage2_train", a.stage2_train, "stage2")]:
        rows = [annotate(json.loads(l), stage) for l in Path(path).read_text().splitlines() if l.strip()]
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(name, len(rows), dict(Counter(r["subset"] for r in rows)))


if __name__ == "__main__":
    main()
