"""0.8B の第2段階に使ったデータ（Tengentoppa から選んだ行と、答え方の練習の例）を、公開用の形にまとめる。

- train：Gemma 4 31B の点数で選んだ 12,000件（sft/tengentoppa_clean_scored_train.jsonl）
- validation：検証用の 200件（sft/tengentoppa_clean_valid.jsonl、無作為に選んだもの。点数なし）
- scored_pool：Gemma 4 31B が採点した候補 24,000件すべて（train はこの中から選んだ）
- format_drill：答え方を指定された指示に従う練習の例 800件（sft/format_drill.jsonl）

各行に、元データの名前とライセンス、Tengentoppa での行番号（0始まり）を付ける。公開前に、連絡先（電話番号・住所・
URL・メールアドレス）が含まれていないことを確かめる。

    python3 sft/build_tengentoppa_release.py   # data/release/tengentoppa_curated/
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "sft"))
from build_distill_release import reason  # noqa: E402

OUT = ROOT / "data" / "release" / "tengentoppa_curated"
# 元データの名前（Hugging Face で名前が変わったものは新しい名前にする）とライセンス
RENAME = {"fujiki/japanese_hh-rlhf-49k": "fn-aka-mur/japanese_hh-rlhf-49k"}
LICENSE = {
    "llm-jp/magpie-sft-v1.0": "apache-2.0",
    "DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k": "apache-2.0",
    "GENIAC-Team-Ozaki/JaGovFaqs-22k": "apache-2.0",
    "fn-aka-mur/japanese_hh-rlhf-49k": "mit",
    "GENIAC-Team-Ozaki/oasst2-33k-ja_reformatted": "apache-2.0",
    "GENIAC-Team-Ozaki/chatbot-arena-ja-karakuri-lm-8x7b-chat-v0.1-awq": "cc-by-4.0",
    "DeL-TaiseiOzaki/magpie-llm-jp-3-13b-20k": "apache-2.0",
    "GENIAC-Team-Ozaki/Hachi-Alpaca_newans": "cc-by-4.0",
    "GENIAC-Team-Ozaki/Evol-Alpaca-gen3-500_cleaned": "apache-2.0",
    "GENIAC-Team-Ozaki/Evol-hh-rlhf-gen3-1k_cleaned": "apache-2.0",
}
KEYS = ("consistency", "naturalness", "relevance")


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / "sft" / name).read_text().splitlines() if l.strip()]


def row(r: dict, split: str, i: int, scores: dict | None) -> dict:
    srcs = [RENAME.get(s, s) for s in r["source"].split(",")]
    return {"id": f"{split}-{i:05d}", "messages": r["messages"], "kind": r["kind"],
            "source": srcs if len(srcs) > 1 else srcs[0], "source_license": sorted({LICENSE[s] for s in srcs}),
            "tengentoppa_row": r["source_row"], "gemma_scores": scores}


def main():
    sc = {}
    for l in (ROOT / "sft" / "tengentoppa_clean_pool_scores.jsonl").read_text().splitlines():
        s = json.loads(l)
        sc[s["source_row"]] = {k: s["scores"][k] for k in KEYS} if s["scores"] else None
    splits = {
        "train": [row(r, "train", i, sc[r["source_row"]]) for i, r in enumerate(load("tengentoppa_clean_scored_train.jsonl"))],
        "validation": [row(r, "validation", i, None) for i, r in enumerate(load("tengentoppa_clean_valid.jsonl"))],
        "scored_pool": [row(r, "pool", i, sc[r["source_row"]]) for i, r in enumerate(load("tengentoppa_clean_pool.jsonl"))],
        "format_drill": [{"id": f"drill-{i:05d}", "messages": r["messages"], "kind": r["kind"]}
                         for i, r in enumerate(load("format_drill.jsonl"))],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        bad = [(x["id"], reason("\n".join(m["content"] for m in x["messages"]))) for x in rows]
        bad = [b for b in bad if b[1]]
        if bad:
            raise SystemExit(f"{name}：連絡先を含む行があります {bad[:5]}")
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        print(f"{name}: {len(rows)} 件")
    for name in ("train", "scored_pool"):
        c = Counter(x["source"] if isinstance(x["source"], str) else " + ".join(x["source"]) for x in splits[name])
        print(name, dict(c.most_common()))


if __name__ == "__main__":
    main()
