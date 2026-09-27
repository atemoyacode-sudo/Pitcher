"""Tengentoppa-sft-v1.0 の各行が、どの元データから来たかを調べる（Tengentoppa には出どころの列がない）。

Tengentoppa は日本語の指示データ16〜17個を統合したもの。元データを Hugging Face から取得し、回答（なければ指示）の
先頭80字（空白を除く）が一致するものを探して、各行の出どころを決める。利用条件の確認が必要な元データだけを除けるようにするため。

アクセスに同意が必要な元データ（weblab-GENIAC/aya-ja-nemotron-dpo-masked、weblab-GENIAC/Open-Platypus-Japanese-masked）は
取得していないので、そこから来た行は「不明」になる。

事前に元データを data/tengentoppa/sources/<作者>__<名前>/ に置く（hf_hub_download で取得）。

    python3 sft/map_tengentoppa_sources.py   # data/tengentoppa/row_sources_full.json（行番号 → 元データ名）
"""

import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TT = ROOT / "data" / "tengentoppa"


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)[:80]


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from strings(v)
    elif hasattr(o, "tolist"):
        yield from strings(o.tolist())


def main():
    index = defaultdict(set)
    for d in sorted(glob.glob(str(TT / "sources" / "*"))):
        name = os.path.basename(d).replace("__", "/")
        for f in glob.glob(d + "/**/*", recursive=True):
            if f.endswith(".parquet"):
                recs = pd.read_parquet(f).to_dict("records")
            elif f.endswith(".jsonl"):
                recs = [json.loads(l) for l in open(f) if l.strip()]
            elif f.endswith(".json"):
                x = json.load(open(f))
                recs = x if isinstance(x, list) else list(x.values()) if isinstance(x, dict) else []
            else:
                continue
            for r in recs:
                for s in strings(r):
                    if len(s) >= 10:
                        index[norm(s)].add(name)
    rows = json.loads((TT / "processed_dataset.json").read_text())
    labels = []
    for r in rows:
        src = index.get(norm(r.get("output") or "")) or index.get(norm(r.get("instruction") or "")) or set()
        labels.append(sorted(src)[0] if len(src) == 1 else ",".join(sorted(src)) if src else "不明")
    (TT / "row_sources_full.json").write_text(json.dumps(labels, ensure_ascii=False))
    for k, v in Counter(labels).most_common():
        print(f"{v:7} {k}")


if __name__ == "__main__":
    main()
