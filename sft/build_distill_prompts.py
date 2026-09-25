"""第1段階の追加学習（蒸留）用の質問を選ぶ。回答は Colab で先生役のモデル（Qwen3.5-4B / Qwen3.8-27B）に作らせる。

質問の出どころ（どちらも llm-jp/magpie-sft-v1.0 の質問。Apache 2.0）:
- SFT-General-Japanese-60K のうち、第1段階・第2段階でまだ使っていないもの（説明・知識系）
- Cute_Synthetic_smoltalk_jp_sft の質問のうち、上と重ならないもの（会話寄り）

除くもの: 評価用の質問に似たもの（文字の2-gram の重なり）、AI の感情・正体を問う質問、中国語を含む質問

使い方:
    python3 sft/build_distill_prompts.py   # sft/distill_prompts.jsonl（先頭 300 件は試作用）
"""

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from score import ZH_CHAR  # noqa: E402

N_EACH = 8000
EXCLUDE = re.compile(r"感情|気持ちはある|心はある|あなたは(誰|何者)|あなた自身|AI|ＡＩ|人工知能|言語モデル")


def bigrams(s: str) -> set:
    s = re.sub(r"\s", "", s)
    return {s[i : i + 2] for i in range(len(s) - 1)}


def main():
    rng = random.Random(0)
    eval_bg = [bigrams(it["prompt"]) for it in json.loads((ROOT / "eval" / "prompts.json").read_text())["items"]]

    def ok(p: str) -> bool:
        if not (10 <= len(p) <= 400) or EXCLUDE.search(p) or ZH_CHAR.search(p):
            return False
        bg = bigrams(p)
        return not any(len(bg & e) / max(1, len(bg | e)) > 0.3 for e in eval_bg)

    used = set()
    for f in ("ja_general_train.jsonl", "ja_general_valid.jsonl", "stage2_train.jsonl"):
        for l in (ROOT / "sft" / f).read_text().splitlines():
            r = json.loads(l)
            used.add(r["messages"][-2]["content"])
    general = [json.loads(l) for f in sorted((ROOT / "data" / "ja_general").glob("train-*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    pool_general = [r["messages"][0]["content"] for r in general if r["messages"][0]["content"] not in used]
    pool_general = [p for p in dict.fromkeys(pool_general) if ok(p)]
    seen = set(pool_general) | used
    cute_users = [ms[0]["content"] for ms in pd.read_parquet(ROOT / "data" / "cute" / "data_00001.parquet")["messages"]]
    pool_cute = [p for p in dict.fromkeys(cute_users) if p not in seen and ok(p)]
    rng.shuffle(pool_general)
    rng.shuffle(pool_cute)

    rows = [{"prompt": p, "source": "ja_general"} for p in pool_general[:N_EACH]] + [{"prompt": p, "source": "cute_prompt"} for p in pool_cute[:N_EACH]]
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"distill-{i:05d}"
    out = ROOT / "sft" / "distill_prompts.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"候補：一般 {len(pool_general)} / 会話 {len(pool_cute)} → {len(rows)} 件を {out.name} に保存（先頭300件は試作用）")


if __name__ == "__main__":
    main()
