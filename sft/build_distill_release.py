"""蒸留データ（sft/distill_sft.jsonl）から、公開版のデータセットを作る。

学習に使ったデータから、さらに次を含む行を除く（先生役がでっち上げている可能性があり、事実確認をしていないため）:
- 電話番号（公的な窓口の番号でも、間違っていると無関係な相手につながるおそれがある）
- 郵便番号・番地つき住所
- 外部 URL（example.com などの説明用と localhost 以外。存在しないドメインが混ざっていた）
- 例示用ではない宛先のメールアドレス（abc.co.jp なども実在の会社のドメインになりうる）

    python3 sft/build_distill_release.py   # data/release/distill/train.jsonl
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "release" / "distill"

PHONE = re.compile(r"(?<!\d)(?:\+81[- ]?|0)\d{1,4}[-‐－ ]\d{1,4}[-‐－ ]\d{3,4}(?!\d)|(?<!\d)0[789]0[-‐－ ]?\d{4}[-‐－ ]?\d{4}(?!\d)")
ADDRESS = re.compile(r"〒\s?\d{3}[-‐－]\d{4}|(都|道|府|県)[^\s、。]{1,12}(市|区|町|村)[^\s、。]{0,12}\d+[-‐－丁目]\d+")
URL = re.compile(r"https?://([^/\s）)」\]\"'`<>:]+)|(?<![A-Za-z0-9.@])www\.[A-Za-z0-9-]+\.[A-Za-z]{2,}")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
# 説明用に予約されたドメイン（RFC 2606 / 6761）と、手元を指すもの
SAFE_HOST = re.compile(r"(^|\.)(example\.(com|org|net)|test|invalid|localhost|local)$|^localhost$|^127\.0\.0\.1$|^0\.0\.0\.0$")


def reason(text: str) -> str | None:
    if PHONE.search(text):
        return "phone"
    if ADDRESS.search(text):
        return "address"
    for m in URL.finditer(text):
        host = (m.group(1) or m.group(0)).lower().removeprefix("www.")
        if not SAFE_HOST.search(host):
            return "url"
    for m in EMAIL.finditer(text):
        if not SAFE_HOST.search(m.group(1).lower()):
            return "email"
    return None


def main():
    rows = [json.loads(l) for l in (ROOT / "sft" / "distill_sft.jsonl").read_text().splitlines() if l.strip()]
    kept, dropped = [], Counter()
    for r in rows:
        why = reason(r["messages"][0]["content"] + "\n" + r["messages"][1]["content"])
        if why:
            dropped[why] += 1
            continue
        kept.append({"id": r["source_row"], "messages": r["messages"],
                     "prompt_source": {"ja_general": "OysterCoreAI/SFT-General-Japanese-60K", "cute_prompt": "RikkaBotan/Cute_Synthetic_smoltalk_jp_sft"}[r["trait"]],
                     "response_model": "Qwen/Qwen3.8-27B"})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept))
    print(f"{len(rows)} 件 → {len(kept)} 件（除外 {dict(dropped)}）")


if __name__ == "__main__":
    main()
