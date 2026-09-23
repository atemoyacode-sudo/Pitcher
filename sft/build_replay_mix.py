"""キャラクターの学習データ（sft/train.jsonl）に、元モデル自身の回答（リプレイ）を混ぜた学習データを作る。

リプレイはキャラ設定なし（system なし）の会話として入れる。途中で切れた回答は使わない。

使い方:
    python3 sft/build_replay_mix.py   # sft/train_replay_<モデル名>.jsonl を作る
"""

import json
import random
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parent


def main():
    persona = [json.loads(l) for l in (SFT_DIR / "train.jsonl").read_text().splitlines() if l.strip()]
    for path in sorted(SFT_DIR.glob("replay_Qwen*.jsonl")):
        replay = []
        for r in map(json.loads, path.read_text().splitlines()):
            text = r["output"].strip()
            if r["finish_reason"] != "stop" or not text or "<think>" in text:
                continue
            replay.append({"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": text}],
                           "dataset": "replay", "trait": r["category"], "source_row": None})
        rows = persona + replay
        random.Random(0).shuffle(rows)
        out = SFT_DIR / f"train_replay_{path.stem.removeprefix('replay_')}.jsonl"
        out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        print(f"{out.name}: キャラ {len(persona)} 件 + リプレイ {len(replay)} 件")


if __name__ == "__main__":
    main()
