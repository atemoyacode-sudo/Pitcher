"""キャラクターの学習データ（sft/train.jsonl）に、元モデル自身の回答（リプレイ）を混ぜた学習データを作る。

リプレイはキャラ設定なし（system なし）の会話として入れる。途中で切れた回答は使わない。
元モデルの回答には中国語が混ざったものがあり（4B で約18%）、そのまま混ぜると中国語の混入が増えたため、
中国語を含む回答を除いた版（train_replay_ja_*）も作る。

使い方:
    python3 sft/build_replay_mix.py   # sft/train_replay_<モデル名>.jsonl と train_replay_ja_<モデル名>.jsonl を作る
"""

import json
import random
import sys
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SFT_DIR.parent / "eval"))
from score import ZH_CHAR  # noqa: E402  評価と同じ基準で中国語の混入を判定する


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
        name = path.stem.removeprefix("replay_")
        replay_ja = [x for x in replay if not ZH_CHAR.search(x["messages"][-1]["content"])]
        for out_name, rep in ((f"train_replay_{name}.jsonl", replay), (f"train_replay_ja_{name}.jsonl", replay_ja)):
            rows = persona + rep
            random.Random(0).shuffle(rows)
            (SFT_DIR / out_name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
            print(f"{out_name}: キャラ {len(persona)} 件 + リプレイ {len(rep)} 件")


if __name__ == "__main__":
    main()
