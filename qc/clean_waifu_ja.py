"""学習データ2（anime-waifu-personality-chat の日本語訳）の品質チェックで見つかった問題を直し、学習用の修正版を作る。

翻訳ファイル本体（学習データ2_日本語.json）は変更しない。行番号は元データと翻訳データで共通の 0 始まりの位置。

使い方:
    python3 qc/clean_waifu_ja.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "anime-waifu-personality-chat.json"
JA = ROOT / "学習データ2_日本語.json"
OUT = ROOT / "学習データ2_日本語_clean.json"
DESC_OUT = ROOT / "学習データ2_属性説明.json"

# セリフではなく属性の解説文（元データの時点で混ざっている）。学習データからは外し、キャラ設定文の素材として別に保存する
DESCRIPTION_ROWS = range(1692, 1712)

# 解説文を除くとセリフが1件しか残らない属性。1件では学習しても身につかないので外す
RARE_TRAITS = {"undere", "sadodere", "bakadere", "hinedere", "nyandere", "oujidere", "mayadere", "byoukidere", "ekidere"}

# 元データの時点で同じ（またはほぼ同じ）セリフ。後に出てくるほうを外す
DUPLICATE_ROWS = {1198, 1507}

# 文法が壊れた語尾（「〜わから。」）と、"anything less" の直訳による誤訳を手で直す
MANUAL_FIXES = {
    508: "もちろん、私は最高に綺麗でしょう？当然じゃない。",  # Did you expect anything less?
    1268: "私の沈黙を無関心だと勘違いしないで。どうでもよかったら、ここにはいないわ。",
    1585: "あなたが尽くすのは当然のことよ。それに満たないものは認めないわ。",
    1595: "ひざまずきたいなら好きにしなさい。もっとも、それくらい当然だと、とっくの昔から思っているけれど。",
}


# 翻訳出力の末尾に JSON の断片（「…だからねっ！」}, {」）が残っている行がある
JSON_TAIL = re.compile(r"[」\"]?\s*\}\s*,?\s*\{?\s*$")


def main():
    src = json.loads(SRC.read_text())
    ja = json.loads(JA.read_text())
    assert len(src) == len(ja), "元データと翻訳の件数が違う"
    assert all(s["trait"] != "" for s in src)

    rows, descriptions, log = [], {}, {"description": 0, "rare": 0, "duplicate": 0, "manual": 0, "boku": 0, "json_tail": 0}
    for i, (s, j) in enumerate(zip(src, ja)):
        if i in DESCRIPTION_ROWS:
            descriptions[j["trait"]] = {"trait_en": s["trait"], "description": j["dialogue"], "source_row": i}
            log["description"] += 1
            continue
        if s["trait"] in RARE_TRAITS:
            log["rare"] += 1
            continue
        if i in DUPLICATE_ROWS:
            log["duplicate"] += 1
            continue

        text = j["dialogue"]
        if JSON_TAIL.search(text):
            text = JSON_TAIL.sub("", text)
            log["json_tail"] += 1
        if i in MANUAL_FIXES:
            text = MANUAL_FIXES[i]
            log["manual"] += 1
        # ボクっ娘の一人称が翻訳で「私」になっていた（39件すべて一人称の用法だったことを確認済み）
        if s["trait"] == "bokukko" and "私" in text:
            text = text.replace("私", "ボク")
            log["boku"] += 1

        rows.append({"trait": j["trait"], "trait_en": s["trait"], "dialogue": text, "source_row": i, "source_en": s["dialogue"]})

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    DESC_OUT.write_text(json.dumps(descriptions, ensure_ascii=False, indent=1))
    print(f"{len(rows)} 件を {OUT.name} に、属性説明 {len(descriptions)} 件を {DESC_OUT.name} に保存しました")
    print("除外・修正:", log)


if __name__ == "__main__":
    main()
