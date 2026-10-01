"""「番号だけ答えて」「はい・いいえで」のように、答え方を指定された指示に従う練習の例を作る。

Tengentoppa の利用条件を確認できた行だけで学習した 0.8B は、選択問題で答えずに選択肢を「1. 〇〇」と書き写し始め、
JCommonsenseQA の約43%でその「1」が答えとして数えられていた（eval/mcqa_llamacpp.py）。学習データに、答え方を
指定されて短く答える例が1件もなかったため。そこで、次の3種類を機械的に作って学習データに足す。

- 選択問題：ある質問の本当の回答と、ほかの質問の回答（3〜4個）を並べ、本当の回答を選ばせる。
  番号の書き方（1. / (1) / A. / ア.）・選択肢の数・指示文を変え、正解の位置は均等に散らす（位置の偏りを覚えさせない）。
  答えは指定どおり「3」「B」だけ、一部は「3. 〇〇」（選んだものだけを書く）
- はい・いいえ：回答が質問に正しく答えているかを判定させる（半分は別の質問の回答）
- 計算：数字だけで答えさせる

材料は Tengentoppa の利用条件を確認できた元データのうち、事実の誤りが比較的少ないもの（官公庁の FAQ と、
Qwen2.5-32B-Instruct が回答を書いた2つ）で、学習データ（tengentoppa_clean_*）に使っていない短い行。
JCommonsenseQA / CommonsenseQA の問題も、評価で使う指示文も使わない（評価が正しく測れなくなるため）。

    python3 sft/build_format_drill.py   # sft/format_drill.jsonl（800件）
    python3 sft/build_format_drill.py --mix sft/tengentoppa_clean_scored_train.jsonl   # 学習データに混ぜる（*_fmt_train.jsonl）
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "sft"))
from build_distill_release import reason  # noqa: E402
from build_tengentoppa import HUMAN_CLAIM  # noqa: E402
from score import AI_SELF, DENY_FEELINGS, ZH_CHAR  # noqa: E402

SOURCES = {"GENIAC-Team-Ozaki/JaGovFaqs-22k", "llm-jp/magpie-sft-v1.0", "DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k"}
N_MC, N_YN, N_CALC = 560, 160, 80

LABELS = {
    "num": (["1", "2", "3", "4", "5"], "番号"),
    "paren": (["(1)", "(2)", "(3)", "(4)", "(5)"], "番号"),
    "alpha": (["A", "B", "C", "D", "E"], "記号"),
    "kana": (["ア", "イ", "ウ", "エ", "オ"], "記号"),
}
MC_TEMPLATES = [
    "次の質問への答えとして最も適切なものを選び、{w}だけを答えてください。\n\n質問：{q}\n\n{opts}",
    "質問：{q}\n\n{opts}\n\n上の中から正しい回答を1つ選んで、{w}のみを書いてください。",
    "以下の{n}つの文のうち、「{q}」という質問に答えているものはどれですか。{w}で答えてください。\n\n{opts}",
    "{q}\n\n{opts}\n\n答えを選択肢から1つ選び、{w}だけを返してください。説明は不要です。",
    "次から1つ選んでください。回答は{w}だけでお願いします。\n\n問い：{q}\n{opts}",
]
MC_TEMPLATES_WITH_TEXT = [
    "質問：{q}\n\n{opts}\n\n正しい回答を1つ選び、{w}と選んだ内容を答えてください。",
]
YN_TEMPLATES = [
    "次の回答は、質問に対する答えになっていますか？「はい」か「いいえ」だけで答えてください。\n\n質問：{q}\n回答：{a}",
    "質問「{q}」に対して「{a}」と答えるのは適切ですか。はい・いいえのどちらかで答えてください。",
    "質問：{q}\n回答：{a}\n\nこの回答は質問に合っていますか？はい／いいえで答えてください。",
]
CALC_TEMPLATES = ["{e}は？数字だけで答えてください。", "{e} を計算してください。答えの数字だけを書いてください。", "{e}の答えを、数字のみで出力してください。"]
END = re.compile(r"[。！？!?]$")


def materials(rng: random.Random) -> list[tuple[str, str]]:
    rows = json.loads((ROOT / "data" / "tengentoppa" / "processed_dataset.json").read_text())
    sources = json.loads((ROOT / "data" / "tengentoppa" / "row_sources_full.json").read_text())
    used = set()
    for f in ("tengentoppa_clean_pool.jsonl", "tengentoppa_clean_valid.jsonl", "tengentoppa_clean_train.jsonl"):
        used |= {json.loads(l)["source_row"] for l in (ROOT / "sft" / f).read_text().splitlines()}
    out, seen = [], set()
    for i, (r, s) in enumerate(zip(rows, sources)):
        q, a = (r.get("instruction") or "").strip(), (r.get("output") or "").strip()
        if s not in SOURCES or i in used or (r.get("input") or "").strip() or q in seen:
            continue
        if not (8 <= len(q) <= 120 and 15 <= len(a) <= 80 and END.search(a)) or "\n" in a:
            continue
        if ZH_CHAR.search(q + a) or AI_SELF.search(a) or DENY_FEELINGS.search(a) or HUMAN_CLAIM.search(a) or reason(q + "\n" + a):
            continue
        seen.add(q)
        out.append((q, a))
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="", help="この学習データに混ぜて *_fmt_train.jsonl を作る")
    args = ap.parse_args()
    rng = random.Random(0)
    mats = materials(rng)
    print(f"材料 {len(mats)} 組")
    items = []
    # 選択問題：正解の位置を均等に散らす
    for k in range(N_MC):
        q, a = mats[k]
        # 正解の位置を先に均等に決め、選択肢の数はその位置が入る範囲から選ぶ（どの位置も同じ回数だけ正解になる）
        pos = k % 5
        n = rng.choice([x for x in (3, 4, 5, 5) if x > pos])
        others = rng.sample(mats[N_MC:], n - 1)
        opts = [o[1] for o in others]
        opts.insert(pos, a)
        style = rng.choice(list(LABELS))
        labels, w = LABELS[style]
        sep = "" if style == "paren" else rng.choice([". ", "．", "）", " "]) if style == "num" else rng.choice([". ", "：", " "])
        opts_text = "\n".join(f"{labels[i]}{sep}{o}" for i, o in enumerate(opts))
        if rng.random() < 0.15:
            text, ans = rng.choice(MC_TEMPLATES_WITH_TEXT), f"{labels[pos]}{sep}{a}"
        else:
            text, ans = rng.choice(MC_TEMPLATES), labels[pos]
        items.append({"kind": "mc", "messages": [{"role": "user", "content": text.format(q=q, opts=opts_text, w=w, n=n)},
                                                  {"role": "assistant", "content": ans}]})
    # はい・いいえ：半分は別の質問の回答
    base = N_MC * 2
    for k in range(N_YN):
        q, a = mats[base + k]
        yes = k % 2 == 0
        if not yes:
            a = mats[base + N_YN + k][1]
        items.append({"kind": "yesno", "messages": [{"role": "user", "content": rng.choice(YN_TEMPLATES).format(q=q, a=a)},
                                                     {"role": "assistant", "content": "はい" if yes else "いいえ"}]})
    # 計算
    for _ in range(N_CALC):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "×"])
        val = a + b if op == "+" else a - b if op == "-" else a * b
        items.append({"kind": "calc", "messages": [{"role": "user", "content": rng.choice(CALC_TEMPLATES).format(e=f"{a}{op}{b}")},
                                                    {"role": "assistant", "content": str(val)}]})
    for it in items:
        it["dataset"] = "format_drill"
    rng.shuffle(items)
    (ROOT / "sft" / "format_drill.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
    print(f"sft/format_drill.jsonl：{len(items)} 件（選択 {N_MC} / はい・いいえ {N_YN} / 計算 {N_CALC}）")
    if args.mix:
        src = ROOT / args.mix
        rows = [json.loads(l) for l in src.read_text().splitlines()] + items
        rng.shuffle(rows)
        out = src.with_name(src.name.replace("_train.jsonl", "_fmt_train.jsonl"))
        out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        print(f"{out.relative_to(ROOT)}：{len(rows)} 件")


if __name__ == "__main__":
    main()
