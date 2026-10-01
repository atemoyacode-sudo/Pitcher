"""選択問題（JCommonsenseQA / CommonsenseQA）を手元の llama.cpp で解かせ、「1番に偏る」原因を調べる。

Tengentoppa の利用条件を確認できた行だけで学習した 0.8B は、JCommonsenseQA の約半分で答えずに選択肢を
「1. 〇〇」と書き写し始め、その「1」が答えとして数えられていた（eval/bench_mcqa_modal.py の結果）。
知識が落ちたのか、答え方（形式）が崩れただけなのかを分けるため、3通りで測る:

- gen    ：bench_mcqa_modal.py と同じ聞き方（thinking なし・貪欲法・16トークン）。書き写しの割合も数える
- prob   ：回答の書き出しを「答え：」に固定し、次に 1〜5 のどれが出やすいか（確率）で答えを決める。
           答え方の癖に左右されずに、知識だけを測る
- rotate ：選択肢の順番を5通りにずらして prob で測る。位置ごとの選ばれやすさ（位置の偏り）と、
           順番を変えても同じ中身を選ぶか（一貫性）を見る

事前に llama-server を起動しておく（--jinja、並列数 8）:
    llama-server -m tools/gguf/qwen08b-tengen-clean-f16.gguf --port 8093 -np 8 -c 16384 --jinja
    python3 eval/mcqa_llamacpp.py --run qwen08b-tengen-clean --lang ja
    → eval/results/mcqa-local_<run>.json（集計）
"""

import argparse
import json
import math
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JA_TEMPLATE = "次の質問に対して、最も適切な答えを選択肢から1つ選び、番号（1〜5）だけを答えてください。\n\n質問：{q}\n{choices}\n\n答え："
EN_TEMPLATE = "Choose the most appropriate answer to the following question from the options, and reply with only its number (1-5).\n\nQuestion: {q}\n{choices}\n\nAnswer:"
PREFIX = {"ja": "答え：", "en": "Answer: "}
COPY = re.compile(r"^\s*1[\.．]\s*\S")  # 答えずに選択肢の一覧を書き写し始めた


def load(lang: str) -> list[dict]:
    """data/mcqa/ に置いた validation（jcqa_valid.parquet / cqa_valid.parquet、bench_mcqa_modal.py と同じ URL から取得）。"""
    rows = []
    if lang == "ja":
        for r in pd.read_parquet(ROOT / "data" / "mcqa" / "jcqa_valid.parquet").itertuples():
            rows.append({"id": str(r.q_id), "q": r.question, "choices": [getattr(r, f"choice{i}") for i in range(5)], "answer": int(r.label)})
    else:
        for r in pd.read_parquet(ROOT / "data" / "mcqa" / "cqa_valid.parquet").itertuples():
            labels = list(r.choices["label"])
            rows.append({"id": r.id, "q": r.question, "choices": list(r.choices["text"]), "answer": labels.index(r.answerKey)})
    return rows


def prompt(x: dict, lang: str, shift: int = 0) -> tuple[str, int]:
    """選択肢を shift だけずらした問題文と、そのときの正解の位置（0始まり）。"""
    ch = x["choices"][shift:] + x["choices"][:shift]
    tmpl = JA_TEMPLATE if lang == "ja" else EN_TEMPLATE
    return tmpl.format(q=x["q"], choices="\n".join(f"{i + 1}. {c}" for i, c in enumerate(ch))), (x["answer"] - shift) % 5


def post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def gen(port: int, text: str) -> str:
    r = post(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": 16,
                                            "chat_template_kwargs": {"enable_thinking": False}})
    return r["choices"][0]["message"]["content"]


def probs(port: int, text: str, lang: str) -> list[float]:
    """回答の書き出しを「答え：」に固定したときの、次のトークンが 1〜5 である確率。"""
    tmpl = post(port, "/apply-template", {"messages": [{"role": "user", "content": text}],
                                          "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    r = post(port, "/completion", {"prompt": tmpl + PREFIX[lang], "n_predict": 1, "n_probs": 50, "temperature": 0,
                                   "post_sampling_probs": False})
    # すぐに回答を終える（終了のトークンを選ぶ）と確率が返らないので、番号の確率はすべて0とする
    top = (r.get("completion_probabilities") or [{}])[0].get("top_logprobs", [])
    p = [0.0] * 5
    for t in top:
        tok = t["token"].strip().translate(str.maketrans("１２３４５", "12345"))
        if tok in "12345" and len(tok) == 1:
            p[int(tok) - 1] += math.exp(t["logprob"])
    return p


def parse(text: str) -> int | None:
    m = re.search(r"[1-5]", text.translate(str.maketrans("１２３４５", "12345")))
    return int(m.group()) - 1 if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lang", default="ja", choices=["ja", "en"])
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    qs = load(args.lang)[: args.limit or None]
    n = len(qs)

    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda x: gen(args.port, prompt(x, args.lang)[0]), qs))
        rot = {s: list(ex.map(lambda x: probs(args.port, prompt(x, args.lang, s)[0], args.lang), qs)) for s in range(5)}

    preds = [parse(o) for o in outs]
    copy = [bool(COPY.match(o)) for o in outs]
    res = {
        "run": args.run, "lang": args.lang, "n": n,
        "gen_acc": sum(p == x["answer"] for p, x in zip(preds, qs)) / n,
        "gen_copy_rate": sum(copy) / n,
        "gen_acc_without_copy": sum(p == x["answer"] for p, x, c in zip(preds, qs, copy) if not c) / max(1, n - sum(copy)),
        "gen_pred_dist": dict(sorted(Counter(p + 1 if p is not None else None for p in preds).items(), key=lambda k: str(k[0]))),
        "prob_acc": sum(max(range(5), key=lambda i: p[i]) == x["answer"] for p, x in zip(rot[0], qs)) / n,
        # 1〜5 のどれかが出る確率（低いと、「答え：」の後に番号以外を書きたがっている）
        "prob_mass_digits": sum(sum(p) for p in rot[0]) / n,
    }
    # 選択肢をずらしたときの正答率・位置ごとの選ばれやすさ・同じ中身を選んだ割合
    picked_pos, picked_content = Counter(), []
    accs = []
    for s in range(5):
        acc = 0
        for p, x in zip(rot[s], qs):
            k = max(range(5), key=lambda i: p[i])
            picked_pos[k + 1] += 1
            acc += k == (x["answer"] - s) % 5
        accs.append(acc / n)
    for j, x in enumerate(qs):
        contents = {(max(range(5), key=lambda i: rot[s][j][i]) + s) % 5 for s in range(5)}
        picked_content.append(len(contents) == 1)
    res |= {"rotate_acc_each": [round(a, 4) for a in accs], "rotate_acc_mean": sum(accs) / 5,
            "rotate_pos_dist": dict(sorted(picked_pos.items())), "rotate_consistent": sum(picked_content) / n}
    out = ROOT / "eval" / "results" / f"mcqa-local_{args.run}_{args.lang}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
