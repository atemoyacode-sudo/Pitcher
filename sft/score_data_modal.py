"""学習データ（質問と回答の組）を Gemma 4 31B に採点させ、質の高い例だけを選べるようにする（AlpaGasus 型の選別）。

評価用の採点（eval/judge_modal.py）のままでは、Gemma は話題の運びが少し唐突なだけの回答まで「支離滅裂」として
1〜2点をつけ、人の採点と「破綻しているか」の判定が30件中10件しか一致しなかった。そこで、学習データの選別用に
指示文を作り直し、まず人が採点した30件（private/human_eval/、git 管理外）で人との一致を確かめる。

GPT-5.6 Luna の採点結果は、学習データの選別には使わない（OpenAI の利用規約。評価の採点にだけ使う）。
人との比較の表に Luna を並べるのは参考のため。

人が採点した30件で比べた結果（2026-10-01）：「破綻しているか」の判定が人と一致したのは、
減点の根拠を引用させる指示文（evidence）で 30件中26件（Luna は27件、評価用の指示文の Gemma は10件）。
点数だけを答えさせる指示文（plain）は18件。選別には evidence を使う。

使い方:
    modal run sft/score_data_modal.py --calibrate   # 指示文の候補ごとに、人が採点した30件との一致を表示
    modal run sft/score_data_modal.py --pool sft/tengentoppa_clean_pool.jsonl   # 候補を採点（途中から再開できる）
    python3 sft/score_data_modal.py select sft/tengentoppa_clean_pool.jsonl     # 点数の高い順に長文 9,000・短文 3,000件を選ぶ
    modal run sft/score_data_modal.py --responses data/dpo/responses_<run>.jsonl   # モデル自身の回答を採点（DPO 用）
"""

import json
import pathlib
import random
import re
import sys

from collections import Counter

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
JUDGE_MODEL = "google/gemma-4-31B-it"

HEAD = """あなたは、日本語の会話AIの学習データを審査する担当者です。ユーザーの質問と、それに対する回答を読み、
次の3つの観点で1〜5点をつけてください。ほとんどの回答は問題がないので、はっきりした問題がない限り減点しないでください。

【評価の観点】
1. 一貫性（consistency）：回答の中で、自分の発言どうしがはっきり矛盾していないか、文章が崩れて話の筋が追えなくなっていないか。
   5＝矛盾も崩れもない／3＝一部で話がつながらない箇所がある／1＝前後で明らかに矛盾する、または筋が追えない
   次のことは一貫性では減点しない：話題の広げ方が少し唐突、例の選び方がいまひとつ、内容の事実の誤り、
   質問とのかみ合わせ（これは適合で減点する）、言い回しの不自然さ（これは自然さで減点する）。
2. 日本語の自然さ（naturalness）：日本語として自然か。
   5＝母語話者が書いたように自然／4＝小さな不自然さが1〜2か所／3＝不自然な言い回しや誤用が目立つ
   2＝文法の崩れが多く読みにくい／1＝意味が取れない、または日本語以外の言語が中心
   日本語では使わない漢字（簡体字）や中国語の語句が混ざっていたら、1か所でも3点以下にする。
3. 質問への適合（relevance）：ユーザーの質問の意図に応えているか。
   5＝的確に応えている／4＝おおむね応えているが、余計な部分や足りない部分がある／3＝一部だけ応えている
   2＝ほとんど応えていない／1＝質問と無関係

【ユーザーの質問】
{prompt}

【回答】
{output}
"""

# 指示文の候補。同じ回でまとめて採点し、人との一致を比べる
VARIANTS = {
    # 点数だけ
    "plain": HEAD + """
次の JSON だけを出力してください。説明は不要です。
{{"consistency": 点数, "naturalness": 点数, "relevance": 点数}}""",
    # 先に減点の根拠を回答から引用させ、引用できない減点をさせない
    "evidence": HEAD + """
減点するときは、その根拠になる箇所を回答から短く引用してください。引用できない減点はしないでください。
次の JSON だけを出力してください。
{{"evidence": "減点の根拠の引用（なければ空文字）", "consistency": 点数, "naturalness": 点数, "relevance": 点数}}""",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0")
    .env({"HF_HUB_CACHE": "/hf-cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
app = modal.App("pitcher-score-data", image=image)


@app.cls(gpu="H100", timeout=60 * 60, scaledown_window=120, volumes={"/hf-cache": hf_cache})
class Scorer:
    @modal.enter()
    def load(self):
        from vllm import LLM

        # judge_modal.py と同じく、重みを FP8 にして KV キャッシュの空きを作る
        self.llm = LLM(model=JUDGE_MODEL, max_model_len=4096, quantization="fp8", gpu_memory_utilization=0.92,
                       limit_mm_per_prompt={"image": 0})

    @modal.method()
    def score(self, prompts: list[str]) -> list[str]:
        from vllm import SamplingParams

        outs = self.llm.chat([[{"role": "user", "content": p}] for p in prompts],
                             SamplingParams(temperature=0.0, max_tokens=300), chat_template_kwargs={"enable_thinking": False})
        return [o.outputs[0].text for o in outs]


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        d = json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        return None
    return d if d and all(isinstance(d.get(k), int) for k in ("consistency", "naturalness", "relevance")) else None


def agreement(name: str, preds: list[dict | None], human: list[dict]):
    keys = ("consistency", "naturalness", "relevance")
    ok = [(p, h) for p, h in zip(preds, human) if p]
    mean = " / ".join(f"{sum(p[k] for p, _ in ok) / len(ok):.2f}" for k in keys)
    mae = " / ".join(f"{sum(abs(p[k] - h[k]) for p, h in ok) / len(ok):.2f}" for k in keys)
    broken = sum(p["consistency"] <= 2 for p, _ in ok)
    agree = sum((p["consistency"] <= 2) == (h["consistency"] <= 2) for p, h in ok)
    print(f"{name:10} 読めた {len(ok):2}/{len(preds)}  平均 {mean}  破綻 {broken:2}件  人と一致 {agree:2}/{len(ok)}  人との差 {mae}")


def scores_path(pool: pathlib.Path) -> pathlib.Path:
    return pool.with_name(pool.stem + "_scores.jsonl")


def score_pool(pool: pathlib.Path, chunk: int = 3000):
    rows = [json.loads(l) for l in pool.read_text().splitlines()]
    out = scores_path(pool)
    done = {json.loads(l)["source_row"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in rows if r["source_row"] not in done]
    print(f"{len(rows)} 件中 {len(done)} 件は採点済み、残り {len(todo)} 件")
    scorer = Scorer()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        texts = scorer.score.remote([VARIANTS["evidence"].format(prompt=r["messages"][0]["content"],
                                                                output=r["messages"][1]["content"].strip()[:2500]) for r in part])
        with out.open("a") as f:
            for r, t in zip(part, texts):
                f.write(json.dumps({"source_row": r["source_row"], "kind": r["kind"], "scores": parse(t), "raw": t}, ensure_ascii=False) + "\n")
        print(f"{min(i + chunk, len(todo))} / {len(todo)}", flush=True)


def select(pool: pathlib.Path, n_long: int = 9000, n_short: int = 3000):
    """3観点の合計点が高い順に選ぶ（同点は無作為）。読めなかった採点は除く。"""
    rows = {json.loads(l)["source_row"]: json.loads(l) for l in pool.read_text().splitlines()}
    sc = [json.loads(l) for l in scores_path(pool).read_text().splitlines()]
    rng = random.Random(0)
    picked = []
    for kind, n in (("long", n_long), ("short", n_short)):
        cand = [s for s in sc if s["kind"] == kind and s["scores"]]
        rng.shuffle(cand)
        cand.sort(key=lambda s: -sum(s["scores"][k] for k in ("consistency", "naturalness", "relevance")))
        tot = [sum(s["scores"][k] for k in ("consistency", "naturalness", "relevance")) for s in cand]
        print(f"{kind}: 採点 {len(cand)} 件、合計点の分布 {dict(sorted(Counter(tot).items()))}、選んだ最低点 {tot[n - 1]}")
        picked += [rows[s["source_row"]] for s in cand[:n]]
    rng.shuffle(picked)
    out = pool.with_name(pool.stem.replace("_pool", "_scored") + "_train.jsonl")
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in picked))
    print(f"{out}：{len(picked)} 件")


def score_responses(path: pathlib.Path, chunk: int = 3000):
    """モデル自身の回答（sft/q08b_v2/dpo_modal.py の generate の出力。1行＝1問、samples に複数の回答）を採点する。
    崩れにくくする学習（DPO）で、良い回答と崩れた回答の組を作るため。途中から再開できる。"""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = scores_path(path)
    done = {(d["id"], d["k"]) for d in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    todo = [(r, s) for r in rows for s in r["samples"] if (r["id"], s["k"]) not in done]
    print(f"回答 {sum(len(r['samples']) for r in rows)} 件中 {len(done)} 件は採点済み、残り {len(todo)} 件")
    scorer = Scorer()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        texts = scorer.score.remote([VARIANTS["evidence"].format(prompt=r["prompt"], output=s["output"].strip()[:2500]) for r, s in part])
        with out.open("a") as f:
            for (r, s), t in zip(part, texts):
                f.write(json.dumps({"id": r["id"], "k": s["k"], "scores": parse(t), "raw": t}, ensure_ascii=False) + "\n")
        print(f"{min(i + chunk, len(todo))} / {len(todo)}", flush=True)


@app.local_entrypoint()
def main(calibrate: bool = False, pool: str = "", responses: str = ""):
    if pool:
        return score_pool(ROOT / pool)
    if responses:
        return score_responses(ROOT / responses)
    items = json.loads((ROOT / "private" / "human_eval" / "items_with_scores.json").read_text())
    res = json.loads((ROOT / "private" / "human_eval" / "human_eval_results.json").read_text())["results"]
    human = [res[str(i + 1)] for i in range(len(items))]
    prompts = [VARIANTS[v].format(prompt=it["prompt"], output=it["output"].strip()[:2500]) for v in VARIANTS for it in items]
    outs = Scorer().score.remote(prompts)
    print("一貫性 / 自然さ / 適合。破綻＝一貫性2点以下")
    agreement("人", human, human)
    agreement("Luna(参考)", [it["luna"] for it in items], human)
    agreement("Gemma(旧)", [it["gemma"] for it in items], human)
    for j, v in enumerate(VARIANTS):
        preds = [parse(t) for t in outs[j * len(items):(j + 1) * len(items)]]
        agreement(v, preds, human)
    out = ROOT / "private" / "human_eval" / "gemma_data_calibration.json"
    out.write_text(json.dumps({v: outs[j * len(items):(j + 1) * len(items)] for j, v in enumerate(VARIANTS)}, ensure_ascii=False, indent=1))
    print(f"生の出力: {out}")


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "select":
    select(ROOT / sys.argv[2])
