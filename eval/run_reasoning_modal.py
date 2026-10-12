"""思考ありの推論の力と、考える長さを Modal 上の vLLM で測る（Ornith-1.5-9B など、Qwen3.5 系の推論モデル用）。

「同じ正答率のまま、考える長さを短くする」学習の前後を比べるためのもの。1回答ごとに、正解したかと、回答全体の
トークン数（思考を含む）を記録する。上限 32,768 トークンで一度測れば、「8k・16k 以内に考え終わって正解した割合」は
記録から計算できる（同じ設定なら、上限を下げても途中までは同じ分布で生成されるため）。

- AIME 2026（MathArena/aime_2026、30問、CC BY-NC-SA 4.0）：実行時に読み込み、問題文は保存しない
- MMLU-Pro（TIGER-Lab/MMLU-Pro、MIT）の物理・化学・生物から無作為に 300問（10択）
- MGSM の日本語（juletxara/mgsm の ja、CC BY-SA 4.0、算数の文章題 250問）：0.8B の日本語の理系を測るため

使い方:
    modal run eval/run_reasoning_modal.py --run-name ornith-1.5-9b --model ornith-ai/Ornith-1.5-9B
    # 0.8B：AIME は解けないので外し、上限 8,192 トークン・L4 で。--no-thinking で思考なし（手順を書いてから答えてよい）
    modal run eval/run_reasoning_modal.py --run-name qwen08b-v2-nothink --model /models/qwen08b-soup0.5/merged \
        --tasks mmlu_pro_sci,mgsm_ja --max-tokens 8192 --gpu L4 --no-thinking
    python3 eval/run_reasoning_modal.py summary ornith-1.5-9b   # 集計だけ
    → eval/results/reasoning_raw/<run>.jsonl（回答の全文、git 管理外）と eval/results/reasoning_<run>.json（集計）
"""

import json
import random
import re
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "reasoning_raw"
MAX_TOKENS = 32768
CAPS = (4096, 8192, 16384, 32768)
N_AIME, N_MMLU = 4, 300
AIME_PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}. The answer is an integer between 0 and 999 inclusive."
MMLU_PROMPT = ("Answer the following multiple choice question. Think step by step, then put the letter of the correct option "
               "(A-J) within \\boxed{{}}.\n\nQuestion: {q}\n\n{opts}")
MGSM_PROMPT = "{q}\n\n考えたあと、最後に答えの数字だけを \\boxed{{}} の中に書いてください。"
LETTERS = "ABCDEFGHIJ"
TASKS = ("aime", "mmlu_pro_sci", "mgsm_ja")
# Ornith-1.5-9B の推奨値（一般のタスク）
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0", "pandas", "pyarrow")
    .env({"HF_HUB_CACHE": "/hf-cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
models = modal.Volume.from_name("pitcher-models", create_if_missing=True)
app = modal.App("pitcher-reasoning", image=image)


def load_tasks(names: tuple[str, ...] = TASKS) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    tasks = []
    if "mgsm_ja" in names:
        df = pd.read_parquet(hf_hub_download("juletxara/mgsm", "ja/test-00000-of-00001.parquet", repo_type="dataset"))
        for i, r in enumerate(df.itertuples()):
            tasks.append({"task": "mgsm_ja", "id": f"mgsm-ja-{i:03d}", "answer": str(int(r.answer_number)),
                          "prompt": MGSM_PROMPT.format(q=r.question), "n": 1})
    if "aime" not in names and "mmlu_pro_sci" not in names:
        return tasks
    df = pd.read_parquet(hf_hub_download("MathArena/aime_2026", "data/train-00000-of-00001.parquet", repo_type="dataset"))
    for r in df.itertuples() if "aime" in names else []:
        tasks.append({"task": "aime", "id": f"aime-{int(r.problem_idx):02d}", "answer": str(int(r.answer)),
                      "prompt": AIME_PROMPT.format(problem=r.problem), "n": N_AIME})
    df = pd.read_parquet("https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro/parquet/default/test/0.parquet")
    df = df[df["category"].isin(["physics", "chemistry", "biology"])]
    rows = list(df.itertuples())
    random.Random(0).shuffle(rows)
    for r in rows[:N_MMLU] if "mmlu_pro_sci" in names else []:
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r.options))
        tasks.append({"task": "mmlu_pro_sci", "id": f"mmlupro-{r.question_id}", "answer": r.answer,
                      "prompt": MMLU_PROMPT.format(q=r.question, opts=opts), "n": 1, "category": r.category})
    return tasks


def last_boxed(text: str) -> str | None:
    """最後の \\boxed{...} の中身（入れ子の括弧にも対応）。"""
    i = text.rfind("\\boxed")
    while i != -1:
        j = text.find("{", i)
        if j != -1:
            depth, k = 0, j
            while k < len(text):
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                if depth == 0:
                    inner = re.sub(r"\\[a-z]+|[{}\s,$]", "", text[j + 1:k])
                    if inner:
                        return inner
                    break
                k += 1
        i = text.rfind("\\boxed", 0, i)
    return None


def grade(task: dict, text: str) -> tuple[str | None, bool]:
    # 思考ありでは「</think>」のあとだけを答えとして読む（考え終わらずに切れたら不正解）。思考なしは全文
    if task.get("thinking", True):
        answer_part = text.split("</think>")[-1] if "</think>" in text else ""
    else:
        answer_part = text
    pred = last_boxed(answer_part)
    if pred is None and task["task"] == "mgsm_ja":
        # \\boxed{} を書かなかったときは「答え：」のあとの数字、それもなければ最後の数字
        m = re.findall(r"答え[:：は]\s*([-\d,]+)", answer_part) or re.findall(r"-?\d[\d,]*", answer_part)
        pred = m[-1].replace(",", "") if m else None
    if pred is None:
        return None, False
    if task["task"] == "aime":
        return pred, pred.lstrip("0") == task["answer"].lstrip("0") or pred == task["answer"]
    if task["task"] == "mgsm_ja":
        num = re.sub(r"\.0+$", "", re.sub(r"[^\d\-.]", "", pred))
        return num, num == task["answer"]
    m = re.search(r"[A-J]", pred.upper())
    return (m.group() if m else pred), bool(m) and m.group() == task["answer"]


@app.function(gpu="H100", timeout=3 * 60 * 60, volumes={"/hf-cache": hf_cache, "/models": models})
def generate(model: str, names: tuple[str, ...] = TASKS, max_tokens: int = MAX_TOKENS, thinking: bool = True) -> list[dict]:
    from vllm import LLM, SamplingParams

    tasks = load_tasks(names)
    print(f"{len(tasks)} 問（回答 {sum(t['n'] for t in tasks)} 件）を解かせます", flush=True)
    llm = LLM(model=model, max_model_len=max_tokens + 2048, gpu_memory_utilization=0.92, limit_mm_per_prompt={"image": 0, "video": 0})
    convs = [[{"role": "user", "content": t["prompt"]}] for t in tasks]
    params = [SamplingParams(**SAMPLING, max_tokens=max_tokens, n=t["n"], seed=i) for i, t in enumerate(tasks)]
    # 公開版の 0.8B は常に思考なしのテンプレートにしてあるので、思考ありは Qwen3.5 の元のテンプレートと同じ形
    # （思考なしの形の「<think>\n\n</think>\n\n」を「<think>\n」にしたもの）を自分で作る
    tok = llm.get_tokenizer()
    prompts = []
    for c in convs:
        p = tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if thinking and p.endswith("<think>\n\n</think>\n\n"):
            p = p.removesuffix("\n</think>\n\n")
        elif thinking:
            p = tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prompts.append(p)
    outs = llm.generate(prompts, params)
    rows = []
    for t, o in zip(tasks, outs):
        for k, c in enumerate(o.outputs):
            rows.append({"task": t["task"], "id": t["id"], "sample": k, "answer": t["answer"], "thinking": thinking, "output": c.text,
                         "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason})
    return rows


def summarize(run: str) -> dict:
    rows = [json.loads(l) for l in (RAW / f"{run}.jsonl").read_text().splitlines()]
    res = {"run": run}
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task]
        if not rs:
            continue
        n = len(rs)
        toks = sorted(r["n_tokens"] for r in rs)
        res[task] = {
            "n": n,
            # 上限ごとの正答率：その上限以内に書き終え、かつ正解した割合
            "acc_at_cap": {str(c): round(sum(r["correct"] and r["n_tokens"] <= c for r in rs) / n, 4) for c in CAPS},
            "acc": round(sum(r["correct"] for r in rs) / n, 4),
            "truncated": round(sum(r["finish_reason"] == "length" for r in rs) / n, 4),
            "tokens_mean": round(sum(toks) / n), "tokens_median": toks[n // 2],
            "tokens_mean_correct": round(sum(r["n_tokens"] for r in rs if r["correct"]) / max(1, sum(r["correct"] for r in rs))),
        }
    (HERE / "results" / f"reasoning_{run}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


@app.local_entrypoint()
def main(run_name: str, model: str, tasks: str = ",".join(TASKS), max_tokens: int = MAX_TOKENS, gpu: str = "H100",
         no_thinking: bool = False):
    names = tuple(t for t in tasks.split(",") if t)
    assert set(names) <= set(TASKS), names
    rows = generate.with_options(gpu=gpu).remote(model, names, max_tokens, not no_thinking)
    for r in rows:
        r["pred"], r["correct"] = grade(r, r["output"])
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{run_name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps(summarize(run_name), ensure_ascii=False, indent=1))


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "summary":
    for run in sys.argv[2:]:
        print(json.dumps(summarize(run), ensure_ascii=False, indent=1))
