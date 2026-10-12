"""理系（算数・数学・理科）の手本を、先生役（Qwen3.8-27B、思考なし）に日本語で作らせる（Modal）。0.8B の v2.5 の準備。

0.8B は MMLU-Pro の理科（10択）で 24% と弱い。答えが決まっている問題を、先生役に「自然な日本語に訳す → 短い手順で解く →
最後に答え」の形で書かせ、正解と一致したものだけを残す（間違った解き方を学ばせないため）。

- 算数の文章題：openai/gsm8k の train（MIT）から 1,500問（評価に使う MGSM は GSM8K の test から作られているので重ならない）
- 数学：data/reasoning/problems_pool.jsonl のうち DeepMath-103K（MIT）の難しさ 5 以下の整数の答えの問題 800問
- 理科：同じく OpenScienceReasoning-2（CC BY 4.0）の選択問題 1,000問（評価に使う MMLU-Pro と文面が重なるものは除外済み）

    modal run sft/gen_sci_modal.py   # data/reasoning/sci_teacher.jsonl
"""

import json
import pathlib
import random
import re

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "reasoning" / "sci_teacher.jsonl"
TEACHER = "Qwen/Qwen3.8-27B"
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)
MATH = """次の問題を、まず自然な日本語の問題文に書き直し、そのあと日本語で手順を短く書いて解いてください。

出力の形式（この形だけを書く）：
問題：（日本語の問題文）
解き方：（必要な式と説明を、3〜8行で）
答え：（数字だけ）

{q}"""
SCI = """次の選択問題を、まず自然な日本語に書き直し（選択肢の記号 A〜J はそのまま）、そのあと日本語で考え方を短く説明して答えてください。

出力の形式（この形だけを書く）：
問題：（日本語の問題文と選択肢。選択肢は1行に1つずつ「A. 〜」の形で）
解き方：（2〜6行）
答え：（記号1文字だけ）

{q}"""

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0", "pandas", "pyarrow")
    .env({"HF_HUB_CACHE": "/hf-cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
app = modal.App("pitcher-gen-sci", image=image)


@app.function(gpu="H100", timeout=2 * 60 * 60, volumes={"/hf-cache": hf_cache})
def run(items: list[dict], n_gsm: int) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset"))
    gsm = [{"task": "gsm8k", "source": "openai/gsm8k", "license": "mit", "answer": a.split("####")[-1].strip().replace(",", ""),
            "question": q} for q, a in zip(df["question"], df["answer"])]
    random.Random(0).shuffle(gsm)
    items = gsm[:n_gsm] + items
    llm = LLM(model=TEACHER, max_model_len=4096, quantization="fp8", gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    convs = [[{"role": "user", "content": (SCI if it["task"] == "sci" else MATH).format(q=it["question"])}] for it in items]
    params = [SamplingParams(**SAMPLING, max_tokens=900, seed=i) for i in range(len(items))]
    outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False})
    return [{**it, "output": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason} for it, o in zip(items, outs)]


def parse(text: str) -> dict | None:
    m = re.search(r"問題[:：]\s*(.+?)\n\s*解き方[:：]\s*(.+?)\n\s*答え[:：]\s*(.+?)\s*$", text.strip(), flags=re.S)
    return {"question": m.group(1).strip(), "solution": m.group(2).strip(), "answer": m.group(3).strip()} if m else None


@app.local_entrypoint()
def main(n_gsm: int = 1500, n_math: int = 800, n_sci: int = 1000):
    pool = [json.loads(l) for l in (ROOT / "data" / "reasoning" / "problems_pool.jsonl").read_text().splitlines()]
    strip = lambda p: p.split("\n\nPlease reason step by step")[0]
    math = [p for p in pool if p["task"] == "math" and p.get("difficulty", 9) <= 5][:n_math]
    sci = [p for p in pool if p["task"] == "sci"][:n_sci]
    items = [{"task": "math", "source": p["source"], "license": "mit", "answer": p["answer"], "question": strip(p["prompt"])} for p in math]
    items += [{"task": "sci", "source": p["source"], "license": "cc-by-4.0", "answer": p["answer"],
               "question": p["prompt"].split("\\boxed{}.\n\n", 1)[-1]} for p in sci]
    rows = run.remote(items, n_gsm)
    keep, stat = [], {}
    for r in rows:
        p = parse(r["output"]) if r["finish_reason"] == "stop" else None
        ok = False
        if p:
            a = p["answer"].strip("。. ")
            ok = (a.upper()[:1] == r["answer"]) if r["task"] == "sci" else (re.sub(r"[^\d\-]", "", a) == r["answer"])
        stat.setdefault(r["task"], [0, 0])
        stat[r["task"]][0] += 1
        stat[r["task"]][1] += ok
        if ok:
            keep.append({"messages": [{"role": "user", "content": p["question"]},
                                      {"role": "assistant", "content": f"{p['solution']}\n\n答え：{p['answer'].strip('。. ')}"}],
                         "task": r["task"], "source": r["source"], "license": r["license"], "dataset": "sci_teacher"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in keep))
    print("正解と一致した数（全体 / 一致）", stat)
    print(f"{OUT}：{len(keep)} 件")
