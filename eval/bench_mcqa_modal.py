"""日本語と英語の常識問題（5択）を同じ形式で解かせて、日本語だけが弱いのかを調べる。

- 日本語: JCommonsenseQA（sbintuitions/JCommonsenseQA、CC BY-SA 4.0）の validation 1,119問
- 英語:   CommonsenseQA（tau/commonsense_qa、MIT）の validation 1,221問
- 理科:   MMLU-Pro（TIGER-Lab/MMLU-Pro、MIT）の物理・化学・生物から無作為に300問（10択、記号で答える）

JCommonsenseQA は CommonsenseQA を参考に作られた日本語版なので、問題の形式がそろっている。
どちらも thinking なし・貪欲法で、選択肢の番号だけを答えさせる。

使い方:
    modal run eval/bench_mcqa_modal.py
    modal run eval/bench_mcqa_modal.py --models openbmb/MiniCPM5-1B
    modal run eval/bench_mcqa_modal.py --models /models/qwen08b-tengen-clean/merged   # 学習したモデル（mcqa_qwen08b-tengen-clean.jsonl）
"""

import json
import pathlib
import re

import modal

HERE = pathlib.Path(__file__).parent
DEFAULT_MODELS = "Qwen/Qwen3.5-0.8B,openbmb/MiniCPM5-1B,Qwen/Qwen3.5-4B,openbmb/MiniCPM5-2B"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0", "pandas", "pyarrow")
    .env({"HF_HUB_CACHE": "/hf-cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("pitcher-models", create_if_missing=True)
app = modal.App("pitcher-bench-mcqa", image=image)

JA_URL = "https://huggingface.co/api/datasets/sbintuitions/JCommonsenseQA/parquet/default/validation/0.parquet"
EN_URL = "https://huggingface.co/api/datasets/tau/commonsense_qa/parquet/default/validation/0.parquet"
JA_TEMPLATE = "次の質問に対して、最も適切な答えを選択肢から1つ選び、番号（1〜5）だけを答えてください。\n\n質問：{q}\n{choices}\n\n答え："
# 理科：MMLU-Pro（TIGER-Lab/MMLU-Pro、MIT）の物理・化学・生物から無作為に300問（eval/run_reasoning_modal.py と同じ選び方）。10択
SCI_URL = "https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro/parquet/default/test/0.parquet"
SCI_TEMPLATE = "Choose the most appropriate answer to the following question from the options, and reply with only its letter (A-J).\n\nQuestion: {q}\n{choices}\n\nAnswer:"
EN_TEMPLATE = "Choose the most appropriate answer to the following question from the options, and reply with only its number (1-5).\n\nQuestion: {q}\n{choices}\n\nAnswer:"


def load_questions() -> list[dict]:
    import pandas as pd

    rows = []
    for r in pd.read_parquet(JA_URL).itertuples():
        choices = [getattr(r, f"choice{i}") for i in range(5)]
        rows.append({"lang": "ja", "id": str(r.q_id), "question": r.question, "choices": choices, "answer": int(r.label) + 1})
    for r in pd.read_parquet(EN_URL).itertuples():
        labels, texts = list(r.choices["label"]), list(r.choices["text"])
        rows.append({"lang": "en", "id": r.id, "question": r.question, "choices": texts, "answer": labels.index(r.answerKey) + 1})
    import random

    sci = pd.read_parquet(SCI_URL)
    sci = list(sci[sci["category"].isin(["physics", "chemistry", "biology"])].itertuples())
    random.Random(0).shuffle(sci)
    for r in sci[:300]:
        opts = list(r.options)
        rows.append({"lang": "sci", "id": f"mmlupro-{r.question_id}", "question": r.question, "choices": opts,
                     "answer": "ABCDEFGHIJ".index(r.answer) + 1,
                     "prompt": SCI_TEMPLATE.format(q=r.question, choices="\n".join(f"{'ABCDEFGHIJ'[i]}. {c}" for i, c in enumerate(opts)))})
    for x in rows:
        if x["lang"] == "sci":
            continue
        tmpl = JA_TEMPLATE if x["lang"] == "ja" else EN_TEMPLATE
        x["prompt"] = tmpl.format(q=x["question"], choices="\n".join(f"{i + 1}. {c}" for i, c in enumerate(x["choices"])))
    return rows


def parse_letter(text: str) -> int | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\b([A-J])\b", text.translate(str.maketrans("ＡＢＣＤＥＦＧＨＩＪ", "ABCDEFGHIJ")))
    return "ABCDEFGHIJ".index(m.group(1)) + 1 if m else None


def parse_choice(text: str) -> int | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.translate(str.maketrans("１２３４５", "12345"))
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None


@app.function(gpu="L4", timeout=60 * 60, volumes={"/hf-cache": hf_cache, "/models": models_vol})
def run(model: str) -> list[dict]:
    from vllm import LLM, SamplingParams

    kwargs = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "Qwen3.5" in model or "qwen08b" in model else {}
    llm = LLM(model=model, max_model_len=4096, **kwargs)
    qs = load_questions()
    outs = llm.chat([[{"role": "user", "content": q["prompt"]}] for q in qs], SamplingParams(temperature=0.0, max_tokens=16),
                    chat_template_kwargs={"enable_thinking": False})
    res = []
    for q, o in zip(qs, outs):
        text = o.outputs[0].text
        pred = parse_letter(text) if q["lang"] == "sci" else parse_choice(text)
        res.append({"model": model, "lang": q["lang"], "id": q["id"], "answer": q["answer"], "pred": pred,
                    "correct": pred == q["answer"], "output": text})
    return res


@app.local_entrypoint()
def main(models: str = DEFAULT_MODELS):
    names = models.split(",")
    out_dir = HERE / "results"
    for name, res in zip(names, run.map(names)):
        # 学習したモデル（/models/<run名>/merged）は run 名で保存する
        (out_dir / f"mcqa_{name.removesuffix('/merged').split('/')[-1]}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res))
        for lang in ("ja", "en", "sci"):
            rs = [r for r in res if r["lang"] == lang]
            acc = sum(r["correct"] for r in rs) / len(rs)
            none = sum(r["pred"] is None for r in rs) / len(rs)
            print(f"{name:24} {lang}: 正答率 {acc:.1%}（{len(rs)}問、番号を答えなかった {none:.1%}）")
