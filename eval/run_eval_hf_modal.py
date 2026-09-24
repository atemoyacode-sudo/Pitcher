"""vLLM が対応していないモデル（Spark-X2.5 など）を、transformers で直接動かして評価する。

Spark-X2.5 は独自構造で、同梱のコード（modeling_spark.py、中身は確認済み）が transformers 4.57 向けに書かれている。
vLLM 0.30 には対応がなく、同梱の transformers 5 系では設定ファイルの読み込みに失敗するため、4.57.1 を使う。
出力の形式は run_eval_modal.py / bench_mcqa_modal.py と同じなので、score.py などはそのまま使える。

使い方:
    modal run eval/run_eval_hf_modal.py::eval140 --run-name spark-x2.5-4b --model XHToken/Spark-X2.5-4B
    modal run eval/run_eval_hf_modal.py::mcqa --model XHToken/Spark-X2.5-4B
"""

import json
import pathlib
import re

import modal

HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.0", "transformers==4.57.1", "accelerate", "pandas", "pyarrow")
    .env({"HF_HUB_CACHE": "/hf-cache"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("pitcher-models", create_if_missing=True)
app = modal.App("pitcher-eval-hf", image=image)

# Spark-X2.5 のモデルカード推奨値（thinking なし）
SAMPLED = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=0, max_new_tokens=512)
GREEDY = dict(do_sample=False, max_new_tokens=256)
N_SAMPLES = {"emotion": 5, "general": 3, "knowledge": 1, "knowledge_en": 1}


@app.function(gpu="A100-80GB", timeout=3 * 60 * 60, volumes={"/hf-cache": hf_cache, "/models": models_vol})
def generate(model_id: str, convs: list[list[dict]], cfgs: list[dict], seed: int = 0, batch_size: int = 48) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False) for c in convs]

    # 同じ生成設定・近い長さのものをまとめて、パディングを減らす
    order = sorted(range(len(texts)), key=lambda i: (json.dumps(cfgs[i], sort_keys=True), len(texts[i])))
    results = [None] * len(texts)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        cfg_groups = {}
        for i in idx:
            cfg_groups.setdefault(json.dumps(cfgs[i], sort_keys=True), []).append(i)
        for cfg_json, ids in cfg_groups.items():
            cfg = json.loads(cfg_json)
            torch.manual_seed(seed + b)
            enc = tok([texts[i] for i in ids], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, **cfg, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            for k, i in enumerate(ids):
                gen = out[k, enc["input_ids"].shape[1]:]
                ended = (gen == tok.eos_token_id).any().item()
                text = tok.decode(gen, skip_special_tokens=True)
                results[i] = {"output": re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip(),
                              "finish_reason": "stop" if ended else "length"}
        print(f"{min(b + batch_size, len(order))}/{len(order)}")
    return results


@app.function(timeout=600)
def mcqa_questions() -> list[dict]:
    # bench_mcqa_modal.py と同じ問題・同じ書式
    import pandas as pd

    ja_t = "次の質問に対して、最も適切な答えを選択肢から1つ選び、番号（1〜5）だけを答えてください。\n\n質問：{q}\n{choices}\n\n答え："
    en_t = "Choose the most appropriate answer to the following question from the options, and reply with only its number (1-5).\n\nQuestion: {q}\n{choices}\n\nAnswer:"
    rows = []
    for r in pd.read_parquet("https://huggingface.co/api/datasets/sbintuitions/JCommonsenseQA/parquet/default/validation/0.parquet").itertuples():
        rows.append({"lang": "ja", "id": str(r.q_id), "question": r.question, "choices": [getattr(r, f"choice{i}") for i in range(5)], "answer": int(r.label) + 1})
    for r in pd.read_parquet("https://huggingface.co/api/datasets/tau/commonsense_qa/parquet/default/validation/0.parquet").itertuples():
        labels, texts = list(r.choices["label"]), list(r.choices["text"])
        rows.append({"lang": "en", "id": r.id, "question": r.question, "choices": texts, "answer": labels.index(r.answerKey) + 1})
    for x in rows:
        t = ja_t if x["lang"] == "ja" else en_t
        x["prompt"] = t.format(q=x["question"], choices="\n".join(f"{i + 1}. {c}" for i, c in enumerate(x["choices"])))
    return rows


@app.local_entrypoint()
def eval140(run_name: str, model: str, seed: int = 0, only: str = ""):
    items = json.loads((HERE / "prompts.json").read_text())["items"]
    prefixes = [p for p in only.split(",") if p]
    if prefixes:
        items = [it for it in items if any(it["id"].startswith(p) for p in prefixes)]
    convs, cfgs, meta = [], [], []
    for it in items:
        msgs = ([{"role": "system", "content": it["system"]}] if it["system"] else []) + [{"role": "user", "content": it["prompt"]}]
        for k in range(N_SAMPLES[it["category"]]):
            convs.append(msgs)
            cfgs.append(GREEDY if it["category"].startswith("knowledge") else SAMPLED)
            meta.append((it, k))
    outs = generate.remote(model, convs, cfgs, seed)
    rows = [{"run": run_name, "model": model, **it, "sample": k, **o} for (it, k), o in zip(meta, outs)]
    out = HERE / "results" / f"{run_name}.jsonl"
    if prefixes and out.exists():
        keys = {(r["id"], r["sample"]) for r in rows}
        rows = [r for r in map(json.loads, out.read_text().splitlines()) if (r["id"], r["sample"]) not in keys] + rows
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} 件を {out} に保存しました")


@app.local_entrypoint()
def mcqa(model: str):
    qs = mcqa_questions.remote()
    outs = generate.remote(model, [[{"role": "user", "content": q["prompt"]}] for q in qs], [dict(do_sample=False, max_new_tokens=16)] * len(qs))
    res = []
    for q, o in zip(qs, outs):
        t = o["output"].translate(str.maketrans("１２３４５", "12345"))
        m = re.search(r"[1-5]", t)
        pred = int(m.group()) if m else None
        res.append({"model": model, "lang": q["lang"], "id": q["id"], "answer": q["answer"], "pred": pred,
                    "correct": pred == q["answer"], "output": o["output"]})
    (HERE / "results" / f"mcqa_{model.split('/')[-1]}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res))
    for lang in ("ja", "en"):
        rs = [r for r in res if r["lang"] == lang]
        print(f"{model} {lang}: 正答率 {sum(r['correct'] for r in rs) / len(rs):.1%}（{len(rs)}問、番号を答えなかった {sum(r['pred'] is None for r in rs) / len(rs):.1%}）")
