"""Mac 上の llama.cpp（llama-server、Metal）で評価用の回答を作る。出力の形式は run_eval_hf_colab.py と同じ。

Spark-X2.5 は transformers だと Mac の GPU でも毎秒十数トークンと遅いので、GGUF に変換して llama.cpp で動かす。
変換（量子化なしの f16）:
    .venv-eval/bin/python tools/llama.cpp-<commit>/convert_hf_to_gguf.py <HF のモデルのフォルダ> --outtype f16 --outfile tools/gguf/<名前>.gguf

使い方:
    python3 eval/run_eval_llamacpp.py eval140 --gguf tools/gguf/spark-ja2-4b-f16.gguf --run-name spark-ja2-4b
    python3 eval/run_eval_llamacpp.py mcqa --gguf tools/gguf/spark-ja2-4b-f16.gguf --run-name spark-ja2-4b
    python3 eval/run_eval_llamacpp.py eval140 --gguf ... --run-name ... --only know-   # 一部の質問だけ
    python3 eval/run_eval_llamacpp.py eval140 --gguf tools/gguf/spark-ja2-4b-f16.gguf --run-name spark-ja2-4b-long --only emo-none,gen --max-tokens 2048
    python3 eval/run_eval_llamacpp.py eval140 --qwen --gguf tools/gguf/qwen08b-tengen-scored-fmt-Q4_K_M.gguf --run-name qwen08b-tengen-scored-fmt-q4km   # 0.8B の公開用 GGUF の確認
"""

import argparse
import json
import re
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
# run_eval_hf_colab.py と同じ生成設定（Spark-X2.5 の推奨値、思考なし）。llama.cpp の top_k=0 は「使わない」
SAMPLED = dict(temperature=1.0, top_p=0.95, top_k=0, max_tokens=512)
# Qwen3.5 系（0.8B など）は run_eval_modal.py と同じ Qwen の推奨値（--qwen）
SAMPLED_QWEN = dict(temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0, max_tokens=512)
GREEDY = dict(temperature=0.0, max_tokens=256)
N_SAMPLES = {"emotion": 5, "general": 3, "knowledge": 1, "knowledge_en": 1}
PARALLEL = 8
PORT = 8091


def start_server(gguf: str, max_tokens: int = 0) -> subprocess.Popen:
    log = open(HERE / "results" / "llama-server.log", "w")
    p = subprocess.Popen(["llama-server", "-m", gguf, "--port", str(PORT), "-ngl", "99", "-np", str(PARALLEL),
                          "-c", str(max(4096, max_tokens + 1024) * PARALLEL), "--jinja", "--no-webui"], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok":
                return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise RuntimeError("llama-server が起動しませんでした（eval/results/llama-server.log を確認）")


def chat(messages: list[dict], cfg: dict, seed: int) -> dict:
    body = {"messages": messages, **cfg, "seed": seed, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]
    text = re.sub(r"<think>.*?</think>\s*", "", r["message"].get("content") or "", flags=re.S).strip()
    return {"output": text, "finish_reason": "length" if r.get("finish_reason") == "length" else "stop"}


def run_all(jobs: list[tuple[list[dict], dict, int]]) -> list[dict]:
    done, t0 = [0], time.time()

    def one(job):
        out = chat(*job)
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"{done[0]}/{len(jobs)}（{time.time() - t0:.0f}秒）", flush=True)
        return out

    with ThreadPoolExecutor(PARALLEL) as ex:
        return list(ex.map(one, jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval140", "mcqa"])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat-penalty", type=float, default=0, help="llama.cpp の repeat_penalty（直近64トークンに出た語を出にくくする）")
    ap.add_argument("--qwen", action="store_true", help="Qwen3.5 系の推奨の生成設定を使う（0.8B など）")
    ap.add_argument("--max-tokens", type=int, default=0, help="回答の長さの上限（既定 512）。打ち切りの影響を調べる再テスト用")
    a = ap.parse_args()

    if a.qwen:
        SAMPLED.clear()
        SAMPLED.update(SAMPLED_QWEN)
    if a.repeat_penalty:  # 公開用の推奨設定の確認用（0.8B は笑い声などをまれに繰り返し続けるため）
        SAMPLED["repeat_penalty"] = a.repeat_penalty
    if a.max_tokens:
        SAMPLED["max_tokens"] = a.max_tokens
    server = start_server(a.gguf, a.max_tokens)
    try:
        if a.mode == "eval140":
            items = json.loads((HERE / "prompts.json").read_text())["items"]
            prefixes = [p for p in a.only.split(",") if p]
            if prefixes:
                items = [it for it in items if any(it["id"].startswith(p) for p in prefixes)]
            jobs, meta = [], []
            for it in items:
                msgs = ([{"role": "system", "content": it["system"]}] if it["system"] else []) + [{"role": "user", "content": it["prompt"]}]
                for k in range(N_SAMPLES[it["category"]]):
                    jobs.append((msgs, GREEDY if it["category"].startswith("knowledge") else SAMPLED, a.seed + k))
                    meta.append((it, k))
            outs = run_all(jobs)
            rows = [{"run": a.run_name, "model": a.gguf, **it, "sample": k, **o} for (it, k), o in zip(meta, outs)]
            path = HERE / "results" / f"{a.run_name}.jsonl"
            if prefixes and path.exists():
                keys = {(r["id"], r["sample"]) for r in rows}
                rows = [r for r in map(json.loads, path.read_text().splitlines()) if (r["id"], r["sample"]) not in keys] + rows
        else:
            import pandas as pd

            ja_t = "次の質問に対して、最も適切な答えを選択肢から1つ選び、番号（1〜5）だけを答えてください。\n\n質問：{q}\n{choices}\n\n答え："
            en_t = "Choose the most appropriate answer to the following question from the options, and reply with only its number (1-5).\n\nQuestion: {q}\n{choices}\n\nAnswer:"
            qs = []
            for r in pd.read_parquet("https://huggingface.co/api/datasets/sbintuitions/JCommonsenseQA/parquet/default/validation/0.parquet").itertuples():
                qs.append({"lang": "ja", "id": str(r.q_id), "choices": [getattr(r, f"choice{i}") for i in range(5)], "answer": int(r.label) + 1, "q": r.question})
            for r in pd.read_parquet("https://huggingface.co/api/datasets/tau/commonsense_qa/parquet/default/validation/0.parquet").itertuples():
                labels = list(r.choices["label"])
                qs.append({"lang": "en", "id": r.id, "choices": list(r.choices["text"]), "answer": labels.index(r.answerKey) + 1, "q": r.question})
            jobs = [([{"role": "user", "content": (ja_t if q["lang"] == "ja" else en_t).format(
                q=q["q"], choices="\n".join(f"{i + 1}. {c}" for i, c in enumerate(q["choices"])))}], dict(temperature=0.0, max_tokens=16), 0) for q in qs]
            outs = run_all(jobs)
            rows = []
            for q, o in zip(qs, outs):
                m = re.search(r"[1-5]", o["output"].translate(str.maketrans("１２３４５", "12345")))
                pred = int(m.group()) if m else None
                rows.append({"model": a.gguf, "lang": q["lang"], "id": q["id"], "answer": q["answer"], "pred": pred,
                             "correct": pred == q["answer"], "output": o["output"]})
            for lang in ("ja", "en"):
                rs = [r for r in rows if r["lang"] == lang]
                print(f"{a.run_name} {lang}: 正答率 {sum(r['correct'] for r in rs) / len(rs):.1%}（{len(rs)}問）", flush=True)
            path = HERE / "results" / f"mcqa_{a.run_name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{len(rows)} 件を {path} に保存しました", flush=True)
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
