"""AIME 2026（MathArena/aime_2026、30問）を、Mac 上の llama.cpp で「思考あり」で解かせる。

日本語化の学習（思考なし）で、元の Spark-X2.5-4B の数学の力が落ちていないかを調べるためのもの。
問題文はライセンス（CC BY-NC-SA 4.0）の都合でリポジトリに含めず、実行時に Hugging Face から読む。
1回答ずつ eval/results/aime_raw/<run名>.jsonl に追記するので、途中で止めても同じコマンドで続きから再開できる。

使い方:
    python3 eval/run_aime_llamacpp.py --gguf tools/gguf/spark-x2.5-4b-Q8_0.gguf --run-name spark-x2.5-4b
    python3 eval/run_aime_llamacpp.py --summary spark-x2.5-4b spark-ja-4b spark-ja2-4b   # 集計だけ
"""

import argparse
import json
import re
import subprocess
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "aime_raw"
PORT = 8092
# MathArena の指示文に合わせる
PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}. The answer is an integer between 0 and 999 inclusive."
# Spark-X2.5 の推奨値（思考あり）。llama.cpp の top_k=0 は「使わない」
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=0)


def load_problems() -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    df = pd.read_parquet(hf_hub_download("MathArena/aime_2026", "data/train-00000-of-00001.parquet", repo_type="dataset"))
    return [{"idx": int(r.problem_idx), "answer": int(r.answer), "problem": r.problem} for r in df.itertuples()]


def start_server(gguf: str, parallel: int, max_tokens: int) -> subprocess.Popen:
    log = open(RAW / "llama-server.log", "w")
    p = subprocess.Popen(["llama-server", "-m", gguf, "--port", str(PORT), "-ngl", "99", "-np", str(parallel),
                          "-c", str((max_tokens + 1024) * parallel), "--jinja", "--no-webui"], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok":
                return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise RuntimeError(f"llama-server が起動しませんでした（{RAW / 'llama-server.log'} を確認）")


def last_boxed(text: str) -> int | None:
    """最後の \\boxed{...} の中身を整数として読む（入れ子の括弧にも対応）。"""
    i = text.rfind("\\boxed")
    while i != -1:
        j = text.find("{", i)
        if j != -1:
            depth, k = 0, j
            while k < len(text):
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                if depth == 0:
                    inner = re.sub(r"\\[a-z]+|[{}\s,$]", "", text[j + 1:k])
                    if re.fullmatch(r"\d+", inner):
                        return int(inner)
                    break
                k += 1
        i = text.rfind("\\boxed", 0, i)
    return None


def solve(prob: dict, sample: int, max_tokens: int) -> dict:
    body = {"messages": [{"role": "user", "content": PROMPT.format(problem=prob["problem"])}], **SAMPLING,
            "max_tokens": max_tokens, "seed": 1000 * sample + prob["idx"], "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=6 * 3600))
    msg = r["choices"][0]["message"]
    content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
    # 思考が区切られずに content に入った場合も、</think> の後ろを回答とみなす
    if "</think>" in content:
        reasoning, content = content.split("</think>", 1)
    pred = last_boxed(content)
    return {"idx": prob["idx"], "sample": sample, "answer": prob["answer"], "pred": pred, "correct": pred == prob["answer"],
            "finish_reason": r["choices"][0].get("finish_reason"), "completion_tokens": r["usage"]["completion_tokens"],
            "seconds": round(time.time() - t0), "content": content, "reasoning": reasoning}


def summary(runs: list[str]):
    print(f"{'':16}{'正答率':>8}{'全問':>6}{'打ち切り':>8}{'平均トークン':>12}{'回答数':>6}")
    for run in runs:
        path = RAW / f"{run}.jsonl"
        if not path.exists():
            print(f"{run:16}（まだ結果がない）")
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        by = defaultdict(list)
        for r in rows:
            by[r["idx"]].append(r["correct"])
        solved_any = sum(any(v) for v in by.values())
        acc = sum(r["correct"] for r in rows) / len(rows)
        cut = sum(r["finish_reason"] == "length" for r in rows) / len(rows)
        tok = sum(r["completion_tokens"] for r in rows) / len(rows)
        print(f"{run:16}{acc:>8.1%}{solved_any:>6}{cut:>8.0%}{tok:>12.0f}{len(rows):>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf")
    ap.add_argument("--run-name")
    ap.add_argument("--samples", type=int, default=4, help="1問あたりの回答数")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--only", default="", help="問題番号をカンマ区切りで（例：1,2,3）")
    ap.add_argument("--summary", nargs="*")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.summary is not None:
        summary(a.summary)
        return

    probs = load_problems()
    if a.only:
        probs = [p for p in probs if p["idx"] in {int(x) for x in a.only.split(",")}]
    out = RAW / f"{a.run_name}.jsonl"
    done = {(r["idx"], r["sample"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    # 1周目で全問を1回ずつ解いてから2周目に進む（途中で止めても問題の偏りが出にくい）
    jobs = [(p, s) for s in range(a.samples) for p in probs if (p["idx"], s) not in done]
    print(f"{a.run_name}: 残り {len(jobs)} 回答（済み {len(done)}）", flush=True)
    if not jobs:
        return

    server = start_server(a.gguf, a.parallel, a.max_tokens)
    lock, t0, n = threading.Lock(), time.time(), [0]

    def one(job):
        r = solve(*job, a.max_tokens)
        with lock:
            with out.open("a") as f:
                f.write(json.dumps({"run": a.run_name, "model": a.gguf, "max_tokens": a.max_tokens, **r}, ensure_ascii=False) + "\n")
            n[0] += 1
            print(f"[{n[0]}/{len(jobs)} {time.time() - t0:.0f}秒] 問{r['idx']} #{r['sample']} 予測={r['pred']} 正解={r['answer']} "
                  f"{'○' if r['correct'] else '×'} {r['completion_tokens']}トークン {r['finish_reason']}", flush=True)

    try:
        with ThreadPoolExecutor(a.parallel) as ex:
            list(ex.map(one, jobs))
    finally:
        server.terminate()
    summary([a.run_name])


if __name__ == "__main__":
    main()
