"""Colab のランタイム上の vLLM で、推論モデルに同じ問題を何回も解かせ、正解したかと考えた長さを記録する。

「考える長さを短くする」学習のデータ作り（正解した中でいちばん短い思考を選ぶ）と、強化学習の下調べに使う。
問題は sft/build_reasoning_problems.py で作った data/reasoning/problems_pool.jsonl。採点は eval/run_reasoning_modal.py と同じ
（思考のあとの答えの最後の \\boxed{} を読む）。chunk 件ずつ out に追記するので、止まっても同じコマンドで続きから再開できる。

    python rollout_colab.py --model ornith-ai/Ornith-1.5-9B --problems problems_pool.jsonl --out rollouts.jsonl --limit 100   # 試し
    python rollout_colab.py --model ornith-ai/Ornith-1.5-9B --problems problems_pool.jsonl --out rollouts.jsonl --start 100 --limit 1000
"""

import argparse
import json
import os
import re
import time
import zlib

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# Ornith-1.5-9B の推奨値（一般のタスク）。eval/run_reasoning_modal.py と同じ
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5)


def last_boxed(text: str) -> str | None:
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


def grade(p: dict, text: str) -> tuple[str | None, bool]:
    pred = last_boxed(text.split("</think>")[-1]) if "</think>" in text else None
    if pred is None:
        return None, False
    if p["task"] == "math":
        return pred, pred.lstrip("+") == p["answer"] or (pred.lstrip("0") or "0") == (p["answer"].lstrip("0") or "0")
    m = re.search(r"[A-J]", pred.upper())
    return (m.group() if m else pred), bool(m) and m.group() == p["answer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4, help="1問あたり何回解かせるか")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=100)
    a = ap.parse_args()

    probs = [json.loads(l) for l in open(a.problems) if l.strip()][a.start:a.start + a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
    todo = [p for p in probs if p["id"] not in done]
    print(f"{len(probs)} 問中 {len(probs) - len(todo)} 問は済み、残り {len(todo)} 問", flush=True)
    if not todo:
        return
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, max_model_len=a.max_tokens + 2048, gpu_memory_utilization=0.92,
              limit_mm_per_prompt={"image": 0, "video": 0})
    for c in range(0, len(todo), a.chunk):
        part = todo[c:c + a.chunk]
        t0 = time.time()
        params = [SamplingParams(**SAMPLING, max_tokens=a.max_tokens, n=a.n, seed=zlib.crc32(p["id"].encode())) for p in part]
        outs = llm.chat([[{"role": "user", "content": p["prompt"]}] for p in part], params,
                        chat_template_kwargs={"enable_thinking": True}, use_tqdm=False)
        n_tok = 0
        with open(a.out, "a") as f:
            for p, o in zip(part, outs):
                samples = []
                for k, s in enumerate(o.outputs):
                    pred, ok = grade(p, s.text)
                    samples.append({"sample": k, "output": s.text, "n_tokens": len(s.token_ids), "finish_reason": s.finish_reason,
                                    "pred": pred, "correct": ok})
                    n_tok += len(s.token_ids)
                f.write(json.dumps({"id": p["id"], "task": p["task"], "answer": p["answer"], "difficulty": p.get("difficulty"),
                                    "samples": samples}, ensure_ascii=False) + "\n")
        dt = time.time() - t0
        print(f"{min(c + a.chunk, len(todo))} / {len(todo)} 問　{dt / 60:.1f} 分　{n_tok / dt:.0f} トークン/秒", flush=True)


if __name__ == "__main__":
    main()
