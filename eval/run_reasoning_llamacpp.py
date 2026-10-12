"""思考ありの理科・算数の力を、手元の llama.cpp（Mac）で測る。LoRA の付け外しも比べられる。

問題と採点は eval/run_reasoning_modal.py と同じ（MMLU-Pro の物理・化学・生物 300問、MGSM の日本語 250問）。
LoRA は llama-server に読み込んでおき、依頼ごとに効き具合を指定する（--lora-scale 0 なら LoRA なし）。
llama.cpp の presence_penalty は質問文のトークンにもかかるので、ここでは 0 にする（vLLM の結果とは条件が違うので、
比べるのはこのスクリプトで測ったものどうしだけにする）。

    llama-server -m tools/gguf/qwen08b-think-f16.gguf --lora lora.gguf --port 8096 -np 8 -c 65536 --jinja
    python3 eval/run_reasoning_llamacpp.py --run qwen08b-think+sci-lora --lora-scale 1
    → eval/results/reasoning_raw/<run>.jsonl と eval/results/reasoning_<run>.json

1問を何回も解かせるときは --samples（回ごとに乱数の種を変える。1回解き終わるごとに保存）：
    python3 eval/run_reasoning_llamacpp.py --run v25b-s4 --samples 4 --seed-base 1000
"""

import argparse
import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_reasoning_modal as R  # noqa: E402

SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0)


def post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--tasks", default="mmlu_pro_sci,mgsm_ja")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--lora-scale", type=float, default=None)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--samples", type=int, default=1, help="1問を解かせる回数")
    ap.add_argument("--seed-base", type=int, default=0, help="乱数の種のずらし（別の実行と重ならないように）")
    ap.add_argument("--tasks-file", help="評価の問題の代わりに使う問題（jsonl。task・answer・prompt、なければ id は行番号）")
    a = ap.parse_args()
    thinking = not a.no_thinking
    if a.tasks_file:
        tasks = [json.loads(l) for l in open(a.tasks_file) if l.strip()]
        for i, t in enumerate(tasks):
            t.setdefault("id", f"{t['task']}-{i:05d}")
    else:
        tasks = R.load_tasks(tuple(a.tasks.split(",")))

    done, lock = [0], threading.Lock()

    def one(it):
        k, i, t = it
        p = post(a.port, "/apply-template", {"messages": [{"role": "user", "content": t["prompt"]}],
                                             "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        if thinking:
            assert p.endswith("<think>\n\n</think>\n\n"), p[-40:]
            p = p.removesuffix("\n</think>\n\n")
        seed = i if a.samples == 1 and a.seed_base == 0 else a.seed_base + k * 100003 + i
        body = {"prompt": p, "n_predict": a.max_tokens, "seed": seed, **SAMPLING}
        if a.lora_scale is not None:
            body["lora"] = [{"id": 0, "scale": a.lora_scale}]
        r = post(a.port, "/completion", body)
        row = {"task": t["task"], "id": t["id"], "sample": a.seed_base + k, "seed": seed, "answer": t["answer"], "thinking": thinking, "output": r["content"],
               "n_tokens": r.get("tokens_predicted", 0), "finish_reason": "length" if r.get("stop_type") == "limit" else "stop"}
        row["pred"], row["correct"] = R.grade(row, row["output"])
        with lock:
            done[0] += 1
            if a.samples > 1 and done[0] % 50 == 0:
                print(f"{done[0]}/{len(tasks) * a.samples} 問", flush=True)
        return row

    R.RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for k in range(a.samples):
            rows += list(ex.map(one, [(k, i, t) for i, t in enumerate(tasks)]))
            (R.RAW / f"{a.run}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            print(f"{k + 1}/{a.samples} 回目まで保存", flush=True)
    res = R.summarize(a.run)
    print(json.dumps({k: (v["acc"], v["truncated"], v["tokens_median"]) for k, v in res.items() if k != "run"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
