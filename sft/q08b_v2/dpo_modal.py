"""0.8B を崩れにくくする学習（DPO）。モデル自身の回答から「良い回答」と「崩れた回答」の組を作り、良いほうを選ぶように学習する。

1. generate：質問（data/dpo/prompts.jsonl、sft/q08b_v2/build_dpo_prompts.py。続きの会話は messages に前のやり取りを入れる）に、学習したいモデル自身に
   4回ずつ答えさせる（評価と同じ生成の設定。上限 1,536 トークン）
2. 採点：sft/score_data_modal.py --responses で、人と判定が合うように調整した Gemma 4 31B の指示文で採点する
3. pairs：質問ごとに、崩れていない良い回答（一貫性・適合が4点以上、繰り返し・「私は人間です」・中国語・打ち切りなし）と、
   いちばん点の低い回答を組にする（合計点の差が3点以上のものだけ）
4. train：DPO（beta 0.1）に、良い回答の確率も下げないための項（良い回答の負の対数尤度 × 0.2）を足して、全部の重みを学習する。
   比べる元（参照モデル）は学習前のモデル

    modal run sft/q08b_v2/dpo_modal.py --step generate --run-name qwen08b-wiki-sft --model /models/qwen08b-wiki-sft/merged
    modal run sft/score_data_modal.py --responses data/dpo/responses_qwen08b-wiki-sft.jsonl
    python3 sft/q08b_v2/dpo_modal.py pairs qwen08b-wiki-sft
    modal run sft/q08b_v2/dpo_modal.py --step train --run-name qwen08b-wiki-dpo --model /models/qwen08b-wiki-sft/merged --pairs-file data/dpo/pairs_qwen08b-wiki-sft.jsonl
"""

import json
import pathlib
import re
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2]
DPO = ROOT / "data" / "dpo"
# run_eval_modal.py と同じ（Qwen3.5 の推奨値、思考なし）
SAMPLED = dict(temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0)
LOOP = re.compile(r"(.{1,3})\1{15,}")
HUMAN = re.compile(r"(私は(おそらく)?人間|人間である私|僕は人間|私も人間)(?!では|じゃ|でな|ではな)|人間であるため|人間なので")
KEYS = ("consistency", "naturalness", "relevance")

gen_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0")
    .env({"HF_HUB_CACHE": "/hf-cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.0", "transformers==5.17.0", "accelerate==1.15.0", "flash-linear-attention==0.5.2")
    .env({"HF_HUB_CACHE": "/hf-cache", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
models = modal.Volume.from_name("pitcher-models", create_if_missing=True)
app = modal.App("pitcher-dpo")


@app.function(image=gen_image, gpu="L40S", timeout=2 * 60 * 60, volumes={"/hf-cache": hf_cache, "/models": models})
def generate(model: str, prompts: list[dict], n: int, max_tokens: int) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=max_tokens + 2048, gpu_memory_utilization=0.92, limit_mm_per_prompt={"image": 0, "video": 0})
    params = [SamplingParams(**SAMPLED, max_tokens=max_tokens, n=n, seed=i) for i in range(len(prompts))]
    convs = [p.get("messages") or [{"role": "user", "content": p["prompt"]}] for p in prompts]
    outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False})
    return [{**p, "samples": [{"k": k, "output": c.text, "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason}
                              for k, c in enumerate(o.outputs)]} for p, o in zip(prompts, outs)]


def flags(sample: dict) -> list[str]:
    from score import ZH_CHAR  # eval/score.py

    t = sample["output"]
    return [name for name, bad in (("繰り返し", LOOP.search(t)), ("人間", HUMAN.search(t)), ("中国語", ZH_CHAR.search(t)),
                                   ("打ち切り", sample["finish_reason"] == "length")) if bad]


def pairs(run: str):
    sys.path.insert(0, str(ROOT / "eval"))
    rows = [json.loads(l) for l in (DPO / f"responses_{run}.jsonl").read_text().splitlines()]
    sc = {(d["id"], d["k"]): d["scores"] for d in map(json.loads, (DPO / f"responses_{run}_scores.jsonl").read_text().splitlines())}
    out, stat = [], {"質問": len(rows), "組": 0, "良い回答なし": 0, "差が小さい": 0}
    flag_count = {}
    for r in rows:
        cands = []
        for s in r["samples"]:
            score = sc.get((r["id"], s["k"]))
            fl = flags(s)
            for f in fl:
                flag_count[f] = flag_count.get(f, 0) + 1
            if score is None:
                continue
            total = sum(score[k] for k in KEYS) - 5 * len(fl)
            cands.append((total, score, fl, s))
        good = [c for c in cands if not c[2] and c[1]["consistency"] >= 4 and c[1]["relevance"] >= 4]
        if not good:
            stat["良い回答なし"] += 1
            continue
        chosen = max(good, key=lambda c: c[0])
        rejected = min(cands, key=lambda c: c[0])
        if chosen[0] - rejected[0] < 3:
            stat["差が小さい"] += 1
            continue
        out.append({"id": r["id"], "prompt": r["prompt"], **({"messages": r["messages"]} if r.get("messages") else {}), "chosen": chosen[3]["output"], "rejected": rejected[3]["output"],
                    "chosen_score": chosen[1], "rejected_score": rejected[1], "rejected_flags": rejected[2]})
    stat["組"] = len(out)
    (DPO / f"pairs_{run}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    n = sum(len(r["samples"]) for r in rows)
    print(stat, "回答", n, "規則で見つけた崩れ", flag_count)
    if out:
        lc = sum(len(x["chosen"]) for x in out) / len(out)
        lr = sum(len(x["rejected"]) for x in out) / len(out)
        print(f"平均の長さ：良い回答 {lc:.0f}字、崩れた回答 {lr:.0f}字　崩れた回答の規則の内訳",
              {f: sum(f in x["rejected_flags"] for x in out) for f in ("繰り返し", "人間", "中国語", "打ち切り")})


@app.function(image=train_image, gpu="H100", timeout=3 * 60 * 60, volumes={"/hf-cache": hf_cache, "/models": models})
def train(run_name: str, model_path: str, data: list[dict], beta: float, lr: float, nll: float, accum: int) -> dict:
    import os
    import random
    import shutil
    import time

    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.backends.cuda.enable_cudnn_sdp(False)
    tok = AutoTokenizer.from_pretrained(model_path)
    policy = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.float32, device_map="cuda")
    ref = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda").eval()
    for n_, p in policy.named_parameters():
        p.requires_grad = "visual" not in n_
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in ref.parameters():
        p.requires_grad = False
    end = tok.convert_tokens_to_ids("<|im_end|>")

    def encode(x: dict, resp: str):
        conv = x.get("messages") or [{"role": "user", "content": x["prompt"]}]
        pre = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        a = tok(pre, add_special_tokens=False)["input_ids"]
        b = tok(resp, add_special_tokens=False)["input_ids"] + [end]
        ids = (a + b)[:3072]
        return torch.tensor([ids], device="cuda"), len(a)

    def logp(model, ids, start, grad: bool):
        """回答部分のトークンの対数確率の合計と、トークン数。確率の表は小分けにして作る。"""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=ids[:, :-1]).last_hidden_state[:, start - 1:]
        y = ids[:, start:]
        w = model.lm_head.weight

        def part(hc, yc):
            lp = torch.log_softmax((hc @ w.t()).float(), -1)
            return lp.gather(-1, yc.unsqueeze(-1)).sum()

        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, h.size(1), 512):
            hc, yc = h[:, i:i + 512], y[:, i:i + 512]
            total = total + (checkpoint(part, hc, yc, use_reentrant=False) if grad else part(hc, yc))
        return total, y.numel()

    random.Random(0).shuffle(data)
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = len(data) // accum
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    hist, t0 = [], time.time()
    policy.train()
    for s in range(steps):
        acc = mar = ls = 0.0
        for x in data[s * accum:(s + 1) * accum]:
            ic, sc_ = encode(x, x["chosen"])
            ir, sr = encode(x, x["rejected"])
            with torch.no_grad():
                rc, _ = logp(ref, ic, sc_, False)
                rr, _ = logp(ref, ir, sr, False)
            pc, nc = logp(policy, ic, sc_, True)
            pr, _ = logp(policy, ir, sr, True)
            margin = beta * ((pc - rc) - (pr - rr))
            loss = (-F.logsigmoid(margin) + nll * (-pc / nc)) / accum
            loss.backward()
            acc += float(margin > 0) / accum
            mar += margin.item() / accum
            ls += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        hist.append({"step": s + 1, "loss": round(ls, 4), "reward_acc": round(acc, 3), "margin": round(mar, 4)})
        if (s + 1) % 5 == 0 or s == steps - 1:
            print(hist[-1], f"{(time.time() - t0) / 60:.1f} 分", flush=True)
    out = f"/models/{run_name}/merged"
    sd = {k: v.to(torch.bfloat16) for k, v in policy.state_dict().items()}
    policy.save_pretrained(out, state_dict=sd, safe_serialization=True)
    tok.save_pretrained(out)
    for f in os.listdir(model_path):
        if f.endswith((".json", ".jinja", ".txt")) and not f.endswith("index.json") and not os.path.exists(f"{out}/{f}"):
            shutil.copy(f"{model_path}/{f}", f"{out}/{f}")
    models.commit()
    return {"run": run_name, "base": model_path, "pairs": len(data), "steps": steps, "beta": beta, "lr": lr, "nll": nll,
            "accum": accum, "train_seconds": time.time() - t0, "history": hist}


@app.local_entrypoint()
def main(step: str, run_name: str, model: str, pairs_file: str = "", n: int = 4, max_tokens: int = 1536,
         beta: float = 0.1, lr: float = 5e-7, nll: float = 0.2, accum: int = 16):
    if step == "generate":
        prompts = [json.loads(l) for l in (DPO / "prompts.jsonl").read_text().splitlines()]
        rows = generate.remote(model, prompts, n, max_tokens)
        out = DPO / f"responses_{run_name}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{out}：{len(rows)} 問・回答 {sum(len(r['samples']) for r in rows)} 件")
    elif step == "train":
        data = [json.loads(l) for l in (ROOT / pairs_file).read_text().splitlines()]
        r = train.remote(run_name, model, data, beta, lr, nll, accum)
        (ROOT / "sft" / "runs" / f"{run_name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(f"組 {r['pairs']}、{r['steps']} ステップ、{r['train_seconds'] / 60:.1f} 分")


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "pairs":
    pairs(sys.argv[2])
