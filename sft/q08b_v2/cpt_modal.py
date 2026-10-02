"""日本語版 Wikipedia を読ませる「追加の事前学習」で、0.8B の日本の知識を増やす（Modal）。

会話形式の学習（SFT）は知っていることを引き出すのは得意だが、新しい知識はあまり増えない。そこで、蒸留後の 0.8B
（qwen08b-distill）に日本語版 Wikipedia の本文を読ませてから、第2段階の学習（Tengentoppa＋答え方の練習）をやり直す。

元データ：wikimedia/wikipedia の 20231101.ja（CC BY-SA 3.0 / GFDL）。記事を無作為に選び、短い記事・一覧・曖昧さ回避と、
「脚注」「参考文献」「外部リンク」「関連項目」以降を除き、記事の間に終わりの記号を入れて 2,048 トークンずつに詰める。

学習の途中で8回ほど、Wikipedia の検証データ（100万トークン）と会話の検証データ（Tengentoppa の検証 100件）の損失を測る。
会話の損失が上がり続けたら、Wikipedia の文体に引っぱられて会話の力を忘れ始めている。1/3・2/3 の時点のモデルも
/models/<run名>/step<N> に保存し、どこまで読ませるのが良いかを常識問題で比べられるようにする。

語彙が約25万語と大きく、確率の表（logits）をまとめて作るとメモリが足りないため、系列を小分けにして損失を計算する
（小分けごとに計算をやり直して、確率の表を保存しない）。

    modal run sft/q08b_v2/cpt_modal.py --prep-data --tokens 100000000         # データを作って Volume に保存
    modal run sft/q08b_v2/cpt_modal.py --run-name qwen08b-wiki --max-steps 20   # 速さを測る
    modal run sft/q08b_v2/cpt_modal.py --run-name qwen08b-wiki                   # 本番（/models/qwen08b-wiki/merged）
"""

import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = "/models/qwen08b-distill/merged"
SEQ = 2048
DATA = "/models/data/wiki_ja_{n}.npy"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.0", "transformers==5.17.0", "accelerate==1.15.0", "flash-linear-attention==0.5.2",
                    "pandas", "pyarrow", "numpy")
    .env({"HF_HUB_CACHE": "/hf-cache", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
models = modal.Volume.from_name("pitcher-models", create_if_missing=True)
app = modal.App("pitcher-cpt", image=image)

SKIP_TITLE = ("一覧", "のリスト", "曖昧さ回避")
CUT = ("\n脚注", "\n出典", "\n参考文献", "\n外部リンク", "\n関連項目", "\n注釈")


def clean(text: str) -> str:
    for c in CUT:
        i = text.find(c)
        if i != -1:
            text = text[:i]
    return text.strip()


@app.function(cpu=8, memory=32768, timeout=2 * 60 * 60, volumes={"/hf-cache": hf_cache, "/models": models})
def prep(tokens: int, seed: int = 0) -> dict:
    import random

    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    eos = tok.eos_token_id
    rng = random.Random(seed)
    files = [f"20231101.ja/train-{i:05d}-of-00015.parquet" for i in range(15)]
    rng.shuffle(files)
    ids, n_art, total = [], 0, 0
    target = tokens + 1_000_000  # 1M トークンは検証用
    for f in files:
        df = pd.read_parquet(hf_hub_download("wikimedia/wikipedia", f, repo_type="dataset"), columns=["title", "text"])
        rows = [(t, clean(x)) for t, x in zip(df["title"], df["text"]) if not any(s in t for s in SKIP_TITLE)]
        rows = [r for r in rows if len(r[1]) >= 500]
        rng.shuffle(rows)
        # 各ファイルから同じ割合だけ取る（15ファイルで目標に届くように）
        take = rows[: max(1, int(len(rows) * 0.12))]
        enc = tok([f"{t}\n\n{x}" for t, x in take], add_special_tokens=False)["input_ids"]
        for e in enc:
            ids.extend(e + [eos])
        n_art += len(take)
        total = len(ids)
        print(f"{f}: 記事 {len(take)}、累計 {total / 1e6:.1f}M トークン", flush=True)
        if total >= target:
            break
    n_seq = min(total, target) // SEQ
    arr = np.array(ids[: n_seq * SEQ], dtype=np.uint32).reshape(n_seq, SEQ)
    out = DATA.format(n=tokens)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    models.commit()
    return {"articles": n_art, "sequences": n_seq, "tokens": n_seq * SEQ, "path": out}


def chunked_ce(h, weight, labels, chunk: int = 512):
    """系列を chunk トークンずつに分けて交差エントロピーを足す。確率の表は保存せず、逆伝播のときに作り直す。"""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def part(hc, yc):
        logits = (hc @ weight.t()).float()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), yc.reshape(-1), reduction="sum")

    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(1), chunk):
        total = total + checkpoint(part, h[:, i:i + chunk], labels[:, i:i + chunk], use_reentrant=False)
    return total / labels.numel()


@app.function(gpu="H100", timeout=4 * 60 * 60, volumes={"/hf-cache": hf_cache, "/models": models})
def train(run_name: str, tokens: int, lr: float, micro: int, accum: int, max_steps: int, grad_ckpt: bool,
          chat_valid: list[dict]) -> dict:
    import math
    import os
    import shutil
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.backends.cuda.enable_cudnn_sdp(False)
    data = np.load(DATA.format(n=tokens), mmap_mode="r")
    n_valid = 1_000_000 // SEQ
    valid, data = data[:n_valid], data[n_valid:]
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.float32, device_map="cuda")
    for n, p in model.named_parameters():
        p.requires_grad = "visual" not in n
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    steps = len(data) // (micro * accum)
    if max_steps:
        steps = min(steps, max_steps)
    warm = max(1, int(steps * 0.03))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / max(1, steps))))))
    lm = model.lm_head.weight
    body = model.model

    def loss_of(batch):
        x = torch.from_numpy(batch.astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = body(input_ids=x[:, :-1]).last_hidden_state
            return chunked_ce(h, lm, x[:, 1:])

    tok = AutoTokenizer.from_pretrained(BASE)
    # 会話の検証データ：チャットテンプレートで並べた全文の損失（Wikipedia の文体に引っぱられて会話の力を忘れていないかの目安）
    chat = [tok.apply_chat_template(r["messages"], tokenize=False, enable_thinking=False) for r in chat_valid]
    chat = [tok(t, add_special_tokens=False, return_tensors="np")["input_ids"][0][:SEQ] for t in chat]

    @torch.no_grad()
    def evaluate(step: int) -> dict:
        model.eval()
        wiki = [loss_of(valid[i:i + micro]).item() for i in range(0, min(len(valid), 64), micro)]
        conv = [loss_of(c[None]).item() for c in chat]
        model.train()
        return {"step": step, "wiki_valid_loss": round(sum(wiki) / len(wiki), 4), "chat_valid_loss": round(sum(conv) / len(conv), 4)}

    def save(path: str):
        # 学習中の重み（fp32）はそのままにして、保存する写しだけ bf16 にする
        sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        model.save_pretrained(path, state_dict=sd, safe_serialization=True)
        del sd
        tok.save_pretrained(path)
        for f in os.listdir(BASE):
            if f.endswith((".json", ".jinja", ".txt")) and not f.endswith("index.json") and not os.path.exists(f"{path}/{f}"):
                shutil.copy(f"{BASE}/{f}", f"{path}/{f}")
        models.commit()

    order = np.random.default_rng(0).permutation(len(data))
    hist = [evaluate(0)]
    eval_every = max(10, steps // 8)  # 学習の途中で8回ほど検証する
    keep = {steps // 3, 2 * steps // 3} if not max_steps else set()  # 1/3・2/3 の時点のモデルも保存して、あとで比べる
    print(hist[-1], flush=True)
    model.train()
    t0 = time.time()
    for s in range(steps):
        tot = 0.0
        for a in range(accum):
            idx = np.sort(order[(s * accum + a) * micro:(s * accum + a + 1) * micro])
            loss = loss_of(data[idx]) / accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if (s + 1) % 10 == 0 or s == steps - 1:
            el = time.time() - t0
            tps = (s + 1) * micro * accum * SEQ / el
            hist.append({"step": s + 1, "loss": tot, "lr": sched.get_last_lr()[0], "tok_per_s": round(tps)})
            print(hist[-1], f"残り約 {(steps - s - 1) * el / (s + 1) / 60:.0f} 分", flush=True)
        if (s + 1) % eval_every == 0 and s + 1 < steps:
            hist.append(evaluate(s + 1))
            print(hist[-1], flush=True)
        if s + 1 in keep:
            save(f"/models/{run_name}/step{s + 1}")
            print(f"途中のモデルを保存：/models/{run_name}/step{s + 1}", flush=True)
    train_sec = time.time() - t0
    hist.append(evaluate(steps))
    print(hist[-1], flush=True)
    result = {"run": run_name, "base": BASE, "tokens_trained": steps * micro * accum * SEQ, "steps": steps, "lr": lr,
              "micro": micro, "accum": accum, "train_seconds": train_sec, "gpu": torch.cuda.get_device_name(), "history": hist}
    if not max_steps:
        save(f"/models/{run_name}/merged")
        with open(f"/models/{run_name}/train_result.json", "w") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=1)
        models.commit()
    return result


@app.local_entrypoint()
def main(prep_data: bool = False, run_name: str = "qwen08b-wiki", tokens: int = 100_000_000, lr: float = 2e-5,
         micro: int = 8, accum: int = 16, max_steps: int = 0, grad_ckpt: bool = False):
    if prep_data:
        print(json.dumps(prep.remote(tokens), ensure_ascii=False))
        return
    chat_valid = [json.loads(l) for l in (ROOT / "sft" / "tengentoppa_clean_valid.jsonl").read_text().splitlines()][:100]
    r = train.remote(run_name, tokens, lr, micro, accum, max_steps, grad_ckpt, chat_valid)
    out = ROOT / "sft" / "runs" / f"{run_name}{'-test' if max_steps else ''}.json"
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"{out} に保存しました")
