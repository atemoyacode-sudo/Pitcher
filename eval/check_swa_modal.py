"""transformers（学習に使った実装）で、Spark-X2.5 の「直近512トークンだけを見る層」（sliding window）が本当に効いているかを調べる。

Pitcher 版（spark-pitcher2-4b）が llama.cpp でだけ長い回答で崩れたため。transformers 側で窓が効いていないと、
学習中のモデルは「窓より前も見える」状態で学習し、窓を正しく守る llama.cpp とずれる。

同じ長い文章（約1,500トークン）を、設定どおり（窓 512）と、窓をなくした設定（全部見える）で読ませ、
各位置の出力（ロジット）の差を比べる。窓が効いていれば、512 トークンより後ろで差が出る。

    modal run eval/check_swa_modal.py --model /models/spark-pitcher2-4b/merged
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.9.1", "transformers==4.57.1", "accelerate")
    .env({"HF_HUB_CACHE": "/hf-cache"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("pitcher-models", create_if_missing=True)
app = modal.App("pitcher-check-swa", image=image)


@app.function(gpu="A100-80GB", timeout=30 * 60, volumes={"/hf-cache": hf_cache, "/models": models_vol})
def check(model_id: str, text: str) -> str:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :1500].cuda()
    out = [f"トークン数 {ids.shape[1]}"]
    logits = {}
    for name, window in [("設定どおり", None), ("窓なし", 10**9)]:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if window:
            cfg.sliding_window = window
        for impl in ("eager", "sdpa"):
            try:
                m = AutoModelForCausalLM.from_pretrained(model_id, config=cfg, trust_remote_code=True, dtype=torch.bfloat16,
                                                         device_map="cuda", attn_implementation=impl)
            except Exception as e:  # noqa: BLE001
                out.append(f"{name} {impl}: 読み込めない（{type(e).__name__}）")
                continue
            with torch.no_grad():
                logits[(name, impl)] = m(ids).logits.float()[0]
            out.append(f"{name} {impl}: 既定の実装 {m.config._attn_implementation}、sliding_window={m.config.sliding_window}")
            del m
            torch.cuda.empty_cache()
    for impl in ("eager", "sdpa"):
        a, b = logits.get(("設定どおり", impl)), logits.get(("窓なし", impl))
        if a is None or b is None:
            continue
        diff = (a - b).abs().max(dim=-1).values
        out.append(f"[{impl}] 窓の有無による出力の差：0〜511番目の最大 {diff[:512].max():.4f}、512番目以降の最大 {diff[512:].max():.4f}、平均 {diff[512:].mean():.4f}")
    return "\n".join(out)


@app.local_entrypoint()
def main(model: str):
    import json
    import pathlib

    rows = [json.loads(l) for l in (pathlib.Path(__file__).parent.parent / "sft" / "distill_sft.jsonl").read_text().splitlines()[:50]]
    text = "\n\n".join(r["messages"][1]["content"] for r in rows)
    print(check.remote(model, text))


@app.function(gpu="A100-80GB", timeout=30 * 60, volumes={"/hf-cache": hf_cache, "/models": models_vol})
def ppl(model_id: str, text: str, n_ctx: int = 1536, chunks: int = 4, act: str = "") -> str:
    """llama.cpp の llama-perplexity と同じ数え方（各区切りの先頭を BOS にし、後半だけを採点）で PPL を出す。
    後半（768番目以降）はすべて窓（512）より後ろなので、窓の扱いが llama.cpp と同じかを比べられる。"""
    import math

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    if act:  # 活性化関数を差し替えて、llama.cpp との計算の違いを再現してみる
        from transformers.activations import ACT2FN

        for layer in m.model.layers:
            layer.mlp.act_fn = ACT2FN[act]
    ids = tok(text, add_special_tokens=False).input_ids
    nll, cnt, out = 0.0, 0, []
    for c in range(chunks):
        x = ids[c * n_ctx:(c + 1) * n_ctx]
        x[0] = tok.bos_token_id
        with torch.no_grad():
            lp = torch.log_softmax(m(torch.tensor([x]).cuda()).logits.float()[0], dim=-1)
        first = n_ctx // 2
        for j in range(first, n_ctx - 1):
            nll -= lp[j, x[j + 1]].item()
            cnt += 1
        out.append(f"[{c + 1}]{math.exp(nll / cnt):.4f}")
    return f"{model_id} {act or 'gelu'}: トークン数 {len(ids)}  " + ",".join(out)


@app.local_entrypoint()
def ppl_main(models: str, n_ctx: int = 1536, chunks: int = 4, act: str = ""):
    import json
    import pathlib

    rows = [json.loads(l) for l in (pathlib.Path(__file__).parent.parent / "sft" / "distill_sft.jsonl").read_text().splitlines()[:50]]
    text = "\n\n".join(r["messages"][1]["content"] for r in rows)
    for r in ppl.map(models.split(","), kwargs={"text": text, "n_ctx": n_ctx, "chunks": chunks, "act": act}):
        print(r)


@app.function(gpu="A100-80GB", timeout=30 * 60, volumes={"/hf-cache": hf_cache, "/models": models_vol})
def pos_logprobs(model_id: str, ids: list[int], positions: list[int]) -> dict:
    """先頭から p トークンを読ませたときの、次のトークンの上位5つの対数確率（llama-server の n_probs と比べる用）。"""
    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    with torch.no_grad():
        lp = torch.log_softmax(m(torch.tensor([ids[:max(positions)]]).cuda()).logits.float()[0], dim=-1)
    res = {}
    for p in positions:
        v, i = lp[p - 1].topk(5)
        res[p] = [(int(a), round(float(b), 3)) for a, b in zip(i, v)]
    return res


@app.local_entrypoint()
def pos_main(model: str, ids_file: str, positions: str):
    import json

    ids = json.loads(open(ids_file).read())
    for p, r in pos_logprobs.remote(model, ids, [int(x) for x in positions.split(",")]).items():
        print(p, "正解", ids[int(p)], r)


@app.function(gpu="A100-80GB", timeout=30 * 60, volumes={"/hf-cache": hf_cache, "/models": models_vol})
def act_range(model_id: str, ids: list[int]) -> str:
    """各層の入力・線形層の入出力の最大値を調べる（llama.cpp が途中を16ビット浮動小数点で計算するときに、はみ出すかどうか）。"""
    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    stats = {}

    def hook(name):
        def f(mod, inp, out):
            a = inp[0].float().abs().max().item()
            b = out.float().abs().max().item() if torch.is_tensor(out) else 0.0
            s = stats.setdefault(name, [0.0, 0.0])
            s[0], s[1] = max(s[0], a), max(s[1], b)
        return f

    for n, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear) or n.endswith("layers." + n.split(".")[-1]) and n.split(".")[-1].isdigit():
            mod.register_forward_hook(hook(n))
    with torch.no_grad():
        m(torch.tensor([ids[:1024]]).cuda())
    top = sorted(stats.items(), key=lambda kv: -max(kv[1]))[:12]
    return model_id + "\n" + "\n".join(f"  {n}: 入力 {a:.0f} / 出力 {b:.0f}" for n, (a, b) in top)


@app.local_entrypoint()
def act_main(models: str, ids_file: str):
    import json

    ids = json.loads(open(ids_file).read())
    for r in act_range.map(models.split(","), kwargs={"ids": ids}):
        print(r)
