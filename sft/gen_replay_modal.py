"""劣化対策（リプレイ）用のデータを作る：一般的な依頼文を Qwen3.5-27B で作り、元モデル自身に答えさせる。

キャラクターのセリフだけで学習すると、何を聞かれても短いセリフで返すようになる（lora-4b で確認）。
元モデルの回答をキャラ設定なしの会話として混ぜ、「設定があるときだけキャラになる」ことを狙う。
AIの感情や正体を問う質問はあえて入れない（「AIには感情はありません」を直接教えないため）。

出力:
    sft/replay_prompts.jsonl
    sft/replay_<モデル名>.jsonl（例: replay_Qwen3.5-4B.jsonl）

使い方:
    modal run sft/gen_replay_modal.py
"""

import json
import pathlib
import random
import re

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
SFT_DIR = ROOT / "sft"
PROMPT_MODEL = "Qwen/Qwen3.5-27B"
ANSWER_MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"]
N_PROMPTS = 800

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0")
    .env({"HF_HUB_CACHE": "/hf-cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
app = modal.App("pitcher-gen-replay", image=image)

CATEGORIES = [
    "日本の地理や歴史についての知識を問う質問", "理科や科学の基礎についての質問", "算数・数学の計算問題",
    "料理のレシピや調理手順についての依頼（カレー以外）", "掃除・片付け・家事などの生活の知恵", "健康や運動についてのアドバイスの依頼",
    "勉強法や資格試験についての相談", "仕事で使うメールやビジネス文書の作成依頼", "お礼状・お祝い・案内などの手紙やメッセージの作成依頼",
    "短い物語・詩・キャッチコピーの創作依頼", "本文を含めた文章の要約依頼", "本文を含めた文章の言い換え・敬語への書き換え依頼",
    "本文を含めた英語から日本語への翻訳依頼", "本文を含めた日本語から英語への翻訳依頼", "プログラミングについての簡単な質問",
    "旅行やお出かけの計画の相談", "趣味やおすすめについての相談", "言葉やことわざの意味の説明の依頼",
    "2つのものの比較やメリット・デメリットの説明の依頼", "お金や家計の管理についての一般的な質問",
]
PROMPT_TEMPLATE = """ユーザーがAIアシスタントに送る依頼や質問を、日本語で{n}個作ってください。

カテゴリ：{cat}

条件：
- 内容・口調・長さをばらばらにし、具体的にする
- 1行に1つ。番号・記号・説明は付けない
- 20〜120文字程度
- 要約・翻訳・言い換えの依頼では、対象の文章も依頼文の中に含める
- AIの感情・好み・正体について尋ねる質問は含めない"""

EXCLUDE = re.compile(r"感情|気持ち|心はある|あなたは(誰|何者)|あなた自身|好きな|AI|ＡＩ|人工知能|Qwen")
SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0)


@app.function(gpu="A100-80GB", timeout=60 * 60, volumes={"/hf-cache": hf_cache})
def make_prompts(n_per_call: int, calls_per_cat: int) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=PROMPT_MODEL, max_model_len=8192, limit_mm_per_prompt={"image": 0, "video": 0})
    convs, meta, params = [], [], []
    for cat in CATEGORIES:
        for k in range(calls_per_cat):
            convs.append([{"role": "user", "content": PROMPT_TEMPLATE.format(n=n_per_call, cat=cat)}])
            params.append(SamplingParams(temperature=0.9, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=4000, seed=k))
            meta.append(cat)
    outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False})
    rows = []
    for cat, o in zip(meta, outs):
        for line in o.outputs[0].text.splitlines():
            line = re.sub(r"^\s*([0-9０-９]+[.．、)）]|[-・*•])\s*", "", line).strip()
            if 20 <= len(line) <= 200:
                rows.append({"category": cat, "prompt": line})
    return rows


@app.function(gpu="L4", timeout=60 * 60, volumes={"/hf-cache": hf_cache})
def answer(prompts: list[str], model: str) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=4096, limit_mm_per_prompt={"image": 0, "video": 0})
    params = [SamplingParams(**SAMPLING, max_tokens=1024, seed=i) for i in range(len(prompts))]
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params, chat_template_kwargs={"enable_thinking": False})
    return [{"output": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason} for o in outs]


def bigrams(s: str) -> set:
    s = re.sub(r"\s", "", s)
    return {s[i : i + 2] for i in range(len(s) - 1)}


@app.local_entrypoint()
def main():
    eval_prompts = [it["prompt"] for it in json.loads((ROOT / "eval" / "prompts.json").read_text())["items"]]
    eval_bg = [bigrams(p) for p in eval_prompts]
    raw = make_prompts.remote(n_per_call=25, calls_per_cat=3)
    seen, kept, dropped = set(), [], {"dup": 0, "exclude": 0, "near_eval": 0}
    for r in raw:
        p = r["prompt"]
        if p in seen:
            dropped["dup"] += 1
            continue
        seen.add(p)
        if EXCLUDE.search(p):
            dropped["exclude"] += 1
            continue
        bg = bigrams(p)
        # 評価用の質問と似たもの（文字の2-gram の重なりが大きいもの）は除外する
        if any(len(bg & e) / max(1, len(bg | e)) > 0.3 for e in eval_bg):
            dropped["near_eval"] += 1
            continue
        kept.append(r)
    random.Random(0).shuffle(kept)
    kept = kept[:N_PROMPTS]
    print(f"依頼文 {len(raw)} 件 → {len(kept)} 件を使用（除外: {dropped}）")
    (SFT_DIR / "replay_prompts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))

    prompts = [r["prompt"] for r in kept]
    for model, outs in zip(ANSWER_MODELS, answer.starmap([(prompts, m) for m in ANSWER_MODELS])):
        path = SFT_DIR / f"replay_{model.split('/')[-1]}.jsonl"
        with path.open("w") as f:
            for r, o in zip(kept, outs):
                f.write(json.dumps({**r, "model": model, **o}, ensure_ascii=False) + "\n")
        print(f"{model}: 途中で切れた回答 {sum(o['finish_reason'] == 'length' for o in outs)} / {len(outs)} → {path.name}")
