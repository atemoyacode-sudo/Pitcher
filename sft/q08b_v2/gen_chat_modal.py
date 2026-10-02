"""一言のつぶやきや気持ちに自然に返す会話と、2〜3往復続く会話の手本を、先生役（Qwen3.8-27B）に作らせる（Modal）。

公開した 0.8B は、「悲しい」に「悲しむは『悲しむ』を意味する」と言葉の意味を答え、続けて話すと崩れた。学習データが
すべて1問1答で、「〇〇とは？」に意味を答える例が多く、つぶやきに寄り添って返す例も、続きの会話もなかったため。

1. つぶやきを作る：テーマ（気持ち・挨拶・愚痴・雑談・小さな相談など）ごとに、ユーザーが送りそうな短い言葉を作らせる
2. 返事を作る：自然な話し言葉で、まず受け止めて短く返す（1〜3文）。意味の説明はしない。人間だと名乗らず、
   「AI なので感情はありません」のような断りも入れない。前の発言と矛盾しない
3. 続きの会話：一部のつぶやきは、ユーザーの次の一言を作らせて、2〜3往復まで続ける

学習に使うときは、指示文（system）は付けず、ユーザーの発言と返事だけにする（指示がなくても同じように返せるようにするため）。

    modal run sft/q08b_v2/gen_chat_modal.py   # data/dpo/chat_teacher.jsonl
"""

import json
import pathlib
import random

import modal

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "dpo" / "chat_teacher.jsonl"
TEACHER = "Qwen/Qwen3.8-27B"
# Qwen3.8 のモデルカードの推奨値（思考なし）
SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)

THEMES = [
    "悲しい気持ち", "疲れた", "眠い", "嬉しいことがあった", "不安なこと", "イライラしている", "寂しい", "暇・退屈",
    "仕事の愚痴", "学校や勉強", "天気", "食べ物・ごはん", "朝の挨拶", "夜の挨拶・寝る前", "励ましてほしい", "褒めてほしい",
    "失敗してしまった", "緊張している", "体調が少し悪い", "趣味の話", "週末の予定", "人間関係のちょっとした悩み",
    "季節の話", "ペットの話", "ゲームの話", "音楽の話", "ちょっとした相談", "ひとりごと", "お礼を言う", "何となく話しかける",
]
LIST_PROMPT = """日本語のチャットで、ユーザーが AI アシスタントに送る短いメッセージの例を20個作ってください。
テーマ：{theme}

条件：
- 話し言葉で、1〜25字くらい。「〜とは？」のような言葉の意味の質問は入れない
- 質問だけでなく、つぶやき・気持ちの吐露・報告・呼びかけなど、いろいろな形を混ぜる
- 絵文字や記号の飾りは使わない
- 番号や記号を付けず、1行に1つずつ、メッセージだけを書く"""
SYSTEM = """あなたは日本語で話す AI アシスタントです。ユーザーのメッセージに、自然な日本語の話し言葉で返事をしてください。
- 気持ちやつぶやきには、まず受け止めて共感し、必要なら一言だけ質問や提案を添える
- 質問には、要点をわかりやすく答える
- 1〜3文、100字くらいまで。箇条書き・見出し・絵文字は使わない
- 言葉の意味を聞かれていないときは、意味の説明をしない
- 自分を人間だと言わない。「AI なので感情はありません」のような断りも入れない
- それまでの自分の発言と矛盾しないようにする"""
FOLLOW = """次の会話の続きとして、ユーザーが次に送りそうな短いメッセージを1つだけ書いてください。
話し言葉で、1〜30字くらい。メッセージだけを書き、説明はいりません。

{dialog}"""

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0")
    .env({"HF_HUB_CACHE": "/hf-cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
app = modal.App("pitcher-gen-chat", image=image)


@app.function(gpu="H100", timeout=2 * 60 * 60, volumes={"/hf-cache": hf_cache})
def run(n_single: int, n_multi: int, eval_prompts: list[str]) -> list[dict]:
    import re

    from vllm import LLM, SamplingParams

    llm = LLM(model=TEACHER, max_model_len=4096, quantization="fp8", gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    kw = {"chat_template_kwargs": {"enable_thinking": False}, "use_tqdm": False}

    def chat(convs, max_tokens, seed0):
        ps = [SamplingParams(**SAMPLING, max_tokens=max_tokens, seed=seed0 + i) for i in range(len(convs))]
        return [o.outputs[0].text.strip() for o in llm.chat(convs, ps, **kw)]

    # 1. つぶやき
    reqs = [(t, s) for t in THEMES for s in range(12)]
    lists = chat([[{"role": "user", "content": LIST_PROMPT.format(theme=t)}] for t, _ in reqs], 600, 0)
    bad = re.compile(r"死にたい|消えたい|自殺|殺|とは[？?]|意味")

    def bigrams(t):
        t = re.sub(r"\s+", "", t)
        return {t[i:i + 2] for i in range(len(t) - 1)}

    ev = [bigrams(p) for p in eval_prompts]
    seen, utts = set(), []
    for (theme, _), text in zip(reqs, lists):
        for line in text.splitlines():
            u = re.sub(r"^[\s\-・*\d.、)）]+", "", line).strip().strip("「」")
            if not (1 <= len(u) <= 30) or u in seen or bad.search(u):
                continue
            b = bigrams(u)
            if any(len(b & e) / max(1, len(b | e)) > 0.3 for e in ev):
                continue
            seen.add(u)
            utts.append({"theme": theme, "text": u})
    random.Random(0).shuffle(utts)
    print(f"つぶやき {len(utts)} 個", flush=True)
    utts = utts[: n_single + n_multi]

    # 2. 返事（1往復目）
    sysmsg = {"role": "system", "content": SYSTEM}
    dialogs = [[{"role": "user", "content": u["text"]}] for u in utts]
    replies = chat([[sysmsg] + d for d in dialogs], 200, 10_000)
    for d, r in zip(dialogs, replies):
        d.append({"role": "assistant", "content": r})

    # 3. 続きの会話（2往復目、半分は3往復目まで）
    multi = list(range(n_single, len(dialogs)))
    for turn in (2, 3):
        idx = multi if turn == 2 else multi[: len(multi) // 2]
        texts = ["\n".join(("ユーザー：" if m["role"] == "user" else "AI：") + m["content"] for m in dialogs[i]) for i in idx]
        follows = chat([[{"role": "user", "content": FOLLOW.format(dialog=t)}] for t in texts], 80, 20_000 * turn)
        for i, f in zip(idx, follows):
            dialogs[i].append({"role": "user", "content": f.strip().strip("「」")})
        replies = chat([[sysmsg] + dialogs[i] for i in idx], 200, 30_000 * turn)
        for i, r in zip(idx, replies):
            dialogs[i].append({"role": "assistant", "content": r})
    return [{"theme": u["theme"], "messages": d} for u, d in zip(utts, dialogs)]


@app.local_entrypoint()
def main(n_single: int = 3000, n_multi: int = 1500):
    evp = [it["prompt"] for it in json.loads((ROOT / "eval" / "prompts.json").read_text())["items"]]
    rows = run.remote(n_single, n_multi, evp)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    turns = [len(r["messages"]) // 2 for r in rows]
    print(f"{OUT}：{len(rows)} 件（1往復 {turns.count(1)}、2往復 {turns.count(2)}、3往復 {turns.count(3)}）")
