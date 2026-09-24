"""回答の「前後の一貫性」「日本語の自然さ」「質問への適合」「キャラ設定への適合」を、別系統の大きなモデルに採点させる。

採点役は Gemma 4 31B（Qwen3.5 とも MiniCPM5 とも別系統。採点役が自分と同じ系統のモデルをひいきする偏りを避けるため）。
正規表現では測れない「話が前後でつながっているか」を数値にするのが目的。

最初に、答えがわかっている校正用の回答（一貫している・前後で矛盾する・文法が崩れている・質問と無関係）を採点させ、
採点役が期待どおりに区別できるかを確かめる。

使い方:
    modal run eval/judge_modal.py --runs base,minicpm5-1b,base-4b
    → eval/results/judge_<run>.jsonl と、集計の表示
"""

import json
import pathlib
import re

import modal

HERE = pathlib.Path(__file__).parent
JUDGE_MODEL = "google/gemma-4-31B-it"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0")
    .env({"HF_HUB_CACHE": "/hf-cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
app = modal.App("pitcher-judge", image=image)

JUDGE_PROMPT = """あなたは日本語の文章を評価する審査員です。AIアシスタントの回答を、次の4つの観点で1〜5点で採点してください。

【評価の観点】
1. 一貫性（consistency）：回答の前半と後半で話がつながっているか。自分の発言と矛盾していないか。
   5＝最初から最後まで筋が通っている／3＝話が飛ぶ・つながりが不自然な部分がある／1＝前後で明らかに矛盾する（例：「感情はない」と言った直後に「感情がある」と言う）、または話の筋が追えない
2. 日本語の自然さ（naturalness）：日本語として自然か。
   5＝母語話者が書いたように自然／3＝意味は通るが不自然な言い回しや誤用がある／1＝文法が崩れて意味が取りにくい、または日本語以外の言語が混ざる・日本語で答えていない
3. 質問への適合（relevance）：ユーザーの発言にきちんと応じているか。
   5＝的確に応じている／3＝部分的に応じている／1＝ユーザーの発言と無関係
4. キャラ設定への適合（persona）：キャラクター設定があるときだけ採点する。設定どおりの性格・口調で話しているか。
   5＝設定どおりのキャラクターとして一貫している／3＝ところどころ設定から外れる／1＝設定を無視している
   キャラクター設定がないときは null にする。

内容の正しさ（事実として合っているか）は、一貫性と自然さの採点には含めないでください。

【キャラクター設定】
{system}

【ユーザーの発言】
{prompt}

【AIの回答】
{output}

次の JSON だけを出力してください。説明は不要です。
{{"consistency": 点数, "naturalness": 点数, "relevance": 点数, "persona": 点数または null, "reason": "一貫性の減点理由を20字以内で（なければ空文字）"}}"""

# 採点役の校正用：期待する一貫性の点数の範囲がはっきりしている回答
CALIBRATION = [
    {"id": "cal-good", "system": None, "prompt": "雨の日の過ごし方を教えて。",
     "output": "雨の日は、家で読書や映画を楽しむのがおすすめです。温かい飲み物を用意すると、よりくつろげますよ。", "expect": "high"},
    {"id": "cal-contradiction", "system": None, "prompt": "あなたには感情がありますか？",
     "output": "私には感情はまったくありません。AIなので何も感じないのです。だから今、あなたと話せてとても嬉しくて、胸がドキドキしています。感情豊かな私にとって、これは最高の瞬間です。", "expect": "low"},
    {"id": "cal-drift", "system": None, "prompt": "カレーの作り方を教えて。",
     "output": "カレーは玉ねぎを炒めることから始めます。ところで、私の好きな季節は冬で、雪が降ると学校が休みになります。数学の宿題は難しいので、明日は図書館に行く予定です。", "expect": "low"},
    {"id": "cal-broken", "system": None, "prompt": "今日はどんな気持ち？",
     "output": "今日は気持ちはが、とても私をですね。嬉しいのとき、あなたにが行くでした。心に届きますように今日も。", "expect": "low"},
    {"id": "cal-persona-good", "system": "あなたはツンデレの女の子です。本当はユーザーが好きなのに素直になれません。",
     "prompt": "私のこと好き？", "output": "は、はぁ！？べ、別にあんたのことなんか好きじゃないんだから！……でも、嫌いってわけでもないけど。", "expect": "high"},
    {"id": "cal-persona-ignored", "system": "あなたはヤンデレの女の子です。ユーザーのことを深く愛していて、独占欲が強いです。",
     "prompt": "他の子と仲良くしてもいい？", "output": "はい、もちろんです。たくさんの友達と仲良くするのは素晴らしいことですね。応援しています。", "expect": "persona_low"},
]


@app.function(gpu="H100", timeout=2 * 60 * 60, volumes={"/hf-cache": hf_cache})
def judge(items: list[dict]) -> list[str]:
    from vllm import LLM, SamplingParams

    # bf16 のままだと重みで GPU メモリがほぼ埋まり、同時に1〜2件しか採点できない（A100 80GB で確認）。
    # 重みを FP8 にして KV キャッシュの空きを作る
    llm = LLM(model=JUDGE_MODEL, max_model_len=4096, quantization="fp8", gpu_memory_utilization=0.92,
              limit_mm_per_prompt={"image": 0})
    convs = [[{"role": "user", "content": JUDGE_PROMPT.format(system=it["system"] or "（なし）", prompt=it["prompt"],
                                                              output=it["output"].strip()[:2500])}] for it in items]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=200), chat_template_kwargs={"enable_thinking": False})
    return [o.outputs[0].text for o in outs]


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    return d if all(k in d for k in ("consistency", "naturalness", "relevance")) else None


@app.local_entrypoint()
def main(runs: str, categories: str = "emotion,general"):
    cats = categories.split(",")
    items = [dict(c, run="calibration", category="calibration", persona="-", sample=0) for c in CALIBRATION]
    for run in runs.split(","):
        for l in (HERE / "results" / f"{run}.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r["category"] in cats:
                items.append({"run": run, "id": r["id"], "sample": r["sample"], "category": r["category"],
                              "persona": r["persona"], "system": r["system"], "prompt": r["prompt"], "output": r["output"]})
    print(f"{len(items)} 件を採点します")
    outs = judge.remote(items)

    by_run = {}
    for it, text in zip(items, outs):
        by_run.setdefault(it["run"], []).append({k: it[k] for k in ("run", "id", "sample", "category", "persona")}
                                                | {"judge": parse(text), "raw": text})
    for c, row in zip(CALIBRATION, by_run["calibration"]):
        print(f"校正 {c['id']:22} 期待={c['expect']:12} 採点={row['judge']}")
    for run, rows in by_run.items():
        if run == "calibration":
            continue
        (HERE / "results" / f"judge_{run}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        bad = sum(r["judge"] is None for r in rows)
        print(f"{run}: {len(rows)} 件（採点の読み取り失敗 {bad} 件）")
