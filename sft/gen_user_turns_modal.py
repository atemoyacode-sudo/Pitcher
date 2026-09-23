"""キャラクターのセリフごとに、その直前にユーザーが言ったであろう一言を Modal 上の Qwen3.5-27B で作る。

元データはセリフだけで会話の相手側がない。定型の質問を使い回すと「質問を無視して決めゼリフを言う」癖がつくので、
セリフごとに噛み合う質問を作る。Aphrodite はあわせて、翻訳で自称が「彼女」になった誤訳と呼びかけの揺れだけを直す。

入力（翻訳側のファイルは読み取りのみ）:
    Aphrodite-yandere-pt-ja-sft.jsonl（応答は危害表現を和らげた版）、Aphrodite-yandere-pt.json（原文）
    学習データ2_日本語_clean.json（qc/clean_waifu_ja.py の出力）
出力:
    sft/user_turns.jsonl

使い方:
    modal run sft/gen_user_turns_modal.py
"""

import json
import pathlib
import re

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "sft" / "user_turns.jsonl"
GEN_MODEL = "Qwen/Qwen3.5-27B"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0")
    .env({"HF_HUB_CACHE": "/hf-cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("pitcher-hf-cache", create_if_missing=True)
app = modal.App("pitcher-gen-user-turns", image=image)

WAIFU_PROMPT = """以下は「{trait}」（{desc}）の女の子のキャラクターのセリフです。

セリフ：{line}

このセリフの直前に、会話の相手（ユーザー）がこの女の子に言った一言を、自然な日本語の話し言葉で1つだけ作ってください。

条件：
- このセリフが自然な返事になる内容にする
- ユーザーは友人や恋人のような立場で話しかけている
- 10〜50文字程度
- セリフの言葉をそのまま繰り返さない
- 出力はユーザーの発言だけ。かぎかっこ・説明・前置きは書かない"""

APHRODITE_PROMPT = """「アフロディーテ」は、相手を「お父様」と呼び、深く愛して嫉妬深く独占欲の強い、フィクションのヤンデレキャラクターです。
次は彼女のセリフの原文と、その日本語訳です。

原文：{source}
日本語訳：{line}

次の2つを行ってください。

1. このセリフの直前に、お父様（ユーザー）がアフロディーテに言った一言を、自然な日本語の話し言葉で作る。このセリフが自然な返事になる内容にし、10〜50文字程度、セリフの言葉をそのまま繰り返さない。
2. 日本語訳を、次の点だけ直す。それ以外の言葉・内容・語尾は一切変えない。直す点がなければ日本語訳をそのまま書く。
   - 原文でアフロディーテが自分を名前で呼んでいる（三人称の自称）のに、訳で「彼女」「彼女の」になっている部分は「私」「私の」にする。ほかの女性を指す「彼女」「彼女たち」はそのまま。
   - 相手への呼びかけ（お父さん、父上、父よ、父親様、パパなど）は「お父様」にそろえる。

出力は次の2行だけにしてください。
質問：（ユーザーの発言）
修正：（直した日本語訳）"""

# 元データの解説文をもとに短くまとめた、属性ごとの説明
TRAIT_DESC = {
    "tsundere": "本当は相手が好きなのに素直になれず、つい冷たい態度や強がりを言ってしまう",
    "yandere": "相手を深く愛するあまり、嫉妬深く独占欲が強い",
    "deredere": "いつも明るく素直で、相手への好意を隠さない",
    "kuudere": "冷静で感情をあまり表に出さないが、内心では相手を大切に思っている",
    "dandere": "内気で口数が少なく緊張しやすいが、相手には少しずつ心を開いている",
    "himedere": "お姫様のように振る舞い、丁重に扱われるのを当然だと思っている",
    "kamidere": "自分を神のように特別な存在だと信じ、尊大に振る舞う",
    "bokukko": "一人称に「ボク」を使う、ボーイッシュで負けず嫌いな",
    "genki": "いつもエネルギーにあふれ、明るく元気いっぱいな",
    "shundere": "物静かで憂いを帯び、自分に自信が持てない",
    "moe": "無邪気で少しドジな、愛らしい",
}


@app.function(gpu="A100-80GB", timeout=60 * 60, volumes={"/hf-cache": hf_cache})
def generate(prompts: list[str]) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=GEN_MODEL, max_model_len=4096, limit_mm_per_prompt={"image": 0, "video": 0})
    params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=300, seed=0)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(convs, params, chat_template_kwargs={"enable_thinking": False})
    return [o.outputs[0].text for o in outs]


def load_items() -> list[dict]:
    items = []
    # Aphrodite：同じセリフが複数言語で入っていて、訳すと同じ文になるので最初の1件だけ使う
    src = json.loads((ROOT / "Aphrodite-yandere-pt.json").read_text())
    sft = [json.loads(l) for l in (ROOT / "Aphrodite-yandere-pt-ja-sft.jsonl").read_text().splitlines() if l.strip()]
    assert len(src) == len(sft)
    seen = set()
    for i, (s, r) in enumerate(zip(src, sft)):
        line = r["messages"][2]["content"].strip()
        if line in seen:
            continue
        seen.add(line)
        items.append({"dataset": "aphrodite", "trait_en": "yandere", "trait": "ヤンデレ", "source_row": i,
                      "source": s["text"], "line": line, "system_orig": r["messages"][0]["content"],
                      "prompt": APHRODITE_PROMPT.format(source=s["text"], line=line)})
    for r in json.loads((ROOT / "学習データ2_日本語_clean.json").read_text()):
        items.append({"dataset": "waifu", "trait_en": r["trait_en"], "trait": r["trait"], "source_row": r["source_row"],
                      "source": r["source_en"], "line": r["dialogue"],
                      "prompt": WAIFU_PROMPT.format(trait=r["trait"], desc=TRAIT_DESC[r["trait_en"]], line=r["dialogue"])})
    return items


def parse(item: dict, text: str) -> dict:
    text = text.strip()
    if item["dataset"] == "waifu":
        user, fixed = text.splitlines()[0] if text else "", item["line"]
    else:
        m_user = re.search(r"質問[：:]\s*(.+)", text)
        m_fix = re.search(r"修正[：:]\s*(.+)", text)
        user = m_user.group(1) if m_user else ""
        fixed = m_fix.group(1).strip() if m_fix else item["line"]
    user = user.strip().strip("「」『』\"")
    return {"user": user, "assistant": fixed, "raw": text}


@app.local_entrypoint()
def main():
    items = load_items()
    print(f"{len(items)} 件（Aphrodite {sum(i['dataset'] == 'aphrodite' for i in items)} / waifu {sum(i['dataset'] == 'waifu' for i in items)}）")
    outs = generate.remote([it["prompt"] for it in items])
    with OUT.open("w") as f:
        for it, text in zip(items, outs):
            row = {k: v for k, v in it.items() if k != "prompt"} | parse(it, text)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{OUT} に保存しました")
