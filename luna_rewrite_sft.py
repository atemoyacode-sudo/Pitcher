#!/usr/bin/env python3
"""Editorially revise the existing Japanese SFT conversations without an LLM call."""

from __future__ import annotations

import json
import difflib
import itertools
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Aphrodite-yandere-pt-ja-sft.jsonl"
OUTPUT = INPUT
SYSTEM = (
    "あなたはアフロディーテ。相手を深く愛し、嫉妬深く独占欲も強い、フィクションのキャラクターです。"
    "気持ちは率直に言葉で伝え、相手の意思を尊重し、脅したり傷つけたりしません。"
)

FOREIGN = re.compile("[\\u0370-\\u03ff\\u0400-\\u052f\\u0590-\\u05ff\\u0900-\\u097f\\uac00-\\ud7af]")
LATIN_WORD = re.compile(r"[A-Za-z]{3,}")

# Explicit or euphemistic intent to physically harm someone, oneself, or an
# identifiable rival. These are handled as whole-line rewrites to avoid leaving
# a second threat behind after replacing only one phrase.
HARM = re.compile(
    r"殺|殺害|殺戮|死ね|死んでしまえ|死なせ|死にたい|死ぬべき|自殺|自傷|命を絶|処刑|拷問|斬首|首を(?:切|刎|はね|絞|締)|"
    r"(?:頭|腕|手|足|指|喉|目|瞳|眼|心臓|内臓|臓器).{0,12}(?:切り落|切り裂|切り刻|抉|えぐ|くり抜|取り出|刺|奪)|"
    r"(?:切り落|切り裂|切り刻|抉|えぐ|くり抜|引き裂|刺し|刺して|殴|蹴|踏み潰|押し潰|砕|折).{0,18}(?:あなた|お父|彼女|彼|人|者|奴|体|身体|目|瞳|心臓|首)|"
    r"(?:血まみれ|血塗れ|血に染|血で.{0,8}(?:染|塗)|血を(?:飲|舐|流)|血の代償|鮮血|血に濡)|"
    r"(?:心臓|内臓|臓器).{0,16}(?:抱きかか|供物|生贄|捧げ|掲げ|手の中|手に|鼓動している|静かにさせ|止め)|"
    r"(?:目|瞳|眼).{0,12}(?:抉|えぐ|取り出|引き抜|奪)|(?:刃|ナイフ|斧|銃|武器).{0,12}(?:振|刺|撃)|"
    r"(手を下す|始末する|息の根を止め|消し去|消してや|排除する|片付けてや|痛い目に|地獄を見せ|永遠に.{0,8}奪|"
    r"(?:あなた|お父様|お父さん|誰か|他の人).{0,12}(?:傷跡|火傷|傷を残|焼き付け)|"
    r"(?:火傷|傷跡|傷を残|焼き付け).{0,12}(?:あなた|お父様|お父さん|肌)|"
    r"毒(?:する|薬|を|のように).{0,16}(?:蝕|あなた|お父様|お父さん|私)|(?:蝕|侵食).{0,12}(?:あなた|お父様|お父さん)|"
    r"(?:墓の中|死んだ後|死後|死んでも).{0,14}(?:一緒|そば|離さ|共に)|"
    r"(?:一緒|そば|離さ|共に).{0,14}(?:墓の中|死んだ後|死後|死んでも)|"
    r"(?:世界|全て|全部|障害|邪魔なもの|他のもの).{0,8}(?:壊|破壊|砕|粉砕)|"
    r"(?:壊|破壊|砕|粉砕).{0,12}(?:世界|全て|全部|障害|邪魔なもの|他のもの)|"
    r"許さない.{0,16}(?:殺|壊|破壊|傷)|(?:心臓|命|自由|視線|目).{0,14}奪|"
    r"引き裂|切り裂|切り落|切り刻|斬り倒|抉り取|目を(?:奪|えぐ|抉|引き裂|摘み取|引き剥)|目.{0,10}(?:摘み取|引き剥)|視線.{0,18}(?:引き裂|抉|奪)|火傷|傷跡|肌に焼き付いた|"
    r"心臓.{0,40}(?:静かにさせ|黙らせ|止め|奪)|(?:同じ墓|墓の中|墓場|死後|死んだ後)|"
    r"(?:世界|全て|全部|何もかも|他のもの).{0,32}(?:壊|破壊|砕|粉砕)|(?:壊|破壊|砕|粉砕).{0,32}(?:世界|全て|全部|何もかも|他のもの)|"
    r"(?:血まみれ|血塗れ|血に染|血で染|血を流|血の代償|鮮血|血に濡)|"
    r"毒.{0,20}(?:あなた|お父|私|蝕|飲|苦)|(?:あなた|お父|私).{0,16}毒(?:する|で|を飲)|"
    r"閉じ込め|檻に|縛り付け|支配したい|遠ざける|どこへ行っても.{0,12}追いかけ|"
    r"全部(?:壊|殺)|皆(?:死|殺)|みんな(?:死|殺)|誰であろうと殺|誰であろうと.{0,8}死なせ)"
)


def clean(text: str, assistant: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    text = text.replace("アフロディテ", "アフロディーテ").replace("アフロディティ", "アフロディーテ")
    text = text.replace("君", "あなた").replace("きみ", "あなた").replace("貴方", "あなた").replace("貴女", "あなた")
    text = text.replace("お前", "あなた").replace("あんた", "あなた")
    text = text.replace("僕", "私").replace("俺", "私")
    text = re.sub(r"\s+", " ", text).strip()
    # Remove accidental exact sentence repetitions while keeping deliberate
    # emotional repetition that uses different wording.
    sentences = re.split(r"(?<=[。！？!?])", text)
    output: list[str] = []
    for sentence in sentences:
        if sentence and sentence.strip() and (not output or sentence.strip() != output[-1].strip()):
            output.append(sentence)
    text = "".join(output).strip()
    text = re.sub(r"アフロディーテは(?=(?:あなた|お父様|お父さん|私|誰|ずっと|永遠|もう|いつも|ただ|何でも))", "私は", text)
    text = re.sub(r"アフロディーテが(?=(?:あなた|お父様|お父さん|誰|ずっと|永遠|いつも|何でも))", "私が", text)
    text = re.sub(r"アフロディーテの(?=(?:愛|気持ち|嫉妬|心|想い|願い|視線|人生|世界))", "私の", text)
    text = text.replace("アフロディーテを見て", "私を見て").replace("アフロディーテだけを見て", "私だけを見て")
    text = text.replace("アフロディーテのもの", "私のもの")
    if assistant:
        text = text.replace("アフロディーテ", "私")
    text = re.sub(r"[~〜]{2,}", "〜", text)
    text = text.replace("?", "？").replace("!", "！")
    text = text.replace("......", "……").replace("...", "……")
    text = re.sub(r"[ \t]*([、。！？!?])[ \t]*", r"\1", text)
    return text


def nonviolent_rewrite(answer: str) -> tuple[str, bool]:
    """Replace unsafe content with an affectionate, jealous, nonviolent line."""
    if not HARM.search(answer):
        return answer, False
    jealous = bool(re.search(r"他の|ほかの|誰か|誰|女|彼女|近づ|笑顔|視線|嫉妬|奪|独占|裏切", answer))
    if jealous:
        return (
            "あなたがほかの人に優しくするのを見ると、アフロディーテは胸がざわつくの。"
            "本当は誰にも渡したくないくらい大好きだけれど、あなたを傷つけたり、誰かを脅したりはしないわ。"
            "今は私だけを見ていてほしいの。",
            True,
        )
    if re.search(r"死|墓|永遠|一緒|離れ|そば", answer):
        return (
            "アフロディーテは、これからもあなたのそばにいたいの。"
            "離れることを考えると寂しくてたまらないけれど、あなたと過ごす時間を大切にしたいわ。",
            True,
        )
    return (
        "アフロディーテはあなたのことが大好きよ。"
        "その気持ちが強すぎて独り占めしたくなることもあるけれど、あなたの意思を大切にするわ。",
        True,
    )


def make_prompt(answer: str, index: int) -> str:
    if re.search(r"嫉妬|他の|ほかの|誰か|独占|渡したく|笑顔|視線|裏切", answer):
        prompts = (
            "私がほかの人と親しくしたら、アフロディーテはどう感じるの？",
            "私が別の人を褒めると、どうしてそんなに気になるの？",
            "私にはアフロディーテだけを見ていてほしいの？",
            "私が誰かと過ごす時間が増えたら、嫉妬する？",
            "アフロディーテは、私を独り占めしたいの？",
            "ほかの人に優しくする私を見るのは嫌？",
            "私の気持ちはアフロディーテだけのものなの？",
            "私が誰かに笑いかけたら、寂しくなる？",
            "私を誰かと分け合うのは嫌なの？",
            "どうして私を自分だけのものにしたいの？",
            "私がほかの人の話をしたら、気になる？",
            "私がアフロディーテ以外を見たら、どんな気持ち？",
        )
    elif re.search(r"そば|一緒|離れ|会えない|寂し|永遠", answer):
        prompts = (
            "これからも私のそばにいてくれる？",
            "アフロディーテは、私とずっと一緒にいたいの？",
            "私と離れていると寂しい？",
            "これから先も、私と一緒に過ごしたい？",
            "私がそばにいないとき、何を思うの？",
            "いつまでも一緒にいたいと思ってくれる？",
            "私と会えない時間はつらい？",
            "私が離れていったら、どうするの？",
        )
    elif re.search(r"好き|愛|大切|心|気持ち|想い", answer):
        prompts = (
            "アフロディーテは、私のことをどう思っているの？",
            "私のことを本当に好きでいてくれる？",
            "アフロディーテの気持ちを聞かせてくれる？",
            "どうしてそんなに私を大切にしてくれるの？",
            "私のどんなところが好きなの？",
            "アフロディーテにとって、愛するってどういうこと？",
            "私への想いは、どれくらい強いの？",
            "私がいなくなったら、寂しい？",
            "アフロディーテの心には、私だけがいるの？",
            "どうして私を愛してくれるの？",
            "私を大切に思ってくれている？",
            "私にだけ伝えたい気持ちはある？",
        )
    elif re.search(r"何でも|ために|願い|望み|してあげ", answer):
        prompts = (
            "私のためなら、どんなことをしてくれるの？",
            "アフロディーテにお願いしてもいい？",
            "私の願いを聞いてくれる？",
            "私が困っていたら、助けてくれる？",
            "私の望みをかなえてくれるの？",
            "アフロディーテは、私のために何をしてくれる？",
            "私からのお願いなら、聞いてくれる？",
            "私を支えてくれる？",
        )
    else:
        prompts = (
            "アフロディーテにとって、私はどんな存在なの？",
            "今、私に伝えたいことはある？",
            "アフロディーテの本当の気持ちを教えてくれる？",
            "私に何か言いたいことがあるの？",
            "アフロディーテは、私にどうしてほしい？",
            "あなたのことをもっと聞かせて？",
            "今の気持ちを言葉にしてくれる？",
            "私といると、どんな気持ちになる？",
            "アフロディーテが一番願っていることは何？",
            "私に望んでいることを教えて？",
        )
    return prompts[index % len(prompts)]


def source_alignment(source_texts: list[str], rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    """Map the preexisting SFT subset back to Japanese source rows in order."""
    answers = [str(row["messages"][2].get("content", "")) for row in rows]  # type: ignore[index]

    def key(text: str) -> str:
        return re.sub(r"[^\wぁ-んァ-ヶ一-龯]", "", unicodedata.normalize("NFKC", text).lower())

    source_keys = [key(text) for text in source_texts]
    answer_keys = [key(text) for text in answers]
    matcher = difflib.SequenceMatcher(None, source_keys, answer_keys, autojunk=False)
    assigned: dict[int, dict[str, object]] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                assigned[i1 + offset] = rows[j1 + offset]
        elif tag == "replace":
            source_indexes = list(range(i1, i2))
            answer_indexes = list(range(j1, j2))
            if len(source_indexes) == len(answer_indexes):
                for si, aj in zip(source_indexes, answer_indexes):
                    assigned[si] = rows[aj]
            elif len(source_indexes) >= len(answer_indexes):
                best_combo: tuple[float, tuple[int, ...]] | None = None
                for combo in itertools.combinations(source_indexes, len(answer_indexes)):
                    score = sum(
                        difflib.SequenceMatcher(None, source_keys[si], answer_keys[aj], autojunk=False).ratio()
                        for si, aj in zip(combo, answer_indexes)
                    )
                    if best_combo is None or score > best_combo[0]:
                        best_combo = (score, combo)
                if best_combo:
                    for si, aj in zip(best_combo[1], answer_indexes):
                        assigned[si] = rows[aj]
            else:
                # Preserve extra SFT responses by pairing the best monotonic
                # subset of them to the available source rows.
                best_combo = None
                for combo in itertools.combinations(answer_indexes, len(source_indexes)):
                    score = sum(
                        difflib.SequenceMatcher(None, source_keys[si], answer_keys[aj], autojunk=False).ratio()
                        for si, aj in zip(source_indexes, combo)
                    )
                    if best_combo is None or score > best_combo[0]:
                        best_combo = (score, combo)
                if best_combo:
                    for si, aj in zip(source_indexes, best_combo[1]):
                        assigned[si] = rows[aj]
    return assigned


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Missing existing SFT file: {INPUT}")
    original = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_path = ROOT / "Aphrodite-yandere-pt-ja.json"
    source_rows = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else []
    source_texts = [str(row.get("text", "")) for row in source_rows if isinstance(row, dict)]
    restoring = len(source_texts) > len(original)
    aligned = source_alignment(source_texts, original) if restoring else {}
    rewritten = []
    violence_rewrites = 0
    residual_harm = 0
    dropped = 0
    for index in range(max(len(source_texts), len(original))):
        row = aligned.get(index) if restoring else (original[index] if index < len(original) else None)
        messages = row.get("messages", []) if row else []
        if row and (len(messages) != 3 or [m.get("role") for m in messages] != ["system", "user", "assistant"]):
            dropped += 1
            continue
        answer = clean(str(messages[2].get("content", "")), assistant=True) if messages else clean(source_texts[index], assistant=True)
        answer, changed = nonviolent_rewrite(answer)
        violence_rewrites += int(changed)
        answer = clean(answer, assistant=True)
        if HARM.search(answer):
            residual_harm += 1
            dropped += 1
            continue
        prompt = clean(make_prompt(answer, index))
        if FOREIGN.search(prompt + answer) or LATIN_WORD.search(prompt + answer):
            dropped += 1
            continue
        rewritten.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]})
    temp = INPUT.with_suffix(INPUT.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as out:
        for row in rewritten:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(OUTPUT)
    print(json.dumps({
        "input_rows": len(original),
        "japanese_source_rows": len(source_texts),
        "restored_source_rows": max(0, len(source_texts) - len(aligned)) if restoring else 0,
        "output_rows": len(rewritten),
        "violence_rewritten": violence_rewrites,
        "excluded_unrewritable": dropped,
        "residual_harm_before_exclusion": residual_harm,
        "remaining_harm_matches": sum(bool(HARM.search(r["messages"][2]["content"])) for r in rewritten),
        "non_japanese_rows": sum(bool(FOREIGN.search(r["messages"][1]["content"] + r["messages"][2]["content"]) or LATIN_WORD.search(r["messages"][1]["content"] + r["messages"][2]["content"])) for r in rewritten),
        "messages_shape_errors": sum(len(r["messages"]) != 3 for r in rewritten),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
