"""採点役のプロンプト・校正用の回答・結果の読み取り（Modal 版の judge_modal.py と Colab 版の vllm_colab.py で共有する）。"""

import json
import re

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


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    return d if all(k in d for k in ("consistency", "naturalness", "relevance")) else None
