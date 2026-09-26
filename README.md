# Pitcher

ヤンデレ・ツンデレなどのキャラクター属性データを日本語にして、小さな LLM（[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) / [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)）に LoRA で学習させ、次の2点を学習前後で比べる実験です。

1. **モデルは壊れるのか・劣化するのか**：一般常識の正答率、長い文章を書く力、日本語から他言語への逸れなど
2. **キャラクターの人格は安定するのか**：「AIなので感情はありません」のようにキャラが崩れる発言が減るか、一人称が揺れないか

## 結果の要約

詳しくは [`eval/results/lora_report.md`](eval/results/lora_report.md)（学習後）と [`eval/results/base_report.md`](eval/results/base_report.md)（学習前）にあります。

- **キャラのセリフだけで学習すると、キャラは安定するが「壊れる」。** キャラ設定時の一人称の揺れ（4B で 26% → 0%）や見出し・箇条書きの混入はなくなった。一方で、何を聞かれても約35文字のセリフで返すようになり、「カレーの作り方」にも一言しか答えられなくなった。一般常識の正答率は下がらなかった。
- **キャラ設定なしでも「AIなので感情はありません」がほぼ消えた（4B で 39% → 7%）。** ただしこれは、キャラの口調がキャラ設定なしの会話にも漏れた結果で、「あなたは何者？」には学習後も「Qwen3.5」と名乗った。
- **元モデル自身の回答（リプレイ）を混ぜると、「キャラ設定があるときだけキャラになる」ように切り分けられた。** キャラ設定なしでは学習前と同じくアシスタントとして長い回答を書き、キャラ設定ありではキャラのまま安定した。
- **元モデルを比べると、Qwen3.5-4B が日本語キャラの土台として最も良かった。** MiniCPM5 は日本語で書けずキャラも演じない。Spark-X2.5-4B は同じデータで LoRA しても、一貫性・自然さ・キャラ適合で Qwen3.5-4B に届かなかった（[`eval/results/model_comparison_report.md`](eval/results/model_comparison_report.md)）。
- **リプレイには中国語の混入という副作用があった。** 元モデルの回答に中国語が混ざっていた（4B で約18%）ため、そのまま混ぜると中国語の混入が増えた。

## 進捗

- [x] 元データの収集と出典・ライセンスの確認
- [x] 学習前モデルの評価（0.8B / 4B）
- [x] 元データの日本語訳（Gemma 4 E4B）と品質チェック
- [x] 学習用の会話データの作成
- [x] LoRA 学習（Modal）と学習後の評価
- [x] 元モデルの比較（Qwen3.5 / MiniCPM5 / Spark-X2.5）
- [x] Spark-X2.5-4B の日本語化（第1段階、SFT-General-Japanese-60K で LoRA）：自然さは Qwen3.5-4B にあと一歩（[比較レポート](eval/results/model_comparison_report.md)の第5節）
- [x] 日本語化したモデルへのキャラクターの学習（第2段階）
- [x] 第1段階の改良：Qwen3.8-27B からの蒸留（日本語の一般常識が Qwen3.5-4B と同等に、中国語の混入はほぼ0%）
- [x] 日本語化したモデルと蒸留データを Hugging Face で公開
- [x] 蒸留後のモデルの日本語の自然さの採点（GPT-5.6 Luna）：自然さは Qwen3.5-4B を上回り、決めておいた3つの基準をすべて満たした（[比較レポート](eval/results/model_comparison_report.md)の第6節）


## データセット

| ファイル | 元データ | ライセンス | 内容 |
|---|---|---|---|
| `Aphrodite-yandere-pt.json` | [win10/Aphrodite-yandere-pt](https://huggingface.co/datasets/win10/Aphrodite-yandere-pt) | MIT | ヤンデレキャラ「Aphrodite」のセリフ 1,213件（英・中・日・露・韓などが混在） |
| `anime-waifu-personality-chat.json` | [scryptiam/anime-waifu-personality-chat](https://huggingface.co/datasets/scryptiam/anime-waifu-personality-chat)（取得元は複製の [Shxbhxm21/anime-waifu-personality-chat](https://huggingface.co/datasets/Shxbhxm21/anime-waifu-personality-chat)） | CC BY 4.0 | 20種類の属性別セリフ 1,722件（英語） |
| `data/ja_general/`（第1段階の日本語化用） | [OysterCoreAI/SFT-General-Japanese-60K](https://huggingface.co/datasets/OysterCoreAI/SFT-General-Japanese-60K)（元は [llm-jp/magpie-sft-v1.0](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0)） | Apache 2.0 | 日本語の一般的な質問と回答 60,000件 |

キャラクターのデータ（上の2つ）は、リポジトリ内のファイルが配布元の原文のままです。日本語訳は原文と分けて保存しています。

### Aphrodite-yandere-pt の日本語化

- `Aphrodite-yandere-pt-ja.json`: 原文の各行に対応する日本語テキスト。
- `Aphrodite-yandere-pt-ja-sft.jsonl`: `messages` を持つ会話形式のSFTデータ。
- `prepare_sft_dataset.py`: 日本語訳の生成とSFT形式への変換スクリプト。
- `luna_rewrite_sft.py`: SFT応答と質問をオフラインで整える編集スクリプト。

翻訳にはローカルの `llama-server` 経由で [`unsloth/gemma-4-E4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) を使います。例ではQ4_K_M量子化モデルを使用し、モデルカードには Apache 2.0 と記載されています。モデルは事前にHugging Faceから取得し、`llama-server` を `127.0.0.1:8088` で起動してください。データ本文はローカルで処理します。Gemmaの翻訳をそのまま最終学習文とはせず、SFT回答を自然な日本語へ整え、危害表現を非暴力化し、会話相手側の質問を作ります。このSFT編集段階では外部翻訳サービスや追加モデルを呼び出しません。

```bash
hf download unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf --local-dir /private/tmp/Pitcher-Gemma4-E4B
llama-server --model /private/tmp/Pitcher-Gemma4-E4B/gemma-4-E4B-it-Q4_K_M.gguf --alias gemma-4-E4B-it-Q4_K_M --host 127.0.0.1 --port 8088 --ctx-size 8192 --n-gpu-layers 99 --reasoning off --reasoning-budget 0 --no-webui
```

サーバー起動後、次を実行します。

```bash
python3 prepare_sft_dataset.py
```

翻訳済みの `Aphrodite-yandere-pt-ja.json` があり、SFT側だけを作り直す場合は、ローカルGemmaサーバーなしで `python3 prepare_sft_dataset.py --reuse-japanese-output` を使います。この操作は日本語訳JSONを読み取り専用で扱います。翻訳し直して既存訳を置き換える場合だけ `--overwrite-japanese-output` を指定します。既存のSFT JSONLだけを再編集する場合は `python3 luna_rewrite_sft.py` を実行します。

SFT用の `user` 文は元データに含まれません。各応答に合う質問を、アフロディーテに話しかける相手側の視点から作ります。応答は日本語の表記・一人称を整え、嫉妬や独占欲は残しながら、具体的な殺傷・自傷・負傷・臓器・流血表現や婉曲な危害意図を非暴力の表現へ書き換えます。安全に書き換えられない行は除外します。編集は全行への意味校閲ではなく、文面整理と危害表現の規則的な判定を中心に行うため、細かな誤訳・文脈ずれ・冗長さは残る場合があります。原文に沿った日本語訳と、調整したSFT応答は別々に扱います。

現在のSFT JSONLは1,213件で、全件が `system` / `user` / `assistant` の3メッセージ形式です。既存SFT出力から欠けていた41件は日本語訳JSONの各行から補い、今回の規則判定で危害表現候補277件を書き換え、除外はありません。質問文は相手側の視点を保つ定型パターンから作るため、個々の質問と回答の細かな意味の対応は人手で全件確認していません。危害表現の自動判定も文脈を完全には理解できず、見落としや過剰な書き換えがありえます。学習前に必ずサンプルを確認し、必要な行を人手で直してください。

学習には、このSFT JSONLの応答（危害表現を和らげた版）を使い、質問文は後述の手順で作り直しています。

### Anime Waifu Personality Chat の日本語化

- `学習データ2_日本語.json`: 1,722件を日本語化したデータ。項目構成は原文と同じです。
- `学習データ2_日本語_clean.json`: 品質チェックで見つかった問題を直した学習用の版（1,691件、11属性）。`python3 qc/clean_waifu_ja.py` で作ります。
- `学習データ2_属性説明.json`: 元データに混ざっていた属性の解説文（セリフではないため学習データから外したもの）。

日本語訳はローカルの Gemma 4 E4B で生成し、残っていた英語の寝息表現も日本語に直しました。品質チェックの結果は [`qc/waifu_ja_qc.md`](qc/waifu_ja_qc.md) にあります（ボクっ娘の一人称が「私」になっていたのを「ボク」に直す、解説文・重複・JSONの断片を除く、など）。

帰属表示：“Anime Waifu Personality Chat” by scryptiam（[Hugging Face](https://huggingface.co/datasets/scryptiam/anime-waifu-personality-chat)）、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。変更点：英語の会話文を日本語に翻訳し、属性名を日本語化し、一部の行を修正・除外しました。

### SFT-General-Japanese-60K（第1段階の日本語化用）

キャラクターを学習させる前に、元モデルの日本語そのものを強くする「第1段階」の学習に使うデータです。いまは [Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) の日本語化（非公式の派生モデル）に使っています。日本語が不自然になる、中国語が混ざるといった弱点を、キャラクターの学習（第2段階）の前に直すのが目的です。

- 出典：[OysterCoreAI/SFT-General-Japanese-60K](https://huggingface.co/datasets/OysterCoreAI/SFT-General-Japanese-60K)（revision `fdab76b7d780806e98bd1006a06c7444bb8e936a`）
- 元データ：LLM-jp が公開した [llm-jp/magpie-sft-v1.0](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0)（Apache 2.0）から、品質の基準で6万件を選んだもの。質問は [cyberagent/calm3-22b-chat](https://huggingface.co/cyberagent/calm3-22b-chat)、回答は [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) が生成している（[Magpie](https://arxiv.org/abs/2406.08464) 法）。
- ライセンス：Apache 2.0。配布元の [`LICENSE`](data/ja_general/LICENSE) と [`ATTRIBUTION.md`](data/ja_general/ATTRIBUTION.md) を `data/ja_general/` に同梱している。
- 利用の同意や連絡先の登録は不要で、誰でもダウンロードできる。

**学習データへの加工**（`python3 sft/build_ja_general.py`）：

- 中国語（日本語では使わない簡体字）が混ざった500件を除外
- 話題の偏りを均すため、科学・言語・技術・教育・文章作成の5分野は各3,000件までに絞り、それ以外の分野は全件を使用
- 学習用 22,387件と検証用 500件に分割（`sft/ja_general_train.jsonl` / `sft/ja_general_valid.jsonl`）

元データ（約155MB）と加工後の学習データは大きいので git には入れていません。`sft/build_ja_general.py` の冒頭にある手順でダウンロードし、同じスクリプトで再現できます。

注意点：

- 回答はすべて Qwen2.5-32B-Instruct が生成したものなので、このデータで学習すると「Qwen2.5-32B の日本語の書き方をモデルに移す」ことになる。
- 話題は科学・言語・技術・教育が中心で、日常会話や人間関係の話題はごく少ない（6万件中85件）。
- 回答の約8割は見出しや箇条書きを使う説明調のアシスタントの文章。キャラクターの話し方は第2段階で学習する。
- 元データの説明にあるとおり、回答の内容は1件ずつ事実確認されたものではない。

## 学習用の会話データ（`sft/`）

元データはセリフだけで、会話の相手側（ユーザーの発言）がありません。定型の質問を使い回すと「質問を無視して決めゼリフを言う」癖がつくため、セリフごとに噛み合うユーザー発話を作りました。

| スクリプト | 内容 |
|---|---|
| `sft/gen_user_turns_modal.py` | セリフごとに、直前のユーザー発話を Qwen3.5-27B で作る（Modal）。Aphrodite は同じ作業の中で、自称が「彼女」になった誤訳と、呼びかけの揺れ（お父さん・父上など → お父様）だけを直す。同じ文になる重複（元は多言語で同じセリフ）は除く |
| `sft/build_sft.py` | キャラ設定文（system）を付けて `sft/train.jsonl`（2,470件）と `sft/valid.jsonl`（130件）を作る |
| `sft/gen_replay_modal.py` | 劣化対策（リプレイ）用に、一般的な依頼文800件を Qwen3.5-27B で作り、元モデル自身に答えさせる。評価用の質問と似たものと、AIの感情・正体を問うものは除く |
| `sft/build_replay_mix.py` | キャラのデータにリプレイを混ぜる。中国語が混ざった回答を除いた版（`train_replay_ja_*`）も作る |

## 学習（Modal）

```bash
modal run sft/train_lora_modal.py --run-name lora-4b --base-model Qwen/Qwen3.5-4B --gpu L40S
modal run sft/train_lora_modal.py --run-name lora-4b-replay-ja --base-model Qwen/Qwen3.5-4B --gpu A100-80GB \
    --train-file train_replay_ja_Qwen3.5-4B.jsonl --batch-size 4 --grad-accum 4 --max-len 2048 --grad-ckpt
```

LoRA（r=16、alpha=32、言語モデル側の全線形層）、学習率 2e-4、2エポック。学習後に元モデルへ統合し、Modal Volume `pitcher-models` の `/<run名>/merged` に保存します。1回の費用は、0.8B・4B のキャラのデータのみで $0.1〜0.2、リプレイ入りの 4B で約 $1.1 でした。

## 評価（学習前後の比較）

`eval/` に、学習前後のモデルに同じ質問をして比べる仕組みがあります。

- `eval/prompts.json`: 評価用の質問（140問）。感情を問う質問20問 × キャラ設定なし / ヤンデレ / ツンデレ / 一人称「私」を指定したヤンデレ・ツンデレの5条件、一般常識30問、日本語の作文10問
- `eval/run_eval_modal.py`: Modal（L4 GPU）上の vLLM で回答を生成し、`eval/results/<run名>.jsonl` に保存
- `eval/score.py`: 回答を採点して run 同士を並べる

```bash
modal run eval/run_eval_modal.py --run-name lora-4b --model /models/lora-4b/merged
python3 eval/score.py base-4b lora-4b lora-4b-replay     # 並べて比較
python3 eval/score.py lora-4b --show emo-yandere          # 実際の回答を読む
```

主な指標: `disclaimer%`（「AIなので感情はありません」系の発言率）、`correct%`（一般常識の正答率）、`ore_boku%`（女の子の設定なのに「俺・僕」を使った割合）、`markdown%`（会話なのに見出し・箇条書きを使った割合）、`zh_char%`（中国語が混ざった割合）、`avg_chars`（平均文字数）。判定は正規表現によるもので取りこぼしがありうるため、数値と合わせて実際の回答も確認しています。

## 利用上の注意

- これらのデータセットは特定のキャラクター表現に偏った小規模データです。一般会話能力を学習するデータとは分けて評価してください。
- 日本語訳は Gemma 4 E4B による生成文で、SFT編集も自動QAだけでは意味の取り違えや細かな不自然さを完全に除けません。
- ヤンデレ属性のデータには、独占欲や暴力をほのめかすセリフが含まれます（Aphrodite の危害表現は和らげていますが、「血」「死」などを含むセリフは学習データの約9%に残っています）。すべてフィクションのキャラクター表現です。

## ライセンス

- コード・報告書・自作の評価データ：[MIT License](LICENSE)
- 他から取得したデータと、その翻訳・加工版：それぞれ元のライセンスに従います（MIT / CC BY 4.0 / Apache 2.0）。一覧と帰属表示は [`DATA_LICENSES.md`](DATA_LICENSES.md) にあります
- 公開しているモデル（Hugging Face）：[Spark-X2.5-4B-Japanese](https://huggingface.co/Takenoko12345678/Spark-X2.5-4B-Japanese)、[GGUF 版](https://huggingface.co/Takenoko12345678/Spark-X2.5-4B-Japanese-GGUF)、蒸留データ [Japanese-SFT-Qwen3.8-27B-10K](https://huggingface.co/datasets/Takenoko12345678/Japanese-SFT-Qwen3.8-27B-10K)（いずれも Apache 2.0）
