# Qwen3.5-0.8B-Japanese-SFT-v2.5 の学習

[Qwen3.5-0.8B-Japanese-SFT-v2.5](https://huggingface.co/Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2.5) を作ったスクリプトです。学習に使ったデータは [Qwen3.5-0.8B-Japanese-SFT-v2.5-Data](https://huggingface.co/datasets/Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2.5-Data) にそのまま公開しています。

公開の [v2](https://huggingface.co/Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2) から、2段階で全部の重みを追加学習しました（画像の部分は固定）。

| 順番 | スクリプト | 内容 |
|---|---|---|
| 1 | `sft/build_reasoning_problems.py` | 理系の問題を集める（GSM8K の train、DeepMath-103K、OpenScienceReasoning-2）。MMLU-Pro と文面が重なる問題は除く |
| 2 | `sft/gen_sci_modal.py` | 先生役（Qwen3.8-27B）が問題を日本語に訳して短く解く。正解と一致したものだけ残す |
| 3 | `sft/build_think_sft.py` | 2. を思考ありの形（解き方を思考の部分へ）と思考なしの形にする |
| 4 | `build_stage1_mix.py` | 1段目のデータ：理系の手本・日本語のコードの説明と確認（Japanese-Code-Review-OSS）・会話の手本・Tengentoppa・答え方の練習 |
| 5 | `train_kaggle.py`（`STAGE=1`） | 1段目：v2 から 9,692件を学習率 1e-5 で1周 |
| 6 | `gen_think_chat_kaggle.py` | 1段目のデータの会話・コード・答え方の練習から 1,600件を選び、質問と答えはそのままに、間の短い思考だけを先生役が書く |
| 7 | `build_stage2_mix.py` | 2段目のデータ：6. の 1,600件と、1段目のデータからの復習 3,000件 |
| 8 | `train_kaggle.py`（`STAGE=2`） | 2段目：1段目のモデルから 4,600件を学習率 5e-6 で1周 |
| 9 | `build_release_data.py`・`check_overlap.py` | 公開用のデータの形にし、評価の問題（MMLU-Pro・MGSM・JCommonsenseQA）と重ならないことを確かめる |

- 2段目を足した理由：1段目だけのモデルは、思考ありで理系以外の質問をすると、96% で答えを考える部分に書いたまま終わりました（思考ありの手本が理系だけだったため）
- 学習の環境：Kaggle の T4。bf16 が使えないので fp32 で計算し、bf16 で保存しました。8bit Adam、出力層の損失は 1,024 トークンずつ計算（`sft/train_lora_colab.py` の `--fp32 --optim adamw_bnb_8bit --chunked-loss`）
- 4.・7. は、手元のファイルから作り直すと公開データと同じ中身になることを確かめています（7. は1バイトも違わず一致。4. は件数と中身が一致し、検証用 150件の分け方だけ公開データの方が実際に使ったもの）
- 6. の入力（選んだ 1,600件の質問と答え）は、公開データの `stage2_train` のうち `subset` が `chat_think_` で始まる行の `messages` と同じです

評価は `eval/run_reasoning_llamacpp.py`（理科・算数、llama.cpp）、`eval/run_eval_llamacpp.py`（会話）で行い、結果はモデルの説明に載せています。
