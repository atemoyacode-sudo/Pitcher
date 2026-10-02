# Qwen3.5-0.8B-Japanese-SFT-v2 の学習

[Qwen3.5-0.8B-Japanese-SFT-v2](https://huggingface.co/Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2) を作ったスクリプトです。v1 の作り方（`sft/` の蒸留・Tengentoppa・答え方の練習）に、次の段階を足しています。Spark-X2.5-4B の学習とは関係ありません。

| 順番 | スクリプト | 内容 | できるモデル |
|---|---|---|---|
| 1 | `sft/train_lora_modal.py`（v1 と同じ） | 蒸留（Qwen3.8-27B の回答） | `qwen08b-distill` |
| 2 | `cpt_modal.py` | 日本語版 Wikipedia（約1億トークン）の追加の事前学習。途中で Wikipedia と会話の検証データの損失を測る | `qwen08b-wiki` |
| 3 | `sft/train_lora_modal.py` | 会話の学び直し（v1 の第2段階と同じデータ） | `qwen08b-wiki-sft` |
| 4 | `gen_chat_modal.py` → `sft/train_lora_modal.py` | 先生役（Qwen3.8-27B）が作った、一言のつぶやきへの返事と2〜3往復の会話の手本で追加学習（3. のデータ 3,000件を混ぜる） | `qwen08b-wiki-chat` |
| 5 | `build_dpo_prompts.py` → `dpo_modal.py` | 自分の回答を Gemma 4 31B（`sft/score_data_modal.py --responses`）と規則で採点し、良い回答と崩れた回答の組で DPO | `qwen08b-wiki-dpo` |
| 6 | `merge_soup.py` | 3. と 5. の重みを半分ずつ平均（5. で減った一般常識を取り戻すため） | v2（`qwen08b-soup0.5`） |

評価結果は `eval/results/` の `qwen08b-wiki-*`・`qwen08b-soup*`・`lc-*`（llama.cpp）、Gemma による崩れにくさの採点は `eval/results/judge-gemma-data_v2_summary.json` にあります。

学習データ（Wikipedia の文章、先生役の会話、DPO の回答）はリポジトリに含めていません。各スクリプトで作り直せます。ライセンスは [`DATA_LICENSES.md`](../../DATA_LICENSES.md) を見てください。
