# データのライセンスと帰属表示

このリポジトリのコード・報告書・自作の評価データは [MIT License](LICENSE) です。ただし、他から取得したデータと、その翻訳・加工版は、それぞれ元のライセンスに従います。

| ファイル | 元データ | ライセンス |
|---|---|---|
| `Aphrodite-yandere-pt.json`（原文のまま） | [win10/Aphrodite-yandere-pt](https://huggingface.co/datasets/win10/Aphrodite-yandere-pt) | MIT（下記） |
| `Aphrodite-yandere-pt-ja.json`、`Aphrodite-yandere-pt-ja-sft.jsonl`（日本語訳と、学習用に書き換えた版） | 同上 | MIT（下記） |
| `anime-waifu-personality-chat.json`（原文のまま） | “Anime Waifu Personality Chat” by scryptiam（[Hugging Face](https://huggingface.co/datasets/scryptiam/anime-waifu-personality-chat)。取得元は複製の [Shxbhxm21/anime-waifu-personality-chat](https://huggingface.co/datasets/Shxbhxm21/anime-waifu-personality-chat)） | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `学習データ2_日本語.json`、`学習データ2_日本語_clean.json`、`学習データ2_属性説明.json`（日本語訳と加工版） | 同上 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。変更点：英語の会話文を日本語に翻訳し、属性名を日本語化し、一部の行を修正・除外 |
| `sft/train.jsonl`、`sft/valid.jsonl` など、上の2つから作った学習データ | 上の2つ | それぞれの行の元データのライセンス（`dataset` 列で区別できる） |
| `data/ja_general/`（LICENSE と ATTRIBUTION.md のみ。データ本体は含まない） | [OysterCoreAI/SFT-General-Japanese-60K](https://huggingface.co/datasets/OysterCoreAI/SFT-General-Japanese-60K)（元は [llm-jp/magpie-sft-v1.0](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0)） | Apache 2.0 |

## 生成に使ったモデル

- 日本語訳：Gemma 4 E4B（ローカルで実行）
- 学習用のユーザー発話・リプレイの依頼文：[Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)（Apache 2.0）
- リプレイの回答（`sft/replay_*.jsonl` など）：各モデル自身（Qwen3.5-0.8B / 4B、Spark-X2.5-4B。いずれも Apache 2.0）
- 評価結果（`eval/results/`）：評価した各モデルの回答と、採点役（Gemma 4 31B、GPT-5.6 Luna）の採点

## このリポジトリに含めていないもの

- JCommonsenseQA（CC BY-SA 4.0）・CommonsenseQA（MIT）の問題文：評価結果には問題番号・正解番号・モデルの回答だけを保存している
- AIME 2026（MathArena/aime_2026、CC BY-NC-SA 4.0）の問題文と解答過程：評価スクリプトが実行時に読み込み、結果は git の管理外に保存している
- 学習済みの重み：Hugging Face で公開（README を参照）

## 注意

- リプレイの回答や評価結果は、モデルが生成した文章をそのまま保存したもので、事実確認をしていません。架空の住所・URL・相談窓口などが含まれることがあります（例：「〒100-0000 東京都千代田区丸の内…△△株式会社」のようなひな形、実在しない URL）。事実として使わないでください。
- ヤンデレ属性のデータには、独占欲や暴力をほのめかすセリフが含まれます。すべてフィクションのキャラクター表現です。

---

## win10/Aphrodite-yandere-pt のライセンス（MIT）

配布元の Hugging Face のデータセットカードで MIT License と表示されています（著作権者の記載はありません）。

```
MIT License

Copyright (c) the contributors of win10/Aphrodite-yandere-pt
(https://huggingface.co/datasets/win10/Aphrodite-yandere-pt)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
