# Pitcher：v2.5 の学習（Kaggle の T4。fp32 で計算し bf16 で保存、8bit Adam、出力層の損失は 1,024 トークンずつ計算）。
# 1段目：公開の v2 から stage1_train（9,692件）を学習率 1e-5 で1周。2段目：1段目のモデルから stage2_train（4,600件）を学習率 5e-6 で1周。
# どちらも全部の重みを学習する（画像の部分は固定）。Kaggle のデータセットに train_lora_colab.py と学習データを置いて使う。
#   STAGE=1 python train_kaggle.py   → /kaggle/working/qwen08b-v25-stage1/
#   STAGE=2 BASE=<1段目の出力> python train_kaggle.py   → /kaggle/working/qwen08b-v25/
import glob, os, shutil, subprocess

OUT = "/kaggle/working"
stage = os.environ.get("STAGE", "1")
work = os.path.dirname(glob.glob("/kaggle/input/**/train_lora_colab.py", recursive=True)[0])
W = "/tmp/w"; shutil.copytree(work, W, dirs_exist_ok=True)
subprocess.run("pip install -q transformers==5.17.0 peft==0.21.0 bitsandbytes && pip uninstall -y torchao", shell=True)
if stage == "1":
    subprocess.run("hf download Takenoko12345678/Qwen3.5-0.8B-Japanese-SFT-v2 --local-dir /tmp/v2 --exclude '*.gguf'", shell=True, check=True)
    name, base, data, lr = "qwen08b-v25-stage1", "/tmp/v2", "stage1_train.jsonl", "1e-5"
else:
    name, base, data, lr = "qwen08b-v25", os.environ["BASE"], "stage2_train.jsonl", "5e-6"
cmd = (f"cd {W} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_lora_colab.py --run-name {name} --base-model {base} "
       f"--train {W}/{data} --valid {W}/stage1_valid.jsonl --lr {lr} --image-text --full --fp32 --chunked-loss --optim adamw_bnb_8bit "
       f"--epochs 1 --batch-size 1 --grad-accum 32 --max-len 4096 --drop-long --grad-ckpt --out /tmp/out")
subprocess.run(cmd, shell=True, check=True)
shutil.copytree(f"/tmp/out/{name}/merged", f"{OUT}/{name}")
