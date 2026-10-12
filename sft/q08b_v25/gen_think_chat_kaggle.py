# Pitcher：v2.5 の2段目用に、理系以外の質問にも「短く考えてから答える」思考ありの手本を作る（先生役 Qwen3.8-27B、llama.cpp、Kaggle の T4×2）。
# 1段目のモデルは思考ありの手本が理系だけだったので、会話・知識の質問を思考ありで出すと、96% で答えを考える部分に書いたまま終わった。
# 1段目の学習データから選んだ会話・コード・答え方の練習の質問と答え（thinkchat_src.jsonl、1,600件）はそのまま使い、
# その間に入る短い思考（1〜3文）だけを先生役に書かせる。
# 入力：Kaggle のデータセットに置いた thinkchat_src.jsonl と、Kaggle 用にビルドした llama.cpp（llama-server）
# 出力：/kaggle/working/thinkchat.jsonl（build_stage2_mix.py の --think-chat）
import glob, json, os, re, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
OUT = "/kaggle/working"
def log(m):
    s = time.strftime("%H:%M:%S ") + str(m); print(s, flush=True); open(f"{OUT}/pipeline.log", "a").write(s + "\n")
def sh(c, timeout=3600):
    t = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)
    log(f"$ {c[:140]} → {r.returncode}（{time.time() - t:.0f}秒）\n{(r.stdout + r.stderr)[-1000:]}"); return r.returncode
binsrc = os.path.dirname(glob.glob("/kaggle/input/**/llama-server", recursive=True)[0])
BIN = "/tmp/bin"
sh(f"mkdir -p {BIN} && cp {binsrc}/* {BIN}/ && chmod +x {BIN}/llama-server && cd {BIN} && "
   "for f in lib*.so.*.*; do b=${f%%.so.*}; v=${f#*.so.}; ln -sf $f $b.so.${v%%.*}; ln -sf $f $b.so; done")
os.environ["LD_LIBRARY_PATH"] = BIN + ":" + os.environ.get("LD_LIBRARY_PATH", "")
sh("hf download ggml-org/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf --local-dir /tmp/q")
lf = open(f"{OUT}/server.log", "w")
p = subprocess.Popen(f"{BIN}/llama-server -m /tmp/q/Qwen3.8-27B-Q4_K_M.gguf --port 8100 --jinja -ngl 99 -sm layer -c 32768 -np 4 -fa on -ub 256 -b 512",
                     shell=True, stdout=lf, stderr=subprocess.STDOUT)
for _ in range(180):
    try: urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=5); break
    except Exception: time.sleep(10)
log("起動")
def post(msgs, max_tokens):
    body = {"messages": msgs, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:8100/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return (json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]["message"]["content"] or "").strip()
ADD = ("次は、日本語の会話AIとユーザーのやり取りです。AI が最後の返事を書く前に、頭の中で短く考えたメモを書いてください。\n"
       "- 1〜3文、日本語で。まず、ユーザーが何を（誰について）聞いているかを押さえ、次に答え方の方針を書く\n"
       "- 返事の文章をそのまま書き写さない。メモだけを出力する\n\n{conv}\n\n【AI の最後の返事】\n{ans}")
def one(r):
    try:
        conv = "\n".join(f"【{'ユーザー' if m['role'] == 'user' else 'AI'}】\n{m['content']}" for m in r["messages"][:-1])
        th = post([{"role": "user", "content": ADD.format(conv=conv, ans=r["messages"][-1]["content"])}], 300)
        return {**r, "thinking": th}
    except Exception as e:
        return {**r, "error": str(e)[:200]}
src = [r for r in (json.loads(l) for l in open(glob.glob("/kaggle/input/**/thinkchat_src.jsonl", recursive=True)[0])) if r["kind"] == "add_think"]
with ThreadPoolExecutor(4) as ex: rows = list(ex.map(one, src))
open(f"{OUT}/thinkchat.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
log(f"完了 {len(rows)} 件、失敗 {sum('error' in r for r in rows)} 件")
p.kill()
