"""Weak-spot plan stage B1: synthesize context sentences for slang terms with the teacher (Qwen3.5-9B, nf4).

Each request names ONE seed term and asks for 10 sentences that use it naturally. Only seed-half words
(see tools/slang_eval.py --split-out) may appear in prompts; eval-half words never do, and seed words that
contain an eval word as a substring are skipped. Raw output is appended to <out>/raw_slang.jsonl and the run
is resumable per (seed, request index). Requests go round-robin over the seeds so a partial run is still balanced.

Usage: python tools/gen_slang_corpus.py --out <dir> --split <split.json> --draft <slang_draft.tsv> --reqs 5
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

CJK = re.compile(r"^[一-鿿]{2,8}$")
STYLES = ["微信或QQ群里的聊天", "视频弹幕", "短视频评论区留言", "微博或小红书式的发帖口吻", "朋友之间当面吐槽", "论坛回帖", "直播间的评论"]
TONES = ["随意口语", "开玩笑", "吐槽", "夸张感叹", "简短省略主语", "无奈", "兴奋", "阴阳怪气但不带脏字"]
TEMPLATE = ("请写 10 句简体中文，场景是{style}，语气：{tone}。每句都要自然地用到网络用语“{term}”。\n"
            "要求：每句 8 到 30 个字，口吻像真人随手打出来的；每句内容各不相同，句式多样，“{term}”在句中的位置也要不同；"
            "不要编号，不要引号，不要解释，每行一句；不要出现真实的人名、电话、地址等个人信息；"
            "只用这个词广为人知的用法，如果不确定它的意思，就把它当作普通的流行说法来造句，不要编造专名、梗的出处或剧情；不要带脏话。")


def load_seeds(split_path, draft_path):
    split = json.load(open(split_path, encoding="utf-8"))
    eval_words = [w for w, v in split.items() if v == "eval"]
    seeds, skipped = [], 0
    seen = set()
    for ln in open(draft_path, encoding="utf-8"):
        if "\t" not in ln:
            continue
        w = ln.rstrip("\n").split("\t")[1]
        if split.get(w) != "seed" or w in seen or not CJK.match(w):
            continue
        seen.add(w)
        if any(e in w for e in eval_words):
            skipped += 1
            continue
        seeds.append(w)
    return seeds, skipped, eval_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--reqs", type=int, default=5, help="requests per seed (10 sentences each)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after this many requests")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds, skipped, eval_words = load_seeds(a.split, a.draft)
    (out / "seeds.json").write_text(json.dumps(seeds, ensure_ascii=False), encoding="utf-8")
    print(f"seeds {len(seeds)}, skipped (contain an eval word) {skipped}", flush=True)
    raw = out / "raw_slang.jsonl"
    done = set()
    if raw.exists():
        for l in open(raw, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["seed"], r["req"]))
    todo = [(s, k) for k in range(a.reqs) for s in seeds if (s, k) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"requests to run {len(todo)} (already done {len(done)})", flush=True)
    if not todo:
        return
    os.environ.setdefault("HF_HOME", "E:/Suzu-data/distill-data/hf")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    kw = {"device_map": {"": 0}, "quantization_config": transformers.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)}
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, **kw)
    except Exception:
        cls = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16, **kw)
    model.eval()
    rng = random.Random(a.seed * 100003 + len(done))
    t0 = time.time()
    n = 0
    with open(raw, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), a.batch):
            if (out.parent / "STOP").exists():
                print("STOP requested, exiting", flush=True)
                break
            part = todo[b:b + a.batch]
            prompts, metas = [], []
            for term, k in part:
                style, tone = rng.choice(STYLES), rng.choice(TONES)
                msgs = [{"role": "user", "content": TEMPLATE.format(style=style, tone=tone, term=term)}]
                try:
                    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                prompts.append(s)
                metas.append((term, k, style, tone))
            enc = tok(prompts, return_tensors="pt", padding=True).to(0)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=420, do_sample=True, temperature=a.temp, top_p=a.top_p, top_k=0,
                                   pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for (term, k, style, tone), t in zip(metas, texts):
                lines = [x.strip() for x in t.splitlines() if x.strip()]
                for line in lines:
                    f.write(json.dumps({"seed": term, "req": k, "style": style, "tone": tone, "t": line}, ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            print(f"{n}/{len(todo)} requests, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
