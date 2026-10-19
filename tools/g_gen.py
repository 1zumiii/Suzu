"""Task G step 1: generate labeled slang evaluation sentences and matched control sentences with Qwen3.5-9B (nf4).

For every slang term in cases_meta.json (428 terms, both word halves) it asks for
  * "pos" sentences: the slang term is used naturally (answer = slang term), and
  * "ctrl" sentences: the same pinyin, but the sentence uses the most common competitor word in its ordinary meaning
    (answer = that ordinary word).
These are EVALUATION data only and never enter training. Resumable per (term, kind, request).
Usage: python tools/g_gen.py --out <dir> --meta <cases_meta.json> [--reqs 2]
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

STYLES = ["微信或QQ群里的聊天", "视频弹幕", "短视频评论区留言", "微博或小红书式的发帖口吻", "朋友之间当面吐槽", "论坛回帖", "直播间的评论"]
TONES = ["随意口语", "开玩笑", "吐槽", "感叹", "简短省略主语", "无奈", "兴奋", "平静叙述"]
COMMON = ("每句 8 到 30 个字，口吻像真人随手打出来的；每句内容各不相同，句式多样；“{w}”前面至少要有 3 个字的铺垫，不要放在句首；"
          "不要编号，不要引号，不要解释，每行一句；不要出现真实的人名、电话、地址等个人信息；不带脏话。")
POS = "请写 10 句简体中文，场景是{style}，语气：{tone}。每句都要自然地用到网络用语“{w}”，用的是这个词广为人知的含义，句子要让人看出这个词在这里是网络用语的用法。\n" + COMMON
CTRL = "请写 10 句简体中文，场景是{style}，语气：{tone}。每句都要用到词语“{w}”，按它平常、字面的意思使用，不要把它当作网络用语或梗来用，也不要写成玩笑或谐音。\n" + COMMON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--reqs", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(a.meta, encoding="utf-8"))
    jobs = []
    for m in meta:
        t = m["term"]
        ctrl = next((w for w in m["dict_order"] if w != t), None)
        if not ctrl:
            continue
        for k in range(a.reqs):
            jobs.append((t, "pos", t, k))
            jobs.append((t, "ctrl", ctrl, k))
    raw = out / "raw_g.jsonl"
    done = set()
    if raw.exists():
        for l in open(raw, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["term"], r["kind"], r["req"]))
    todo = [j for j in jobs if (j[0], j[1], j[3]) not in done]
    random.Random(1).shuffle(todo)
    print(f"requests {len(jobs)}, to run {len(todo)}", flush=True)
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
    rng = random.Random(3)
    t0 = time.time()
    n = 0
    with open(raw, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), a.batch):
            if (out.parent / "STOP").exists():
                print("STOP requested, exiting", flush=True)
                break
            part = todo[b:b + a.batch]
            prompts = []
            for term, kind, w, k in part:
                msgs = [{"role": "user", "content": (POS if kind == "pos" else CTRL).format(style=rng.choice(STYLES), tone=rng.choice(TONES), w=w)}]
                try:
                    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
                except TypeError:
                    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = tok(prompts, return_tensors="pt", padding=True).to(0)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=420, do_sample=True, temperature=1.0, top_p=0.95, top_k=0, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for (term, kind, w, k), t in zip(part, texts):
                for line in (x.strip() for x in t.splitlines()):
                    if line:
                        f.write(json.dumps({"term": term, "kind": kind, "target": w, "req": k, "t": line}, ensure_ascii=False) + "\n")
            f.flush()
            n += len(part)
            print(f"{n}/{len(todo)} requests, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
