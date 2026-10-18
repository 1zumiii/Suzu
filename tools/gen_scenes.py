"""第三轮放量：让教师为每个话题列出具体情景（小标题），作为合成语料的细粒度种子，避免放量后话题饱和。
用法：python tools/gen_scenes.py --out <scenes_xxx.jsonl> --domain anime [--rounds 2]"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", "D:/distill-data/hf")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gen_domain_corpus import TOPICS, DOMAIN_DESC
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    q = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=q)
    except Exception:
        cls = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=q)
    topics = TOPICS[a.domain]
    seen, rows = set(), []
    for r in range(a.rounds):
        prompts = []
        for t in topics:
            p = (f"场景是{DOMAIN_DESC[a.domain]}，话题：{t}。请列出 40 个互不相同的具体情景或小话题，每个 6 到 20 个字，"
                 "覆盖面尽量广，每行一个，不要编号，不要解释。")
            try:
                prompts.append(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                prompts.append(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
        for i in range(0, len(prompts), 16):
            enc = tok(prompts[i:i + 16], return_tensors="pt", padding=True).to(0)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=900, do_sample=True, temperature=1.0, top_p=0.95, top_k=0, pad_token_id=tok.pad_token_id)
            for t, txt in zip(topics[i:i + 16], tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                for line in txt.splitlines():
                    line = line.strip().lstrip("-*•0123456789.、) ").strip()
                    if 4 <= len(line) <= 30 and (t, line) not in seen:
                        seen.add((t, line))
                        rows.append(dict(topic=t, scene=line))
            print(f"round {r} {i + 16}/{len(prompts)} 共 {len(rows)} 条", flush=True)
    Path(a.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
