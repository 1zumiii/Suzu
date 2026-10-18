"""Round 3: per-character negative log-likelihood of each filtered sentence under Qwen3.5-4B-Base (nf4).
Fabricated names and garbled terms get high NLL; the filter drops the worst fraction.
Usage: python tools/r3_ppl.py --sentences <domain.txt> --out <ppl_domain.tsv>   (resumable; one 'nll<TAB>sentence' per line)"""
import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", "D:/distill-data/hf")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sents = [l.strip() for l in open(a.sentences, encoding="utf-8") if l.strip()]
    out = Path(a.out)
    done = 0
    if out.exists():
        done = sum(1 for _ in open(out, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    q = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=q)
    except Exception:
        cls = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=q)
    model.eval()
    start = tok.convert_tokens_to_ids("<|endoftext|>")
    with open(out, "a", encoding="utf-8") as f:
        for i in range(done, len(sents), a.batch):
            if (out.parent / "STOP").exists() or (out.parent.parent / "STOP").exists():
                break
            part = sents[i:i + a.batch]
            ids = [[start] + tok(s, add_special_tokens=False)["input_ids"] for s in part]
            L = max(len(x) for x in ids)
            inp = torch.zeros(len(ids), L, dtype=torch.long)
            att = torch.zeros(len(ids), L, dtype=torch.long)
            for r, x in enumerate(ids):
                inp[r, :len(x)] = torch.tensor(x)
                att[r, :len(x)] = 1
            inp, att = inp.to(0), att.to(0)
            with torch.no_grad():
                lg = model(input_ids=inp, attention_mask=att).logits[:, :-1].float().log_softmax(-1)
            nll = -lg.gather(-1, inp[:, 1:, None])[..., 0] * att[:, 1:]
            for r, s in enumerate(part):
                f.write(f"{nll[r].sum().item() / max(len(s), 1):.4f}\t{s}\n")
            f.flush()
            if (i // a.batch) % 50 == 0:
                print(f"{i + len(part)}/{len(sents)}", flush=True)


if __name__ == "__main__":
    main()
