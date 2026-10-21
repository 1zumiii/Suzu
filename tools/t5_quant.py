"""T5: quantization comparison for one student checkpoint.

For every GGUF file in --dir (made by llama-quantize from the F32 export): file size, desktop latency (student_check.exe, 7 threads,
30-token prefix, 8 candidates), and pure-model accuracy on evaluation sets. Accuracy is measured in PyTorch with the weights
DEQUANTIZED from the quantized file, so it shows the effect of the stored weights (llama.cpp's own kernels add small extra differences).
Usage: python tools/t5_quant.py --ckpt <pt> --dir <dir of *.gguf> --out <md>
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import os
sys.path.insert(0, os.environ.get("GGUF_PY", "llama.cpp/gguf-py"))   # llama.cpp 的 gguf-py 目录
STUDENT_CHECK = os.environ.get("STUDENT_CHECK", "student_check.exe")   # 核对用的可执行文件（本机构建的路径）


def tname(n):
    m = re.match(r"blk\.(\d+)\.(.+)", n)
    top = {"token_embd.weight": "emb.weight", "position_embd.weight": "pos.weight", "output_norm.weight": "ln.weight", "output_norm.bias": "ln.bias"}
    if n in top:
        return top[n]
    i, rest = m.groups()
    mp = {"attn_norm.weight": "norm1.weight", "attn_norm.bias": "norm1.bias", "attn_qkv.weight": "self_attn.in_proj_weight",
          "attn_qkv.bias": "self_attn.in_proj_bias", "attn_output.weight": "self_attn.out_proj.weight", "attn_output.bias": "self_attn.out_proj.bias",
          "ffn_norm.weight": "norm2.weight", "ffn_norm.bias": "norm2.bias", "ffn_up.weight": "linear1.weight", "ffn_up.bias": "linear1.bias",
          "ffn_down.weight": "linear2.weight", "ffn_down.bias": "linear2.bias"}
    return f"blocks.layers.{i}.{mp[rest]}"


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="*", default=[])
    a = ap.parse_args()
    import gguf
    import torch
    from student_model import Student, load_student, score_items
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base, vocab = load_student(a.ckpt, dev)
    ck = torch.load(a.ckpt, map_location="cpu")
    c = ck["cfg"]
    sets = []
    for spec in a.sets:
        name, p = spec.split("=", 1)
        sets.append((name, rj(p)))
    wiki = rj("D:/distill-data-r2/items_eval.jsonl")[:4000]
    sets.append(("wiki eval (4000)", wiki))

    def acc(model):
        out = []
        for _, its in sets:
            sc = score_items(model, vocab, its, dev)
            out.append(float(np.mean([s.index(max(s)) == it["t"] for it, s in zip(its, sc)])) * 100)
        return out

    ids = ",".join(str(i) for i in range(3, 33))
    rows = []
    files = sorted(Path(a.dir).glob("*.gguf"), key=lambda p: -p.stat().st_size)
    ref = acc(base)
    for p in files:
        rd = gguf.GGUFReader(str(p))
        sd = {}
        for t in rd.tensors:
            arr = gguf.quants.dequantize(t.data, t.tensor_type)
            sd[tname(t.name)] = torch.from_numpy(np.asarray(arr, dtype=np.float32).copy())
        m = Student(len(vocab), c["student_d"], c["student_layers"], c["student_heads"], c["student_ctx"]).to(dev)
        own = m.state_dict()
        for k in own:
            if k in sd and sd[k].shape != own[k].shape:
                sd[k] = sd[k].reshape(own[k].shape)
        m.load_state_dict({k: sd[k].to(dev) for k in own if k in sd}, strict=False)
        m.eval()
        accs = acc(m)
        lat = subprocess.run([STUDENT_CHECK, str(p), ids, "7"], capture_output=True, timeout=300)
        out = lat.stdout.decode("utf-8", "replace")
        mb = re.search(r"TIMING threads=7 batched\s+([\d.]+) ms", out)
        mp = re.search(r"前缀 ([\d.]+) ms.*?候选 ([\d.]+) ms", out)
        rows.append((p.stem, p.stat().st_size / 2**20, accs, mb.group(1) if mb else "n/a", f"{mp.group(1)} / {mp.group(2)}" if mp else "n/a"))
    L = ["# T5 quantization comparison (Suzu-2.1 structure)", "",
         "Accuracy: pure model in PyTorch with weights dequantized from each file; difference is against the unquantized PyTorch checkpoint. "
         "Latency: student_check.exe, 7 threads, 30-token prefix, 8 candidates, batched path (ms per item), prefix / candidate-batch split in the last column.", "",
         "| file | size MiB | " + " | ".join(n for n, _ in sets) + " | batched ms | prefix / candidates ms |", "|---|---|" + "---|" * len(sets) + "---|---|"]
    L.append("| PyTorch checkpoint (fp32) | - | " + " | ".join(f"{x:.1f}" for x in ref) + " | - | - |")
    for name, mib, accs, b, pc in rows:
        L.append(f"| {name} | {mib:.1f} | " + " | ".join(f"{x:.1f} ({x - r:+.1f})" for x, r in zip(accs, ref)) + f" | {b} | {pc} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
