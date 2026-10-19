"""Natural-weighted evaluation on the 16000-item wiki evaluation set (the round-2 report's method) for several checkpoints:
mixed accuracy (lambda 0.25 on the corpus-frequency prior, natural weighting of easy and hard items) and pure-model hard-item accuracy.
Natural weights come from meta.json eval_natural (easy : hard in the full evaluation split).
Usage: python tools/t_nat.py --ckpt NAME=pt [...] --out <md>"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default="D:/distill-data-r2/items_eval.jsonl")
    ap.add_argument("--meta", default="D:/distill-data-r2/meta.json")
    a = ap.parse_args()
    import torch
    from student_model import load_student, score_items
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
    nat = json.load(open(a.meta))["eval_natural"]
    w = [nat[0] / sum(nat), nat[1] / sum(nat)]
    L = ["# Natural-weighted evaluation (16000 wiki items)", "", f"Natural weights easy {w[0]:.3f}, hard {w[1]:.3f}. Mixed: score + 0.25 * ln(1 + corpus count). Intervals not computed here.", "",
         "| model | mixed natural-weighted | mixed easy | mixed hard | pure hard |", "|---|---|---|---|---|"]
    for spec in a.ckpt:
        name, p = spec.split("=", 1)
        m, v = load_student(p, dev)
        sc = score_items(m, v, items, dev)
        acc = {}
        for lam, tag in ((0.25, "mix"), (0.0, "pure")):
            c, n = [0, 0], [0, 0]
            for it, s in zip(items, sc):
                k = max(range(len(s)), key=lambda j: s[j] + lam * math.log1p(it["f"][j]))
                n[it["s"]] += 1
                c[it["s"]] += int(k == it["t"])
            acc[tag] = [c[0] / n[0], c[1] / n[1]]
        nw = w[0] * acc["mix"][0] + w[1] * acc["mix"][1]
        L.append(f"| {name} | {nw * 100:.1f} | {acc['mix'][0] * 100:.1f} | {acc['mix'][1] * 100:.1f} | {acc['pure'][1] * 100:.1f} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
