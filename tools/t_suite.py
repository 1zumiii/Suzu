"""Standard evaluation suite for the scale-up tasks: wiki evaluation, three old domain sets, three new held-out domain sets.
Reports, per checkpoint, mixed accuracy (lambda 0.25 on the corpus-frequency prior) and pure-model accuracy, then the
largest difference between checkpoints per set (for seed-noise estimates) and the difference to the first checkpoint.
Usage: python tools/t_suite.py --ckpt NAME=pt [--ckpt NAME=pt ...] --out <md> [--json-out <json>]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
HOLD = "D:/distill-data-r3/run3/items_holdout.jsonl"


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sets():
    out = [("wiki eval", rj("D:/distill-data-r2/items_eval.jsonl"))]
    dom = [i for i in rj("D:/distill-data-domain/items_domain.jsonl") if i["L"] >= 2]
    hold = rj(HOLD)
    for d, n in (("tech", "tech"), ("anime", "anime"), ("chat", "chat")):
        out.append((f"{n} (old)", [i for i in dom if i["domain"] == d]))
    for d, n in (("tech", "tech"), ("anime", "anime"), ("chat", "chat")):
        out.append((f"{n} (new hold-out)", [i for i in hold if i["domain"] == d]))
    return out


def correct(items, sc, lam):
    return np.array([int(max(range(len(s)), key=lambda k: s[k] + lam * math.log1p(it["f"][k])) == it["t"]) for it, s in zip(items, sc)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out")
    a = ap.parse_args()
    import torch
    from student_model import load_student, score_items
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = sets()
    names, res = [], {}
    for spec in a.ckpt:
        name, p = spec.split("=", 1)
        if not Path(p).exists():
            continue
        m, v = load_student(p, dev)
        names.append(name)
        res[name] = {}
        for sn, its in S:
            sc = score_items(m, v, its, dev)
            res[name][sn] = (correct(its, sc, 0.25), correct(its, sc, 0.0))
        del m
        torch.cuda.empty_cache()
    L = ["# Evaluation suite", "", "Cells: mixed accuracy / pure-model accuracy (percent). Sets: " + ", ".join(f"{n} ({len(i)})" for n, i in S), "",
         "| model | " + " | ".join(n for n, _ in S) + " |", "|---|" + "---|" * len(S)]
    for n in names:
        L.append(f"| {n} | " + " | ".join(f"{res[n][sn][0].mean() * 100:.1f} / {res[n][sn][1].mean() * 100:.1f}" for sn, _ in S) + " |")
    if len(names) > 1:
        L.append("| **max difference between models** | " + " | ".join(
            f"{(max(res[n][sn][0].mean() for n in names) - min(res[n][sn][0].mean() for n in names)) * 100:.1f} / "
            f"{(max(res[n][sn][1].mean() for n in names) - min(res[n][sn][1].mean() for n in names)) * 100:.1f}" for sn, _ in S) + " |")
        L += ["", f"Paired difference to the first model ({names[0]}), mixed accuracy, points (95% interval):", "",
              "| model | " + " | ".join(n for n, _ in S) + " |", "|---|" + "---|" * len(S)]
        for n in names[1:]:
            cells = []
            for sn, _ in S:
                d = res[n][sn][0].astype(float) - res[names[0]][sn][0].astype(float)
                se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0
                cells.append(f"{d.mean() * 100:+.1f} (±{1.96 * se * 100:.1f})")
            L.append(f"| {n} | " + " | ".join(cells) + " |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    if a.json_out:
        json.dump({n: {sn: [float(res[n][sn][0].mean()), float(res[n][sn][1].mean())] for sn, _ in S} for n in names}, open(a.json_out, "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
