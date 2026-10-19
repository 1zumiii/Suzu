"""Weak-spot plan stage C / E step 1: hard-example mining.

Scores KD items with a student checkpoint (pure model) and compares its first choice with the teacher's. An item is
"hard" when the student's first choice differs from the teacher's AND the teacher is confident (best minus second best
teacher score > --margin, default 1.0), so label noise is mostly excluded. Writes the hard id list and a per-domain table.

Usage: python tools/r4_mine.py --ckpt <pt> --items <jsonl> [<jsonl> ...] --scores <jsonl> [<jsonl> ...] --out <prefix>
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=1.0)
    a = ap.parse_args()
    import torch
    from student_model import load_student, score_items
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_student(a.ckpt, dev)
    ts = {}
    for p in a.scores:
        ts.update({r["id"]: r["s"] for r in rj(p)})
    items = []
    for p in a.items:
        items += [i for i in rj(p) if i["id"] in ts]
    sc = score_items(model, vocab, items, dev)
    tot, dis, conf, hard = Counter(), Counter(), Counter(), []
    for it, s in zip(items, sc):
        t = ts[it["id"]]
        d = it["domain"]
        tot[d] += 1
        top_t = max(range(len(t)), key=t.__getitem__)
        top_s = max(range(len(s)), key=s.__getitem__)
        if top_t != top_s:
            dis[d] += 1
            srt = sorted(t, reverse=True)
            if len(srt) > 1 and srt[0] - srt[1] > a.margin:
                conf[d] += 1
                hard.append(it["id"])
    Path(a.out + "_hard_ids.json").write_text(json.dumps(hard), encoding="utf-8")
    lines = [f"# Hard-example mining ({Path(a.ckpt).name})", "", f"Margin {a.margin}. Hard = student first choice differs from teacher first choice and teacher gap > margin.", "",
             "| domain | items | disagree | hard (confident) | hard share |", "|---|---|---|---|---|"]
    for d in sorted(tot):
        lines.append(f"| {d} | {tot[d]} | {dis[d]} | {conf[d]} | {conf[d] / tot[d] * 100:.1f}% |")
    lines.append(f"| all | {sum(tot.values())} | {sum(dis.values())} | {len(hard)} | {len(hard) / max(sum(tot.values()), 1) * 100:.1f}% |")
    Path(a.out + "_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
