"""Task G step 3: labeled slang evaluation (pure model) for Suzu-2 / 2.1 / 2.2 and a dictionary-prior sweep.

pos accuracy  = slang answer ranked first on sentences that use the slang term;
ctrl accuracy = ordinary same-pinyin word ranked first on matched control sentences;
balanced      = mean of the two. 95% intervals are bootstrap intervals over sentences (2000 resamples; paired for differences).
Prior sweep   = score + lambda * ln(dictionary weight), lambda in {0, 0.25, 0.5, 1, 2}; the wiki evaluation set (16000 items,
candidates weighted from the rime-ice dictionary, missing = 100) must not drop.
Usage: python tools/g_eval.py --items <items_g.jsonl> --out <md> [--json-out <json>]
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
CKPTS = [("Suzu-2", "E:/Suzu-data/distill-data-r2/ckpt/student_best.pt"),
         ("Suzu-2.1", "E:/Suzu-data/distill-data-r3/run3/ckpt/student_best.pt"),
         ("Suzu-2.2", "E:/Suzu-data/distill-data-r4/runD/ckpt/student_last.pt")]
LAMS = [0, 0.25, 0.5, 1, 2]
import os
DICT_DIR = Path(os.environ.get("CN_DICTS", "cn_dicts"))   # 词库目录：rime-suzu 的词库包安装到用户目录后的 cn_dicts（含 base.dict.yaml 等）


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def dict_weights():
    """word -> max dictionary weight (base/ext/others as written, tencent default 100), like r4_build_slang."""
    w = {}

    def body(path):
        on = False
        for ln in open(path, encoding="utf-8"):
            if ln.startswith("..."):
                on = True
            elif on and ln.strip() and not ln.startswith("#"):
                yield ln.rstrip("\n").split("\t")

    for name in ("8105", "base", "ext", "others"):
        for f in body(DICT_DIR / f"{name}.dict.yaml"):
            if len(f) >= 2:
                wt = int(f[2]) if len(f) > 2 and f[2].isdigit() else 100
                w[f[0]] = max(w.get(f[0], 0), wt)
    for f in body(DICT_DIR / "tencent.dict.yaml"):
        w.setdefault(f[0], int(f[1]) if len(f) > 1 and f[1].isdigit() else 100)
    return w


def correct(items, scores, lam, weights=None):
    out = np.zeros(len(items), dtype=bool)
    for i, (it, s) in enumerate(zip(items, scores)):
        ws = it["w"] if weights is None else [weights.get(c, 100) for c in it["c"]]
        v = [x + lam * math.log(max(w, 1)) for x, w in zip(s, ws)]
        out[i] = v.index(max(v)) == it["t"]
    return out


def boot(pos_list, ctrl_list, n=2000, seed=0):
    """pos_list / ctrl_list: lists of correctness arrays (one per model, same items). Returns (point, lo, hi) of the
    balanced accuracy of the FIRST model minus nothing, or of a difference if two models are given as [a, b]."""
    rs = np.random.RandomState(seed)
    np_, nc = len(pos_list[0]), len(ctrl_list[0])
    f = (lambda p, c: (p.mean() + c.mean()) / 2) if len(pos_list) == 1 else (lambda p, c: 0)
    vals = []
    for _ in range(n):
        ip, ic = rs.randint(0, np_, np_), rs.randint(0, nc, nc)
        if len(pos_list) == 1:
            vals.append((pos_list[0][ip].mean() + ctrl_list[0][ic].mean()) / 2)
        else:
            vals.append(((pos_list[1][ip].mean() + ctrl_list[1][ic].mean()) - (pos_list[0][ip].mean() + ctrl_list[0][ic].mean())) / 2)
    pt = (pos_list[0].mean() + ctrl_list[0].mean()) / 2 if len(pos_list) == 1 else ((pos_list[1].mean() + ctrl_list[1].mean()) - (pos_list[0].mean() + ctrl_list[0].mean())) / 2
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return pt * 100, lo * 100, hi * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out")
    ap.add_argument("--lams", default="0,0.25,0.5,1,2", help="comma-separated lambda values")
    ap.add_argument("--wiki", default="D:/distill-data-r2/items_eval.jsonl")
    a = ap.parse_args()
    import torch
    from student_model import load_student, score_items
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    LAMS[:] = [float(x) for x in a.lams.split(",")]
    items = rj(a.items)
    wiki = rj(a.wiki)
    dw = dict_weights()
    S, SW = {}, {}
    for name, path in CKPTS:
        model, vocab = load_student(path, dev)
        S[name] = score_items(model, vocab, items, dev)
        SW[name] = score_items(model, vocab, wiki, dev)
        del model
        torch.cuda.empty_cache()
    groups = {h: {k: [i for i, it in enumerate(items) if it["kind"] == k and (h == "all" or it["half"] == h)] for k in ("pos", "ctrl")}
              for h in ("eval", "seed", "all")}
    HN = {"eval": "evaluation half (unseen words)", "seed": "seed half (trained words)", "all": "all words"}
    L = ["# Task G: labeled slang evaluation (pure model)", "",
         f"Sentences: {len(items)} (pos {sum(i['kind'] == 'pos' for i in items)}, ctrl {sum(i['kind'] == 'ctrl' for i in items)}). "
         "Intervals: bootstrap 95% over sentences.", ""]
    res = {}

    def acc_for(name, lam, h):
        c = correct(items, S[name], lam)
        return c[groups[h]["pos"]], c[groups[h]["ctrl"]]

    L += ["## Lambda = 0 (pure model)", "", "| half | model | n pos / ctrl | slang acc | control acc | balanced (95% CI) |", "|---|---|---|---|---|---|"]
    for h in ("eval", "seed", "all"):
        for name, _ in CKPTS:
            p, c = acc_for(name, 0, h)
            b = boot([p], [c])
            res[f"{h}/{name}/0"] = dict(pos=p.mean(), ctrl=c.mean(), bal=b[0])
            L.append(f"| {h} | {name} | {len(p)} / {len(c)} | {p.mean() * 100:.1f} | {c.mean() * 100:.1f} | {b[0]:.1f} ({b[1]:.1f} to {b[2]:.1f}) |")
    L += ["", "## Paired balanced-accuracy differences (lambda = 0)", "", "| half | comparison | difference, points (95% CI) |", "|---|---|---|"]
    for h in ("eval", "seed"):
        for x, y in (("Suzu-2", "Suzu-2.1"), ("Suzu-2.1", "Suzu-2.2"), ("Suzu-2", "Suzu-2.2")):
            px, cx = acc_for(x, 0, h)
            py, cy = acc_for(y, 0, h)
            d = boot([px, py], [cx, cy])
            L.append(f"| {h} | {y} minus {x} | {d[0]:+.1f} ({d[1]:+.1f} to {d[2]:+.1f}) |")
    L += ["", "## Dictionary-prior sweep: score + lambda * ln(weight)", "",
          "| model | lambda | eval-half slang | eval-half control | eval-half balanced | seed-half balanced | wiki acc | wiki change vs 0 (95% CI) |", "|---|---|---|---|---|---|---|---|"]
    rec = {}
    for name, _ in CKPTS:
        w0 = correct(wiki, SW[name], 0, dw)
        for lam in LAMS:
            pe, ce = acc_for(name, lam, "eval")
            ps, cs = acc_for(name, lam, "seed")
            wl = correct(wiki, SW[name], lam, dw)
            d = wl.astype(float) - w0.astype(float)
            se = d.std(ddof=1) / math.sqrt(len(d))
            bal_e, bal_s = (pe.mean() + ce.mean()) / 2 * 100, (ps.mean() + cs.mean()) / 2 * 100
            res[f"sweep/{name}/{lam}"] = dict(bal_eval=bal_e, bal_seed=bal_s, wiki=wl.mean() * 100, wiki_delta=d.mean() * 100)
            L.append(f"| {name} | {lam} | {pe.mean() * 100:.1f} | {ce.mean() * 100:.1f} | {bal_e:.1f} | {bal_s:.1f} | {wl.mean() * 100:.1f} | {d.mean() * 100:+.2f} (±{1.96 * se * 100:.2f}) |")
            rec.setdefault(name, []).append((lam, bal_e, d.mean() * 100))
    L += ["", "## Recommended lambda range", "",
          "Rule: eval-half balanced accuracy within 1 point of the best lambda for that model, and wiki change not below -0.3 points.", ""]
    for name, rows in rec.items():
        best = max(r[1] for r in rows)
        ok = [r[0] for r in rows if r[1] >= best - 1.0 and r[2] >= -0.3]
        L.append(f"- {name}: best eval-half balanced {best:.1f}; lambda values that pass: {ok if ok else 'none'}")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    if a.json_out:
        json.dump(res, open(a.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
