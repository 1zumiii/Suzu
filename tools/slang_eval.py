"""Stage A of the weak-spot plan: slang suppression baseline.

For each slang term with same-length homophone competitors (cases_meta.json),
score [term] + competitors under several contexts with the pure student model
(sum of per-character log-probs, no frequency fusion) and report:

  suppressed rate : among terms that lead the dictionary order, the share whose
                    model-first count is not above half of the contexts
  lifted count    : among terms NOT leading the dictionary, how many are model-first
                    in more than half of the contexts
  gap median      : median over (term, context) of term score - best competitor score

Also writes a deterministic category-stratified word split (seed half / eval half)
so that eval-half words never reach any generation prompt or training corpus.

Usage:
  python tools/slang_eval.py --meta <cases_meta.json> --draft <slang_draft.tsv> \
      --ckpt NAME=<pt> [--ckpt NAME=<pt> ...] --split-out <split.json> --out <md>
"""
import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

# The 8 generic colloquial contexts used for the Mac baseline.
CTX_BASE = ["哈哈哈", "我觉得", "这个视频", "今天真的", "他说", "笑死我了", "最近在追", "太好看了"]
# Extra chat / bullet-comment / comment-section style contexts (hand-written, no personal data).
CTX_EXTRA = ["评论区都在说", "家人们谁懂啊", "刷到一个视频", "朋友圈里全是", "室友刚才吐槽",
             "群里有人说", "老板居然说", "看完这集我直接", "弹幕全在刷", "网友评论说"]


def make_split(draft_path):
    """Category-stratified, deterministic: sort words in each category by md5, alternate seed/eval."""
    by_cat = defaultdict(list)
    for ln in open(draft_path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln or "\t" not in ln:
            continue
        c, w = ln.split("\t")[:2]
        by_cat[c].append(w)
    split = {}
    for c, ws in by_cat.items():
        ws = sorted(set(ws), key=lambda w: hashlib.md5(("slang-split-v1:" + w).encode("utf-8")).hexdigest())
        for i, w in enumerate(ws):
            split[w] = "seed" if i % 2 == 0 else "eval"
    return split


def run_model(ckpt, items, dev):
    from student_model import load_student, score_items
    model, vocab = load_student(ckpt, dev)
    return score_items(model, vocab, items, dev)


def metrics(meta, scores, nctx, subset=None):
    """scores[(ti, ci)] = list of candidate scores, candidate 0 is the term."""
    sup_n = sup_hit = lift_n = lift_hit = 0
    gaps = []
    for ti, m in enumerate(meta):
        if subset is not None and m["term"] not in subset:
            continue
        first = 0
        for ci in range(nctx):
            s = scores[(ti, ci)]
            if s[0] >= max(s[1:]):
                first += 1
            gaps.append(s[0] - max(s[1:]))
        if m["dict_order"][0] == m["term"]:
            sup_n += 1
            sup_hit += int(first < nctx / 2)
        else:
            lift_n += 1
            lift_hit += int(first >= nctx / 2)
    med = statistics.median(gaps) if gaps else float("nan")
    return dict(lead_n=sup_n, suppressed=sup_hit, nonlead_n=lift_n, lifted=lift_hit, gap_median=med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=path")
    ap.add_argument("--split-out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(a.meta, encoding="utf-8"))
    split = make_split(a.draft)
    Path(a.split_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(split, open(a.split_out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    n_seed = sum(v == "seed" for v in split.values())
    n_eval = len(split) - n_seed
    eval_terms = {m["term"] for m in meta if split.get(m["term"]) == "eval"}
    seed_terms = {m["term"] for m in meta if split.get(m["term"]) == "seed"}
    missing = [m["term"] for m in meta if m["term"] not in split]
    if missing:
        print("WARNING: terms not in draft:", len(missing))

    ctxs_all = CTX_BASE + CTX_EXTRA
    items, keys = [], []
    for ti, m in enumerate(meta):
        cands = [m["term"]] + [x for x in m["dict_order"] if x != m["term"]]
        for ci, ctx in enumerate(ctxs_all):
            items.append({"ctx": ctx, "c": cands})
            keys.append((ti, ci))

    results = {}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        sc = run_model(path, items, dev)
        scores = dict(zip(keys, sc))
        r = {}
        for label, nctx in (("8 contexts", len(CTX_BASE)), ("18 contexts", len(ctxs_all))):
            for sub, S in (("all", None), ("eval half", eval_terms), ("seed half", seed_terms)):
                r[f"{label} / {sub}"] = metrics(meta, scores, nctx, S)
        results[name] = r

    L = ["# Slang suppression baseline (stage A)", "",
         f"Terms with competitors: {len(meta)} (eval half {len(eval_terms)}, seed half {len(seed_terms)}). "
         f"Word split over the whole draft list: seed {n_seed}, eval {n_eval}; category-stratified, md5 alternate.",
         "Pure model, no frequency fusion. Suppressed = lead-in-dictionary terms the model ranks first in fewer than half of the contexts (not first in a strict majority). "
         "Lifted = non-lead terms the model ranks first in at least half of the contexts. Gap = term score minus best competitor, median over term x context.", ""]
    names = list(results)
    for key in results[names[0]]:
        L += [f"## {key}", "", "| model | lead terms | suppressed | rate | non-lead | lifted | gap median |", "|---|---|---|---|---|---|---|"]
        for n in names:
            r = results[n][key]
            rate = r["suppressed"] / r["lead_n"] * 100 if r["lead_n"] else float("nan")
            L.append(f"| {n} | {r['lead_n']} | {r['suppressed']} | {rate:.1f}% | {r['nonlead_n']} | {r['lifted']} | {r['gap_median']:+.2f} |")
        L.append("")
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    if a.json_out:
        json.dump(results, open(a.json_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
