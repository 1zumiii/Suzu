"""Weak-spot plan stage B2: filter the synthesized slang sentences.

Same cleaning as filter_domain_corpus.py (t2s, charset, length 8-30, junk, exact/near duplicates, leak check against the
holdout and domain test sentences) plus slang-specific rules:
  * the sentence must contain its seed term (and the term must not be glued into a longer eval word);
  * no sentence may contain ANY eval-half word (substring test) - the eval half never reaches the corpus;
  * at most --per-seed sentences are kept per seed term;
  * optional perplexity cut (r3_ppl.py output) drops the worst fraction.

Outputs <out>.txt (one sentence per line) and <out>.seed.tsv (sentence TAB seed).
Usage: python tools/filter_slang_corpus.py --raw <raw_slang.jsonl> --split <split.json> --out <prefix> [--ppl <tsv> --ppl-drop 0.045]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_domain_corpus import ALLOWED, BAD, clean, grams, han_len  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--per-seed", type=int, default=40)
    ap.add_argument("--max-dup-ratio", type=float, default=0.5)
    ap.add_argument("--leak-ratio", type=float, default=0.5)
    ap.add_argument("--ppl")
    ap.add_argument("--ppl-drop", type=float, default=0.0)
    a = ap.parse_args()
    from opencc import OpenCC
    cc = OpenCC("t2s")
    split = json.load(open(a.split, encoding="utf-8"))
    eval_words = [w for w, v in split.items() if v == "eval" and len(w) >= 2]
    ppl, cut = {}, None
    if a.ppl and a.ppl_drop > 0:
        for l in open(a.ppl, encoding="utf-8"):
            v, _, t = l.rstrip("\n").partition("\t")
            ppl[t] = float(v)
        vals = sorted(ppl.values())
        cut = vals[int(len(vals) * (1 - a.ppl_drop))] if vals else None
    test_grams = set()
    files = list((ROOT / "tests" / "domain").glob("*.txt")) + list((ROOT / "tests" / "holdout").glob("*.txt")) + [Path(p) for p in a.exclude]
    for p in files:
        for l in open(p, encoding="utf-8"):
            if l.strip() and not l.startswith("#"):
                test_grams |= grams(l.replace("【", "").replace("】", ""))
    rows = [json.loads(l) for l in open(a.raw, encoding="utf-8")]
    random.Random(7).shuffle(rows)       # so the per-seed cap does not favour early requests
    stat = dict(total=0, shape=0, junk=0, no_seed=0, eval_word=0, ppl=0, dup=0, near=0, leak=0, cap=0)
    seen, seen_grams, per, kept = set(), {}, {}, []
    for r in rows:
        stat["total"] += 1
        t = clean(r["t"], cc)
        n = han_len(t)
        if not 8 <= n <= 30 or n < 0.7 * len(t):
            stat["shape"] += 1
            continue
        if not ALLOWED.match(t) or any(b in t for b in BAD) or re.search(r"(.)\1{3,}", t) or len(re.findall(r"[A-Za-z]", t)) > 12:
            stat["junk"] += 1
            continue
        if r["seed"] not in t:
            stat["no_seed"] += 1
            continue
        if any(e in t for e in eval_words):
            stat["eval_word"] += 1
            continue
        if cut is not None and ppl.get(t, 0) > cut:
            stat["ppl"] += 1
            continue
        if t in seen:
            stat["dup"] += 1
            continue
        g = grams(t)
        if g and len(g & test_grams) / len(g) >= a.leak_ratio:
            stat["leak"] += 1
            continue
        if g and sum(1 for x in g if x in seen_grams) / len(g) > a.max_dup_ratio:
            stat["near"] += 1
            continue
        if per.get(r["seed"], 0) >= a.per_seed:
            stat["cap"] += 1
            continue
        per[r["seed"]] = per.get(r["seed"], 0) + 1
        seen.add(t)
        for x in g:
            seen_grams[x] = 1
        kept.append((t, r["seed"]))
    Path(a.out + ".txt").write_text("\n".join(t for t, _ in kept) + "\n", encoding="utf-8")
    Path(a.out + ".seed.tsv").write_text("\n".join(f"{t}\t{s}" for t, s in kept) + "\n", encoding="utf-8")
    stat["kept"] = len(kept)
    stat["seeds_with_data"] = len(per)
    print(json.dumps(stat, ensure_ascii=False))


if __name__ == "__main__":
    main()
