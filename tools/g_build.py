"""Task G step 2: filter the generated sentences and build labeled evaluation items.

Item = (context before the target word, candidates = term + same-pinyin competitors from cases_meta, answer index).
kind "pos": answer is the slang term; kind "ctrl": answer is the ordinary competitor word.
Filters: t2s, length 8-30, junk, target present, >= 2 chars of context, ctrl sentences must not contain the slang term,
no near-duplicates, no overlap (4-gram ratio >= 0.5) with the slang TRAINING corpus, at most --cap per (term, kind).
--drop <json list of item keys> removes sentences rejected in the manual spot check. Writes items_g.jsonl and spot_check_g.txt.
Usage: python tools/g_build.py --raw <raw_g.jsonl> --meta <cases_meta.json> --split <split.json> --train-sents <slang.txt> --out <dir> [--drop <json>]
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_domain_corpus import ALLOWED, BAD, clean, grams, han_len  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--train-sents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--drop")
    ap.add_argument("--table", help="word_table.pkl; with --min-ctrl-freq drops control sentences whose answer word is rare")
    ap.add_argument("--min-ctrl-freq", type=float, default=0)
    ap.add_argument("--paired-only", action="store_true", help="keep only terms that have both slang and control sentences")
    a = ap.parse_args()
    from opencc import OpenCC
    cc = OpenCC("t2s")
    meta = {m["term"]: m for m in json.load(open(a.meta, encoding="utf-8"))}
    split = json.load(open(a.split, encoding="utf-8"))
    drop = set(json.load(open(a.drop, encoding="utf-8"))) if a.drop and Path(a.drop).exists() else set()
    train_grams = set()
    for l in open(a.train_sents, encoding="utf-8"):
        train_grams |= grams(l.strip())
    rows = [json.loads(l) for l in open(a.raw, encoding="utf-8")]
    random.Random(9).shuffle(rows)
    stat = Counter()
    seen, seen_grams, per, items = set(), {}, Counter(), []
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
        tgt, term, kind = r["target"], r["term"], r["kind"]
        pos = t.find(tgt)
        if pos < 2:
            stat["no_target_or_short_ctx"] += 1
            continue
        if kind == "ctrl" and term in t:
            stat["ctrl_has_slang"] += 1
            continue
        key = f"{kind}|{term}|{t}"
        if key in drop:
            stat["manual_drop"] += 1
            continue
        if t in seen:
            stat["dup"] += 1
            continue
        g = grams(t)
        if g and len(g & train_grams) / len(g) >= 0.5:
            stat["train_overlap"] += 1
            continue
        if g and sum(1 for x in g if x in seen_grams) / len(g) > 0.5:
            stat["near_dup"] += 1
            continue
        if per[(term, kind)] >= a.cap:
            stat["cap"] += 1
            continue
        per[(term, kind)] += 1
        seen.add(t)
        for x in g:
            seen_grams[x] = 1
        m = meta[term]
        cands = [term] + [w for w in m["dict_order"] if w != term]
        items.append(dict(id=len(items), key=key, kind=kind, term=term, target=tgt, half=split.get(term, "?"), ctx=t[:pos][-30:], sent=t,
                          c=cands, t=cands.index(tgt), w=[m["weights"].get(w, 100) for w in cands]))
    if a.table and a.min_ctrl_freq > 0:
        import pickle
        tab = pickle.loads(Path(a.table).read_bytes())
        before = sum(i["kind"] == "ctrl" for i in items)
        items = [i for i in items if i["kind"] == "pos" or tab.get(i["target"], 0) >= a.min_ctrl_freq]
        stat["ctrl_rare_word_dropped"] = before - sum(i["kind"] == "ctrl" for i in items)
    if a.paired_only:
        have = {k: {i["term"] for i in items if i["kind"] == k} for k in ("pos", "ctrl")}
        both = have["pos"] & have["ctrl"]
        stat["unpaired_dropped"] = sum(i["term"] not in both for i in items)
        items = [i for i in items if i["term"] in both]
    for n, it in enumerate(items):
        it["id"] = n
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items_g.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    rng = random.Random(5)
    with open(out / "spot_check_g.txt", "w", encoding="utf-8") as f:
        for kind in ("pos", "ctrl"):
            sel = [i for i in items if i["kind"] == kind]
            f.write(f"## {kind} ({len(sel)})\n")
            for it in rng.sample(sel, min(50, len(sel))):
                f.write(f"{it['id']}\t[{it['term']}→{it['target']}]\t{it['sent']}\n")
    cnt = Counter((i["kind"], i["half"]) for i in items)
    stat.update({f"items_{k}_{h}": v for (k, h), v in cnt.items()})
    stat["items"] = len(items)
    stat["terms_pos"] = len({i["term"] for i in items if i["kind"] == "pos"})
    print(json.dumps(dict(stat), ensure_ascii=False))


if __name__ == "__main__":
    main()
