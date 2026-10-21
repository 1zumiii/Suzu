"""Dictionary evaluation: rime-ice (A) against rime-frost as a whole (B) and the ice word list with frost word frequencies (C).
Only the dictionary changes; the question sets and the models (Suzu-2.1, Suzu-2.4) stay the same.  Prints and writes numbers only,
no word lists.

Question sets (the answer must be a word of dictionary A, as in the abbreviation teacher-ceiling evaluation):
  abbr   1300 two-letter abbreviation questions (cache_student.json of the teacher-ceiling run: ctx, answer, initials, domain)
  mixed  the same words, input = first syllable in full + initial of the second syllable
  full   the same words, input = both syllables in full
  wiki   16000 full-pinyin items of the wiki evaluation; six domain sets (old and new hold-out) as in t_suite.py
Per question the pool is every dictionary word of the variant that matches the input.  Ranking modes: dictionary weight only,
model score only, model score + beta * ln(1 + weight) for beta 0.1 / 0.2 / 0.3.  Model scores of a context/word pair are computed
once and shared by the variants (words found in the earlier abbreviation caches are reused).

  python tools/frost_eval.py build  --work DIR --frost DIR/cn_dicts --ice DIR/cn_dicts --abbrev-cache cache_student.json
  python tools/frost_eval.py score  --work DIR --model NAME=gguf [--reuse abbrev-cache.json] --lua lua54 --so-dir DIR --vocab FILE --pipe-dir DIR
  python tools/frost_eval.py report --work DIR --out report.md --model NAME ...
Every step resumes where it stopped.
"""
import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

INVALID = -1e8
BETAS = (0.1, 0.2, 0.3)
BATCH = 40
VARIANTS = ("A ice", "B frost", "C ice words + frost freq")


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_dict(d, names=("base", "ext")):
    """(word, reading) -> weight; the larger weight wins when an entry appears twice (same rule as the abbreviation prototype)."""
    ent = {}
    for name in names:
        body = False
        for ln in open(f"{d}/{name}.dict.yaml", encoding="utf-8"):
            if ln.startswith("..."):
                body = True
                continue
            if not body or ln.startswith("#"):
                continue
            p = ln.rstrip("\r\n").split("\t")
            if len(p) < 2 or not p[1].strip():
                continue
            w = int(p[2]) if len(p) > 2 and p[2].strip().isdigit() else 0
            k = (p[0], p[1].strip())
            if k not in ent or w > ent[k]:
                ent[k] = w
    return ent


def make_c(ice, frost):
    """Ice word list, frost frequencies.  Entries both dictionaries have take the frost weight.  An ice-only entry (frost dropped it)
    gets its ice weight times the median frost/ice ratio of the shared entries, capped at the lowest frost weight under the same reading
    (a dropped word stays below every kept word of that reading)."""
    ratios = sorted(frost[k] / ice[k] for k in ice if k in frost and ice[k] > 0 and frost[k] > 0)
    scale = ratios[len(ratios) // 2]
    low = {}
    for (w, r), v in frost.items():
        low[r] = min(low.get(r, v), v)
    out = {}
    for k, v in ice.items():
        if k in frost:
            out[k] = frost[k]
        else:
            x = v * scale
            out[k] = max(0, min(x, low[k[1]])) if k[1] in low else x
    return out, scale


class Dict:
    """Pools by input form.  Pool order: weight descending, ties in file order (Rime's order for equal weights)."""

    def __init__(self, ent):
        self.full = defaultdict(dict)       # reading -> {word: w}
        self.abbr = defaultdict(dict)       # initials of 2-syllable words -> {word: w}
        self.mixed = defaultdict(dict)      # (syl1, initial of syl2) -> {word: w}
        for (w, r), v in ent.items():
            self.full[r][w] = max(v, self.full[r].get(w, -1))
            sy = r.split()
            if len(w) == 2 and len(sy) == 2:
                k = sy[0][0] + sy[1][0]
                self.abbr[k][w] = max(v, self.abbr[k].get(w, -1))
                k2 = (sy[0], sy[1][0])
                self.mixed[k2][w] = max(v, self.mixed[k2].get(w, -1))
        self.ent = ent

    @staticmethod
    def order(pool):
        return sorted(pool.items(), key=lambda kv: -kv[1])


def questions(args, ice_dict):
    """Question list: dict(set, id, ctx, ans, form, key, dom).  form = how the input looks; key = pool key."""
    qs = []
    cache = json.load(open(args.abbrev_cache, encoding="utf-8"))
    by_word = defaultdict(list)
    for (w, r), v in ice_dict.ent.items():
        by_word[w].append((v, r))
    for t in cache:
        cands = [(v, r) for v, r in by_word[t["ans"]] if len(r.split()) == 2 and r.split()[0][0] + r.split()[1][0] == t["key"]]
        v, r = max(cands)
        s1, s2 = r.split()
        base = dict(ctx=t["ctx"], ans=t["ans"], dom=t["dom"], ci=t["i"])
        qs.append(dict(base, set="abbr", key=t["key"], form="abbr"))
        qs.append(dict(base, set="mixed", key=[s1, s2[0]], form="mixed"))
        qs.append(dict(base, set="full", key=r, form="full"))
    for it in rj("D:/distill-data-r2/items_eval.jsonl"):
        qs.append(dict(set="wiki", ctx=it["ctx"], ans=it["c"][it["t"]], key=it["py"], form="full", dom="-", ci=None))
    dom = [i for i in rj("D:/distill-data-domain/items_domain.jsonl") if i["L"] >= 2]
    hold = rj("D:/distill-data-r3/run3/items_holdout.jsonl")
    for d in ("tech", "anime", "chat"):
        for it in dom:
            if it["domain"] == d:
                qs.append(dict(set=f"{d} (old)", ctx=it["ctx"], ans=it["c"][it["t"]], key=it["py"], form="full", dom=d, ci=None))
        for it in hold:
            if it["domain"] == d:
                qs.append(dict(set=f"{d} (new)", ctx=it["ctx"], ans=it["c"][it["t"]], key=it["py"], form="full", dom=d, ci=None))
    return qs


def pool_of(D, q):
    if q["form"] == "abbr":
        return D.abbr.get(q["key"], {})
    if q["form"] == "mixed":
        return D.mixed.get((q["key"][0], q["key"][1]), {})
    return D.full.get(q["key"], {})


def cmd_build(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ice = load_dict(a.ice)
    frost = load_dict(a.frost)
    names = ("base", "ext") + (("others",) if a.with_others else ())
    if a.with_others:
        ice = load_dict(a.ice, names)
        frost = load_dict(a.frost, names)
    c, scale = make_c(ice, frost)
    print(f"ice {len(ice)} entries, frost {len(frost)}, shared {sum(1 for k in ice if k in frost)}, frost/ice median ratio {scale:.4f}")
    dicts = {VARIANTS[0]: Dict(ice), VARIANTS[1]: Dict(frost), VARIANTS[2]: Dict(c)}
    qs = questions(a, dicts[VARIANTS[0]])
    out = []
    need = []                                  # (ctx, words) to score, keyed by question index
    for n, q in enumerate(qs):
        pools = {v: Dict.order(pool_of(D, q)) for v, D in dicts.items()}
        union = []
        seen = set()
        for v in VARIANTS:
            for w, _ in pools[v]:
                if w not in seen:
                    seen.add(w)
                    union.append(w)
        out.append(dict(n=n, set=q["set"], dom=q["dom"], ans=q["ans"], ci=q["ci"], form=q["form"],
                        pools={v: pools[v] for v in VARIANTS}))
        need.append(dict(n=n, ctx=q["ctx"], c=union, ci=q["ci"]))
    with open(work / "questions.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(work / "need.jsonl", "w", encoding="utf-8") as f:
        for r in need:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sizes = defaultdict(list)
    for r in out:
        for v in VARIANTS:
            sizes[(r["set"], v)].append(len(r["pools"][v]))
    meta = {f"{s}|{v}": dict(median=statistics.median(x), max=max(x), n=len(x)) for (s, v), x in sizes.items()}
    json.dump(dict(scale=scale, pool_sizes=meta, with_others=a.with_others), open(work / "build_meta.json", "w"), ensure_ascii=False)
    print(f"{len(qs)} questions, {sum(len(r['c']) for r in need)} (context, word) pairs to score per model")


def cmd_score(a):
    work = Path(a.work)
    name, _, gguf = a.model.partition("=")
    need = rj(work / "need.jsonl")
    # scores already known from the abbreviation caches (same model file, same contexts)
    known = {}
    if a.reuse:
        for it in json.load(open(a.reuse, encoding="utf-8")):
            known[it["i"]] = dict(zip(it["words"], it["s"]))
    out = work / f"scores_{name}.jsonl"
    done = sum(1 for _ in open(out, encoding="utf-8")) if out.exists() else 0
    todo = []
    for r in need:
        kn = known.get(r["ci"], {}) if r["ci"] is not None else {}
        miss = [w for w in r["c"] if w not in kn]
        todo.append((r, kn, miss))
    p = subprocess.Popen([a.lua, f"{a.pipe_dir}/score_pipe.lua", a.so_dir, gguf, a.vocab, str(BATCH), str(a.threads)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    total_words = sum(len(m) for _, _, m in todo)
    got = 0
    with open(out, "a", encoding="utf-8") as f:
        for n, (r, kn, miss) in enumerate(todo):
            if n < done:
                continue
            sc = {w: kn[w] for w in r["c"] if w in kn}
            for i in range(0, len(miss), BATCH):
                chunk = miss[i:i + BATCH]
                p.stdin.write(r["ctx"] + "\t" + "|".join(chunk) + "\n")
                p.stdin.flush()
                ln = p.stdout.readline().strip()
                vals = [INVALID] * len(chunk) if ln == "skip" else [float(x) for x in ln.split("|")]
                sc.update(zip(chunk, vals))
                got += len(chunk)
            f.write(json.dumps(dict(n=r["n"], s=[sc[w] for w in r["c"]]), ensure_ascii=False) + "\n")
            f.flush()
            if n % 500 == 0:
                print(f"{name}: {n}/{len(todo)} questions, {got}/{total_words} new words", flush=True)
    p.stdin.close()
    (work / f"score_done_{name}.flag").write_text(str(len(todo)), encoding="ascii")


def rank_hits(ordered, scores, ans, beta):
    """ordered: [(word, weight)] in dictionary order.  Returns (rank of the answer or None).  Mode beta=None: dictionary only."""
    if ans not in dict(ordered):
        return None
    if beta is None or all(s <= INVALID / 2 for s in scores):
        key = [(-w, i) for i, (_, w) in enumerate(ordered)]
    else:
        key = [(-(s + beta * math.log1p(max(w, 0))) if s > INVALID / 2 else 1e9, i) for i, ((_, w), s) in enumerate(zip(ordered, scores))]
    order = sorted(range(len(ordered)), key=lambda i: key[i])
    for r, i in enumerate(order, 1):
        if ordered[i][0] == ans:
            return r


def boot_ci(d, n=2000, seed=7):
    import numpy as np
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.sort(np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]))
    return 100 * d.mean(), 100 * means[int(0.025 * n)], 100 * means[int(0.975 * n)]


def cmd_report(a):
    work = Path(a.work)
    qs = rj(work / "questions.jsonl")
    need = rj(work / "need.jsonl")
    meta = json.load(open(work / "build_meta.json"))
    models = a.model
    sc = {}
    for m in models:
        ss = {r["n"]: r["s"] for r in rj(work / f"scores_{m}.jsonl")}
        sc[m] = ss
    sets = []
    for q in qs:
        if q["set"] not in sets:
            sets.append(q["set"])
    modes = [("dict", None)] + [("model", 0.0)] + [(f"beta {b}", b) for b in BETAS]
    L = []
    P = L.append
    P("# Frost dictionary evaluation")
    P("")
    P(f"Dictionaries: A rime-ice (base+ext), B rime-frost (base+ext{'+others' if meta.get('with_others') else ''}), C ice word list with frost frequencies "
      f"(ice-only entries: ice weight x {meta['scale']:.4f}, capped at the lowest frost weight of the same reading).  Model score alone, and plus beta*ln(1+weight).")
    P("")
    res = {}        # (set, variant, model, mode) -> list of (hit1, hit5) per question index (None = not in pool)
    cover = defaultdict(list)
    byset = defaultdict(list)
    for q, nd in zip(qs, need):
        byset[q["set"]].append((q, nd))
    for s in sets:
        for v in VARIANTS:
            cover[(s, v)] = [q["ans"] in dict(q["pools"][v]) for q, _ in byset[s]]
    P("## Coverage of the answer in the pool, and pool size (median / max)")
    P("")
    P("| set | n | " + " | ".join(f"{v} cover | pool" for v in VARIANTS) + " |")
    P("|---|---|" + "---|---|" * len(VARIANTS))
    for s in sets:
        row = [s, str(len(byset[s]))]
        for v in VARIANTS:
            sz = meta["pool_sizes"][f"{s}|{v}"]
            row += [f"{100 * sum(cover[(s, v)]) / len(cover[(s, v)]):.1f}%", f"{sz['median']:.0f} / {sz['max']}"]
        P("| " + " | ".join(row) + " |")
    P("")
    # accuracy
    for m in [None] + models:
        P(f"## First choice / top 5 ({'dictionary order only' if m is None else m})")
        P("")
        md = [("dict", None)] if m is None else modes[1:]
        P("| set | variant | " + " | ".join(f"{n} top1 | top5" for n, _ in md) + " |")
        P("|---|---|" + "---|---|" * len(md))
        for s in sets:
            for v in VARIANTS:
                row = [s, v]
                for nm, b in md:
                    h1, h5 = [], []
                    for q, nd in byset[s]:
                        sco = sc[m][q["n"]] if m is not None else None
                        pool = q["pools"][v]
                        words = [w for w, _ in pool]
                        if m is not None:
                            allw = nd["c"]
                            idx = {w: i for i, w in enumerate(allw)}
                            scores = [sco[idx[w]] for w in words]
                        else:
                            scores = [0.0] * len(words)
                        r = rank_hits([(w, wt) for w, wt in pool], scores, q["ans"], b)
                        res.setdefault((s, v, m, nm), []).append(r)
                        h1.append(r == 1)
                        h5.append(r is not None and r <= 5)
                    k = len(h1)
                    row += [f"{100 * sum(h1) / k:.1f}%", f"{100 * sum(h5) / k:.1f}%"]
                P("| " + " | ".join(row) + " |")
        P("")
    # paired differences against A, with CI
    P("## Paired difference to A (percentage points of first choice, 95% interval over questions)")
    P("")
    P("| set | variant | model | mode | top1 diff | top5 diff |")
    P("|---|---|---|---|---|---|")
    for s in sets:
        for v in VARIANTS[1:]:
            for m in [None] + models:
                md = [("dict", None)] if m is None else modes[1:]
                for nm, b in md:
                    ra = res[(s, VARIANTS[0], m, nm)]
                    rv = res[(s, v, m, nm)]
                    d1 = [int(y == 1) - int(x == 1) for x, y in zip(ra, rv)]
                    d5 = [int(y is not None and y <= 5) - int(x is not None and x <= 5) for x, y in zip(ra, rv)]
                    c1, c5 = boot_ci(d1), boot_ci(d5)
                    P(f"| {s} | {v} | {m or '-'} | {nm} | {c1[0]:+.1f} ({c1[1]:+.1f}..{c1[2]:+.1f}) | {c5[0]:+.1f} ({c5[1]:+.1f}..{c5[2]:+.1f}) |")
    # coverage drops by domain
    P("")
    P("## Coverage drop B-A and C-A per set (points)")
    P("")
    P("| set | B-A | C-A |")
    P("|---|---|---|")
    for s in sets:
        ca = sum(cover[(s, VARIANTS[0])]) / len(cover[(s, VARIANTS[0])])
        P(f"| {s} | {100 * (sum(cover[(s, VARIANTS[1])]) / len(cover[(s, VARIANTS[1])]) - ca):+.1f} | "
          f"{100 * (sum(cover[(s, VARIANTS[2])]) / len(cover[(s, VARIANTS[2])]) - ca):+.1f} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({"|".join(map(str, k)): v for k, v in res.items()}, open(work / "ranks.json", "w"))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--work", required=True); p.add_argument("--frost", required=True); p.add_argument("--ice", required=True)
    p.add_argument("--abbrev-cache", required=True); p.add_argument("--with-others", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("--work", required=True); p.add_argument("--model", required=True); p.add_argument("--reuse")
    p.add_argument("--lua", required=True); p.add_argument("--so-dir", required=True); p.add_argument("--vocab", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--pipe-dir", required=True, help="folder with score_pipe.lua (private repo ime/data/abbrev-proto)")
    p = sub.add_parser("report")
    p.add_argument("--work", required=True); p.add_argument("--out", required=True); p.add_argument("--model", action="append", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "score": cmd_score, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
