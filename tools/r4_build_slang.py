"""Weak-spot plan stage B3: build slang KD items from the filtered sentences.

One item per sentence: context = text before the first occurrence of the seed term (>= 2 chars), candidates = the 8
most frequent same-pinyin words from the round-3 word table, with the slang term itself ALWAYS included. Item ids start
at --id-base so they never collide with the round-3 items. Also writes a spot-check sample (30 per category) as text.

Usage: python tools/r4_build_slang.py --sents <prefix> --table <word_table.pkl> --draft <slang_draft.tsv> --out <dir>
"""
import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sents", required=True, help="prefix of filter_slang_corpus output")
    ap.add_argument("--table", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dict-dir", required=True, help="rime-ice cn_dicts: competitors are the same-pinyin same-length words users actually see")
    ap.add_argument("--id-base", type=int, default=1_000_000)
    a = ap.parse_args()
    from pypinyin import lazy_pinyin
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pickle.loads(Path(a.table).read_bytes())

    def pykey(w, cache={}):
        if w not in cache:
            p = lazy_pinyin(w)
            cache[w] = " ".join(p) if len(p) == len(w) and all(x.isascii() and x.isalpha() for x in p) else None
        return cache[w]

    # rime-ice dictionary index: pinyin key -> {word: weight}, plus word -> best key
    chars, index, wkey = {}, defaultdict(dict), {}
    dd = Path(a.dict_dir)

    def body(path):
        on = False
        for ln in open(path, encoding="utf-8"):
            if ln.startswith("..."):
                on = True
                continue
            if on and ln.strip() and not ln.startswith("#"):
                yield ln.rstrip("\n").split("\t")

    for name in ("8105", "base", "ext", "others"):
        for f in body(dd / f"{name}.dict.yaml"):
            if len(f) < 2:
                continue
            w, py = f[0], f[1]
            wt = int(f[2]) if len(f) > 2 and f[2].isdigit() else 100
            if name == "8105" and w not in chars:
                chars[w] = py
            if len(w) == len(py.split()):
                index[py][w] = max(wt, index[py].get(w, 0))
    for f in body(dd / "tencent.dict.yaml"):
        w = f[0]
        if all(c in chars for c in w):
            py = " ".join(chars[c] for c in w)
            index[py].setdefault(w, int(f[1]) if len(f) > 1 and f[1].isdigit() else 100)
    for py, d in index.items():
        for w, wt in d.items():
            if w not in wkey or wt > index[wkey[w]][w]:
                wkey[w] = py

    def pykey(w, cache={}):
        if w in wkey:
            return wkey[w]
        if w not in cache:
            p = lazy_pinyin(w)
            cache[w] = " ".join(p) if len(p) == len(w) and all(x.isascii() and x.isalpha() for x in p) else None
        return cache[w]

    cat = {}
    for ln in open(a.draft, encoding="utf-8"):
        if "\t" in ln:
            c, w = ln.rstrip("\n").split("\t")[:2]
            cat[w] = c
    items, skip = [], defaultdict(int)
    per_cat = defaultdict(list)
    for ln in open(a.sents + ".seed.tsv", encoding="utf-8"):
        if "\t" not in ln:
            continue
        t, seed = ln.rstrip("\n").split("\t")
        pos = t.find(seed)
        if pos < 2:
            skip["ctx_short"] += 1
            continue
        if not 2 <= len(seed) <= 4:
            skip["len"] += 1
            continue
        k = pykey(seed)
        if not k:
            skip["no_pinyin"] += 1
            continue
        comp = sorted(((wt, w) for w, wt in index.get(k, {}).items() if w != seed and len(w) == len(seed)), key=lambda x: (-x[0], x[1]))[:7]
        if not comp:
            skip["no_competitor"] += 1
            continue
        fr = lambda w: max(int(round(table.get(w, 0))), 1)
        top = sorted([(fr(w), w) for _, w in comp] + [(fr(seed), seed)], key=lambda x: (-x[0], x[1]))
        c = [w for _, w in top]
        tix = c.index(seed)
        it = dict(ctx=t[:pos][-30:], py=k, c=c, f=[f for f, _ in top], t=tix, s=int(tix > 0), L=len(seed),
                  word=seed, domain="slang", id=a.id_base + len(items))
        items.append(it)
        per_cat[cat.get(seed, "?")].append(t)
    with open(out / "items_slang.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    rng = random.Random(5)
    with open(out / "spot_check.txt", "w", encoding="utf-8") as f:
        for c, ts in sorted(per_cat.items()):
            f.write(f"## {c} ({len(ts)})\n")
            for t in rng.sample(ts, min(30, len(ts))):
                f.write(t + "\n")
    print(json.dumps(dict(items=len(items), hard=sum(i["s"] for i in items), skipped=dict(skip),
                          words=len({i["word"] for i in items})), ensure_ascii=False))


if __name__ == "__main__":
    main()
