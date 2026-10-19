"""T2 (candidate-count sweep) and T3 (error profile) on one evaluation pool.

Pool: wiki eval items (sampled), single-character items cut from the wiki evaluation sentences, and the bracketed words in
tests/domain and tests/holdout. Candidates = rime-ice words with the same pinyin and length, weight descending, first 12.
T2: per syllable-count group and K in {3,5,8,10,12}: share of answers inside the top K, pure-model hit rate (answer outside top K = miss),
    dictionary-first baseline, latency estimate (batch cost scaled from the measured 8-candidate figure).
T3: for errors at K=10: category shares (answer not in candidates / rare wrong word / common wrong word, wrong context) and a
    corpus-frequency-floor simulation (penalty P for candidates with corpus count below a floor).
Usage: python tools/t23_scan.py --ckpt <pt> --out <md> [--json-out <json>]
"""
import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "pipeline"))
import t_common as T  # noqa: E402

KS = [3, 5, 8, 10, 12]
MARK = re.compile(r"【([^】]+)】")


def pool(index, entries, seed=1):
    import jieba
    jieba.setLogLevel(30)
    rng = random.Random(seed)
    items = []
    wiki = [json.loads(l) for l in open("D:/distill-data-r2/items_eval.jsonl", encoding="utf-8")]
    for it in rng.sample(wiki, 4000):
        items.append(dict(src="wiki", ctx=it["ctx"], ans=it["c"][it["t"]]))
    sents = [l.strip() for l in open("D:/distill-data/eval_sentences.txt", encoding="utf-8") if l.strip()]
    rng.shuffle(sents)
    per = Counter()
    n1 = 0
    for s in sents:
        pos = 0
        for w in jieba.cut(s, HMM=False):
            ctx = s[:pos]
            pos += len(w)
            if len(w) == 1 and T.CJK.match(w) and len(ctx) >= 2 and per[w] < 30 and w in entries:
                per[w] += 1
                items.append(dict(src="single", ctx=ctx[-30:], ans=w))
                n1 += 1
        if n1 >= 1500:
            break
    for d in ("domain", "holdout"):
        for p in sorted((ROOT / "tests" / d).glob("*.txt")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                text, last, posn = [], 0, 0
                for m in MARK.finditer(line):
                    text.append(line[last:m.start()])
                    posn += len(line[last:m.start()])
                    items.append(dict(src=d, ctx=("".join(text))[-30:], ans=m.group(1)))
                    text.append(m.group(1))
                    posn += len(m.group(1))
                    last = m.end()
    out = []
    for it in items:
        if len(it["ctx"]) < 2 or not T.CJK.match(it["ans"]):
            continue
        py, cands = T.candidates(index, entries, it["ans"], 12)
        it["c"], it["py"] = cands, py
        it["rank"] = cands.index(it["ans"]) if it["ans"] in cands else None   # None: not within top 12 (or not in dictionary)
        it["known"] = it["ans"] in entries
        out.append(it)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out")
    a = ap.parse_args()
    import torch
    from student_model import load_student, score_items
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    index, entries, chars = T.build_index()
    items = pool(index, entries)
    print("pool", len(items), dict(Counter(i["src"] for i in items)), flush=True)
    model, vocab = load_student(a.ckpt, dev)
    ev = [i for i in items if i["c"]]
    # answer may be outside the top 12: score the 12 plus the answer so the miss is a genuine pick among shown ones
    sc = score_items(model, vocab, [{"ctx": i["ctx"], "c": i["c"]} for i in ev], dev)
    for i, s in zip(ev, sc):
        i["s"] = s
    grp = lambda n: 1 if n == 1 else (2 if n == 2 else 3)
    GN = {1: "1 syllable", 2: "2 syllables", 3: "3+ syllables"}
    L = ["# T2 candidate-count sweep and T3 error profile (Suzu-2.1, pure model)", "",
         f"Pool: {len(items)} items ({dict(Counter(i['src'] for i in items))}); {len(ev)} have dictionary candidates. Candidates are rime-ice same-pinyin same-length words by weight. "
         "Hit = the model's first choice among the first K candidates equals the answer; an answer outside the first K is a miss. "
         "Latency is estimated by scaling the measured 8-candidate batch cost (9.7 ms, 7 threads) linearly with K; it is not a separate measurement.", ""]
    res = {}
    L += ["## T2", "", "| group | N | K | answer in top K | pure-model hit | dictionary-first | est. batch ms |", "|---|---|---|---|---|---|---|"]
    for g in (1, 2, 3):
        its = [i for i in items if grp(len(i["ans"])) == g]
        n = len(its)
        dfirst = sum(1 for i in its if i["rank"] == 0) / max(n, 1) * 100
        for K in KS:
            inK = sum(1 for i in its if i["rank"] is not None and i["rank"] < K)
            hit = 0
            for i in its:
                if i.get("s") and i["rank"] is not None and i["rank"] < K:
                    s = i["s"][:K]
                    hit += int(s.index(max(s)) == i["rank"])
            res[f"{g}/{K}"] = dict(n=n, inK=inK / max(n, 1), hit=hit / max(n, 1))
            L.append(f"| {GN[g]} | {n} | {K} | {inK / max(n, 1) * 100:.1f} | {hit / max(n, 1) * 100:.1f} | {dfirst:.1f} | {9.7 * K / 8:.1f} |")
    # ---- T3
    K = 10
    wrong, cat = [], Counter()
    for i in items:
        if i["rank"] is None or i["rank"] >= K or not i.get("s"):
            cat["answer not in the first 10 candidates" + (" (word not in dictionary)" if not i["known"] else "")] += 1
            continue
        s = i["s"][:K]
        p = s.index(max(s))
        if p != i["rank"]:
            wrong.append((i, p))
    words = {i["c"][p] for i, p in wrong} | {c for i in ev for c in i["c"][:K]}
    cnt = T.corpus_counts(words)
    for i, p in wrong:
        w = i["c"][p]
        cat["wrong pick is rare in corpus (count < 3)" if cnt[w] < 3 else "wrong pick is common, context mismatch"] += 1
    total_err = sum(cat.values())
    L += ["", "## T3 (errors at K = 10)", "", f"Items: {len(items)}; errors incl. answer outside candidates: {total_err} ({total_err / len(items) * 100:.1f}%).", "",
          "| category | count | share of errors |", "|---|---|---|"]
    for k, v in cat.most_common():
        L.append(f"| {k} | {v} | {v / max(total_err, 1) * 100:.1f}% |")
    zero = sum(1 for i, p in wrong if cnt[i["c"][p]] == 0)
    first = sum(1 for i, p in wrong if p == 0)
    L += ["", f"Among {len(wrong)} wrong picks inside candidates: corpus count 0 for {zero}; the wrong pick is the dictionary-first word in {first}; mean weight rank of the wrong pick {sum(p for _, p in wrong) / max(len(wrong), 1):.2f}; "
          f"length distribution {dict(Counter(len(i['c'][p]) for i, p in wrong))}.", "",
          "### Corpus-frequency floor simulation (K = 10)", "", "penalty P is subtracted from the score of candidates whose corpus count is below the floor.", "",
          "| floor | P | hit rate | change vs no penalty (points) |", "|---|---|---|---|"]
    base_hits = None
    ev10 = [i for i in ev if i["rank"] is not None and i["rank"] < K]
    for floor, P in [(0, 0), (1, 1), (1, 2), (1, 4), (3, 1), (3, 2), (3, 4), (10, 1), (10, 2)]:
        h = 0
        for i in ev10:
            s = [x - (P if cnt[c] < floor else 0) for x, c in zip(i["s"][:K], i["c"][:K])]
            h += int(s.index(max(s)) == i["rank"])
        base_hits = h if base_hits is None else base_hits
        L.append(f"| {floor} | {P} | {h / len(items) * 100:.2f}% of all items | {(h - base_hits) / len(items) * 100:+.2f} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    if a.json_out:
        json.dump(res, open(a.json_out, "w", encoding="utf-8"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
