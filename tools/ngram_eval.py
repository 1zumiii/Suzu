"""Offline evaluation for the hashed n-gram embedding experiment (Student with extra lookup tables).
Everything runs in PyTorch (fp32) on checkpoints, so the baselines and the new arms are scored by the same code.
No word lists are written, only numbers.

  python tools/ngram_eval.py check0 --base BASE.pt --items items_dev.jsonl [--n 64]
      step 0 equivalence: the base weights loaded into an n-gram model (zero-initialised projections) score like the base model
  python tools/ngram_eval.py pool --pool pool_cache.json --ckpt NAME=pt [--ckpt ...] --out-dir DIR
      scores the whole candidate pool of every abbreviation question (pool_cache.json: list of {i, dom, key, ctx, ans, words, s})
  python tools/ngram_eval.py report --pool pool_cache.json --dir DIR --ref NAME --name NAME [--name ...] --out report.md
          [--enc enc_cache.npz --vocab vocab.json]       # n-gram statistics of the training corpus for the "seen collocation" slices
  python tools/ngram_eval.py best --pool pool_cache.json --dir DIR --name NAME [--name ...]     prints the name with the highest top1
  python tools/ngram_eval.py bench --ckpt NAME=pt [--tokens 25]
      CPU time of assembling the n-gram embedding for one 25-token sequence (hash, lookup, projection), single thread
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

CHUNK = 60
BATCH = 24


def cmd_check0(a):
    import torch
    from student_model import Student, cand_scores
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.base, map_location=dev)
    c, vocab = ck["cfg"], ck["vocab"]
    vi = {x: i for i, x in enumerate(vocab)}
    base = Student(len(vocab), c["student_d"], c["student_layers"], c["student_heads"], c["student_ctx"]).to(dev)
    base.load_state_dict(ck["model"])
    base.eval()
    ng = dict(ns=[2, 3], log_rows=14, dim=128, heads=2)
    m = Student(len(vocab), c["student_d"], c["student_layers"], c["student_heads"], c["student_ctx"], ng).to(dev)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")][:a.n]
    with torch.no_grad():
        x, y = cand_scores(base, vi, items, dev), cand_scores(m, vi, items, dev)
    ok = x > -1e3
    print(f"step 0 max abs difference over {int(ok.sum())} candidate scores: {(x - y).abs()[ok].max().item():.3g}")


def pool_scores(model, vi, items, dev):
    import torch
    from student_model import cand_scores
    chunks = []
    for n, t in enumerate(items):
        for k in range(0, len(t["words"]), CHUNK):
            chunks.append((n, k, {"ctx": t["ctx"], "c": t["words"][k:k + CHUNK]}))
    out = [[0.0] * len(t["words"]) for t in items]
    model.eval()
    with torch.no_grad():
        for a in range(0, len(chunks), BATCH):
            part = chunks[a:a + BATCH]
            sc = cand_scores(model, vi, [p[2] for p in part], dev).tolist()
            for (n, k, it), row in zip(part, sc):
                out[n][k:k + len(it["c"])] = row[:len(it["c"])]
    for n, t in enumerate(items):      # the inference core scores a candidate with a character outside the vocabulary as invalid
        for j, w in enumerate(t["words"]):
            if any(c not in vi for c in w):
                out[n][j] = -1e9
    return out


def cmd_pool(a):
    import torch
    from student_model import load_student
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = json.load(open(a.pool, encoding="utf-8"))
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for spec in a.ckpt:
        name, _, path = spec.partition("=")
        out = Path(a.out_dir) / f"pool_{name}.json"
        if out.exists():
            continue
        model, vocab = load_student(path, dev)
        vi = {c: i for i, c in enumerate(vocab)}
        t0 = time.time()
        sc = pool_scores(model, vi, items, dev)
        json.dump(sc, open(out, "w"))
        print(f"{name}: {len(items)} questions scored in {time.time() - t0:.0f} s", flush=True)
        del model
        torch.cuda.empty_cache()


def ranks_of(items, scores):
    r = []
    for t, s in zip(items, scores):
        i = t["words"].index(t["ans"])
        r.append(1 + sum(1 for x in s if x > s[i]))
    return np.array(r)


def boot(d, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    means = np.sort(np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]))
    return 100 * d.mean(), 100 * means[int(0.025 * n)], 100 * means[int(0.975 * n)]


def seen_sets(a):
    """Sorted arrays of the 2-gram and 3-gram keys that occur in the training sentences (vocabulary indices, BOS-bounded)."""
    z = np.load(a.enc)
    flat, off = z["flat"].astype(np.int64), z["off"]
    V = 8192
    is_bos = flat == 2
    k2 = flat[:-1] * V + flat[1:]
    ok2 = ~is_bos[1:]
    k3 = (flat[:-2] * V + flat[1:-1]) * V + flat[2:]
    ok3 = ~is_bos[1:-1] & ~is_bos[2:]
    return np.unique(k2[ok2]), np.unique(k3[ok3]), V


def in_sorted(w, k):
    """Membership of the keys w in the sorted unique array k."""
    w = np.asarray(w, dtype=np.int64)
    if w.size == 0:
        return np.zeros(0, dtype=bool)
    i = np.searchsorted(k, w).clip(max=len(k) - 1)
    return k[i] == w


def question_windows(items, vocab, V):
    vi = {c: i for i, c in enumerate(vocab)}
    out = []
    for t in items:
        ctx = [vi.get(c, 1) for c in t["ctx"]][-2:]
        ans = [vi.get(c, 1) for c in t["ans"]]
        seq = ctx + ans
        la = len(ctx)
        w2, w3 = [], []
        for j in range(la, len(seq)):
            if j - 1 >= 0:
                w2.append(seq[j - 1] * V + seq[j])
            if j - 2 >= 0:
                w3.append((seq[j - 2] * V + seq[j - 1]) * V + seq[j])
        out.append((w2, w3))
    return out


def cmd_report(a):
    items = json.load(open(a.pool, encoding="utf-8"))
    d = Path(a.dir)
    sc = {n: json.load(open(d / f"pool_{n}.json")) for n in a.name}
    rk = {n: ranks_of(items, sc[n]) for n in a.name}
    L = []
    P = L.append
    P("# n-gram embedding experiment: abbreviation questions (whole candidate pool, model score only)")
    P("")
    P(f"{len(items)} two-letter abbreviation questions; reference arm: {a.ref}.")
    P("")
    P("| arm | top1 | top5 | top10 | top1 diff to ref (95%) | top5 diff to ref (95%) |")
    P("|---|---|---|---|---|---|")
    for n in a.name:
        r = rk[n]
        d1, d5 = boot((r == 1).astype(int) - (rk[a.ref] == 1).astype(int)), boot((r <= 5).astype(int) - (rk[a.ref] <= 5).astype(int))
        P(f"| {n} | {100 * (r == 1).mean():.1f}% | {100 * (r <= 5).mean():.1f}% | {100 * (r <= 10).mean():.1f}% | "
          f"{d1[0]:+.1f} ({d1[1]:+.1f}..{d1[2]:+.1f}) | {d5[0]:+.1f} ({d5[1]:+.1f}..{d5[2]:+.1f}) |")
    doms = sorted({t["dom"] for t in items})
    P("")
    P("Top1 by domain:")
    P("")
    P("| arm | " + " | ".join(f"{x} (n={sum(t['dom'] == x for t in items)})" for x in doms) + " |")
    P("|---|" + "---|" * len(doms))
    for n in a.name:
        P(f"| {n} | " + " | ".join(f"{100 * np.mean([rk[n][i] == 1 for i, t in enumerate(items) if t['dom'] == x]):.1f}%" for x in doms) + " |")
    if a.enc:
        vocab = json.load(open(a.vocab, encoding="utf-8"))
        k2, k3, V = seen_sets(a)
        wins = question_windows(items, vocab, V)
        seen2 = np.array([all(in_sorted(w2, k2)) if w2 else False for w2, _ in wins])
        seen3 = np.array([all(in_sorted(w3, k3)) if w3 else False for _, w3 in wins])
        frac2 = np.mean([in_sorted(w2, k2).mean() if w2 else 0 for w2, _ in wins])
        frac3 = np.mean([in_sorted(w3, k3).mean() if w3 else 0 for _, w3 in wins])
        P("")
        P("## Training-corpus coverage of the question windows (windows over the end of the context plus the answer)")
        P("")
        P(f"Average share of 2-gram windows seen in the training sentences: {100 * frac2:.1f}%; 3-gram windows: {100 * frac3:.1f}%.")
        P(f"Questions whose windows are all seen: 2-gram {100 * seen2.mean():.1f}% ({int(seen2.sum())}), 3-gram {100 * seen3.mean():.1f}% ({int(seen3.sum())}).")
        for tag, mask in (("3-gram collocation seen", seen3), ("3-gram collocation not seen", ~seen3),
                          ("2-gram collocation seen", seen2), ("2-gram collocation not seen", ~seen2)):
            P("")
            P(f"### {tag} (n={int(mask.sum())})")
            P("")
            P("| arm | top1 | top1 diff to ref (95%) | top5 | top5 diff to ref (95%) |")
            P("|---|---|---|---|---|")
            for n in a.name:
                r, f = rk[n][mask], rk[a.ref][mask]
                if mask.sum() < 5:
                    continue
                d1, d5 = boot((r == 1).astype(int) - (f == 1).astype(int)), boot((r <= 5).astype(int) - (f <= 5).astype(int))
                P(f"| {n} | {100 * (r == 1).mean():.1f}% | {d1[0]:+.1f} ({d1[1]:+.1f}..{d1[2]:+.1f}) | {100 * (r <= 5).mean():.1f}% | {d5[0]:+.1f} ({d5[1]:+.1f}..{d5[2]:+.1f}) |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def cmd_best(a):
    items = json.load(open(a.pool, encoding="utf-8"))
    d = Path(a.dir)
    top = {n: float((ranks_of(items, json.load(open(d / f"pool_{n}.json"))) == 1).mean()) for n in a.name}
    print(max(top, key=top.get))


def cmd_bench(a):
    import torch
    from student_model import load_student
    torch.set_num_threads(1)
    for spec in a.ckpt:
        name, _, path = spec.partition("=")
        model, vocab = load_student(path, "cpu")
        if model.ngram is None:
            continue
        ids = torch.randint(3, len(vocab), (1, a.tokens))
        prev = torch.arange(-1, a.tokens - 1)[None]
        with torch.no_grad():
            for _ in range(20):
                model.ngram(ids, prev)
            t0 = time.perf_counter()
            for _ in range(a.reps):
                model.ngram(ids, prev)
            dt = (time.perf_counter() - t0) / a.reps * 1000
            x = torch.zeros(1, a.tokens, 1)
            t0 = time.perf_counter()
            for _ in range(5):
                model.hidden(ids)
            full = (time.perf_counter() - t0) / 5 * 1000
        print(f"{name}: n-gram embedding for {a.tokens} tokens {dt:.3f} ms (single thread, PyTorch, not the production code path); "
              f"whole forward of the same sequence {full:.1f} ms", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check0"); p.add_argument("--base", required=True); p.add_argument("--items", required=True); p.add_argument("--n", type=int, default=64)
    p = sub.add_parser("pool"); p.add_argument("--pool", required=True); p.add_argument("--ckpt", action="append", required=True); p.add_argument("--out-dir", required=True)
    p = sub.add_parser("report"); p.add_argument("--pool", required=True); p.add_argument("--dir", required=True); p.add_argument("--ref", required=True)
    p.add_argument("--name", action="append", required=True); p.add_argument("--out", required=True); p.add_argument("--enc"); p.add_argument("--vocab")
    p = sub.add_parser("best"); p.add_argument("--pool", required=True); p.add_argument("--dir", required=True); p.add_argument("--name", action="append", required=True)
    p = sub.add_parser("bench"); p.add_argument("--ckpt", action="append", required=True); p.add_argument("--tokens", type=int, default=25); p.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    {"check0": cmd_check0, "pool": cmd_pool, "report": cmd_report, "best": cmd_best, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
