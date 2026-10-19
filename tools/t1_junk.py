"""T1: dictionary junk list. Words of length >= 2 with weight >= 3000 in base and ext that are rare in the training corpus
and score badly without context under Suzu-2.1. Output TSV goes to the PRIVATE repo (ime/data/dict-junk/), never to a public repo.
Order: corpus count 0 first, then lowest mean per-character log-probability. Top 500.
Usage: python tools/t1_junk.py --ckpt <pt> --out <tsv> [--min-weight 3000] [--top 500]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import t_common as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-weight", type=int, default=3000)
    ap.add_argument("--top", type=int, default=500)
    a = ap.parse_args()
    index, entries, chars = T.build_index()
    words = {}
    for name in ("base", "ext"):
        for f in T.read_dict(name):
            w = f[0]
            wt = int(f[2]) if len(f) > 2 and f[2].isdigit() else 100
            if len(w) >= 2 and T.CJK.match(w) and wt >= a.min_weight:
                words[w] = max(wt, words.get(w, 0))
    print("words", len(words), flush=True)
    cnt = T.corpus_counts(set(words))
    print("corpus scan done; zero-count words", sum(1 for w in words if cnt[w] == 0), flush=True)
    import torch
    from student_model import load_student, score_items
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_student(a.ckpt, dev)
    ws = sorted(words)
    sc = score_items(model, vocab, [{"ctx": "", "c": [w]} for w in ws], dev)
    rows = []
    for w, s in zip(ws, sc):
        py = T.best_key(entries, w)
        comp = [x for x in sorted(index[py].items(), key=lambda t: (-t[1], t[0])) if x[0] != w][:3] if py else []
        rows.append((cnt[w], s[0] / len(w), w, py or "", words[w], comp))
    rows.sort(key=lambda r: (r[0] != 0, r[1]) if r[0] == 0 else (True, 0, r[1]))
    zero = [r for r in rows if r[0] == 0]
    zero.sort(key=lambda r: r[1])
    out = zero[:a.top]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("word\tpinyin\tweight\tcorpus_count\tmean_logprob\tcompetitors(word:weight)\n")
        for c, m, w, py, wt, comp in out:
            f.write(f"{w}\t{py}\t{wt}\t{c}\t{m:.2f}\t{' '.join(f'{x}:{y}' for x, y in comp)}\n")
    print(f"zero-count words {len(zero)} of {len(words)}; wrote {len(out)} rows to {a.out}")
    print("mean logprob of listed rows: %.2f; of all words: %.2f" % (sum(r[1] for r in out) / max(len(out), 1), sum(r[1] for r in rows) / len(rows)))


if __name__ == "__main__":
    main()
