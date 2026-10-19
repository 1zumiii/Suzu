"""T4: single-character training items, held-out single-character set, run assembly and held-out evaluation.

  build     : make single-char KD items (candidates = rime-ice same-pinyin single chars, top 10 by weight, answer included) from wiki
              and daily-chat synthetic training sentences, plus a held-out set (wiki evaluation sentences and the bracketed words in tests/)
              that never enters training.
  assemble  : previous run (Suzu-2.1's run3) items + new items, teacher scores merged, train_sentences copied -> run directory.
  eval      : pure-model accuracy on the held-out set (candidates = top 10) for several checkpoints, paired bootstrap 95% intervals.
Usage: python tools/t4_single.py build --out <dir> | assemble --prev <run3> --label <dir> --out <run> | eval --holdout <jsonl> --ckpt NAME=pt ...
"""
import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "pipeline"))
import t_common as T  # noqa: E402

MARK = re.compile(r"【([^】]+)】")


def single_cands(index, entries, ch, k=10):
    py = T.best_key(entries, ch)
    if py is None:
        return None, []
    ranked = sorted(((w, wt) for w, wt in index[py].items() if len(w) == 1), key=lambda x: (-x[1], x[0]))
    c = [w for w, _ in ranked][:k]
    if ch not in c and c:
        c = c[:k - 1] + [ch]
    return py, c


def mk(ctx, ch, index, entries, domain, idn):
    py, c = single_cands(index, entries, ch)
    if len(c) < 3 or ch not in c:
        return None
    wts = {w: wt for w, wt in index[py].items() if len(w) == 1}
    c = sorted(c, key=lambda w: (-wts.get(w, 0), w))
    return dict(ctx=ctx[-30:], py=py, c=c, f=[wts.get(w, 1) for w in c], t=c.index(ch), s=int(c.index(ch) > 0), L=1, word=ch, domain=domain, id=idn)


def cmd_build(a):
    import jieba
    jieba.setLogLevel(30)
    index, entries, chars = T.build_index()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(21)
    items, per = [], Counter()
    srcs = [(r"D:\distill-data-r2\train_sentences.txt", 0.5), (r"D:\distill-data-r3\big\chat.txt", 0.5)]
    for path, share in srcs:
        lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
        rng.shuffle(lines)
        target = int(a.n * share)
        got = 0
        for s in lines:
            pos = 0
            for w in jieba.cut(s, HMM=False):
                ctx = s[:pos]
                pos += len(w)
                if len(w) == 1 and w in chars and len(ctx) >= 2 and per[w] < a.cap:
                    it = mk(ctx, w, index, entries, "single", 2_000_000 + len(items))
                    if it:
                        per[w] += 1
                        items.append(it)
                        got += 1
                        if got >= target:
                            break
            if got >= target:
                break
    rng.shuffle(items)
    for n, it in enumerate(items):
        it["id"] = 2_000_000 + n
    with open(out / "items_train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # held-out set: never used in training
    hold, per2 = [], Counter()
    sents = [l.strip() for l in open("D:/distill-data/eval_sentences.txt", encoding="utf-8") if l.strip()]
    rng.shuffle(sents)
    for s in sents:
        pos = 0
        for w in jieba.cut(s, HMM=False):
            ctx = s[:pos]
            pos += len(w)
            if len(w) == 1 and w in chars and len(ctx) >= 2 and per2[w] < 6 and len(hold) < a.hold:
                it = mk(ctx, w, index, entries, "single_hold", 3_000_000 + len(hold))
                if it:
                    per2[w] += 1
                    hold.append(it)
    for d in ("domain", "holdout"):
        for p in sorted((ROOT / "tests" / d).glob("*.txt")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                text, last = "", 0
                for m in MARK.finditer(line):
                    text += line[last:m.start()]
                    if len(m.group(1)) == 1 and len(text) >= 2:
                        it = mk(text, m.group(1), index, entries, "single_hold", 3_000_000 + len(hold))
                        if it:
                            hold.append(it)
                    text += m.group(1)
                    last = m.end()
    with open(out / "items_holdout_single.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in hold:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    trn = {(i["ctx"], i["word"]) for i in items}
    overlap = sum(1 for h in hold if (h["ctx"], h["word"]) in trn)
    print(json.dumps(dict(train_items=len(items), distinct_chars=len(per), holdout=len(hold), overlap_with_train=overlap,
                          train_hard=sum(i["s"] for i in items)), ensure_ascii=False))


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def cmd_assemble(a):
    prev, lab, out = Path(a.prev), Path(a.label), Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    for n in ("items_dev.jsonl", "meta.json", "gates.json", "train_sentences.txt"):
        if not (out / n).exists():
            shutil.copy(prev / n, out / n)
    if not (out / "ckpt" / "vocab.json").exists():
        shutil.copy(prev / "ckpt" / "vocab.json", out / "ckpt" / "vocab.json")
    t = "q35_4b_base__train.jsonl"
    scores = {r["id"]: r for r in rj(prev / "scores" / t)}
    scores.update({r["id"]: r for r in rj(lab / "scores" / t)})
    items = [i for i in rj(prev / "items_train.jsonl") + rj(lab / "items_train.jsonl") if i["id"] in scores]
    random.Random(4).shuffle(items)
    with open(out / "items_train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out / "scores" / t, "w", encoding="utf-8", newline="\n") as f:
        for r in scores.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n1 = sum(1 for i in items if i["domain"] == "single")
    print(json.dumps(dict(items=len(items), single=n1, single_share=round(n1 / len(items), 3))))


def cmd_eval(a):
    import torch
    from student_model import load_student, score_items
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hold = rj(a.holdout)
    corr = {}
    for spec in a.ckpt:
        name, p = spec.split("=", 1)
        m, v = load_student(p, dev)
        sc = score_items(m, v, hold, dev)
        corr[name] = np.array([s.index(max(s)) == it["t"] for it, s in zip(hold, sc)])
    names = list(corr)
    L = ["# T4 single-character held-out evaluation (pure model, top-10 candidates)", "", f"Held-out items: {len(hold)}. Paired bootstrap 95% over items (2000 resamples).", "",
         "| model | accuracy |", "|---|---|"]
    for n in names:
        L.append(f"| {n} | {corr[n].mean() * 100:.1f} |")
    rs = np.random.RandomState(0)
    L += ["", "| comparison | difference, points (95% CI) |", "|---|---|"]
    for i in range(1, len(names)):
        d = corr[names[i]].astype(float) - corr[names[0]].astype(float)
        bs = [d[rs.randint(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        L.append(f"| {names[i]} minus {names[0]} | {d.mean() * 100:+.1f} ({lo * 100:+.1f} to {hi * 100:+.1f}) |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=15000)
    b.add_argument("--cap", type=int, default=60)
    b.add_argument("--hold", type=int, default=500)
    s = sp.add_parser("assemble")
    s.add_argument("--prev", required=True)
    s.add_argument("--label", required=True)
    s.add_argument("--out", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--holdout", required=True)
    e.add_argument("--ckpt", action="append", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "assemble": cmd_assemble, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
