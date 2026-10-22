"""Diagnosis of the two-letter abbreviation task: how far is the student from the training questions and from the teacher.
Inference only, no training.  Writes numbers (and, for check 3, the question text of public held-out questions), nothing else.

  python tools/ngram_diag.py sample  --items A.jsonl --items B.jsonl --dict-dir CN_DICTS --out train_cache.json [--n 1300 --seed 11]
      draws training questions of the abbreviation recipe (two-letter input, two-character answer found in the dictionary pool)
      and writes them in the same layout as the evaluation cache (words = whole pool by dictionary weight)
  python tools/t_abbrev.py items --cache train_cache.json --data TEACHER_DIR
  python tools/t_abbrev.py label --data TEACHER_DIR --model <teacher model> --name <name>      (teacher scores of the pool, chunked)
  python tools/ngram_eval.py pool --pool train_cache.json --out-dir TRAIN_DIR --ckpt NAME=pt ...  (student scores of the same pool)
  python tools/ngram_diag.py teacher --cache cache.json --data TEACHER_DIR --name <name> --out pool_teacher.json
      turns the chunked teacher scores into one score list per question (same layout as ngram_eval pool files)
  python tools/ngram_diag.py report  --eval-cache C --eval-dir D --eval-teacher T --train-cache C --train-dir D --train-teacher T
         --model NAME ... --out report.md --tsv teacher_disagree.tsv [--sample 60 --seed 3]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CHUNK = 60


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_pools(d):
    pools = defaultdict(dict)
    for name in ("base", "ext"):
        body = False
        for ln in open(f"{d}/{name}.dict.yaml", encoding="utf-8"):
            if ln.startswith("..."):
                body = True
                continue
            if not body or ln.startswith("#"):
                continue
            p = ln.rstrip("\r\n").split("\t")
            if len(p) < 2 or len(p[0]) != 2:
                continue
            sy = p[1].split()
            if len(sy) != 2:
                continue
            w = int(p[2]) if len(p) > 2 and p[2].isdigit() else 0
            k = sy[0][0] + sy[1][0]
            pools[k][p[0]] = max(w, pools[k].get(p[0], 0))
    return pools


def cmd_sample(a):
    pools = load_pools(a.dict_dir)
    cands = []
    for f in a.items:
        for it in rj(f):
            py = it["py"]
            if len(py) != 2 or " " in py or not py.isalpha():
                continue
            ans = it["c"][it["t"]]
            if len(ans) != 2 or ans not in pools.get(py, {}):
                continue
            cands.append((it["id"], it["ctx"], py, ans))
    rng = random.Random(a.seed)
    rng.shuffle(cands)
    out = []
    for n, (i, ctx, py, ans) in enumerate(cands[:a.n]):
        pool = pools[py]
        words = sorted(pool, key=lambda w: -pool[w])
        out.append({"i": n, "src_id": i, "dom": "train", "key": py, "ctx": ctx, "ans": ans, "words": words, "w": [pool[w] for w in words]})
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(cands)} usable training questions, wrote {len(out)}")


def teacher_lists(items, scores_jsonl):
    chunks = {}
    for ln in open(scores_jsonl, encoding="utf-8"):
        r = json.loads(ln)
        chunks[r["id"]] = r["s"][:-1]                 # drop the sentinel candidate
    out = []
    for n, t in enumerate(items):
        k = (len(t["words"]) + CHUNK - 1) // CHUNK
        parts = [chunks.get(n * 1000 + j) for j in range(k)]
        out.append([x for p in parts for x in p] if all(p is not None for p in parts) else None)
    return out


def cmd_teacher(a):
    items = json.load(open(a.cache, encoding="utf-8"))
    sc = teacher_lists(items, Path(a.data) / "scores" / f"{a.name}__abbrev.jsonl")
    miss = sum(s is None for s in sc)
    if miss:
        print(f"{miss} of {len(items)} questions are missing teacher scores", file=sys.stderr)
        sys.exit(1)
    json.dump(sc, open(a.out, "w"))
    print(f"teacher scores for {len(items)} questions")


def ranks(items, scores):
    r = []
    for t, s in zip(items, scores):
        i = t["words"].index(t["ans"])
        r.append(1 + sum(1 for x in s if x > s[i]))
    return np.array(r)


def load_set(cache, d, teacher, names):
    items = json.load(open(cache, encoding="utf-8"))
    sc = {n: json.load(open(Path(d) / f"pool_{n}.json")) for n in names}
    sc["teacher"] = json.load(open(teacher))
    for n, s in sc.items():
        if len(s) != len(items):
            raise SystemExit(f"{n}: {len(s)} score lists for {len(items)} questions")
        bad = [i for i, x in enumerate(s) if len(x) != len(items[i]["words"])]
        if bad:
            raise SystemExit(f"{n}: score list length differs from the pool in {len(bad)} questions")
        empty = [i for i, x in enumerate(s) if not x or max(x) < -1e8]
        if empty:
            raise SystemExit(f"{n}: {len(empty)} questions without any valid score (empty pool or all candidates invalid)")
    return items, sc, {n: ranks(items, s) for n, s in sc.items()}


def pct(x):
    return f"{100 * float(np.mean(x)):.1f}%"


def pm(p, n):
    return 1.96 * 100 * np.sqrt(p * (1 - p) / n)


def jaccard(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def cmd_report(a):
    names = a.model
    ei, es, er = load_set(a.eval_cache, a.eval_dir, a.eval_teacher, names)
    ti, ts, tr = load_set(a.train_cache, a.train_dir, a.train_teacher, names)
    models = names + ["teacher"]
    L = []
    P = L.append
    P("# Diagnosis of the two-letter abbreviation task (inference only)")
    P("")
    P(f"Evaluation questions: {len(ei)} (held-out, whole candidate pool, model score only). Training questions: {len(ti)} drawn at random from the "
      "abbreviation training questions (two-letter input, two-character answer in the dictionary pool), same pool and scoring.")
    P("")
    P("## Check 1: accuracy on training questions and on evaluation questions")
    P("")
    P("| model | train top1 | eval top1 | train - eval top1 (95%) | train top5 | eval top5 | train - eval top5 |")
    P("|---|---|---|---|---|---|---|")
    for m in models:
        t1, e1, t5, e5 = (tr[m] == 1).mean(), (er[m] == 1).mean(), (tr[m] <= 5).mean(), (er[m] <= 5).mean()
        se1 = np.sqrt(pm(t1, len(ti)) ** 2 + pm(e1, len(ei)) ** 2)
        se5 = np.sqrt(pm(t5, len(ti)) ** 2 + pm(e5, len(ei)) ** 2)
        P(f"| {m} | {100 * t1:.1f}% (±{pm(t1, len(ti)):.1f}) | {100 * e1:.1f}% (±{pm(e1, len(ei)):.1f}) | {100 * (t1 - e1):+.1f} (±{se1:.1f}) | "
          f"{100 * t5:.1f}% | {100 * e5:.1f}% | {100 * (t5 - e5):+.1f} (±{se5:.1f}) |")
    P("")
    P("Band of the top1 gap (train minus eval), rule written in advance: above 5 points, between 2 and 5 points, below 2 points:")
    P("")
    for m in models:
        g = 100 * ((tr[m] == 1).mean() - (er[m] == 1).mean())
        P(f"- {m}: {g:+.1f} points: " + ("above 5" if g > 5 else "below 2" if g < 2 else "between 2 and 5"))

    P("")
    P("## Check 2: question by question comparison on the evaluation questions")
    students = [m for m in names]
    right = {m: er[m] == 1 for m in models}
    P("")
    P("### Student against teacher (first choice correct: both / teacher only / student only / neither)")
    P("")
    P("| student | both right | teacher right, student wrong | student right, teacher wrong | both wrong |")
    P("|---|---|---|---|---|")
    for m in students:
        s, t = right[m], right["teacher"]
        P(f"| {m} | {int((s & t).sum())} | {int((~s & t).sum())} | {int((s & ~t).sum())} | {int((~s & ~t).sum())} |")
    P("")
    P("### Overlap of the error sets (intersection over union)")
    P("")
    sel = [m for m in students if m != "Suzu-2.1"]
    P("| | " + " | ".join(sel) + " |")
    P("|---|" + "---|" * len(sel))
    for m in sel:
        P(f"| {m} | " + " | ".join(f"{jaccard(~right[m], ~right[o]):.2f}" for o in sel) + " |")
    allw = np.logical_and.reduce([~right[m] for m in sel])
    P("")
    P(f"Questions that all of {', '.join(sel)} get wrong: {int(allw.sum())} of {len(ei)}.")
    P("")
    P("### Teacher right, student wrong: distribution")
    P("")
    sizes = np.array([len(t["words"]) for t in ei])
    bins = [(0, 200), (200, 500), (500, 800), (800, 10 ** 9)]
    blabel = ["<=200", "201-500", "501-800", ">800"]
    doms = sorted({t["dom"] for t in ei})
    P("| student | n | " + " | ".join(f"dom {d}" for d in doms) + " | " + " | ".join(f"pool {b}" for b in blabel) + " |")
    P("|---|---|" + "---|" * (len(doms) + len(bins)))
    P("| (all evaluation questions) | " + str(len(ei)) + " | " + " | ".join(str(sum(t["dom"] == d for t in ei)) for d in doms) + " | "
      + " | ".join(str(int(((sizes > lo) & (sizes <= hi)).sum())) for lo, hi in bins) + " |")
    sets = {}
    for m in students:
        mask = ~right[m] & right["teacher"]
        sets[m] = mask
        P(f"| {m} | {int(mask.sum())} | " + " | ".join(str(sum(1 for i in np.where(mask)[0] if ei[i]["dom"] == d)) for d in doms) + " | "
          + " | ".join(str(int((mask & (sizes > lo) & (sizes <= hi)).sum())) for lo, hi in bins) + " |")
    P("")
    P("Same questions across students? Intersection over union of the 'teacher right, student wrong' sets:")
    P("")
    P("| | " + " | ".join(sel) + " |")
    P("|---|" + "---|" * len(sel))
    for m in sel:
        P(f"| {m} | " + " | ".join(f"{jaccard(sets[m], sets[o]):.2f}" for o in sel) + " |")
    inter = np.logical_and.reduce([sets[m] for m in sel])
    P("")
    P(f"In all of {', '.join(sel)}: {int(inter.sum())} questions.")
    P("")
    P(f"### Questions all of {', '.join(sel)} get wrong ({int(allw.sum())}): teacher result")
    P("")
    if allw.sum():
        P(f"Teacher top1 {pct(er['teacher'][allw] == 1)}, top5 {pct(er['teacher'][allw] <= 5)} (teacher on all evaluation questions: top1 {pct(er['teacher'] == 1)}, top5 {pct(er['teacher'] <= 5)}).")

    # check 3: teacher disagrees with the reference answer
    wrong_t = [i for i in range(len(ei)) if er["teacher"][i] != 1]
    rng = random.Random(a.seed)
    pick = sorted(rng.sample(wrong_t, min(a.sample, len(wrong_t))))
    with open(a.tsv, "w", encoding="utf-8", newline="") as f:
        f.write("input\tcontext\treference_answer\tteacher_top1\tteacher_top2\tteacher_top3\n")
        for i in pick:
            t = ei[i]
            order = sorted(range(len(t["words"])), key=lambda k: -es["teacher"][i][k])[:3]
            f.write("\t".join([t["key"], t["ctx"].replace("\t", " "), t["ans"]] + [t["words"][k] for k in order]) + "\n")
    P("")
    P("## Check 3: reference answer against the teacher's first choice")
    P("")
    P(f"The teacher's first choice differs from the reference answer in {len(wrong_t)} of {len(ei)} evaluation questions "
      f"({100 * len(wrong_t) / len(ei):.1f}%). {len(pick)} of them were drawn with a fixed seed ({a.seed}) and written to {Path(a.tsv).name} "
      "(input, context, reference answer, teacher top 3); the judgement is left to the reader.")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample"); p.add_argument("--items", action="append", required=True); p.add_argument("--dict-dir", required=True)
    p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=1300); p.add_argument("--seed", type=int, default=11)
    p = sub.add_parser("teacher"); p.add_argument("--cache", required=True); p.add_argument("--data", required=True); p.add_argument("--name", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    for k in ("eval-cache", "eval-dir", "eval-teacher", "train-cache", "train-dir", "train-teacher", "out", "tsv"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--model", action="append", required=True); p.add_argument("--sample", type=int, default=60); p.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    {"sample": cmd_sample, "teacher": cmd_teacher, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
