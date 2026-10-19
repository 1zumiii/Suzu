"""Weak-spot plan stage D / E step 3: assemble a continued-training run directory.

Inputs: the previous run directory (items, teacher scores, dev set, vocab), the slang items and their teacher scores,
the domain corpora, the hard-example id list. Output layout is what pipeline/s4_train.py expects:
  items_train.jsonl, scores/<teacher>__train.jsonl, train_sentences.txt, items_dev.jsonl, meta.json, gates.json, ckpt/vocab.json
Hard items appear twice in items_train.jsonl (same id, so the teacher score is shared) = double sampling.
Sentence mix defaults to wiki 70, chat 10, tech 5, anime 5, slang 6 (the remaining 4 of the plan's 100 is the hard-item doubling).

Usage: python tools/r4_build_run.py --prev <run dir> --out <dir> --slang-items <jsonl> --slang-scores <jsonl> --slang-sents <txt>
        --corpus-dir <dir with chat.txt tech.txt anime.txt> --wiki-train <txt> [--hard <json> ...] [--extra-items <jsonl> --extra-scores <jsonl> --extra-sents <txt>]
"""
import argparse
import json
import random
import shutil
from pathlib import Path


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def lines(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="q35_4b_base")
    ap.add_argument("--slang-items", nargs="*", default=[])
    ap.add_argument("--slang-scores", nargs="*", default=[])
    ap.add_argument("--slang-sents", nargs="*", default=[])
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--wiki-train", required=True)
    ap.add_argument("--wiki-n", type=int, default=700000)
    ap.add_argument("--mix", default="wiki=70,chat=10,tech=5,anime=5,slang=6")
    ap.add_argument("--hard", nargs="*", default=[])
    a = ap.parse_args()
    prev, out = Path(a.prev), Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    for n in ("items_dev.jsonl", "meta.json", "gates.json"):
        shutil.copy(prev / n, out / n)
    if not (out / "ckpt" / "vocab.json").exists():
        shutil.copy(prev / "ckpt" / "vocab.json", out / "ckpt" / "vocab.json")

    items = rj(prev / "items_train.jsonl")
    for p in a.slang_items:
        items += rj(p)
    scores = {}
    for p in [prev / "scores" / f"{a.teacher}__train.jsonl"] + [Path(x) for x in a.slang_scores]:
        for r in rj(p):
            scores[r["id"]] = r
    hard = set()
    for p in a.hard:
        hard |= set(json.load(open(p)))
    dup = [it for it in items if it["id"] in hard]
    items = [it for it in items if it["id"] in scores]
    dup = [it for it in dup if it["id"] in scores]
    rng = random.Random(11)
    allit = items + dup
    rng.shuffle(allit)
    with open(out / "items_train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in allit:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out / "scores" / f"{a.teacher}__train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in scores.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mix = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
    wiki = []
    for l in open(a.wiki_train, encoding="utf-8"):
        if l.strip():
            wiki.append(l.strip())
        if len(wiki) >= a.wiki_n:
            break
    sents = list(wiki)
    pools = {d: lines(Path(a.corpus_dir) / f"{d}.txt") for d in ("chat", "tech", "anime")}
    pools["slang"] = [s for p in a.slang_sents for s in lines(p)]
    reps = {}
    for d, pool in pools.items():
        want = int(len(wiki) * mix.get(d, 0) / mix["wiki"])
        if want and pool:
            sents += [pool[i % len(pool)] for i in range(want)]
            reps[d] = round(want / len(pool), 1)
    rng.shuffle(sents)
    (out / "train_sentences.txt").write_text("\n".join(sents) + "\n", encoding="utf-8")
    print(json.dumps(dict(items=len(allit), hard_duplicated=len(dup), sentences=len(sents), repeat_factor=reps,
                          items_by_domain={d: sum(1 for i in allit if i["domain"] == d) for d in sorted({i["domain"] for i in allit})}),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
