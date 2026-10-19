"""T10: bigger corpus. Two subcommands.
  wiki     : download wikimedia/wikipedia 20231101.zh shards 1-5 (public corpus, user-authorised; announced in the log), convert to simplified,
             cut sentences with the pipeline's own cleaner, drop sentences that overlap evaluation text (4-gram ratio >= 0.5) and duplicates.
  assemble : training directory = a previous run directory (items, teacher scores, dev set, vocab) with a NEW train_sentences.txt made of the
             old sentences plus the given extra sentence files (each repeated --rep times, default 1).
Usage: python tools/t10_corpus.py wiki --out <txt> [--max 1800000]
       python tools/t10_corpus.py assemble --prev <run3> --out <dir> --extra <txt> [<txt> ...]
"""
import argparse
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DISTILL_ROUND", "2")
os.environ.setdefault("DISTILL_DATA", r"D:\distill-data-r4\t10")
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def grams(s, n=4):
    import re
    s = re.sub(r"[^一-鿿A-Za-z0-9]", "", s)
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def cmd_wiki(a):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from opencc import OpenCC
    import common as C
    from s0_fetch import clean_sentences
    cc = OpenCC("t2s")
    ev = set()
    files = [r"D:\distill-data\eval_sentences.txt"] + [str(p) for d in ("domain", "holdout") for p in sorted((ROOT / "tests" / d).glob("*.txt"))]
    for p in files:
        for l in open(p, encoding="utf-8"):
            if l.strip() and not l.startswith("#"):
                ev |= grams(l.replace("【", "").replace("】", ""))
    rng = random.Random(10)
    seen, rows, dropped = set(), [], 0
    for shard in range(1, 6):
        print(f"downloading wikimedia/wikipedia 20231101.zh shard {shard} (about 230 MB, public corpus)", flush=True)
        path = hf_hub_download("wikimedia/wikipedia", f"20231101.zh/train-0000{shard}-of-00006.parquet", repo_type="dataset")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["text"]):
            if (Path(r"D:\distill-data-r4") / "STOP").exists():
                print("STOP requested", flush=True)
                return
            for text in batch.column("text").to_pylist():
                cands = clean_sentences(cc.convert(text), C.CFG)
                rng.shuffle(cands)
                for s in cands[: C.CFG["per_article"]]:
                    if s in seen:
                        continue
                    g = grams(s)
                    if g and len(g & ev) / len(g) >= 0.5:
                        dropped += 1
                        continue
                    seen.add(s)
                    rows.append(s)
            if len(rows) >= a.max:
                break
        print(f"shard {shard} done: {len(rows)} sentences, {dropped} dropped for overlap with evaluation text", flush=True)
        if len(rows) >= a.max:
            break
    Path(a.out).write_text("\n".join(rows[: a.max]) + "\n", encoding="utf-8")
    print("wrote", min(len(rows), a.max), "sentences")


def cmd_assemble(a):
    prev, out = Path(a.prev), Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    for n in ("items_train.jsonl", "items_dev.jsonl", "meta.json", "gates.json"):
        shutil.copy(prev / n, out / n)
    shutil.copy(prev / "ckpt" / "vocab.json", out / "ckpt" / "vocab.json")
    for n in prev.glob("scores/*.jsonl"):
        shutil.copy(n, out / "scores" / n.name)
    sents = [l.strip() for l in open(prev / "train_sentences.txt", encoding="utf-8") if l.strip()]
    n0 = len(sents)
    for p in a.extra:
        sents += [l.strip() for l in open(p, encoding="utf-8") if l.strip()] * a.rep
    random.Random(2).shuffle(sents)
    (out / "train_sentences.txt").write_text("\n".join(sents) + "\n", encoding="utf-8")
    print(f"sentences {n0} -> {len(sents)} (x{len(sents) / n0:.2f})")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("wiki")
    w.add_argument("--out", required=True)
    w.add_argument("--max", type=int, default=1800000)
    s = sp.add_parser("assemble")
    s.add_argument("--prev", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--extra", nargs="+", required=True)
    s.add_argument("--rep", type=int, default=1)
    a = ap.parse_args()
    {"wiki": cmd_wiki, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    main()
