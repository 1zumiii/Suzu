"""长上文训练题与容量实验的训练目录。上文下限 10 个汉字、上限 64 个字，可以跨句、跨标点，保留标点。
题型同臂 C（tools/t_abbrev_train.py）：两字母简拼 70%、首字全拼加次字首字母 30%，候选取法同臂 C 的两批（40% 为“答案 + 权重前 24 + 随机 23”，
60% 为“答案 + 学生分前 20 + 权重前 14 + 随机 13”）；另加同样数量的全拼题（取法同 tools/r3_build.py：结巴分词的 2 到 4 字词，
同音词按词频表取前 8 个，难例约 75%）。泄漏检查：与评测句、留出题、实机用例共享 12 字片段的文字整段丢弃。
所有路径走参数。

  wiki   --parquet SHARD.parquet --exclude EVAL_SENTENCES.txt ... --out wiki_passages.txt [--n 40000 --seed 3]
      维基长窗口：按段落取连续文字（繁转简，只保留汉字与常用中文标点的连续片段，40 到 160 字），每篇最多 3 段
  items  --passages DOMAIN=FILE ... --dict-dir CN_DICTS --word-table word_table.pkl --student-ckpt S.pt --out DIR
         [--abbrev wiki=30000,chat=10000,tech=10000,anime=10000] [--full （同上）] [--exclude FILE ...] [--holdout-dir tests/holdout] [--cases cases.tsv]
      写 DIR/items_long.jsonl（id 从 30000000 起），每段最多 2 道简拼题、2 道全拼题
  mix    --base ARM_DIR --long DIR --long-scores long__q35_4b_base.jsonl --out DIR2 --ctx-max 30|64 [--seed 11]
      长上文题（上文截到最后 ctx-max 个字）与 ARM_DIR 的训练题按 1 比 1 混合（从 ARM_DIR 按原比例随机抽同样多的题），其余文件照抄 ARM_DIR
  mbase  --set DIR[@ITEMS=SCORES,ITEMS=SCORES] ... --dev-from DIR --out DIR2
      从零训练用的合并目录：各目录的训练句子去重合并，训练题重新编号后合并（同一上文、同一答案、同一候选的题只留后出现的一份，
      即较新的老师标注），老师分数写成 scores/mix__train.jsonl
"""
import argparse
import json
import pickle
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HAN = re.compile(r"[一-鿿]")
CJK2 = re.compile(r"^[一-鿿]{2}$")
KEEP = re.compile(r"[一-鿿，、：；“”‘’（）《》…—·。！？]+")
ID0 = 30_000_000
MAXCTX = 64


def han(s):
    return len(HAN.findall(s))


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def shingle_set(texts, n=12):
    s = set()
    for t in texts:
        for j in range(max(1, len(t) - n + 1)):
            s.add(t[j:j + n])
    return s


def leaked(t, sh, n=12):
    return any(t[j:j + n] in sh for j in range(max(1, len(t) - n + 1)))


def exclusion(files, holdout_dir=None, cases=None):
    texts = []
    for f in files:
        texts += [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
    if holdout_dir:
        for f in Path(holdout_dir).glob("*.txt"):
            texts += [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
    if cases:
        for l in open(cases, encoding="utf-8"):
            f = l.rstrip("\n").split("\t")
            if len(f) >= 3:
                texts.append(f[2] + (f[3].split("|")[0] if len(f) > 3 else ""))
    return shingle_set(texts)


def cmd_wiki(a):
    import pyarrow.parquet as pq
    from opencc import OpenCC
    cc = OpenCC("t2s")
    sh = exclusion(a.exclude)
    rng = random.Random(a.seed)
    out, stat = [], Counter()
    pf = pq.ParquetFile(a.parquet)
    for batch in pf.iter_batches(batch_size=256, columns=["text"]):
        arts = batch.column("text").to_pylist()
        rng.shuffle(arts)
        for text in arts:
            if rng.random() > a.take:
                continue
            got = 0
            paras = [p for p in cc.convert(text).split("\n") if p.strip()]
            rng.shuffle(paras)
            for p in paras:
                if got >= 3:
                    break
                for m in KEEP.findall(p):
                    if len(m) < 40 or han(m) < 30:
                        continue
                    m = m[:160]
                    stat["candidates"] += 1
                    if leaked(m, sh):
                        stat["leaked"] += 1
                        continue
                    out.append(m)
                    got += 1
                    break
        if len(out) >= a.n:
            break
    out = out[:a.n]
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wiki passages {len(out)}; candidates {stat['candidates']}, dropped (shared 12-char piece with the evaluation text) {stat['leaked']}; "
          f"median length {sorted(len(x) for x in out)[len(out) // 2]}")


def load_dict(d):
    words = defaultdict(list)
    pa, pm = defaultdict(dict), defaultdict(dict)
    for name in ("base", "ext"):
        body = False
        for ln in open(Path(d) / f"{name}.dict.yaml", encoding="utf-8"):
            if ln.startswith("..."):
                body = True
                continue
            if not body or ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 2 or not CJK2.match(p[0]):
                continue
            sy = p[1].split()
            if len(sy) != 2:
                continue
            w = int(p[2]) if len(p) > 2 and p[2].isdigit() else 0
            ka, km = sy[0][0] + sy[1][0], sy[0] + " " + sy[1][0]
            words[p[0]].append((w, ka, km))
            pa[ka][p[0]] = max(w, pa[ka].get(p[0], 0))
            pm[km][p[0]] = max(w, pm[km].get(p[0], 0))
    return words, pa, pm


def parse_quota(s):
    return {k: int(v) for k, v in (x.split("=") for x in s.split(","))}


def cmd_items(a):
    import jieba
    from pypinyin import lazy_pinyin
    jieba.setLogLevel(60)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    words, pa, pm = load_dict(a.dict_dir)
    sh = exclusion(a.exclude, a.holdout_dir, a.cases)
    table = pickle.loads(Path(a.word_table).read_bytes())
    pycache = {}

    def pykey(w):
        if w not in pycache:
            p = lazy_pinyin(w)
            pycache[w] = " ".join(p) if len(p) == len(w) and all(x.isascii() and x.isalpha() for x in p) else None
        return pycache[w]

    by_key = defaultdict(list)
    for w, c in table.items():
        if c >= 3:
            k = pykey(w)
            if k:
                by_key[k].append((int(round(c)), w))
    for k in by_key:
        by_key[k].sort(key=lambda x: (-x[0], x[1]))
    qa, qf = parse_quota(a.abbrev), parse_quota(a.full)
    texts = {}
    for spec in a.passages:
        dom, _, f = spec.partition("=")
        ts = [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
        n0 = len(ts)
        ts = [t for t in ts if not leaked(t, sh)]
        rng.shuffle(ts)
        texts[dom] = ts
        print(f"{dom}: {n0} passages, {n0 - len(ts)} dropped by the leak check", flush=True)

    recs, full_items = [], []
    per_a, per_f = defaultdict(int), defaultdict(int)
    for dom, ts in texts.items():
        ga = 0
        hard, easy = [], []
        nh = int(qf.get(dom, 0) * 0.75)
        for t in ts:
            # abbreviation positions
            if ga < qa.get(dom, 0):
                pos = [i for i in range(1, len(t) - 1) if t[i:i + 2] in words and han(t[max(0, i - MAXCTX):i]) >= 10]
                rng.shuffle(pos)
                took = 0
                for i in pos:
                    if took >= 2 or ga >= qa[dom]:
                        break
                    w = t[i:i + 2]
                    if per_a[w] >= 60:
                        continue
                    mode = "abbr" if rng.random() < 0.7 else "mix"
                    best = max(words[w])
                    key = best[1] if mode == "abbr" else best[2]
                    pool = (pa if mode == "abbr" else pm)[key]
                    if len(pool) < 6 or w not in pool:
                        continue
                    recs.append({"ctx": t[max(0, i - MAXCTX):i], "w": w, "key": key, "mode": mode, "dom": dom, "pool": pool,
                                 "style": "hard" if rng.random() < 0.6 else "top"})
                    per_a[w] += 1
                    ga += 1
                    took += 1
            # full-pinyin positions (r3_build recipe, context at least 10 Chinese characters)
            if len(hard) < 1.3 * nh or len(easy) < 1.3 * (qf.get(dom, 0) - nh):   # some spare for the per-word cap
                pos, cur = [], 0
                for w in jieba.cut(t, HMM=False):
                    ctx = t[max(0, cur - MAXCTX):cur]
                    cur += len(w)
                    if not 2 <= len(w) <= 4 or not all("一" <= c <= "鿿" for c in w) or han(ctx) < 10:
                        continue
                    k = pykey(w)
                    lst = by_key.get(k) if k else None
                    if not lst or len(lst) < 2:
                        continue
                    top = lst[:8]
                    c = [x[1] for x in top]
                    if w not in c:
                        continue
                    ti = c.index(w)
                    pos.append(dict(ctx=ctx, py=k, c=c, f=[x[0] for x in top], t=ti, s=int(ti > 0), L=len(w), word=w, domain=dom))
                rng.shuffle(pos)
                hp = [p for p in pos if p["s"]][:2]
                ep = [p for p in pos if not p["s"]][:2]
                hard += hp
                easy += ep
        sel = []
        for it in hard:
            if per_f[it["word"]] < 25 and len(sel) < nh:
                per_f[it["word"]] += 1
                sel.append(it)
        for it in easy:
            if len(sel) >= qf.get(dom, 0):
                break
            if per_f[it["word"]] < 25:
                per_f[it["word"]] += 1
                sel.append(it)
        full_items += sel
        print(f"{dom}: abbreviation {ga}/{qa.get(dom, 0)}, full pinyin {len(sel)}/{qf.get(dom, 0)} (hard {sum(i['s'] for i in sel)})", flush=True)

    # student hard candidates for the 60% "hard" style abbreviation items (whole pool scored by the student)
    sys.path.insert(0, str(ROOT / "pipeline"))
    import torch
    from student_model import load_student, score_items
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_student(a.student_ckpt, dev)
    chunk_items, owner = [], []
    for n, r in enumerate(recs):
        ws = sorted(r["pool"], key=lambda x: -r["pool"][x])
        r["ranked"] = ws
        if r["style"] == "hard":
            for k in range(0, len(ws), 100):
                chunk_items.append({"ctx": r["ctx"], "c": ws[k:k + 100]})
                owner.append((n, k))
    scores = [dict() for _ in recs]
    with torch.no_grad():
        for a0 in range(0, len(chunk_items), 64):
            part = chunk_items[a0:a0 + 64]
            for (n, _), it, sc in zip(owner[a0:a0 + 64], part, score_items(model, vocab, part, dev)):
                for wd, v in zip(it["c"], sc):
                    scores[n][wd] = v
            if (a0 // 64) % 200 == 0:
                print(f"  student scoring {a0}/{len(chunk_items)} chunks", flush=True)
    items, nid = [], ID0
    for r, sc in zip(recs, scores):
        w, pool = r["w"], r["pool"]
        rest = [x for x in r["ranked"] if x != w]
        if r["style"] == "hard":
            hard = sorted(rest, key=lambda x: -sc.get(x, -1e9))[:20]
            taken = set(hard)
            top = [x for x in rest if x not in taken][:14]
            taken |= set(top)
            left = [x for x in rest if x not in taken]
            cands = set(hard + top + rng.sample(left, min(13, len(left))) + [w])
        else:
            top = rest[:24]
            cands = set(top + (rng.sample(rest[24:], min(23, len(rest) - 24)) if len(rest) > 24 else []) + [w])
        cands = sorted(cands, key=lambda x: -pool[x])
        items.append({"id": nid, "ctx": r["ctx"], "py": r["key"], "c": cands, "f": [pool[x] for x in cands], "t": cands.index(w),
                      "s": int(cands.index(w) > 0), "L": 2, "word": w, "domain": r["dom"], "mode": r["mode"], "kind": "abbrev_" + r["style"]})
        nid += 1
    for it in full_items:
        it["id"] = nid
        it["kind"] = "full"
        nid += 1
    items += full_items
    rng.shuffle(items)
    with open(out / "items_long.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    ab = [i for i in items if i["kind"] != "full"]
    fu = [i for i in items if i["kind"] == "full"]
    lens = sorted(han(i["ctx"]) for i in items)
    print(f"items {len(items)}: abbreviation {len(ab)} (two-letter {sum(i['mode'] == 'abbr' for i in ab)}, mixed {sum(i['mode'] == 'mix' for i in ab)}, "
          f"student-hard style {sum(i['kind'] == 'abbrev_hard' for i in ab)}; candidates mean {sum(len(i['c']) for i in ab) / max(len(ab), 1):.1f}), "
          f"full pinyin {len(fu)} (hard {sum(i['s'] for i in fu)}); context Chinese characters median {lens[len(lens) // 2]}, min {lens[0]}, max {lens[-1]}")
    print("by domain: " + ", ".join(f"{d} {sum(1 for i in items if i['domain'] == d)}" for d in texts))


def cmd_mix(a):
    base, out = Path(a.base), Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    long_items = rj(Path(a.long) / "items_long.jsonl")
    lsc = {r["id"]: r for r in rj(Path(a.long) / "scores" / a.long_scores)}
    long_items = [it for it in long_items if it["id"] in lsc]
    for it in long_items:
        it["ctx"] = it["ctx"][-a.ctx_max:]
    old = rj(base / "items_train.jsonl")
    osc = {r["id"]: r for r in rj(base / "scores" / f"{a.teacher}__train.jsonl")}
    old = [it for it in old if it["id"] in osc]
    rep = rng.sample(old, min(len(long_items), len(old)))
    items = long_items + rep
    rng.shuffle(items)
    with open(out / "items_train.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out / "scores" / f"{a.teacher}__train.jsonl", "w", encoding="utf-8") as f:
        for it in long_items:
            f.write(json.dumps(lsc[it["id"]], ensure_ascii=False) + "\n")
        for it in rep:
            f.write(json.dumps(osc[it["id"]], ensure_ascii=False) + "\n")
    for n in ("items_dev.jsonl", "meta.json", "gates.json", "train_sentences.txt"):
        if not (out / n).exists():
            shutil.copy(base / n, out / n)
    shutil.copy(base / "ckpt" / "vocab.json", out / "ckpt" / "vocab.json")
    short = sum(1 for it in items if han(it["ctx"]) < 4)
    print(f"{out.name}: long-context items {len(long_items)} (context cut to {a.ctx_max}) + original items {len(rep)} "
          f"(abbreviation {sum(1 for i in rep if i['id'] >= 10_000_000)}) = {len(items)}; context under 4 Chinese characters {short}")


def cmd_mbase(a):
    out = Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    seen_s, n_s = set(), 0
    with open(out / "train_sentences.txt", "w", encoding="utf-8") as f:
        for spec in a.set:
            d = Path(spec.split("@")[0])
            if not (d / "train_sentences.txt").exists():
                continue
            for l in open(d / "train_sentences.txt", encoding="utf-8"):
                s = l.strip()
                if s and s not in seen_s:
                    seen_s.add(s)
                    f.write(s + "\n")
                    n_s += 1
    merged = {}
    nid = total = 0
    for spec in a.set:
        d, _, rest = spec.partition("@")
        d = Path(d)
        files = rest.split(",") if rest else [None]
        for sp in files:
            items_f, scores_f = (sp.split("=") if sp else ("items_train.jsonl", None))
            its = rj(d / items_f)
            sc = {r["id"]: r["s"] for r in rj(d / "scores" / (scores_f or "q35_4b_base__train.jsonl"))}
            got = 0
            for it in its:
                if it["id"] not in sc:
                    continue
                k = (it["ctx"], it["py"], tuple(it["c"]))
                it = dict(it, src=f"{d.name}/{items_f}", src_id=it["id"])
                merged.pop(k, None)
                merged[k] = (it, sc[it["src_id"]])
                got += 1
            total += got
            print(f"{d.name}/{items_f}: {got} items with teacher scores", flush=True)
    with open(out / "items_train.jsonl", "w", encoding="utf-8") as fi, open(out / "scores" / "mix__train.jsonl", "w", encoding="utf-8") as fs:
        for it, s in merged.values():
            it["id"] = nid
            fi.write(json.dumps(it, ensure_ascii=False) + "\n")
            fs.write(json.dumps({"id": nid, "s": s}) + "\n")
            nid += 1
    dev = Path(a.dev_from)
    for n in ("items_dev.jsonl", "meta.json", "gates.json"):
        shutil.copy(dev / n, out / n)
    shutil.copy(dev / "ckpt" / "vocab.json", out / "ckpt" / "vocab.json")
    print(f"merged: sentences {n_s}, items {nid} (duplicates removed {total - nid})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("wiki"); p.add_argument("--parquet", required=True); p.add_argument("--exclude", nargs="+", required=True)
    p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=40000); p.add_argument("--seed", type=int, default=3)
    p.add_argument("--take", type=float, default=0.5, help="每篇文章被选中的概率（打散来源）")
    p = sub.add_parser("items"); p.add_argument("--passages", nargs="+", required=True); p.add_argument("--dict-dir", required=True)
    p.add_argument("--word-table", required=True); p.add_argument("--student-ckpt", required=True); p.add_argument("--out", required=True)
    p.add_argument("--abbrev", default="wiki=30000,chat=10000,tech=10000,anime=10000"); p.add_argument("--full", default="wiki=30000,chat=10000,tech=10000,anime=10000")
    p.add_argument("--exclude", nargs="*", default=[]); p.add_argument("--holdout-dir", default=str(ROOT / "tests" / "holdout")); p.add_argument("--cases")
    p.add_argument("--seed", type=int, default=23)
    p = sub.add_parser("mix"); p.add_argument("--base", required=True); p.add_argument("--long", required=True); p.add_argument("--long-scores", required=True)
    p.add_argument("--out", required=True); p.add_argument("--ctx-max", type=int, required=True); p.add_argument("--seed", type=int, default=11)
    p.add_argument("--teacher", default="q35_4b_base")
    p = sub.add_parser("mbase"); p.add_argument("--set", action="append", required=True, help="DIR 或 DIR@ITEMS=SCORES[,ITEMS=SCORES]（文件相对 DIR 与 DIR/scores）")
    p.add_argument("--dev-from", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"wiki": cmd_wiki, "items": cmd_items, "mix": cmd_mix, "mbase": cmd_mbase}[a.cmd](a)


if __name__ == "__main__":
    main()
