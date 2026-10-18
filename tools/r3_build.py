"""第三轮数据构造：补全词频表、出领域训练题、出新留出题、混合训练句子。
词频表 = 维基词频（word_counts.pkl，训练句统计）+ jieba 词典频次（按比例缩放，只补维基里没有或很少的词）
        + 合成领域语料词频（缩放到各领域应占的份额）。
用法：python tools/r3_build.py --out <目录> --wiki-counts <word_counts.pkl> --wiki-train <train_sentences.txt>
        --wiki-items <r2 items_train.jsonl> --corpus-dir <含 chat.txt tech.txt anime.txt 的目录> --wiki-n 20000
        [--dom-items chat=10000 tech=6000 anime=4000]"""
import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARK_PARSE = None


def is_cjk(s):
    return bool(s) and all("一" <= c <= "鿿" for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wiki-counts", required=True)
    ap.add_argument("--wiki-train", required=True)
    ap.add_argument("--wiki-items", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--wiki-n", type=int, default=20000)
    ap.add_argument("--dom-items", nargs="*", default=["chat=10000", "tech=6000", "anime=4000"])
    ap.add_argument("--wiki-sent-n", type=int, default=300000)
    ap.add_argument("--mix", default="wiki=70,chat=14,tech=9,anime=7", help="训练句子采样比例（试点）")
    ap.add_argument("--dom-token-share", type=float, default=0.25, help="领域词频合计相对维基词频合计的份额（按混合比例分配）")
    a = ap.parse_args()
    import jieba
    from pypinyin import lazy_pinyin
    jieba.setLogLevel(30)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(3)
    doms = ["chat", "tech", "anime"]
    sents = {d: [l.strip() for l in open(Path(a.corpus_dir) / f"{d}.txt", encoding="utf-8") if l.strip()] for d in doms}

    # ---- 词频表
    wiki = pickle.loads(Path(a.wiki_counts).read_bytes())
    wtot = sum(wiki.values())
    table = Counter({w: float(c) for w, c in wiki.items()})
    jf = {w: f for w, f in jieba.dt.FREQ.items() if 1 < len(w) <= 4 and is_cjk(w) and f > 0}
    jscale = wtot / max(sum(jf.values()), 1) * 0.3
    for w, f in jf.items():
        table[w] += f * jscale
    mix = dict(kv.split("=") for kv in a.mix.split(","))
    mix = {k: float(v) for k, v in mix.items()}
    dom_cnt = {}
    for d in doms:
        c = Counter()
        for s in sents[d]:
            for w in jieba.cut(s, HMM=False):
                if len(w) <= 4 and is_cjk(w):
                    c[w] += 1
        dom_cnt[d] = c
        share = a.dom_token_share * mix[d] / sum(mix[x] for x in doms)
        scale = share * wtot / max(sum(c.values()), 1)
        for w, n in c.items():
            table[w] += n * scale
    pickle.dump(dict(table), open(out / "word_table.pkl", "wb"))
    min_f = 3

    def pykey(w, cache={}):
        if w not in cache:
            p = lazy_pinyin(w)
            cache[w] = " ".join(p) if len(p) == len(w) and all(x.isascii() and x.isalpha() for x in p) else None
        return cache[w]

    by_key = defaultdict(list)
    for w, c in table.items():
        if c >= min_f:
            k = pykey(w)
            if k:
                by_key[k].append((int(round(c)), w))
    for k in by_key:
        by_key[k].sort(key=lambda x: (-x[0], x[1]))
    print("词表", sum(len(v) for v in by_key.values()), "词")

    def item(ctx, w, k, top):
        c = [x[1] for x in top]
        t = c.index(w)
        return dict(ctx=ctx[-30:], py=k, c=c, f=[x[0] for x in top], t=t, s=int(t > 0), L=len(w), word=w)

    # ---- 领域训练题：每个句子的每个 2-4 字词一个候选位置，难例优先（约 3:1）
    train_items = []
    for d in doms:
        n_target = int(dict(kv.split("=") for kv in a.dom_items)[d])
        easy, hard = [], []
        for s in sents[d]:
            pos = 0
            for w in jieba.cut(s, HMM=False):
                ctx = s[:pos]
                pos += len(w)
                if not 2 <= len(w) <= 4 or not is_cjk(w) or len(ctx) < 2:
                    continue
                lst = by_key.get(pykey(w))
                if not lst or len(lst) < 2:
                    continue
                top = lst[:8]
                if w not in [x[1] for x in top]:
                    continue
                it = item(ctx, w, pykey(w), top)
                it["domain"] = d
                (hard if it["s"] else easy).append(it)
        rng.shuffle(easy)
        rng.shuffle(hard)
        per = Counter()
        sel = []
        nh = int(n_target * 0.75)
        for it in hard:
            if per[it["word"]] < 25 and len(sel) < nh:
                per[it["word"]] += 1
                sel.append(it)
        for it in easy:
            if len(sel) >= n_target:
                break
            if per[it["word"]] < 25:
                per[it["word"]] += 1
                sel.append(it)
        print(d, "训练题", len(sel), "其中难例", sum(i["s"] for i in sel), f"（可选 难{len(hard)} 易{len(easy)}）")
        train_items += sel
    # ---- 维基训练题：沿用第二轮的题（同分布），随机取一部分，保持 3:1 难例比例
    w_items = [json.loads(l) for l in open(a.wiki_items, encoding="utf-8")]
    wh = [i for i in w_items if i["s"]]
    we = [i for i in w_items if not i["s"]]
    rng.shuffle(wh)
    rng.shuffle(we)
    sel = wh[: int(a.wiki_n * 0.75)] + we[: a.wiki_n - int(a.wiki_n * 0.75)]
    for it in sel:
        it["domain"] = "wiki"
    train_items += sel
    rng.shuffle(train_items)
    for i, it in enumerate(train_items):
        it["id"] = i
    with open(out / "items_train.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in train_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # ---- 混合训练句子（试点：按比例重复领域句子）
    wiki_s = []
    with open(a.wiki_train, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if l.strip():
                wiki_s.append(l.strip())
            if len(wiki_s) >= a.wiki_sent_n:
                break
    mixed = list(wiki_s)
    for d in doms:
        want = int(len(wiki_s) * mix[d] / mix["wiki"])
        reps = [sents[d][i % len(sents[d])] for i in range(want)]
        mixed += reps
    rng.shuffle(mixed)
    (out / "train_sentences.txt").write_text("\n".join(mixed) + "\n", encoding="utf-8")
    print("训练句子", len(mixed), "其中领域句重复倍数", {d: round(int(len(wiki_s) * mix[d] / mix["wiki"]) / len(sents[d]), 1) for d in doms})

    # ---- 新留出题（tests/holdout，用补全后的词频表）
    import re
    mk = re.compile(r"【([^】]+)】")
    hold, cover = [], defaultdict(Counter)
    for d in doms:
        for line in (ROOT / "tests" / "holdout" / f"{d}.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            text, targets, pos, last = [], [], 0, 0
            for m in mk.finditer(line):
                text.append(line[last:m.start()])
                pos += len(line[last:m.start()])
                targets.append((m.group(1), pos))
                text.append(m.group(1))
                pos += len(m.group(1))
                last = m.end()
            text.append(line[last:])
            sent = "".join(text)
            for w, p in targets:
                cover[d]["标记词"] += 1
                k = pykey(w) if is_cjk(w) else None
                lst = by_key.get(k, []) if k else []
                if w not in [x[1] for x in lst]:
                    cover[d]["不在词频表"] += 1
                    continue
                if len(lst) < 2:
                    cover[d]["无同音词"] += 1
                    continue
                top = lst[:8]
                if w not in [x[1] for x in top]:
                    cover[d]["排名超出前8"] += 1
                    continue
                it = item(sent[:p], w, k, top)
                it["domain"] = d
                hold.append(it)
                cover[d]["出题"] += 1
    for i, it in enumerate(hold):
        it["id"] = i
    with open(out / "items_holdout.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in hold:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    for d in doms:
        its = [i for i in hold if i["domain"] == d]
        print("留出", d, len(its), "题，难例", sum(i["s"] for i in its), dict(cover[d]))


if __name__ == "__main__":
    main()
