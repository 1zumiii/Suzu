"""阶段 1：构造评测题和训练题 → data/items_eval.jsonl、items_train.jsonl、meta.json。

一道题 = 在上文 ctx 之后输入一个词的拼音 py，候选是同拼音、同语料词频排序的词（最多 k_cands 个），
正确答案 t 是语料里真实出现的那个词。词频只用训练句统计，评测句不参与，避免泄漏。
候选按词频降序排列，所以 t = 0 表示「词频第一就是对的」（易例），t > 0 是需要上下文才能选对的难例。
"""
import random
from collections import Counter, defaultdict
from functools import lru_cache

import common as C
from common import CFG, DATA, is_cjk, log


def load_nlp():
    import jieba
    from pypinyin import lazy_pinyin
    if hasattr(jieba, "setLogLevel"):
        jieba.setLogLevel(30)
    return jieba, lazy_pinyin


def run():
    jieba, lazy_pinyin = load_nlp()
    sents = [l.rstrip("\n") for l in open(DATA / "sentences.txt", encoding="utf-8") if l.strip()]
    train, ev, dev = [], [], []
    for s in sents:
        if C.md5_mod(s, CFG["eval_mod"]) == 0:
            # 评测句再对半分：开发集只用于挑检查点，评测集只用于最终报告，互不相交
            (dev if C.md5_mod("d" + s, 2) == 0 else ev).append(s)
        else:
            train.append(s)
    C.atomic_write(DATA / "train_sentences.txt", "\n".join(train) + "\n")
    C.atomic_write(DATA / "eval_sentences.txt", "\n".join(ev) + "\n")
    log(f"句子 {len(sents)}：训练 {len(train)}，评测 {len(ev)}，开发 {len(dev)}")

    @lru_cache(maxsize=None)
    def pykey(w):
        p = lazy_pinyin(w)
        if len(p) == len(w) and all(x.isascii() and x.isalpha() for x in p):
            return " ".join(p)
        return None

    cnt = Counter()
    for i, s in enumerate(train):
        if i % 20000 == 0:
            C.check_stop()
        for w in jieba.cut(s, HMM=False):
            if len(w) <= 4 and is_cjk(w):
                cnt[w] += 1
    by_key = defaultdict(list)
    for w, c in cnt.items():
        if c >= CFG["min_count"]:
            k = pykey(w)
            if k:
                by_key[k].append((c, w))
    for k in by_key:
        by_key[k].sort(key=lambda x: (-x[0], x[1]))
    log(f"词表 {sum(len(v) for v in by_key.values())} 词，{len(by_key)} 个拼音")

    K = CFG["k_cands"]

    def sites(sentences):
        for si, s in enumerate(sentences):
            if si % 20000 == 0:
                C.check_stop()
            pos = 0
            for w in jieba.cut(s, HMM=False):
                ctx = s[:pos]
                pos += len(w)
                if not (CFG["min_word_len"] <= len(w) <= 4) or not is_cjk(w) or len(ctx) < CFG["min_ctx"]:
                    continue
                lst = by_key.get(pykey(w))
                if not lst or len(lst) < 2:
                    continue
                top = lst[:K]
                ws = [x[1] for x in top]
                if w not in ws:
                    continue
                yield ctx[-CFG["ctx_chars"]:], w, pykey(w), ws, [x[0] for x in top]

    rng = random.Random(1)

    def pick(recs, quota):
        rng.shuffle(recs)
        per, taken = Counter(), []
        for r in recs:
            if per[r[1]] >= CFG["max_per_truth"]:
                continue
            per[r[1]] += 1
            taken.append(r)
            if len(taken) >= quota:
                break
        return taken

    def four_sites(sentences):
        """四字输入：相邻两个两字词合成一题，候选为两个词各取前 3 个同音词的组合（按词频积降序）。"""
        for si, s in enumerate(sentences):
            if si % 20000 == 0:
                C.check_stop()
            ws = list(jieba.cut(s, HMM=False))
            pos = 0
            for k in range(len(ws) - 1):
                a, b = ws[k], ws[k + 1]
                ctx = s[:pos]
                pos += len(a)
                if len(a) != 2 or len(b) != 2 or not is_cjk(a + b) or len(ctx) < CFG["min_ctx"]:
                    continue
                la, lb = by_key.get(pykey(a)), by_key.get(pykey(b))
                if not la or not lb or len(la) < 2 or len(lb) < 2:
                    continue
                combos = sorted(((x[0] * y[0], x[1] + y[1]) for x in la[:3] for y in lb[:3]), key=lambda z: (-z[0], z[1]))
                cands = [c[1] for c in combos]
                if a + b not in cands:
                    continue
                yield ctx[-CFG["ctx_chars"]:], a + b, pykey(a) + " " + pykey(b), cands, [c[0] for c in combos]

    def make(sentences, per_stratum, path, gen=None):
        strata, natural = ([], []), [0, 0]
        for ctx, w, k, ws, fs in (gen or sites)(sentences):
            t = ws.index(w)
            st = 1 if t > 0 else 0
            natural[st] += 1
            strata[st].append((ctx, w, k, ws, fs))
        pe, ph = per_stratum if isinstance(per_stratum, tuple) else (per_stratum, per_stratum)
        rows = pick(strata[0], pe) + pick(strata[1], ph)
        rng.shuffle(rows)
        items = [dict(id=i, ctx=r[0], py=r[2], c=r[3], f=r[4], t=r[3].index(r[1]), s=1 if r[3].index(r[1]) > 0 else 0, L=len(r[1]))
                 for i, r in enumerate(rows)]
        C.write_jsonl(path, items)
        return natural, [sum(1 for it in items if it["s"] == s) for s in (0, 1)]

    if not C.ROUND2:   # 第二轮沿用第一轮的评测题和开发题，保证前后可比
        nat_e, samp_e = make(ev, CFG["eval_per_stratum"], DATA / "items_eval.jsonl")
        log(f"评测题：易例 {samp_e[0]}、难例 {samp_e[1]}（评测句里自然分布 {nat_e[0]} : {nat_e[1]}）")
        make(dev, CFG["dev_per_stratum"], DATA / "items_dev.jsonl")
    half = CFG["train_items"] // 2
    nat_t, samp_t = make(train[: CFG["train_scan"]], (CFG.get("train_easy", half), CFG.get("train_hard", half)),
                         DATA / "items_train.jsonl")
    log(f"训练题：易例 {samp_t[0]}、难例 {samp_t[1]}")
    meta = dict(train_sampled=samp_t, n_sentences=len(sents), n_train=len(train), n_eval=len(ev))
    if C.ROUND2:
        _, samp_f = make(ev[:300_000], CFG["four_per_stratum"], DATA / "items_four.jsonl", gen=four_sites)
        log(f"四字评测题：易例 {samp_f[0]}、难例 {samp_f[1]}")
        meta["four_sampled"] = samp_f
    else:
        meta.update(eval_natural=nat_e, eval_sampled=samp_e)
    C.write_json(DATA / ("meta_r2.json" if C.ROUND2 else "meta.json"), meta)
