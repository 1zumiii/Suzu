"""分数层组合的离线实验：在已有的逐题打分文件上，对每个候选取两个模型得分的加权和，再叠加词频融合，
看组合能否优于任何单个模型。不需要重新跑任何模型，不含个人数据。

对每套题（维基、技术、二次元、日常口语）：
  - 为避免在评测集上调权后数字偏乐观，把题目按难易分层后对半切开，在一半上选权重（及词频融合权重），在另一半上报告；
    两个方向各做一次，合并成每题的“样本外”正误，再计算准确率与同题配对的置信区间。
  - 单个模型也用同样的办法选词频融合权重，保证比较公平。
  - 另用“留一个领域”检验默认权重是否跨领域稳妥：在三套题上选权重，在第四套题上检验。
用法：
  python tools/score_combine.py --wiki-items <items_eval.jsonl> --domain-items <items_domain.jsonl> \\
      --wiki 系统名=打分.jsonl ... --domain-dir <领域打分目录> [--pair A,B ...]
系统名：student（Suzu-2）、q35_0p8b、q35_4b；领域打分目录下是 <系统名>.jsonl。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

WS = np.round(np.linspace(0, 1, 11), 2)
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0]
NAME = {"student": "Suzu-2", "q35_0p8b": "Qwen3.5-0.8B", "q35_4b": "Qwen3.5-4B"}
DOMS = [("wiki", "维基"), ("tech", "技术"), ("anime", "二次元"), ("chat", "日常口语")]
PAD = -1e9


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def to_arrays(items, scores, K=8):
    n = len(items)
    S = np.full((n, K), PAD)
    LF = np.full((n, K), PAD)
    for r, it in enumerate(items):
        s = scores[it["id"]]
        S[r, :len(s)] = s
        LF[r, :len(it["f"])] = np.log1p(it["f"])
    return S, LF, np.array([it["t"] for it in items]), np.array([it["s"] for it in items])


def correct(SA, SB, LF, t, w, lam):
    v = w * SA + (1 - w) * SB + lam * LF
    v[(SA <= PAD / 2)] = PAD
    return (v.argmax(1) == t).astype(float)


def grid_acc(SA, SB, LF, t, idx, single=None):
    """返回 {(w, lam): 在 idx 上的准确率}。single 为 'A' 或 'B' 时只用该模型（w 固定）。"""
    ws = [1.0] if single == "A" else [0.0] if single == "B" else WS
    return {(w, lam): correct(SA[idx], SB[idx], LF[idx], t[idx], w, lam).mean() for w in ws for lam in LAMS}


def crossfit(SA, SB, LF, t, strat, single=None, seed=0):
    """对半切（按 strat 分层）、两个方向各选一次参数，返回每题的样本外正误与各折选出的参数。"""
    rng = np.random.RandomState(seed)
    fold = np.zeros(len(t), dtype=int)
    for g in (0, 1):
        ii = np.where(strat == g)[0]
        fold[ii] = (rng.rand(len(ii)) < 0.5).astype(int)
    out, chosen = np.zeros(len(t)), []
    for f in (0, 1):
        sel, rep = np.where(fold == f)[0], np.where(fold != f)[0]
        res = grid_acc(SA, SB, LF, t, sel, single)
        best = max(res, key=lambda k: (res[k], -abs(k[0] - 0.5)))
        chosen.append(best)
        out[rep] = correct(SA[rep], SB[rep], LF[rep], t[rep], best[0], best[1])
    return out, chosen


def ci(d):
    se = math.sqrt(d.var(ddof=1) / len(d)) if len(d) > 1 else 0.0
    return 1.96 * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki-items", required=True)
    ap.add_argument("--domain-items", required=True)
    ap.add_argument("--wiki", action="append", default=[], help="系统名=维基打分文件")
    ap.add_argument("--domain-dir", required=True)
    ap.add_argument("--pair", action="append", default=[], help="A,B，默认 student,q35_0p8b 与 student,q35_4b 与 q35_0p8b,q35_4b")
    ap.add_argument("--out")
    a = ap.parse_args()
    pairs = [tuple(p.split(",")) for p in a.pair] or [("student", "q35_0p8b"), ("student", "q35_4b"), ("q35_0p8b", "q35_4b")]
    wiki_sc = {}
    for spec in a.wiki:
        k, p = spec.split("=", 1)
        wiki_sc[k] = {r["id"]: r["s"] for r in load_jsonl(p)}
    dom_items = [i for i in load_jsonl(a.domain_items) if i["L"] >= 2]
    wiki_items = load_jsonl(a.wiki_items)
    data = {}
    for key, _ in DOMS:
        if key == "wiki":
            items, get = wiki_items, lambda s, w=wiki_sc: w[s]
        else:
            items = [i for i in dom_items if i["domain"] == key]
            get = lambda s, key=key: {r["id"]: r["s"] for r in load_jsonl(Path(a.domain_dir) / "scores" / f"{s}.jsonl")}
        data[key] = (items, get)
    cache = {}

    def arrays(key, A, B):
        items, get = data[key]
        sc = {}
        for s in (A, B):
            if (key, s) not in cache:
                cache[(key, s)] = get(s) if key != "wiki" else wiki_sc[s]
            sc[s] = cache[(key, s)]
        both = [i for i in items if i["id"] in sc[A] and i["id"] in sc[B]]
        SA, LF, t, st = to_arrays(both, sc[A])
        SB, _, _, _ = to_arrays(both, sc[B])
        return SA, SB, LF, t, st

    out = ["# 分数层组合的离线实验", "",
           "每套题按难易分层后对半切开，在一半上选权重（模型权重 w 与词频融合 λ），在另一半上报告，两个方向各一次，合并成每题的样本外正误。"
           f"w 取 {[float(w) for w in WS]}，λ 取 {LAMS}；融合得分 = w·A 得分 + (1-w)·B 得分 + λ·ln(1+词频)。单个模型用同样办法只选 λ。"
           "置信区间为同题配对的 95% 区间。", ""]
    for A, B in pairs:
        out += [f"## {NAME[A]} + {NAME[B]}", "",
                f"| 题集 | 题数 | {NAME[A]} 单独 | {NAME[B]} 单独 | 组合 | 组合相对较好者 | 各折选出的 w（对 A） | 全集最优 w |",
                "|---|---|---|---|---|---|---|---|"]
        per_dom, curves = {}, {}
        for key, label in DOMS:
            SA, SB, LF, t, st = arrays(key, A, B)
            ca, _ = crossfit(SA, SB, LF, t, st, single="A")
            cb, _ = crossfit(SA, SB, LF, t, st, single="B")
            cc, chosen = crossfit(SA, SB, LF, t, st)
            better = ca if ca.mean() >= cb.mean() else cb
            d = cc - better
            full = grid_acc(SA, SB, LF, t, np.arange(len(t)))
            bw = max(full, key=lambda k: (full[k], -abs(k[0] - 0.5)))
            per_dom[key] = (SA, SB, LF, t, full)
            curves[key] = {w: max(full[(w, lam)] for lam in LAMS) for w in WS}
            out.append(f"| {label} | {len(t)} | {ca.mean() * 100:.1f}% | {cb.mean() * 100:.1f}% | {cc.mean() * 100:.1f}% | "
                       f"{d.mean() * 100:+.1f}（±{ci(d) * 100:.1f}） | {chosen[0][0]}, {chosen[1][0]} | {bw[0]}（λ={bw[1]}） |")
        # 留一个领域检验默认权重
        out += ["", "**默认权重是否跨领域稳妥**（在其余三套题上选 w 与 λ，在留出的一套题上检验；括号为相对该题集上较好单模型的差）：", "",
                "| 留出 | 在其余三套上选出的 (w, λ) | 留出题集准确率 | 较好单模型（该题集自己选 λ） |", "|---|---|---|---|"]
        keys = [k for k, _ in DOMS]
        macro = {}
        for hold in keys:
            train = [k for k in keys if k != hold]
            best, bacc = None, -1
            for w in WS:
                for lam in LAMS:
                    m = np.mean([correct(*[per_dom[k][i] for i in (0, 1, 2)], per_dom[k][3], w, lam).mean() for k in train])
                    if m > bacc:
                        best, bacc = (w, lam), m
            SA, SB, LF, t, full = per_dom[hold]
            acc = correct(SA, SB, LF, t, *best).mean()
            sing = max(max(correct(SA, SB, LF, t, w0, lam).mean() for lam in LAMS) for w0 in (0.0, 1.0))
            out.append(f"| {dict(DOMS)[hold]} | w={best[0]}, λ={best[1]} | {acc * 100:.1f}% | {sing * 100:.1f}%（{(acc - sing) * 100:+.1f}） |")
        # 四套题宏平均下的默认建议
        best, bacc = None, -1
        for w in WS:
            for lam in LAMS:
                m = np.mean([correct(*[per_dom[k][i] for i in (0, 1, 2)], per_dom[k][3], w, lam).mean() for k in keys])
                if m > bacc:
                    best, bacc = (w, lam), m
        out += ["", f"四套题宏平均下的默认权重建议：w={best[0]}（对 {NAME[A]}），λ={best[1]}，宏平均准确率 {bacc * 100:.1f}%（在这四套题上选出，偏乐观）。",
                "", "各领域在不同 w 下的准确率（每个 w 取最优 λ，全集，偏乐观，仅用来看最优权重在不同领域是否差别很大）：", "",
                "| 题集 | " + " | ".join(f"w={w}" for w in WS[::2]) + " |", "|---|" + "---|" * len(WS[::2])]
        for key, label in DOMS:
            out.append(f"| {label} | " + " | ".join(f"{curves[key][w] * 100:.1f}" for w in WS[::2]) + " |")
        out.append("")
    text = "\n".join(out) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
