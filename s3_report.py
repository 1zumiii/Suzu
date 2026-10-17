"""阶段 3：汇总准确率 → report.md、gates.json。

指标：首选准确率（top-1）。题目是分层抽样的（易例/难例各取一半），所以「自然加权」按评测句里
易例:难例的真实比例重新加权，得到接近真实使用的数字。「混合」= 模型分 + λ·ln(1+词频)，λ 在评测集上选最优
（略乐观，只用于相对比较），对应实际部署里模型分与词频融合的做法。
"""
import json
import math

import common as C
from common import CFG, DATA, SCORES, log

LAMBDAS = [0, 0.25, 0.5, 1, 2, 4]
# 判定阈值（可按需调整）：教师相对小模型的提升空间 / 小模型相对 n-gram 的价值
G1_NAT, G1_HARD, G2_NAT = 0.015, 0.03, 0.02


def pick(it, s, lam):
    best, bi = -1e18, 0
    for i, x in enumerate(s):
        v = x + lam * math.log1p(it["f"][i])
        if v > best:
            best, bi = v, i
    return bi


def metrics(items, scores, lam, w):
    c, n = [0, 0], [0, 0]
    for it in items:
        s = scores.get(it["id"])
        if s is None:
            continue
        n[it["s"]] += 1
        c[it["s"]] += pick(it, s, lam) == it["t"]
    a = [c[i] / n[i] if n[i] else 0.0 for i in (0, 1)]
    nat = w[0] * a[0] + w[1] * a[1]
    se = math.sqrt(sum((w[i] ** 2) * a[i] * (1 - a[i]) / n[i] for i in (0, 1) if n[i]))
    return dict(easy=a[0], hard=a[1], nat=nat, se=se, n=n)


def load_scores(name):
    p = SCORES / f"{name}.jsonl"
    if not p.exists():
        return {}
    return {r["id"]: r["s"] for r in C.read_jsonl(p)}


def run(final=False):
    items = C.read_jsonl(DATA / "items_eval.jsonl")
    meta = C.read_json(DATA / "meta.json")
    nat = meta["eval_natural"]
    w = [nat[0] / (nat[0] + nat[1]), nat[1] / (nat[0] + nat[1])]
    ms = C.models()
    systems = {"ngram": "字符三元模型（文法模型基线）"}
    for k, v in ms.items():
        systems[k] = f"{v['hf']}（{v.get('quant', 'none')}，角色 {v.get('role', '-')}）"
    if final:
        systems["student"] = "蒸馏学生"
    rows, res = [], {}
    # 词频基线：候选本身按词频降序，选第 0 个
    res["freq"] = dict(easy=1.0, hard=0.0, nat=w[0], se=0.0, n=meta["eval_sampled"])
    res["random"] = dict(easy=0, hard=0, nat=sum(1 / len(it["c"]) for it in items) / len(items), se=0, n=meta["eval_sampled"])
    for name in systems:
        sc = load_scores(name)
        if len(sc) < 0.99 * len(items):
            res[name] = None
            continue
        pure = metrics(items, sc, 0, w)
        best = max((metrics(items, sc, lam, w) | {"lam": lam} for lam in LAMBDAS), key=lambda m: m["nat"])
        res[name] = dict(pure=pure, fused=best)

    def pct(x):
        return f"{x*100:.1f}%"

    L = ["# 输入法排序蒸馏：评测报告", "",
         f"评测题 {sum(meta['eval_sampled'])} 道（词长 ≥ {CFG['min_word_len']} 字；易例 {meta['eval_sampled'][0]}、难例 {meta['eval_sampled'][1]}）；"
         f"评测句内自然比例 易例 {pct(w[0])} : 难例 {pct(w[1])}。", "",
         "| 系统 | 纯模型：自然加权 | 难例 | 易例 | 混合词频：自然加权（λ） | 95% 区间 |", "|---|---|---|---|---|---|",
         f"| 词频第一（Rime 默认顺序近似） | {pct(res['freq']['nat'])} | 0.0% | 100.0% | - | - |",
         f"| 随机 | {pct(res['random']['nat'])} | - | - | - | - |"]
    for name, desc in systems.items():
        r = res.get(name)
        if r is None:
            L.append(f"| {name}：{desc} | 缺失或未完成 | | | | |")
            continue
        p, f = r["pure"], r["fused"]
        L.append(f"| {name}：{desc} | {pct(p['nat'])} | {pct(p['hard'])} | {pct(p['easy'])} | "
                 f"{pct(f['nat'])}（{f['lam']}） | ±{pct(1.96*f['se'])} |")
    # 按词长分桶：难例上的首选准确率（纯模型 / 混合），看大模型的收益是否集中在某个词长
    L += ["", "## 按词长分桶（难例首选准确率；纯模型 / 混合）", "", "| 系统 | 2 字 | 3–4 字 |", "|---|---|---|"]
    buckets = (lambda it: it.get("L", 2) == 2, lambda it: it.get("L", 2) >= 3)
    for name in systems:
        r = res.get(name)
        if r is None:
            continue
        sc = load_scores(name)
        cells = []
        for b in buckets:
            sub = [it for it in items if it["s"] == 1 and b(it) and it["id"] in sc]
            if not sub:
                cells.append("-")
                continue
            acc = lambda lam: sum(pick(it, sc[it["id"]], lam) == it["t"] for it in sub) / len(sub)
            cells.append(f"{pct(acc(0))} / {pct(acc(r['fused']['lam']))}（n={len(sub)}）")
        L.append(f"| {name} | {cells[0]} | {cells[1]} |")
    roles = {v.get("role"): k for k, v in ms.items()}
    t, s = res.get(roles.get("teacher")), res.get(roles.get("small"))
    g = res.get("ngram")
    gates = dict(proceed=False, reason="")
    if not (t and s and g):
        gates["reason"] = "教师、小模型或 n-gram 的打分缺失，无法判定"
    else:
        d_nat = t["fused"]["nat"] - s["fused"]["nat"]
        d_hard = t["pure"]["hard"] - s["pure"]["hard"]
        v_nat = s["fused"]["nat"] - g["fused"]["nat"]
        g1 = d_nat >= G1_NAT or d_hard >= G1_HARD
        g2 = v_nat >= G2_NAT
        gates = dict(proceed=bool(g1 and g2), d_nat=d_nat, d_hard=d_hard, v_nat=v_nat, g1=bool(g1), g2=bool(g2))
        L += ["", "## 判定", "",
              f"- 教师相对小模型：混合自然加权 {d_nat*100:+.1f} 个百分点，难例 {d_hard*100:+.1f} 个百分点"
              f"（门槛 {G1_NAT*100:.1f} 或 {G1_HARD*100:.1f}）→ {'有提升空间' if g1 else '提升空间不足'}",
              f"- 小模型相对 n-gram：混合自然加权 {v_nat*100:+.1f} 个百分点（门槛 {G2_NAT*100:.1f}）→ "
              f"{'大模型确有价值' if g2 else '大模型相对文法模型优势不明显'}",
              f"- 结论：{'继续做蒸馏（训练学生）' if gates['proceed'] else '不建议继续蒸馏，改走词表裁剪等无需训练的方案'}"]
    mid = res.get(roles.get("mid"))
    if mid and s and t:
        L += ["", f"规模曲线（混合自然加权）：小 {pct(s['fused']['nat'])} → 中 {pct(mid['fused']['nat'])} → 教师 {pct(t['fused']['nat'])}。"]
    if final and res.get("student"):
        st = res["student"]["fused"]
        refs = [f"{n} {pct(r['fused']['nat'])}" for n, r in (("n-gram", g), ("小模型", s), ("教师", t)) if r]
        L += ["", "## 学生模型", "",
              f"混合自然加权 {pct(st['nat'])}（{'，'.join(refs) if refs else '无对照系统'}）。"]
        if g and t and t["fused"]["nat"] > g["fused"]["nat"]:
            keep = (st["nat"] - g["fused"]["nat"]) / (t["fused"]["nat"] - g["fused"]["nat"])
            L.append(f"相对 n-gram，保留了教师提升量的 {keep*100:.0f}%。")
        bench = C.read_json(C.STATE / "student_bench.json")
        if bench:
            L.append(f"学生参数量 {bench['params_m']:.1f} M；本机 CPU 单线程每题（一组候选）约 {bench['ms_per_item']:.1f} ms。"
                     "手机端实际延迟仍需另行实测。")
    if res["freq"]:
        L += ["", "注：混合行的 λ 在评测集上选取，数字略乐观，适合相对比较，不宜当作绝对指标。"]
    C.atomic_write(DATA / ("report.md" if final else "report_stage0.md"), "\n".join(L) + "\n")
    C.write_json(DATA / "gates.json", gates)
    log("报告已写出：" + ("report.md" if final else "report_stage0.md") + "；判定：" + json.dumps(gates, ensure_ascii=False))
