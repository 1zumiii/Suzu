"""第三轮评测：用学生检查点在 维基评测、三套旧领域题、新留出题 上打分，与基线（Suzu-2 已有打分）比较。
纯模型与混合（词频融合 λ，固定 0.25）两种准确率，配对 95% 区间，难例单列。
用法：python tools/r3_eval.py --ckpt <pt> --out <md> --name Suzu-3试点"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def corr(items, sc, lam):
    out = []
    for it in items:
        s = sc[it["id"]]
        v = [x + lam * math.log1p(it["f"][i]) for i, x in enumerate(s)]
        out.append(int(v.index(max(v)) == it["t"]))
    return out


def m(x):
    return sum(x) / max(len(x), 1)


def cell(a, b):
    d = [y - x for x, y in zip(a, b)]
    mu = m(d)
    se = math.sqrt(sum((v - mu) ** 2 for v in d) / max(len(d) - 1, 1) / max(len(d), 1))
    return mu * 100, 1.96 * se * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="新模型")
    ap.add_argument("--r2-scores", default="D:/distill-data-r2/scores/student.jsonl")
    ap.add_argument("--dom-items", default="D:/distill-data-domain/items_domain.jsonl")
    ap.add_argument("--dom-base", default="D:/distill-data-domain/scores/student.jsonl")
    ap.add_argument("--wiki-items", default="D:/distill-data-r2/items_eval.jsonl")
    ap.add_argument("--holdout", default="D:/distill-data-r3/run/items_holdout.jsonl")
    a = ap.parse_args()
    import torch
    from student_model import load_student, score_items
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_student(a.ckpt, dev)
    base_s = {r["id"]: r["s"] for r in rj(a.r2_scores)}
    dom_base = {r["id"]: r["s"] for r in rj(a.dom_base)}
    sets = [("维基评测", rj(a.wiki_items), base_s)]
    dom = [i for i in rj(a.dom_items) if i["L"] >= 2]
    for d, n in (("tech", "技术"), ("anime", "二次元"), ("chat", "日常口语")):
        sets.append((n + "（旧）", [i for i in dom if i["domain"] == d], dom_base))
    hold = rj(a.holdout)
    # 新留出题没有 Suzu-2 的打分，现场用 Suzu-2 检查点补
    r2_ck = "D:/distill-data-r2/ckpt/student_best.pt"
    m2, v2 = load_student(r2_ck, dev)
    hb = {it["id"]: s for it, s in zip(hold, score_items(m2, v2, hold, dev))}
    for d, n in (("tech", "技术"), ("anime", "二次元"), ("chat", "日常口语")):
        sets.append((n + "（新留出）", [i for i in hold if i["domain"] == d], hb))
    lam = 0.25
    lines = [f"# {a.name} 与 Suzu-2 的对比", "", f"检查点 {a.ckpt}。混合 λ={lam}（Suzu-2 在维基上选出的固定值，不再调）。区间为同题配对 95%。小样本题集区间很宽，只作参考。", "",
             "| 题集 | 题数（难例） | Suzu-2 混合 | 新 混合 | 差（区间） | Suzu-2 纯 | 新 纯 | 差（区间） | 难例 混合 Suzu-2 → 新 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, items, base in sets:
        new_s = {it["id"]: s for it, s in zip(items, score_items(model, vocab, items, dev))}
        hard = [i for i in items if i["s"]]
        b1, n1 = corr(items, base, lam), corr(items, new_s, lam)
        b0, n0 = corr(items, base, 0), corr(items, new_s, 0)
        d1, d0 = cell(b1, n1), cell(b0, n0)
        bh, nh = m(corr(hard, base, lam)), m(corr(hard, new_s, lam))
        lines.append(f"| {name} | {len(items)}（{len(hard)}） | {m(b1)*100:.1f} | {m(n1)*100:.1f} | {d1[0]:+.1f}（±{d1[1]:.1f}） | "
                     f"{m(b0)*100:.1f} | {m(n0)*100:.1f} | {d0[0]:+.1f}（±{d0[1]:.1f}） | {bh*100:.1f} → {nh*100:.1f} |")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
