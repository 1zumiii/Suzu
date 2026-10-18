"""教师的领域提示实验：给教师打分时在上文前加一句领域背景（提示文字不参与评分），比较有无提示的首选准确率；
并把各教师在三个领域评测题上的准确率按领域拆开。提示只用于给教师打分，学生仍用原样上下文训练。
用法：
  python tools/domain_teacher_prompt.py score  --work <工作目录> --domain-items <items_domain.jsonl> --models <模型名=HF 仓库名> ...
  python tools/domain_teacher_prompt.py report --work <工作目录> --domain-items <items_domain.jsonl> --scores 系统名=打分.jsonl ...
score 需要 GPU 与 HF_HOME 指向已下载的模型缓存；模型一律按 nf4 加载。"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = {"tech": "以下是软件开发与技术社区的讨论：", "anime": "以下是二次元社区的日常聊天：", "chat": "以下是朋友之间的日常聊天："}
NAMES = {"tech": "技术", "anime": "二次元", "chat": "日常口语"}
LAMBDA = {"q35_4b": 0.25, "q35_9b": 0.5, "q35_4b_base": 0.25, "q35_9b_base": 0.0}


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def acc(items, scores, lam):
    ok = []
    for it in items:
        s = scores[it["id"]]
        v = [x + lam * math.log1p(it["f"][i]) for i, x in enumerate(s)]
        ok.append(int(v.index(max(v)) == it["t"]))
    return ok


def mean(x):
    return sum(x) / max(len(x), 1)


def cmd_score(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    spec = {}
    for m in a.models:
        k, hf = m.split("=", 1)
        spec[k] = dict(hf=hf, quant="nf4", role="teacher")
    (work / "models_prompt.json").write_text(json.dumps(spec), encoding="utf-8")
    os.environ["DISTILL_DATA"] = str(work)
    os.environ["DISTILL_MODELS"] = str(work / "models_prompt.json")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import common as C
    import s2_score
    C.install_signals()
    items = [i for i in load_jsonl(a.domain_items) if i["L"] >= 2]
    plain = [dict(i) for i in items]
    prompted = [dict(i, ctx=PROMPTS[i["domain"]] + i["ctx"]) for i in items]
    for name, rows in (("plain", plain), ("prompt", prompted)):
        (work / f"items_{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    for model in spec:
        scorer = s2_score.make_scorer(model)
        for name in ("plain", "prompt"):
            s2_score.score_unit(f"{model}__{name}", work / f"items_{name}.jsonl", lambda s=scorer: s)
        del scorer


def cmd_report(a):
    items = [i for i in load_jsonl(a.domain_items) if i["L"] >= 2]
    sc = {}
    for spec in a.scores:
        k, p = spec.split("=", 1)
        sc[k] = {r["id"]: r["s"] for r in load_jsonl(p)}
    out = ["# 教师的领域提示实验与分领域教师准确率", "",
           "提示文字只用于给教师打分，不参与评分。准确率为首选准确率，混合使用各教师在维基评测上选出的固定词频融合权重。区间为同题配对 95%。", ""]
    out += ["## 一、各教师分领域准确率（无提示，混合 / 纯模型 / 难例混合）", "",
            "| 教师 | " + " | ".join(f"{NAMES[d]} 混合 / 纯 / 难例" for d in NAMES) + " |", "|---|---|---|---|"]
    plain_names = [k for k in sc if not k.endswith("__prompt")]
    for k in plain_names:
        base = k.replace("__plain", "")
        row = []
        for d in NAMES:
            its = [i for i in items if i["domain"] == d]
            hard = [i for i in its if i["s"]]
            row.append(f"{mean(acc(its, sc[k], LAMBDA[base])) * 100:.1f} / {mean(acc(its, sc[k], 0)) * 100:.1f} / "
                       f"{mean(acc(hard, sc[k], LAMBDA[base])) * 100:.1f}")
        out.append(f"| {base} | " + " | ".join(row) + " |")
    out += ["", "## 二、领域提示的效果（同一教师，有提示对无提示）", "",
            "| 教师 | 领域 | 题数（难例） | 无提示 混合 | 有提示 混合 | 差（区间） | 难例 无提示 → 有提示 |", "|---|---|---|---|---|---|---|"]
    for k in plain_names:
        base = k.replace("__plain", "")
        pk = f"{base}__prompt"
        if pk not in sc:
            continue
        for d in NAMES:
            its = [i for i in items if i["domain"] == d]
            hard = [i for i in its if i["s"]]
            p0, p1 = acc(its, sc[k], LAMBDA[base]), acc(its, sc[pk], LAMBDA[base])
            diff = [y - x for x, y in zip(p0, p1)]
            m = mean(diff)
            se = math.sqrt(sum((x - m) ** 2 for x in diff) / max(len(diff) - 1, 1) / len(diff))
            h0, h1 = mean(acc(hard, sc[k], LAMBDA[base])), mean(acc(hard, sc[pk], LAMBDA[base]))
            out.append(f"| {base} | {NAMES[d]} | {len(its)}（{len(hard)}） | {mean(p0) * 100:.1f}% | {mean(p1) * 100:.1f}% | "
                       f"{m * 100:+.1f}（±{1.96 * se * 100:.1f}） | {h0 * 100:.1f}% → {h1 * 100:.1f}% |")
    out += ["", "## 三、二次元题上 9B-Base 对 4B-Base（无提示）", ""]
    if "q35_9b_base__plain" in sc and "q35_4b_base__plain" in sc:
        its = [i for i in items if i["domain"] == "anime"]
        hard = [i for i in its if i["s"]]
        x, y = acc(its, sc["q35_4b_base__plain"], LAMBDA["q35_4b_base"]), acc(its, sc["q35_9b_base__plain"], LAMBDA["q35_9b_base"])
        diff = [q - p for p, q in zip(x, y)]
        m = mean(diff)
        se = math.sqrt(sum((v - m) ** 2 for v in diff) / max(len(diff) - 1, 1) / len(diff))
        out.append(f"9B-Base 相对 4B-Base：混合 {m * 100:+.1f}（±{1.96 * se * 100:.1f}），难例 "
                   f"{mean(acc(hard, sc['q35_4b_base__plain'], LAMBDA['q35_4b_base'])) * 100:.1f}% → "
                   f"{mean(acc(hard, sc['q35_9b_base__plain'], LAMBDA['q35_9b_base'])) * 100:.1f}%（难例 {len(hard)} 题）")
    text = "\n".join(out) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--work", required=True)
    s.add_argument("--domain-items", required=True, dest="domain_items")
    s.add_argument("--models", nargs="+", required=True)
    r = sub.add_parser("report")
    r.add_argument("--domain-items", required=True, dest="domain_items")
    r.add_argument("--scores", nargs="+", required=True)
    r.add_argument("--out")
    a = ap.parse_args()
    {"score": cmd_score, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
