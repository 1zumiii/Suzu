"""教师候选评测：用已有的评测题给一个候选教师打分，并与当前教师等对照系统比较，按阈值判定是否值得换教师。
打分结果追加写入，可随时中断，重跑自动接着做。中断办法：在数据目录下新建名为 STOP 的空文件，或直接结束进程。
用法：
  python tools/eval_teacher.py <系统名> <HF 仓库名> <量化 nf4|none> --data <数据目录> --items <items_eval.jsonl> \\
      --meta <meta.json> --baseline q35_9b=<打分.jsonl> [--baseline q35_4b=<打分.jsonl> ...]
判定（与 s3_report 同口径）：候选相对当前教师（第一个 --baseline）混合自然加权高出 1.0 个百分点，或难例（纯模型）高出 3.0 个百分点。"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("hf")
    ap.add_argument("quant", choices=["nf4", "none"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--baseline", action="append", default=[], help="系统名=打分文件，第一个是当前教师")
    a = ap.parse_args()
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    for src, dst in ((a.items, "items_eval.jsonl"), (a.meta, "meta.json")):
        if not (data / dst).exists():
            shutil.copy(src, data / dst)
    (data / "models_candidate.json").write_text(
        json.dumps({a.name: dict(hf=a.hf, quant=a.quant, role="candidate")}, ensure_ascii=False), encoding="utf-8")
    os.environ["DISTILL_DATA"] = str(data)
    os.environ["DISTILL_MODELS"] = str(data / "models_candidate.json")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import common as C
    import s2_score
    import s3_report as R

    C.install_signals()
    try:
        s2_score.run_eval(a.name)
    except C.StopRequested:
        C.log(f"[{a.name}] 已按请求停止，进度已保存，重跑同一命令接着做")
        return 75

    items = C.read_jsonl(data / "items_eval.jsonl")
    nat = C.read_json(data / "meta.json")["eval_natural"]
    w = [nat[0] / sum(nat), nat[1] / sum(nat)]
    systems = {a.name: data / "scores" / f"{a.name}.jsonl"}
    for b in a.baseline:
        k, v = b.split("=", 1)
        systems[k] = Path(v)
    scores = {k: {r["id"]: r["s"] for r in C.read_jsonl(p)} for k, p in systems.items()}
    # 只在所有系统都打过分的题目上比较，避免候选没跑完（被中断）时拿不同的题集互相比较
    common = set.intersection(*[set(sc) for sc in scores.values()])
    items = [it for it in items if it["id"] in common]
    rows, res = [], {}
    for k, sc in scores.items():
        pure = R.metrics(items, sc, 0, w)
        best = max((R.metrics(items, sc, lam, w) | {"lam": lam} for lam in R.LAMBDAS), key=lambda m: m["nat"])
        res[k] = (pure, best, len(sc))
        rows.append(f"| {k} | {pure['nat'] * 100:.1f}% | {pure['hard'] * 100:.1f}% | {pure['easy'] * 100:.1f}% | "
                    f"{best['nat'] * 100:.1f}%（λ={best['lam']}） | {len(items)}（共打分 {len(sc)}） |")
    out = ["| 系统 | 纯模型 自然加权 | 难例 | 易例 | 混合 自然加权 | 比较题数 |", "|---|---|---|---|---|---|"] + rows
    if len(scores[a.name]) < len(scores[list(systems)[1]] if len(systems) > 1 else scores[a.name]):
        out += ["", f"注意：候选只打了 {len(scores[a.name])} 题（未跑完），以上只在 {len(items)} 道共同题目上比较，判定仅供参考。"]
    ref = list(systems)[1] if len(systems) > 1 else None
    if ref:
        cp, cb, _ = res[a.name]
        rp, rb, _ = res[ref]
        d_nat, d_hard = cb["nat"] - rb["nat"], cp["hard"] - rp["hard"]
        worth = d_nat >= 0.01 or d_hard >= 0.03
        out += ["", f"相对当前教师 {ref}：混合自然加权 {d_nat * 100:+.1f} 个百分点（门槛 +1.0），难例 {d_hard * 100:+.1f} 个百分点（门槛 +3.0）"
                f" → {'值得用它重标训练题' if worth else '不值得'}"]
    text = "\n".join(out)
    (data / f"teacher_eval_{a.name}.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
