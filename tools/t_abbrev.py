"""教师天花板（简拼）：用 4B-Base 与 9B-Base 教师按“上文加候选词”给整个候选池打分，与学生比较首选与前 5 名。只推理，不训练。
候选池与学生分数来自原型脚本（proto.py score）写出的缓存 json：
每题 {i, dom, key, ctx, ans, words（按词库权重降序）, w, s（学生分，同序）, first}。
用法：
  python tools/t_abbrev.py items  --cache cache.json --data <目录>             写 <目录>/items.jsonl（id、ctx、c）
  python tools/t_abbrev.py label  --data <目录> --model Qwen/Qwen3.5-4B-Base --name q35_4b_base   续跑式打分，写 <目录>/scores/<name>__abbrev.jsonl
  python tools/t_abbrev.py report --cache cache.json --data <目录> --out report.md --name q35_4b_base --name q35_9b_base
输出只含数字与留出集、词典里的公开内容。"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_cache(p):
    return json.load(open(p, encoding="utf-8"))


CHUNK = 60          # 每个打分块的候选数：整个候选池（最多 1000 多个）一次前向会爆显存
SENTINEL = "。"     # 每块都加一个哨兵候选，保证各块“公共前缀”只到上文为止（否则整块都以同一个字开头时，这个字不计分，块与块之间不可比）


def cmd_items(a):
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    items = load_cache(a.cache)
    nchunks = 0
    with open(data / "items.jsonl", "w", encoding="utf-8") as f:
        for n, t in enumerate(items):
            w = t["words"]
            for k in range(0, len(w), CHUNK):
                f.write(json.dumps({"id": n * 1000 + k // CHUNK, "ctx": t["ctx"], "c": w[k:k + CHUNK] + [SENTINEL]}, ensure_ascii=False) + "\n")
                nchunks += 1
    (data / "expected_chunks.txt").write_text(str(nchunks), encoding="ascii")
    tot = sum(len(t["words"]) for t in items)
    print(f"{len(items)} 题，候选共 {tot} 个（每题平均 {tot / len(items):.0f}，最大 {max(len(t['words']) for t in items)}），分成 {nchunks} 块")


def cmd_label(a):
    data = Path(a.data)
    (data / "scores").mkdir(parents=True, exist_ok=True)
    (data / "models_label.json").write_text(json.dumps({a.name: dict(hf=a.model, quant="nf4", role="teacher")}), encoding="utf-8")
    os.environ["DISTILL_DATA"] = str(data)
    os.environ["DISTILL_MODELS"] = str(data / "models_label.json")
    os.environ.setdefault("HF_HOME", "D:/distill-data/hf")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import common as C
    import s2_score
    C.install_signals()
    s2_score.score_unit(f"{a.name}__abbrev", data / "items.jsonl", lambda: s2_score.make_scorer(a.name))
    expected = int((data / "expected_chunks.txt").read_text())
    got = len(C.load_done_ids(data / "scores" / f"{a.name}__abbrev.jsonl"))
    if got >= expected:
        (data / f"done_{a.name}.flag").write_text(f"{got}/{expected}", encoding="ascii")
    else:
        print(f"未打完：{got}/{expected} 块", flush=True)
        sys.exit(1)


def ranks(scores, ans_idx):
    """答案的名次（1 起）；分数相同按先出现者在前。无效分（<= -1e8）排最后。"""
    s = scores[ans_idx]
    better = sum(1 for x in scores if x > s)
    return better + 1


def boot_ci(xs, ys, n=2000, seed=7):
    """成对差（ys - xs）的均值与 95% 区间（对题自助重采样），单位百分点。"""
    rnd = random.Random(seed)
    d = [y - x for x, y in zip(xs, ys)]
    m = len(d)
    means = sorted(sum(d[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return 100 * sum(d) / m, 100 * means[int(0.025 * n)], 100 * means[int(0.975 * n)]


def cmd_report(a):
    items = load_cache(a.cache)
    data = Path(a.data)
    models = {"学生 Suzu-2.1": [t["s"] for t in items]}
    for spec in a.model_cache or []:
        nm, _, path = spec.partition("=")
        other = load_cache(path)
        if len(other) != len(items) or any(x["ans"] != y["ans"] or x["ctx"] != y["ctx"] for x, y in zip(items, other)):
            print(f"警告：{nm} 的题与基线不一致，跳过", file=sys.stderr)
            continue
        models[nm] = [t["s"] for t in other]
    for name in a.name or []:
        chunks = {}
        for ln in open(data / "scores" / f"{name}__abbrev.jsonl", encoding="utf-8"):
            r = json.loads(ln)
            chunks[r["id"]] = r["s"][:-1]          # 去掉哨兵候选
        sc = {}
        for n, t in enumerate(items):
            k = (len(t["words"]) + CHUNK - 1) // CHUNK
            parts = [chunks.get(n * 1000 + j) for j in range(k)]
            if all(p is not None for p in parts):
                sc[n] = [x for p in parts for x in p]
        missing = [n for n in range(len(items)) if n not in sc]
        if missing:
            print(f"警告：{name} 缺 {len(missing)} 题，报告只用已完成的题", file=sys.stderr)
        models[name] = [sc.get(n) for n in range(len(items))]
    done = [n for n in range(len(items)) if all(models[m][n] is not None for m in models)]
    lines = []
    out = lines.append
    out("# 简拼评测（模型分，整个候选池）" if not a.name else "# 简拼教师天花板")
    out("")
    out(f"题数 {len(done)}（答案为两字词，输入为两个字的首字母；候选池为词典里首字母相同的全部两字词，每题平均 "
        f"{sum(len(items[n]['words']) for n in done) / len(done):.0f} 个，最多 {max(len(items[n]['words']) for n in done)} 个）。"
        "只用模型分排序，不加词库先验；分数相同按词库权重靠前者在前。")
    out("")
    out("| 模型 | 首选 | 前 5 名 | 前 10 名 |")
    out("|---|---|---|---|")
    hit1, hit5 = {}, {}
    for m, sc in models.items():
        h1, h5, h10 = [], [], []
        for n in done:
            t = items[n]
            ai = t["words"].index(t["ans"])
            r = ranks(sc[n], ai)
            h1.append(r == 1)
            h5.append(r <= 5)
            h10.append(r <= 10)
        hit1[m], hit5[m] = h1, h5
        k = len(done)
        out(f"| {m} | {100 * sum(h1) / k:.1f}% | {100 * sum(h5) / k:.1f}% | {100 * sum(h10) / k:.1f}% |")
    base = "学生 Suzu-2.1"
    out("")
    out("相对学生的成对差（百分点，95% 区间；对题自助重采样）：")
    out("")
    out("| 模型 | 首选差 | 前 5 名差 |")
    out("|---|---|---|")
    for m in models:
        if m == base:
            continue
        d1 = boot_ci([int(x) for x in hit1[base]], [int(x) for x in hit1[m]])
        d5 = boot_ci([int(x) for x in hit5[base]], [int(x) for x in hit5[m]])
        out(f"| {m} | {d1[0]:+.1f}（{d1[1]:+.1f} 到 {d1[2]:+.1f}） | {d5[0]:+.1f}（{d5[1]:+.1f} 到 {d5[2]:+.1f}） |")
    doms = sorted({items[n].get("dom") or "-" for n in done})
    if len(doms) > 1:
        out("")
        out("按领域的首选（学生 / 各教师）：")
        out("")
        out("| 领域 | 题数 | " + " | ".join(models) + " |")
        out("|---|---|" + "---|" * len(models))
        for d in doms:
            idx = [j for j, n in enumerate(done) if (items[n].get("dom") or "-") == d]
            out(f"| {d} | {len(idx)} | " + " | ".join(f"{100 * sum(hit1[m][j] for j in idx) / len(idx):.1f}%" for m in models) + " |")
    out("")
    out("判定线（Mac 任务书 2.3）：教师首选比学生高 5 个点以上，才值得做简拼专项训练；高不到 2 个点，说明简拼的歧义是上限。")
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("items"); p.add_argument("--cache", required=True); p.add_argument("--data", required=True)
    p = sub.add_parser("label"); p.add_argument("--data", required=True); p.add_argument("--model", required=True); p.add_argument("--name", required=True)
    p = sub.add_parser("report"); p.add_argument("--cache", required=True); p.add_argument("--data", required=True)
    p.add_argument("--out", required=True); p.add_argument("--name", action="append")
    p.add_argument("--model-cache", action="append", help="NAME=cache.json：另一个学生模型在同一批题上的候选池打分（proto.py score 写出）")
    a = ap.parse_args()
    {"items": cmd_items, "label": cmd_label, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
