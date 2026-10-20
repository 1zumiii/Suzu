"""简拼专项训练的数据：从训练句里出“上文 + 两字词”的简拼题，教师标注，与原有全拼题混合成训练目录。
题的形式与原有训练题相同（ctx、c 候选、f 词库权重、t 答案下标、s 难例标记），学生只看上文与候选词的文字，所以拼音只是元数据。
输入形式：70% 两字母简拼（zm），30% 首字全拼加次字首字母（jing l）。候选 = 答案 + 词库权重靠前的 24 个 + 随机 23 个（同一缩写的词）。
评测集的句子（tests/holdout、cases.tsv 的上文）用 12 字滑窗排除，防止泄漏。
用法：
  python tools/t_abbrev_train.py build --out <目录> [--n 24000]
  python tools/t_abbrev_train.py label --out <目录>                         续跑式教师打分（4B-Base，领域提示同 r3_label）
  python tools/t_abbrev_train.py mix   --abbrev <目录> --out <臂目录> --replay-n 48000
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN3 = Path("D:/distill-data-r3/run3")
BIG = Path("D:/distill-data-r3/big")
DICTS = Path("D:/AppData/Rime/cn_dicts")
CASES = Path("D:/distill-data-r4/core/cases.tsv")
HOLDOUT = ROOT / "tests" / "holdout"
PROMPTS = {"tech": "以下是软件开发与技术社区的讨论：", "anime": "以下是二次元社区的日常聊天：",
           "slang": "以下是网络社区里的聊天、弹幕和评论："}
ID0 = 10_000_000
CJK = re.compile("^[一-鿿]+$")


def load_dict():
    """两字词的读音与权重：words[词] = [(权重, 简拼键, 混合键)]；pools_abbr[键] = {词: 权重}（所有读音）。"""
    words = defaultdict(list)
    pa, pm = defaultdict(dict), defaultdict(dict)
    for name in ("base", "ext"):
        body = False
        for ln in open(DICTS / f"{name}.dict.yaml", encoding="utf-8"):
            if ln.startswith("..."):
                body = True
                continue
            if not body or ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 2 or len(p[0]) != 2 or not CJK.match(p[0]):
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


def shingles():
    s = set()
    texts = []
    for f in HOLDOUT.glob("*.txt"):
        texts += [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
    for l in open(CASES, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        if len(f) >= 3:
            texts.append(f[2] + (f[3].split("|")[0] if len(f) > 3 else ""))
    for t in texts:
        for j in range(max(1, len(t) - 11)):
            s.add(t[j:j + 12])
    return s


def leaked(sent, sh):
    return any(sent[j:j + 12] in sh for j in range(max(1, len(sent) - 11)))


def cmd_build(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    words, pa, pm = load_dict()
    sh = shingles()
    quota = {"wiki": int(a.n * 0.30), "chat": int(a.n * 0.25), "tech": int(a.n * 0.25)}
    quota["anime"] = a.n - sum(quota.values())
    src = {"wiki": RUN3 / "train_sentences.txt", "chat": BIG / "chat.txt", "tech": BIG / "tech.txt", "anime": BIG / "anime.txt"}
    items, per_word = [], defaultdict(int)
    nid = ID0
    for dom, q in quota.items():
        sents = [l.strip() for l in open(src[dom], encoding="utf-8") if 6 <= len(l.strip()) <= 120]
        rng.shuffle(sents)
        got = 0
        for s in sents:
            if got >= q:
                break
            if leaked(s, sh):
                continue
            pos = [i for i in range(4, len(s) - 1) if s[i:i + 2] in words and CJK.match(s[i - 4:i + 2])]
            if not pos:
                continue
            i = rng.choice(pos)
            w = s[i:i + 2]
            if per_word[w] >= 40:
                continue
            mode = "abbr" if rng.random() < 0.7 else "mix"
            best = max(words[w])
            key = best[1] if mode == "abbr" else best[2]
            pool = (pa if mode == "abbr" else pm)[key]
            if len(pool) < 6 or w not in pool:
                continue
            ranked = sorted(pool, key=lambda x: -pool[x])
            rest = [x for x in ranked if x != w]
            top = rest[:24]
            rnd = rng.sample(rest[24:], min(23, len(rest) - 24)) if len(rest) > 24 else []
            cands = sorted(set(top + rnd + [w]), key=lambda x: -pool[x])
            items.append({"id": nid, "ctx": s[max(0, i - 40):i], "py": key, "c": cands, "f": [pool[x] for x in cands],
                          "t": cands.index(w), "s": int(cands.index(w) > 0), "L": 2, "word": w, "domain": dom, "mode": mode})
            nid += 1
            per_word[w] += 1
            got += 1
        print(f"{dom}: {got}/{q}", flush=True)
    rng.shuffle(items)
    with open(out / "items_abbrev.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tot = sum(len(it["c"]) for it in items)
    print(f"共 {len(items)} 题，候选 {tot} 个（平均 {tot / len(items):.1f}），答案是词库首位的占 {100 * sum(1 for it in items if it['t'] == 0) / len(items):.1f}%，"
          f"简拼 {sum(1 for it in items if it['mode'] == 'abbr')}、混合 {sum(1 for it in items if it['mode'] == 'mix')}")


def cmd_build2(a):
    """第二批题：候选里加入学生自己（Suzu-2.1）在整个候选池里排得靠前的词（难例），与第一批的题不重复。
    候选 = 答案 + 学生分最高的 20 个 + 词库权重最高的 14 个 + 随机 13 个（同缩写的词，去重）。"""
    sys.path.insert(0, str(ROOT / "pipeline"))
    import torch
    from student_model import load_student, score_items
    out = Path(a.out)
    rng = random.Random(a.seed)
    words, pa, pm = load_dict()
    sh = shingles()
    seen = set()
    for l in open(out / a.exclude, encoding="utf-8"):
        r = json.loads(l)
        seen.add((r["ctx"], r["word"]))
    quota = {"wiki": int(a.n * 0.30), "chat": int(a.n * 0.25), "tech": int(a.n * 0.25)}
    quota["anime"] = a.n - sum(quota.values())
    src = {"wiki": RUN3 / "train_sentences.txt", "chat": BIG / "chat.txt", "tech": BIG / "tech.txt", "anime": BIG / "anime.txt"}
    recs, per_word = [], defaultdict(int)
    for dom, q in quota.items():
        sents = [l.strip() for l in open(src[dom], encoding="utf-8") if 6 <= len(l.strip()) <= 120]
        rng.shuffle(sents)
        got = 0
        for s in sents:
            if got >= q:
                break
            if leaked(s, sh):
                continue
            pos = [i for i in range(4, len(s) - 1) if s[i:i + 2] in words and CJK.match(s[i - 4:i + 2])]
            if not pos:
                continue
            i = rng.choice(pos)
            w = s[i:i + 2]
            ctx = s[max(0, i - 40):i]
            if per_word[w] >= 60 or (ctx, w) in seen:
                continue
            mode = "abbr" if rng.random() < 0.7 else "mix"
            best = max(words[w])
            key = best[1] if mode == "abbr" else best[2]
            pool = (pa if mode == "abbr" else pm)[key]
            if len(pool) < 6 or w not in pool:
                continue
            recs.append({"ctx": ctx, "w": w, "key": key, "mode": mode, "dom": dom, "pool": pool})
            per_word[w] += 1
            got += 1
        print(f"{dom}: {got}/{q}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_student(a.student_ckpt, dev)
    print(f"学生给 {len(recs)} 题的整个候选池打分……", flush=True)
    chunk_items, owner = [], []
    for n, r in enumerate(recs):
        ws = sorted(r["pool"], key=lambda x: -r["pool"][x])
        r["ranked"] = ws
        for k in range(0, len(ws), 100):
            chunk_items.append({"ctx": r["ctx"], "c": ws[k:k + 100]})
            owner.append((n, k))
    scores = [dict() for _ in recs]
    with torch.no_grad():
        for a0 in range(0, len(chunk_items), 64):
            part = chunk_items[a0:a0 + 64]
            res = score_items(model, vocab, part, dev)
            for (n, k), it, sc in zip(owner[a0:a0 + 64], part, res):
                for wd, v in zip(it["c"], sc):
                    scores[n][wd] = v
            if (a0 // 64) % 100 == 0:
                print(f"  打分 {a0}/{len(chunk_items)} 块", flush=True)
    items, nid = [], a.id0
    for r, sc in zip(recs, scores):
        w, pool = r["w"], r["pool"]
        rest = [x for x in r["ranked"] if x != w]
        hard = sorted(rest, key=lambda x: -sc.get(x, -1e9))[:20]
        taken = set(hard)
        top = [x for x in rest if x not in taken][:14]
        taken |= set(top)
        left = [x for x in rest if x not in taken]
        rnd = rng.sample(left, min(13, len(left)))
        cands = sorted(set(hard + top + rnd + [w]), key=lambda x: -pool[x])
        items.append({"id": nid, "ctx": r["ctx"], "py": r["key"], "c": cands, "f": [pool[x] for x in cands],
                      "t": cands.index(w), "s": int(cands.index(w) > 0), "L": 2, "word": w, "domain": r["dom"], "mode": r["mode"],
                      "student_rank": 1 + sum(1 for x in rest if sc.get(x, -1e9) > sc.get(w, -1e9))})
        nid += 1
    rng.shuffle(items)
    with open(out / a.name, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tot = sum(len(it["c"]) for it in items)
    first = sum(1 for it in items if it["student_rank"] == 1)
    print(f"共 {len(items)} 题，候选 {tot} 个（平均 {tot / len(items):.1f}），学生已排第一的占 {100 * first / len(items):.1f}%，"
          f"简拼 {sum(1 for it in items if it['mode'] == 'abbr')}、混合 {sum(1 for it in items if it['mode'] == 'mix')}")


def cmd_label(a):
    out = Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    name = "q35_4b_base"
    (out / "models_label.json").write_text(json.dumps({name: dict(hf="Qwen/Qwen3.5-4B-Base", quant="nf4", role="teacher")}), encoding="utf-8")
    os.environ["DISTILL_DATA"] = str(out)
    os.environ["DISTILL_MODELS"] = str(out / "models_label.json")
    os.environ.setdefault("HF_HOME", "D:/distill-data/hf")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import common as C
    import s2_score
    C.install_signals()
    rows = [json.loads(l) for l in open(out / a.items, encoding="utf-8")]
    for r in rows:
        r["ctx"] = PROMPTS.get(r["domain"], "") + r["ctx"]
    pf = out / (Path(a.items).stem + "_prompted.jsonl")
    C.write_jsonl(pf, rows)
    s2_score.score_unit(a.score_name, pf, lambda: s2_score.make_scorer(name))
    got = len(C.load_done_ids(out / "scores" / (a.score_name + ".jsonl")))
    if got >= len(rows):
        (out / a.flag).write_text(f"{got}/{len(rows)}", encoding="ascii")
    else:
        print(f"未打完：{got}/{len(rows)}", flush=True)
        sys.exit(1)


def cmd_mix(a):
    ab, out = Path(a.abbrev), Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    ab_items = [json.loads(l) for l in open(ab / "items_abbrev.jsonl", encoding="utf-8")]
    score_files = [ab / "scores" / "abbrev_train__q35_4b_base.jsonl"]
    for spec in a.extra or []:
        fn, _, sn = spec.partition(":")
        ab_items += [json.loads(l) for l in open(ab / fn, encoding="utf-8")]
        score_files.append(ab / "scores" / (sn + ".jsonl"))
    old = [json.loads(l) for l in open(RUN3 / "items_train.jsonl", encoding="utf-8")]
    old_scores = {}
    for l in open(RUN3 / "scores" / "q35_4b_base__train.jsonl", encoding="utf-8"):
        r = json.loads(l)
        old_scores[r["id"]] = r
    old = [it for it in old if it["id"] in old_scores]
    replay = rng.sample(old, min(a.replay_n, len(old)))
    items = ab_items + replay
    rng.shuffle(items)
    with open(out / "items_train.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out / "scores" / "q35_4b_base__train.jsonl", "w", encoding="utf-8") as f:
        for sf in score_files:
            for l in open(sf, encoding="utf-8"):
                f.write(l if l.endswith("\n") else l + "\n")
        for it in replay:
            f.write(json.dumps(old_scores[it["id"]], ensure_ascii=False) + "\n")
    import shutil
    for n in ("items_dev.jsonl", "meta.json", "gates.json", "train_sentences.txt"):
        if not (out / n).exists():
            shutil.copy(RUN3 / n, out / n)
    shutil.copy(RUN3 / "ckpt" / "vocab.json", out / "ckpt" / "vocab.json")
    print(f"{out.name}: 简拼题 {len(ab_items)} + 原有题 {len(replay)} = {len(items)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=24000); p.add_argument("--seed", type=int, default=5)
    p = sub.add_parser("build2"); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=36000)
    p.add_argument("--seed", type=int, default=17); p.add_argument("--id0", type=int, default=20_000_000)
    p.add_argument("--name", default="items_abbrev2.jsonl"); p.add_argument("--exclude", default="items_abbrev.jsonl")
    p.add_argument("--student-ckpt", required=True)
    p = sub.add_parser("label"); p.add_argument("--out", required=True)
    p.add_argument("--items", default="items_abbrev.jsonl"); p.add_argument("--score-name", default="abbrev_train__q35_4b_base")
    p.add_argument("--flag", default="label_done.flag")
    p = sub.add_parser("mix"); p.add_argument("--abbrev", required=True); p.add_argument("--out", required=True)
    p.add_argument("--replay-n", type=int, default=48000); p.add_argument("--seed", type=int, default=11)
    p.add_argument("--extra", action="append", help="ITEMS文件:打分文件名（不带 .jsonl），追加的简拼题")
    a = ap.parse_args()
    {"build": cmd_build, "build2": cmd_build2, "label": cmd_label, "mix": cmd_mix}[a.cmd](a)


if __name__ == "__main__":
    main()
