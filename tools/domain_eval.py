"""领域评测：用合成题量化 Suzu-2 在维基百科以外的领域（技术、二次元、日常口语）相对 Qwen3.5-0.8B 的差距。

题目在 tests/domain/*.txt：每行一句自编句子，【】标出要考的词，全部为合成，不含任何个人数据。
出题方式与维基评测同口径：上文 = 该词之前的文字，候选 = 同拼音、按训练语料词频降序的前 8 个词，答案 = 句中应有的词。
同时统计 jieba 与词频表对这些领域词的覆盖（被切碎、不在词典、不在词频表、排在前 8 之外），因为这会影响以后用领域语料出题。

分三步（各自可重跑）：
  python tools/domain_eval.py build  --work <工作目录> --train-sentences <train_sentences.txt>
  python tools/domain_eval.py score  --work <工作目录> --student-ckpt-dir <含 student_best.pt 的目录>
  python tools/domain_eval.py report --work <工作目录> --wiki 系统名=打分.jsonl ... --wiki-items <items_eval.jsonl>
score 需要 GPU 与已下载的 Qwen3.5 模型（HF_HOME 指向缓存）。"""
import argparse
import json
import math
import os
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ["tech", "anime", "chat"]
NAMES = {"tech": "技术", "anime": "二次元", "chat": "日常口语"}
# 各系统在维基评测上选出的词频融合权重（固定使用，不在小样本上重新调）
LAMBDA = {"ngram": 1, "q35_0p8b": 0.5, "q35_4b": 0.25, "q35_9b": 0.5, "student": 0.25}
SYSTEMS = ["ngram", "q35_0p8b", "q35_4b", "q35_9b", "student"]
LABEL = {"freq": "词频第一", "ngram": "n-gram", "q35_0p8b": "Qwen3.5-0.8B", "q35_4b": "Qwen3.5-4B",
         "q35_9b": "Qwen3.5-9B（教师）", "student": "Suzu-2"}
MARK = re.compile(r"【([^】]+)】")


def is_cjk(s):
    return bool(s) and all("一" <= c <= "鿿" for c in s)


# ------------------------------------------------------------------ build
def count_words(train_path, work, jieba):
    cache = work / "word_counts.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    cnt = Counter()
    for i, line in enumerate(open(train_path, encoding="utf-8")):
        for w in jieba.cut(line.strip(), HMM=False):
            if len(w) <= 4 and is_cjk(w):
                cnt[w] += 1
    cache.write_bytes(pickle.dumps(cnt))
    return cnt


def parse_line(line):
    """返回 (去掉标记后的整句, [(词, 词在整句中的起始位置)])。"""
    text, targets, pos, last = [], [], 0, 0
    for m in MARK.finditer(line):
        text.append(line[last:m.start()])
        pos += len(line[last:m.start()])
        targets.append((m.group(1), pos))
        text.append(m.group(1))
        pos += len(m.group(1))
        last = m.end()
    text.append(line[last:])
    return "".join(text), targets


def cmd_build(a):
    import jieba
    from pypinyin import lazy_pinyin
    jieba.setLogLevel(30)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    cnt = count_words(a.train_sentences, work, jieba)
    min_count, K = 3, 8
    by_key = defaultdict(list)

    def pykey(w):
        p = lazy_pinyin(w)
        return " ".join(p) if len(p) == len(w) and all(x.isascii() and x.isalpha() for x in p) else None

    for w, c in cnt.items():
        if c >= min_count:
            k = pykey(w)
            if k:
                by_key[k].append((c, w))
    for k in by_key:
        by_key[k].sort(key=lambda x: (-x[0], x[1]))
    items, cover, rows = [], {}, []
    for dom in DOMAINS:
        st = Counter()
        lines = [l.strip() for l in (ROOT / "tests" / "domain" / f"{dom}.txt").read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        for line in lines:
            sent, targets = parse_line(line)
            toks = list(jieba.cut(sent, HMM=False))
            spans, p = set(), 0
            for t in toks:
                spans.add((p, p + len(t)))
                p += len(t)
            for w, pos in targets:
                st["标记词总数"] += 1
                if not is_cjk(w):
                    st["非汉字词（不适用）"] += 1
                    continue
                in_dict = jieba.dt.FREQ.get(w, 0) > 0
                single = (pos, pos + len(w)) in spans
                st["在 jieba 词典里"] += int(in_dict)
                st["被 jieba 整词切出（不被切碎）"] += int(single)
                key = pykey(w)
                lst = by_key.get(key, [])
                words = [x[1] for x in lst]
                if w not in words:
                    st["不在词频表（语料中词频低于 3 或被切碎）"] += 1
                    rows.append((dom, w, "不在词频表"))
                    continue
                st["在词频表里"] += 1
                if len(lst) < 2:
                    st["无同音词（没有歧义，不出题）"] += 1
                    continue
                top = lst[:K]
                if w not in [x[1] for x in top]:
                    st["同音词排在前 8 之外（输入法首页看不到）"] += 1
                    rows.append((dom, w, "排名超出前 8"))
                    continue
                c = [x[1] for x in top]
                t = c.index(w)
                st["产出评测题"] += 1
                items.append(dict(id=len(items), domain=dom, ctx=sent[:pos][-30:], py=key, c=c, f=[x[0] for x in top],
                                  t=t, s=int(t > 0), L=len(w), word=w))
        cover[dom] = dict(st)
    with open(work / "items_domain.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (work / "coverage.json").write_text(json.dumps(cover, ensure_ascii=False, indent=1), encoding="utf-8")
    (work / "uncovered_words.tsv").write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
    for dom in DOMAINS:
        its = [i for i in items if i["domain"] == dom]
        print(f"{NAMES[dom]}：{len(its)} 题，难例 {sum(i['s'] for i in its)}，单字词 {sum(i['L'] == 1 for i in its)}；{cover[dom]}")


# ------------------------------------------------------------------ score
def link_dir(src, dst):
    dst = Path(dst)
    if dst.exists():
        return
    if os.name == "nt":   # mklink 只认反斜杠路径，先规范化
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst.resolve()), str(Path(src).resolve())],
                       check=True, capture_output=True)
    else:
        os.symlink(src, dst, target_is_directory=True)


def cmd_score(a):
    work = Path(a.work)
    os.environ["DISTILL_DATA"] = str(work)
    sys.path.insert(0, str(ROOT / "pipeline"))
    import shutil
    if not (work / "train_sentences.txt").exists():
        shutil.copy(a.train_sentences, work / "train_sentences.txt")
    link_dir(a.student_ckpt_dir, work / "ckpt")
    import common as C
    import s2_score
    C.install_signals()
    for name in SYSTEMS:
        s2_score.run_eval(name, items="items_domain")


# ------------------------------------------------------------------ report
def acc(items, scores, lam, import_pick=None):
    ok = []
    for it in items:
        s = scores[it["id"]]
        best, bi = -1e18, 0
        for i, x in enumerate(s):
            v = x + lam * math.log1p(it["f"][i])
            if v > best:
                best, bi = v, i
        ok.append(int(bi == it["t"]))
    return ok


def mean(x):
    return sum(x) / max(len(x), 1)


def load_scores(p):
    return {json.loads(l)["id"]: json.loads(l)["s"] for l in open(p, encoding="utf-8") if l.strip()}


def cmd_report(a):
    work = Path(a.work)
    items = [json.loads(l) for l in open(work / "items_domain.jsonl", encoding="utf-8")]
    sc = {n: load_scores(work / "scores" / f"{n}.jsonl") for n in SYSTEMS}
    cover = json.loads((work / "coverage.json").read_text(encoding="utf-8"))
    out = ["# 领域评测报告", "",
           "题目为合成题（tests/domain/），出题方式与维基评测同口径。准确率为首选准确率；混合使用各系统在维基评测上选出的固定词频融合权重，"
           "不在小样本上重新调。主要比较只含不少于 2 个字的词（与维基评测一致），单字词另列。", ""]
    verdicts = []
    for dom in DOMAINS:
        its = [i for i in items if i["domain"] == dom and i["L"] >= 2]
        one = [i for i in items if i["domain"] == dom and i["L"] == 1]
        hard = [i for i in its if i["s"]]
        easy = [i for i in its if not i["s"]]
        out += [f"## {NAMES[dom]}", "", f"题数 {len(its)}（难例 {len(hard)}，易例 {len(easy)}）；单字词 {len(one)} 题另列。", "",
                "| 系统 | 纯模型 | 混合 | 难例（混合） | 易例（混合） | 单字词（混合） |", "|---|---|---|---|---|---|",
                f"| {LABEL['freq']} | {mean([int(i['t'] == 0) for i in its]) * 100:.1f}% | - | 0.0% | 100.0% | "
                f"{mean([int(i['t'] == 0) for i in one]) * 100:.1f}% |"]
        mixed = {}
        for n in SYSTEMS:
            pure = acc(its, sc[n], 0)
            mix = acc(its, sc[n], LAMBDA[n])
            mixed[n] = mix
            out.append(f"| {LABEL[n]} | {mean(pure) * 100:.1f}% | {mean(mix) * 100:.1f}% | "
                       f"{mean(acc(hard, sc[n], LAMBDA[n])) * 100:.1f}% | {mean(acc(easy, sc[n], LAMBDA[n])) * 100:.1f}% | "
                       f"{mean(acc(one, sc[n], LAMBDA[n])) * 100:.1f}% |")
        d = [x - y for x, y in zip(mixed["student"], mixed["q35_0p8b"])]
        diff = mean(d)
        se = math.sqrt(sum((x - diff) ** 2 for x in d) / max(len(d) - 1, 1) / max(len(d), 1))
        verdict = diff <= -0.03
        verdicts.append((dom, diff, se, verdict, len(its)))
        out += ["", f"Suzu-2 相对 Qwen3.5-0.8B（混合，同题配对）：{diff * 100:+.1f} 个百分点（95% 区间 ±{1.96 * se * 100:.1f}），"
                f"判定线 -3.0 → **{'低于判定线，建议补该领域语料' if verdict else '未低于判定线'}**", ""]
    out += ["## 维基评测同口径对照", ""]
    if a.wiki:
        w_items = [json.loads(l) for l in open(a.wiki_items, encoding="utf-8")]
        w_items = [i for i in w_items if len(i["c"][i["t"]]) >= 2 or True]
        out += ["维基评测集是易例与难例各一半（分层抽样），下面给出同样的原始准确率（不做自然加权），便于与上面各领域的原始准确率对照。", "",
                "| 系统 | 维基 混合 | 维基 难例 | 维基 易例 |", "|---|---|---|---|"]
        for spec in a.wiki:
            n, p = spec.split("=", 1)
            s = load_scores(p)
            hd = [i for i in w_items if i["s"] and i["id"] in s]
            ey = [i for i in w_items if not i["s"] and i["id"] in s]
            out.append(f"| {LABEL.get(n, n)} | {mean(acc(hd + ey, s, LAMBDA[n])) * 100:.1f}% | "
                       f"{mean(acc(hd, s, LAMBDA[n])) * 100:.1f}% | {mean(acc(ey, s, LAMBDA[n])) * 100:.1f}% |")
    keys = ["标记词总数", "非汉字词（不适用）", "在 jieba 词典里", "被 jieba 整词切出（不被切碎）", "在词频表里",
            "不在词频表（语料中词频低于 3 或被切碎）", "无同音词（没有歧义，不出题）",
            "同音词排在前 8 之外（输入法首页看不到）", "产出评测题"]
    out += ["", "## 分词与词频表覆盖（按名字对齐，各栏互相不排斥，含单字词）", "",
            "| 领域 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for dom in DOMAINS:
        out.append(f"| {NAMES[dom]} | " + " | ".join(str(cover[dom].get(k, 0)) for k in keys) + " |")
    out += ["", "未覆盖的词见工作目录的 uncovered_words.tsv。"]
    text = "\n".join(out) + "\n"
    (work / "domain_eval_report.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--train-sentences", required=True, dest="train_sentences")
    s = sub.add_parser("score")
    s.add_argument("--work", required=True)
    s.add_argument("--train-sentences", required=True, dest="train_sentences")
    s.add_argument("--student-ckpt-dir", required=True, dest="student_ckpt_dir")
    r = sub.add_parser("report")
    r.add_argument("--work", required=True)
    r.add_argument("--wiki", action="append", default=[], help="系统名=维基评测打分文件")
    r.add_argument("--wiki-items", dest="wiki_items")
    a = ap.parse_args()
    {"build": cmd_build, "score": cmd_score, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
