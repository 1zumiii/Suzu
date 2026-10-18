"""第三轮：过滤合成领域语料。繁转简（OpenCC t2s）、去编号/引号/标点前缀，长度 8–30 个字，
去掉含拉丁字母过多、重复字符、拒答/解释类句子的行；整句去重，并按 4 字片段的重叠做近重复去重；
再与评测/开发用的领域句子（tests/domain/*.txt 与 --exclude 指定的文件）做 4 字片段重叠过滤，防止泄漏。
用法：python tools/filter_domain_corpus.py --raw <raw_xxx.jsonl> --out <输出txt> [--exclude 文件 ...] [--max-dup-ratio 0.5]"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HAN = re.compile(r"[一-鿿]")
PREFIX = re.compile(r"^\s*(?:[-*•·]+|\(?\d+[.、)）:：]|[（(]\d+[)）])\s*")
ALLOWED = re.compile(r"^[一-鿿　-〿＀-￯ -~—…‘’“”]+$")
BAD = ("以下是", "好的", "当然", "这里是", "当然可以", "希望", "如果你", "请注意", "例如：", "句子", "示例")


def clean(line, cc):
    line = PREFIX.sub("", line.strip()).strip(" \t\"“”'‘’「」『』")
    line = cc.convert(line)
    return line


def grams(s, n=4):
    s = re.sub(r"[^一-鿿A-Za-z0-9]", "", s)
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def han_len(s):
    return len(HAN.findall(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--max-dup-ratio", type=float, default=0.5, help="与已留句子共享的 4 字片段占比上限")
    ap.add_argument("--leak-ratio", type=float, default=0.5, help="与测试句共享的 4 字片段占比达到此值则丢弃")
    a = ap.parse_args()
    from opencc import OpenCC
    cc = OpenCC("t2s")
    test_grams = set()
    files = list((ROOT / "tests" / "domain").glob("*.txt")) + [Path(p) for p in a.exclude]
    for p in files:
        for l in open(p, encoding="utf-8"):
            if l.strip() and not l.startswith("#"):
                test_grams |= grams(l.replace("【", "").replace("】", ""))
    seen, seen_grams, kept = set(), {}, []
    stat = dict(total=0, short_long=0, junk=0, dup=0, near=0, leak=0)
    for r in open(a.raw, encoding="utf-8"):
        t = clean(json.loads(r)["t"], cc)
        stat["total"] += 1
        n = han_len(t)
        if not 8 <= n <= 30 or n < 0.7 * len(t):
            stat["short_long"] += 1
            continue
        if not ALLOWED.match(t) or any(b in t for b in BAD) or re.search(r"(.)\1{3,}", t) or len(re.findall(r"[A-Za-z]", t)) > 12:
            stat["junk"] += 1
            continue
        if t in seen:
            stat["dup"] += 1
            continue
        g = grams(t)
        if g and len(g & test_grams) / len(g) >= a.leak_ratio:
            stat["leak"] += 1
            continue
        if g:
            hit = sum(1 for x in g if x in seen_grams)
            if hit / len(g) > a.max_dup_ratio:
                stat["near"] += 1
                continue
        seen.add(t)
        for x in g:
            seen_grams[x] = 1
        kept.append(t)
    Path(a.out).write_text("\n".join(kept) + "\n", encoding="utf-8")
    stat["kept"] = len(kept)
    print(json.dumps(stat, ensure_ascii=False))


if __name__ == "__main__":
    main()
