"""Shared helpers for the afternoon tasks T1 to T5: rime-ice dictionary index, corpus counts, pinyin keys."""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import os
DICT_DIR = Path(os.environ.get("CN_DICTS", "cn_dicts"))   # 词库目录（rime-suzu 的词库包安装后的 cn_dicts）
CORPUS_FILES = [r"D:\distill-data-r2\train_sentences.txt", r"D:\distill-data-r3\big\chat.txt", r"D:\distill-data-r3\big\tech.txt",
                r"D:\distill-data-r3\big\anime.txt", r"D:\distill-data-r4\slang\slang.txt"]
CJK = re.compile(r"^[一-鿿]+$")


def read_dict(name):
    on = False
    for ln in open(DICT_DIR / f"{name}.dict.yaml", encoding="utf-8"):
        if ln.startswith("..."):
            on = True
        elif on and ln.strip() and not ln.startswith("#"):
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 2:
                yield f


def build_index():
    """index[pinyin key] = {word: weight}; entries[word] = list of (pinyin key, weight, dict name); chars = char -> first pinyin."""
    chars, index, entries = {}, defaultdict(dict), defaultdict(list)
    for name in ("8105", "base", "ext", "others"):
        for f in read_dict(name):
            w, py = f[0], f[1]
            wt = int(f[2]) if len(f) > 2 and f[2].isdigit() else 100
            if name == "8105" and w not in chars:
                chars[w] = py
            if len(w) == len(py.split()):
                index[py][w] = max(wt, index[py].get(w, 0))
                entries[w].append((py, wt, name))
    for f in read_dict("tencent"):
        w = f[0]
        if CJK.match(w) and all(c in chars for c in w):
            py = " ".join(chars[c] for c in w)
            wt = int(f[1]) if len(f) > 1 and f[1].isdigit() else 100
            index[py].setdefault(w, wt)
            entries[w].append((py, wt, "tencent"))
    return index, entries, chars


def best_key(entries, w):
    return max(entries[w], key=lambda e: e[1])[0] if w in entries else None


def candidates(index, entries, w, k=12):
    """Same-pinyin same-length words, weight descending (ties by word), first k."""
    py = best_key(entries, w)
    if py is None:
        return None, []
    d = index[py]
    ranked = sorted(d.items(), key=lambda x: (-x[1], x[0]))
    return py, [x[0] for x in ranked if len(x[0]) == len(w)][:k]


def corpus_counts(words, files=CORPUS_FILES, maxlen=8):
    """Occurrences (substring matches) of each word in the training corpus."""
    by_len = defaultdict(set)
    for w in words:
        by_len[len(w)].add(w)
    lens = sorted(n for n in by_len if n <= maxlen)
    cnt = Counter()
    for p in files:
        try:
            f = open(p, encoding="utf-8")
        except OSError:
            continue
        with f:
            for ln in f:
                s = ln.strip()
                L = len(s)
                for n in lens:
                    ws = by_len[n]
                    for i in range(L - n + 1):
                        sub = s[i:i + n]
                        if sub in ws:
                            cnt[sub] += 1
    return cnt
