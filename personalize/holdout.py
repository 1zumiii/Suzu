"""个人语料的留出集划分（不依赖 torch，可单独测试）。

默认按时间留出：取 hard 条目时间戳的分位点作为截止时间，截止时间及以后的条目（hard 与 anchor）都不参与训练，
其中的 hard 条目用来验收。按 id 哈希随机抽取时，同一个常用词前后几次选择会分到训练和留出两边，
留出集命中率偏乐观；按时间留出更接近“微调之后的日常使用”。
时间戳缺失（0）的条目一律留在训练集。
"""
import hashlib


def hash_holdout(r):
    return int(hashlib.md5(r["id"].encode("utf-8")).hexdigest()[:8], 16) % 10 == 0


def time_cutoff(recs, frac=0.1):
    """返回截止时间戳：hard 条目里最新的 frac 比例从这一刻开始。hard 条目没有时间戳时返回 None。"""
    ts = sorted(r["ts"] for r in recs if r["typ"] == "hard" and r.get("ts", 0) > 0)
    if not ts:
        return None
    k = min(len(ts) - 1, max(0, int(len(ts) * (1 - frac))))
    return ts[k]


def split(recs, mode="time", frac=0.1):
    """返回 (训练条目, 留出条目, 截止时间戳或 None)。留出条目里 anchor 也在内，调用方只拿 hard 做验收。"""
    if mode == "hash":
        tr = [r for r in recs if not hash_holdout(r)]
        ho = [r for r in recs if hash_holdout(r)]
        return tr, ho, None
    cut = time_cutoff(recs, frac)
    if cut is None:
        return split(recs, "hash", frac)
    ho = [r for r in recs if r.get("ts", 0) > 0 and r["ts"] >= cut]
    tr = [r for r in recs if not (r.get("ts", 0) > 0 and r["ts"] >= cut)]
    return tr, ho, cut


def to_unix(ts):
    """select_corpus 的时间戳是分钟；也兼容秒。"""
    return ts * 60 if 0 < ts < 100_000_000 else ts
