"""阶段 0：准备干净语料 → data/sentences.txt（一行一句，简体）。
默认下载维基百科中文版第一个分片（约 590 MB，HF 数据集 wikimedia/wikipedia），
也可用环境变量 DISTILL_CORPUS 或 pipeline.py --corpus 指定本地语料（一行一句）。
"""
import os
import random
import re
import shutil

import common as C
from common import CFG, DATA, log

ALLOWED = set("，、：；“”‘’（）《》…—·")
SENT_RE = re.compile(r"[^。！？\n]+[。！？]")


def clean_sentences(text, cfg):
    """从一篇文章里抽出合格句子（去掉句末标点）。纯函数，便于测试。"""
    out = []
    for m in SENT_RE.findall(text):
        s = m[:-1].strip()
        if not (cfg["min_len"] <= len(s) <= cfg["max_len"]):
            continue
        if all(("一" <= c <= "鿿") or c in ALLOWED for c in s):
            out.append(s)
    return out


def run():
    dst = DATA / "sentences.txt"
    local = os.environ.get("DISTILL_CORPUS")
    if local:
        seen, rows = set(), []
        for line in open(local, encoding="utf-8"):
            s = line.strip()
            if s and s not in seen:
                seen.add(s)
                rows.append(s)
        C.atomic_write(dst, "\n".join(rows[: CFG["n_sentences"]]) + "\n")
        log(f"使用本地语料 {local}，{min(len(rows), CFG['n_sentences'])} 句")
        return

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from opencc import OpenCC

    log("下载/校验维基百科中文分片（已下载过则秒过，中断可续传）…")
    cc = OpenCC("t2s")
    rng = random.Random(0)
    seen, rows = set(), []
    for shard in range(6 if C.ROUND2 else 1):   # 第二轮用全部 6 个分片
        path = hf_hub_download("wikimedia/wikipedia", f"20231101.zh/train-0000{shard}-of-00006.parquet", repo_type="dataset")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["text"]):
            C.check_stop()
            for text in batch.column("text").to_pylist():
                cands = clean_sentences(cc.convert(text), CFG)   # 先转简体再过滤，保证字集一致
                rng.shuffle(cands)
                for s in cands[: CFG["per_article"]]:
                    if s not in seen:
                        seen.add(s)
                        rows.append(s)
            if len(rows) >= CFG["n_sentences"]:
                break
        log(f"分片 {shard} 处理完，累计 {len(rows)} 句")
        if len(rows) >= CFG["n_sentences"]:
            break
    rows = rows[: CFG["n_sentences"]]
    C.atomic_write(dst, "\n".join(rows) + "\n")
    log(f"语料完成：{len(rows)} 句 → {dst}")
