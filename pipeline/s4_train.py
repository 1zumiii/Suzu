"""阶段 4：蒸馏学生。损失 = 字符语言模型交叉熵 + kd_weight × 候选排序的 KL（教师分数的 softmax 为目标）。
断点续跑：每 ckpt_every 步原子写一次 ckpt/student_last.pt（模型、优化器、步数、配置）；
每一步的随机抽样只取决于步数，所以续跑后数据顺序与不中断时一致。
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from common import CFG, CKPT, DATA, SCORES, log
from student_model import BOS, PAD, Student, build_vocab, cand_scores, encode, pad_batch


class EncStore:
    """预编码的语料：一维 int16 数组加偏移，按下标取出整句，不再占用数百万个 Python 列表。"""

    def __init__(self, flat, off):
        self.flat, self.off = flat, off

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        return self.flat[self.off[i]:self.off[i + 1]]


def load_or_build_enc(sents, vi, vocab, ctx):
    """句子编码（[BOS] + 字符编号，截到 ctx）缓存到 ckpt/enc_cache.npz，键含词表与句子指纹，不一致则重建。"""
    import hashlib
    key = hashlib.md5(("|".join(vocab) + f"|{ctx}|{len(sents)}|{sents[0]}|{sents[-1]}|{sum(map(len, sents[::997]))}"
                       ).encode("utf-8")).hexdigest()
    path = CKPT / "enc_cache.npz"
    if path.exists():
        z = np.load(path)
        if str(z["key"]) == key:
            return EncStore(z["flat"], z["off"])
    rows = [[BOS] + encode(vi, s)[: ctx - 1] for s in sents]
    off = np.zeros(len(rows) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(r) for r in rows])
    flat = np.fromiter((x for r in rows for x in r), dtype=np.int16, count=int(off[-1]))
    np.savez(path, flat=flat, off=off, key=np.array(key))
    return EncStore(flat, off)


def teacher_name():
    if os.environ.get("DISTILL_TEACHER"):
        return os.environ["DISTILL_TEACHER"]
    ms = C.models()
    return next(k for k, v in ms.items() if v.get("role") == "teacher") if any(
        v.get("role") == "teacher" for v in ms.values()) else next(iter(ms))


def evaluate(model, vi, items, w, dev):
    model.eval()
    c, n = [0, 0], [0, 0]
    with torch.no_grad():
        for a in range(0, len(items), 64):
            part = items[a:a + 64]
            sc = cand_scores(model, vi, part, dev).argmax(1).tolist()
            for it, p in zip(part, sc):
                n[it["s"]] += 1
                c[it["s"]] += p == it["t"]
    model.train()
    acc = [c[i] / max(n[i], 1) for i in (0, 1)]
    return dict(easy=acc[0], hard=acc[1], nat=w[0] * acc[0] + w[1] * acc[1])


def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CFG
    sents = [l.strip() for l in open(DATA / "train_sentences.txt", encoding="utf-8") if l.strip()]
    vpath = CKPT / "vocab.json"
    if vpath.exists():
        vocab = json.loads(vpath.read_text(encoding="utf-8"))
    else:
        vocab = build_vocab(sents, cfg["vocab_size"])
        C.write_json(vpath, vocab)
    vi = {c: i for i, c in enumerate(vocab)}
    enc = load_or_build_enc(sents, vi, vocab, cfg["student_ctx"])

    seed = int(os.environ.get("DISTILL_SEED", "0"))      # 0 keeps the original sampling order
    torch.manual_seed(seed)
    tname = teacher_name()
    ts = {r["id"]: r["s"] for r in C.read_jsonl(SCORES / f"{tname}__train.jsonl")}
    kd_items = [it for it in C.read_jsonl(DATA / "items_train.jsonl") if it["id"] in ts]
    for it in kd_items:
        it["ts"] = ts[it["id"]]
    maxc = int(os.environ.get("DISTILL_KD_MAXCAND", "0"))   # sweep: at most this many candidates per item, the answer always kept
    if maxc:
        for it in kd_items:
            if len(it["c"]) > maxc:
                idx = list(range(maxc))
                if it["t"] >= maxc:
                    idx[-1] = it["t"]
                it["ts"] = [it["ts"][i] for i in idx]
                it["c"] = [it["c"][i] for i in idx]
                it["f"] = [it["f"][i] for i in idx]
                it["t"] = idx.index(it["t"])
    ev_items = C.read_jsonl(DATA / "items_dev.jsonl")   # 开发集：只用于挑检查点，不参与最终评测
    meta = C.read_json(DATA / "meta.json")
    nat = meta["eval_natural"]
    w = [nat[0] / sum(nat), nat[1] / sum(nat)]
    log(f"学生训练：句子 {len(enc)}，蒸馏题 {len(kd_items)}（教师 {tname}），词表 {len(vocab)}，设备 {dev}")

    model = Student(len(vocab), cfg["student_d"], cfg["student_layers"], cfg["student_heads"], cfg["student_ctx"]).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    log(f"学生参数量 {nparams/1e6:.1f} M")
    if os.environ.get("DISTILL_INIT") and not (CKPT / "student_last.pt").exists():   # 试点：从已有学生（如 Suzu-2）的权重起步
        model.load_state_dict(torch.load(os.environ["DISTILL_INIT"], map_location=dev)["model"])
        log(f"从 {os.environ['DISTILL_INIT']} 初始化权重")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["student_lr"], betas=(0.9, 0.95), weight_decay=0.1,
                            fused=(dev == "cuda"))
    steps = cfg["student_steps"]
    step = 0
    last = CKPT / "student_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        log(f"从第 {step} 步续跑")
    best_path = CKPT / "student_best.pt"
    best = C.read_json(CKPT / "best.json", {"nat": -1})

    def save(path, extra=None, with_opt=True):
        tmp = path.with_name(path.name + ".tmp")
        state = dict(model=model.state_dict(), step=step, cfg=cfg, vocab=vocab, **(extra or {}))
        if with_opt:                           # student_best.pt 只用于导出和打分，不存优化器状态
            state["opt"] = opt.state_dict()
        torch.save(state, tmp)
        os.replace(tmp, path)

    def lr_at(s):
        warm = min(500, steps // 10)
        if s < warm:
            return cfg["student_lr"] * (s + 1) / warm
        fl = float(os.environ.get("DISTILL_LR_FLOOR", "0.1"))
        return cfg["student_lr"] * (fl + (1 - fl) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))

    model.train()
    prog = C.Progress("train", steps, step)
    while step < steps:
        try:
            C.check_stop()
        except C.StopRequested:
            save(last)
            log(f"收到停止请求，已在第 {step} 步存盘")
            raise
        rs = np.random.RandomState(step + seed * 1_000_003)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = rs.randint(0, len(enc), cfg["student_batch_lm"])
        ids = pad_batch([enc[i] for i in idx], dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            lg = model(ids[:, :-1])
            lm = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=PAD)
            kd = torch.zeros((), device=dev)
            if kd_items:
                part = [kd_items[i] for i in rs.randint(0, len(kd_items), cfg["student_batch_kd"])]
                sc = cand_scores(model, vi, part, dev)
                K = sc.shape[1]
                tgt_np = np.full((len(part), K), -1e4, dtype=np.float32)
                for n, it in enumerate(part):
                    tgt_np[n, :len(it["ts"])] = it["ts"]
                tgt = torch.from_numpy(tgt_np).to(dev)   # 在 numpy 里拼好，只传一次显卡
                pt = F.softmax(tgt / cfg["kd_temp"], -1)
                kdv = F.kl_div(F.log_softmax(sc / cfg["kd_temp"], -1), pt, reduction="none").sum(-1)
                wv = torch.tensor([1.0 + cfg.get("kd_hard_weight", 0.0) * it["s"] for it in part], device=dev)   # 难例加权
                kd = (kdv * wv).sum() / wv.sum()
            loss = lm + cfg["kd_weight"] * kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        prog.tick()
        if step % 100 == 0:
            C.write_json(C.STATE / "train_last.json", dict(step=step, lm=lm.item(), kd=kd.detach().item(), ts=time.time()))
        if step % cfg["eval_every"] == 0 or step == steps:
            m = evaluate(model, vi, ev_items, w, dev)
            log(f"[train] step {step} lm={lm.item():.3f} kd={kd.detach().item():.3f} 评测 自然加权 {m['nat']*100:.1f}% 难例 {m['hard']*100:.1f}%")
            with open(C.LOGS / "train_metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(step=step, lm=lm.item(), kd=kd.detach().item(), **m)) + "\n")
            if m["nat"] > best["nat"]:
                best = dict(nat=m["nat"], step=step)
                save(best_path, with_opt=False)
                C.write_json(CKPT / "best.json", best)
        if step % cfg["ckpt_every"] == 0:
            save(last)
    save(last)
    if not best_path.exists():
        save(best_path, with_opt=False)

    # 粗测 CPU 单线程延迟：一题 = 一组候选
    from student_model import load_student, score_items
    torch.set_num_threads(1)
    m2, v2 = load_student(best_path, "cpu")
    its = ev_items[:200]
    score_items(m2, v2, its[:10], "cpu")
    t0 = time.time()
    score_items(m2, v2, its, "cpu")
    C.write_json(C.STATE / "student_bench.json", dict(params_m=nparams / 1e6, ms_per_item=(time.time() - t0) / len(its) * 1000))
