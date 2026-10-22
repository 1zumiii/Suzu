"""训练一步的耗时剖析：与 pipeline/s4_train.py 相同的数据、抽样与损失，分段计时（每段之后同步显卡），另测不同步时的真实步速。
不写检查点，只打印数字。环境变量与 s4_train 相同（DISTILL_DATA、DISTILL_RUNDIR、DISTILL_ROUND、DISTILL_TEACHER、DISTILL_STUDENT_* 等），
DISTILL_RUNDIR 下要有 ckpt/vocab.json（若有 ckpt/enc_cache.npz 则直接用）。
用法：python tools/train_profile.py [--steps 60] [--warmup 10] [--pack 160]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pack", type=int, default=0, help="row capacity of the packed batches (0 = padded, as in training without DISTILL_PACK)")
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    import common as C
    from common import CFG, CKPT, DATA, SCORES
    from s4_train import load_or_build_enc
    from student_model import PAD, Student, cand_batch, cand_scores_run, lm_batch_packed, pad_batch
    import json
    import os
    dev = "cuda"
    cfg = CFG
    t0 = time.time()
    sents = [l.strip() for l in open(DATA / "train_sentences.txt", encoding="utf-8") if l.strip()]
    vocab = json.loads((CKPT / "vocab.json").read_text(encoding="utf-8"))
    vi = {c: i for i, c in enumerate(vocab)}
    enc = load_or_build_enc(sents, vi, vocab, cfg["student_ctx"])
    tname = os.environ["DISTILL_TEACHER"]
    ts = {r["id"]: r["s"] for r in C.read_jsonl(SCORES / f"{tname}__train.jsonl")}
    kd_items = [it for it in C.read_jsonl(DATA / "items_train.jsonl") if it["id"] in ts]
    for it in kd_items:
        it["ts"] = ts[it["id"]]
        it["kw"] = 1.0
    print(f"data loaded in {time.time() - t0:.0f} s: sentences {len(enc)}, items {len(kd_items)}", flush=True)
    model = Student(len(vocab), cfg["student_d"], cfg["student_layers"], cfg["student_heads"], cfg["student_ctx"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M, shape d={cfg['student_d']} L={cfg['student_layers']} "
          f"H={cfg['student_heads']} ctx={cfg['student_ctx']}, batch lm={cfg['student_batch_lm']} kd={cfg['student_batch_kd']}", flush=True)

    def batches(step):
        rs = np.random.RandomState(step)
        idx = rs.randint(0, len(enc), cfg["student_batch_lm"])
        lm = [enc[i] for i in idx]
        part = [kd_items[i] for i in rs.randint(0, len(kd_items), cfg["student_batch_kd"])]
        return lm, part

    sec = {k: [] for k in ("prep_lm", "prep_kd", "fwd_lm", "fwd_kd", "backward", "optimizer")}
    sizes = []

    def one(step, timed):
        def mark(name, t):
            if timed:
                torch.cuda.synchronize()
                sec[name].append(time.perf_counter() - t)
            return time.perf_counter()
        t = time.perf_counter()
        lm, part = batches(step)
        if a.pack:
            pb = {k: torch.from_numpy(v).to(dev) for k, v in lm_batch_packed(lm, a.pack).items()}
            ids = pb["ids"]
        else:
            ids = pad_batch(lm, dev)
        t = mark("prep_lm", t)
        b = cand_batch(model.ctx, vi, part, a.pack)
        K = b["K"]
        tgt_np = np.full((len(part), K), -1e4, dtype=np.float32)
        for n, it in enumerate(part):
            tgt_np[n, :len(it["ts"])] = it["ts"]
        tgt = torch.from_numpy(tgt_np).to(dev)
        if timed:
            sizes.append((ids.shape[1], b["ids"].shape[1]))
        t = mark("prep_kd", t)
        with torch.autocast(dev, dtype=torch.bfloat16):
            if a.pack:
                H = model.hidden(pb["ids"], pos=pb["pp"], mask=pb["mk"])
                lmv = F.cross_entropy((H[pb["r"], pb["c"]] @ model.emb.weight.T).float(), pb["t"])
            else:
                lg = model(ids[:, :-1])
                lmv = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=PAD)
            t = mark("fwd_lm", t)
            sc = cand_scores_run(model, b, dev)
            pt = F.softmax(tgt, -1)
            kdv = F.kl_div(F.log_softmax(sc, -1), pt, reduction="none").sum(-1)
            wv = torch.tensor([1.0 + cfg.get("kd_hard_weight", 0.0) * it["s"] for it in part], device=dev)
            loss = lmv + (kdv * wv).sum() / wv.sum()
            t = mark("fwd_kd", t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        t = mark("backward", t)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        mark("optimizer", t)

    model.train()
    for s in range(a.warmup):
        one(s, False)
    torch.cuda.synchronize()
    for s in range(a.warmup, a.warmup + a.steps):
        one(s, True)
    tot = sum(sum(v) for v in sec.values())
    print("section breakdown (each section followed by a GPU sync):")
    for k, v in sec.items():
        print(f"  {k:10s} {1000 * np.mean(v):7.1f} ms  {100 * sum(v) / tot:5.1f}%")
    print(f"  total      {1000 * tot / a.steps:7.1f} ms per step")
    lens = np.array(sizes)
    print(f"padded lengths: LM batch median {int(np.median(lens[:, 0]))}, KD batch median {int(np.median(lens[:, 1]))} (max {lens[:, 1].max()})")
    torch.cuda.synchronize()
    t = time.perf_counter()
    for s in range(a.warmup + a.steps, a.warmup + 2 * a.steps):
        one(s, False)
    torch.cuda.synchronize()
    print(f"real step time without syncs: {1000 * (time.perf_counter() - t) / a.steps:.1f} ms ({a.steps / (time.perf_counter() - t):.2f} steps/s)")
    print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
