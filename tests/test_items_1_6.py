"""脚本改进清单第 1 到 6 项的等价性验证与速度对比（需要 torch，建议在显卡上运行）。
旧实现原样内嵌在本文件里，作为对照。用法：python tests/test_items_1_6.py [--bench]"""
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

TMP = Path(tempfile.mkdtemp(prefix="items16_"))
os.environ["DISTILL_DATA"] = str(TMP)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402
import s4_train  # noqa: E402
import student_model as S  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 旧实现（原样）
def old_pad_batch(seqs, device):
    L = max(len(s) for s in seqs)
    t = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, :len(s)] = torch.tensor(s)
    return t.to(device)


def old_cand_scores(model, vocab_index, items, device):
    seqs, spans, owner = [], [], []
    for n, it in enumerate(items):
        ctx = S.encode(vocab_index, it["ctx"])[-(model.ctx - 6):]
        for k, c in enumerate(it["c"]):
            cs = S.encode(vocab_index, c)
            seqs.append([S.BOS] + ctx + cs)
            spans.append((len(ctx), len(cs)))
            owner.append((n, k))
    ids = old_pad_batch(seqs, device)
    logits = model(ids[:, :-1])
    lp = F.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    j = torch.arange(lp.shape[1], device=device)[None]
    lo = torch.tensor([a for a, _ in spans], device=device)[:, None]
    hi = torch.tensor([a + b for a, b in spans], device=device)[:, None]
    tot = (lp * ((j >= lo) & (j < hi))).sum(1)
    K = max(len(it["c"]) for it in items)
    out = torch.full((len(items), K), -1e4, device=device)
    for s, (n, k) in enumerate(owner):
        out[n, k] = tot[s]
    return out


def old_tgt(part, K, dev):
    tgt = torch.full((len(part), K), -1e4, device=dev)
    for n, it in enumerate(part):
        tgt[n, :len(it["ts"])] = torch.tensor(it["ts"], device=dev)
    return tgt


# ---------------------------------------------------------------- 合成数据
rng = np.random.RandomState(0)
VOCAB = ["<pad>", "<unk>", "<bos>"] + [chr(0x4E00 + i) for i in range(300)]
VI = {c: i for i, c in enumerate(VOCAB)}


def rand_text(n):
    return "".join(VOCAB[3 + rng.randint(300)] for _ in range(n))


def rand_items(n, maxk=8):
    items = []
    for _ in range(n):
        k = rng.randint(2, maxk + 1)
        items.append(dict(ctx=rand_text(rng.randint(2, 30)), c=[rand_text(rng.randint(1, 4)) for _ in range(k)],
                          t=0, s=int(rng.rand() < 0.5), ts=list(rng.randn(k))))
    return items


def tiny_model():
    torch.manual_seed(0)
    return S.Student(len(VOCAB), 64, 2, 4, 64).to(DEV)


# ---------------------------------------------------------------- 各项验证
def test_cand_scores():
    m = tiny_model()
    items = rand_items(40)
    new, old = S.cand_scores(m, VI, items, DEV), old_cand_scores(m, VI, items, DEV)
    assert new.shape == old.shape
    d = (new - old).abs().max().item()
    assert d < 1e-4, f"cand_scores 输出差异 {d}"
    m.zero_grad()
    (new.clamp(min=-100)).sum().backward()
    g_new = m.emb.weight.grad.clone()
    m.zero_grad()
    (old.clamp(min=-100)).sum().backward()
    gd = (g_new - m.emb.weight.grad).abs().max().item()
    assert gd < 1e-4, f"梯度差异 {gd}"
    print(f"第 4 项 cand_scores：输出最大差 {d:.2e}，梯度最大差 {gd:.2e}  OK")


def test_pad_batch():
    seqs = [list(rng.randint(0, 300, rng.randint(1, 50))) for _ in range(64)]
    assert torch.equal(S.pad_batch(seqs, "cpu"), old_pad_batch(seqs, "cpu"))
    arrs = [np.array(s, dtype=np.int16) for s in seqs]
    assert torch.equal(S.pad_batch(arrs, "cpu"), old_pad_batch(seqs, "cpu"))
    print("第 4 项 pad_batch：与旧实现逐元素相同（列表与 int16 数组）  OK")


def test_tgt():
    part = rand_items(48)
    K = max(len(it["ts"]) for it in part)
    tgt_np = np.full((len(part), K), -1e4, dtype=np.float32)
    for n, it in enumerate(part):
        tgt_np[n, :len(it["ts"])] = it["ts"]
    d = (torch.from_numpy(tgt_np) - old_tgt(part, K, "cpu").float()).abs().max().item()
    assert d < 1e-6, d
    print(f"第 4 项 tgt 拼装：最大差 {d:.1e}  OK")


def test_enc_cache():
    sents = ["".join(rand_text(rng.randint(5, 60))) for _ in range(2000)]
    old = [[S.BOS] + S.encode(VI, s)[:63] for s in sents]
    t0 = time.time()
    st = s4_train.load_or_build_enc(sents, VI, VOCAB, 64)
    t_build = time.time() - t0
    assert (C.CKPT / "enc_cache.npz").exists()
    t0 = time.time()
    st2 = s4_train.load_or_build_enc(sents, VI, VOCAB, 64)
    t_load = time.time() - t0
    assert len(st) == len(st2) == len(old)
    for i in range(len(old)):
        assert list(st[i]) == old[i] == list(st2[i]), i
    # 词表变化时必须重建
    v2 = VOCAB[:-1] + ["<x>"]
    st3 = s4_train.load_or_build_enc(sents, {c: i for i, c in enumerate(v2)}, v2, 64)
    assert len(st3) == len(old)
    print(f"第 5 项 预编码：{len(old)} 句逐条相同；构建 {t_build:.2f}s，缓存加载 {t_load:.3f}s，词表变化会重建  OK")


def test_fused_adamw():
    if DEV != "cuda":
        print("第 6 项 fused AdamW：无显卡，跳过")
        return
    ma, mb = tiny_model(), tiny_model()
    oa = torch.optim.AdamW(ma.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    ob = torch.optim.AdamW(mb.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    g = torch.Generator(device=DEV).manual_seed(1)
    for _ in range(5):
        # 给两个优化器喂完全相同的梯度：显卡上的反传本身有微小的非确定性，会被 Adam 放大，不能用来比较优化器
        grads = [torch.randn(p.shape, device=DEV, generator=g) * 1e-2 for p in ma.parameters()]
        for m, o in ((ma, oa), (mb, ob)):
            for p, gr in zip(m.parameters(), grads):
                p.grad = gr.clone()
            o.step()
    d = max((p - q).abs().max().item() for p, q in zip(ma.parameters(), mb.parameters()))
    assert d < 1e-5, d
    print(f"第 6 项 fused AdamW：5 步后参数最大差 {d:.2e}  OK")
    # 旧检查点（含未融合优化器状态）能被融合优化器加载
    ob.load_state_dict(oa.state_dict())
    print("第 6 项：未融合优化器的状态可被 fused 优化器加载  OK")


def test_progress_window():
    clock = [1000.0]
    with mock.patch("common.time.time", lambda: clock[0]):
        p = C.Progress("t", 10**6)
        for sec in range(900):              # 前 600 秒每秒 5 步，后 300 秒每秒 1 步
            clock[0] += 1
            p.tick(5 if sec < 600 else 1)
        t_a, d_a = p.hist[0]
        win = (p.done - d_a) / (clock[0] - t_a)
        cum = (p.done - p.done0) / (clock[0] - p.t0)
    assert abs(win - 1.0) < 0.1, win
    print(f"第 1 项 进度速度：窗口速度 {win:.2f}/s（真实 1/s），旧的累计平均会显示 {cum:.2f}/s  OK")


def bench():
    if DEV != "cuda":
        print("无显卡，跳过速度对比")
        return
    V = len(VOCAB)
    results = {}
    for mode in ("old", "new"):
        torch.manual_seed(0)
        m = S.Student(6000, 512, 12, 8, 64).to(DEV)
        vi = {c: i for i, c in enumerate(VOCAB)}
        fused = mode == "new"
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=fused)
        cs = old_cand_scores if mode == "old" else S.cand_scores
        pad = old_pad_batch if mode == "old" else S.pad_batch
        lm_seqs = [[S.BOS] + list(rng.randint(3, V, rng.randint(20, 60))) for _ in range(96)]
        kd = rand_items(48)
        for it in kd:
            it["c"] = it["c"] + [rand_text(2) for _ in range(8 - len(it["c"]))]
            it["ts"] = list(rng.randn(8))

        def step():
            ids = pad(lm_seqs, DEV)
            with torch.autocast(DEV, dtype=torch.bfloat16):
                lg = m(ids[:, :-1])
                lm = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=S.PAD)
                sc = cs(m, vi, kd, DEV)
                K = sc.shape[1]
                if mode == "old":
                    tgt = old_tgt(kd, K, DEV)
                else:
                    t = np.full((len(kd), K), -1e4, dtype=np.float32)
                    for n, it in enumerate(kd):
                        t[n, :len(it["ts"])] = it["ts"]
                    tgt = torch.from_numpy(t).to(DEV)
                kdl = F.kl_div(F.log_softmax(sc, -1), F.softmax(tgt, -1), reduction="batchmean")
                loss = lm + kdl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        for _ in range(8):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(40):
            step()
        torch.cuda.synchronize()
        results[mode] = 40 / (time.time() - t0)
        print(f"  {mode}: {results[mode]:.2f} 步/秒")
        del m, opt
        torch.cuda.empty_cache()
    print(f"速度对比（40.9M 学生，LM 批 96，蒸馏批 48×8 候选）：旧 {results['old']:.2f} → 新 {results['new']:.2f} 步/秒，"
          f"提升 {(results['new'] / results['old'] - 1) * 100:+.1f}%")


def main():
    test_cand_scores()
    test_pad_batch()
    test_tgt()
    test_enc_cache()
    test_fused_adamw()
    test_progress_window()
    print("等价性验证全部通过")
    if "--bench" in sys.argv:
        bench()


if __name__ == "__main__":
    main()
