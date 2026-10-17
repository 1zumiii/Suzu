"""脚本改进清单第 7、8、9 项的等价性验证与速度对比（需要 torch，建议在显卡上运行）。
基准是第 1 到 6 项之后的 cand_scores（逐候选各算一整句）。用法：python tests/test_items_7_9.py [--bench]"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("DISTILL_DATA", tempfile.mkdtemp(prefix="items79_"))
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import student_model as S  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.RandomState(0)
VOCAB = ["<pad>", "<unk>", "<bos>"] + [chr(0x4E00 + i) for i in range(300)]
VI = {c: i for i, c in enumerate(VOCAB)}


def rand_text(n):
    return "".join(VOCAB[3 + rng.randint(300)] for _ in range(n))


def rand_items(n, maxk=8, edge=False):
    items = []
    for _ in range(n):
        k = rng.randint(2, maxk + 1)
        ctx = rand_text(rng.randint(0 if edge else 2, 80 if edge else 30))     # edge：含空上文与超长上文（会被截断）
        cands = [rand_text(rng.randint(0 if edge else 1, 5)) for _ in range(k)]
        items.append(dict(ctx=ctx, c=cands, ts=list(rng.randn(k))))
    return items


def model_of(d, L, H, seed=0):
    torch.manual_seed(seed)
    return S.Student(len(VOCAB), d, L, H, 64).to(DEV)


def check(name, fn, m, items, tol=1e-4):
    ref, new = S.cand_scores_ref(m, VI, items, DEV), fn(m, VI, items, DEV)
    assert ref.shape == new.shape, (ref.shape, new.shape)
    valid = ref > -1e3
    assert torch.equal(valid, new > -1e3)
    d = (ref - new)[valid].abs().max().item()
    assert d < tol, f"{name} 输出差异 {d}"
    for p in m.parameters():
        p.grad = None
    ref.clamp(min=-100).sum().backward()
    g_ref = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in m.parameters()]
    for p in m.parameters():
        p.grad = None
    new.clamp(min=-100).sum().backward()
    gd = max((a - (p.grad if p.grad is not None else torch.zeros_like(p))).abs().max().item()
             for a, p in zip(g_ref, m.parameters()))
    assert gd < tol, f"{name} 梯度差异 {gd}"
    return d, gd


def test_equivalence():
    for tag, (d, L, H) in {"小模型": (64, 2, 4), "中模型": (128, 4, 4)}.items():
        m = model_of(d, L, H)
        for edge in (False, True):
            items = rand_items(60, edge=edge)
            for name, fn in (("第 8 项", S.cand_scores_8), ("第 9 项", S.cand_scores_9)):
                dd, gd = check(name, fn, m, items)
                print(f"{name} {tag} {'含空上文/空候选/超长上文' if edge else '常规'}：输出最大差 {dd:.1e}，梯度最大差 {gd:.1e}  OK")
    if DEV == "cuda":   # bf16 混合精度（训练时的实际精度）：只要求在 bf16 误差内
        m = model_of(128, 4, 4)
        items = rand_items(48)
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
            ref = S.cand_scores_ref(m, VI, items, DEV)
            for name, fn in (("第 8 项", S.cand_scores_8), ("第 9 项", S.cand_scores_9)):
                new = fn(m, VI, items, DEV)
                dd = (ref - new)[ref > -1e3].abs().max().item()
                assert dd < 0.1, dd
                print(f"{name} bf16 混合精度：输出最大差 {dd:.3f}（bf16 误差范围内）  OK")


def make_batches(n):
    batches = []
    for _ in range(n):
        lm = [[S.BOS] + list(rng.randint(3, len(VOCAB), rng.randint(20, 60))) for _ in range(96)]
        kd = rand_items(48)
        for it in kd:
            it["c"] = (it["c"] + [rand_text(rng.randint(1, 4)) for _ in range(8)])[:8]
            it["ts"] = list(rng.randn(8))
        batches.append((lm, kd))
    return batches


def bench_variant(fn, pad_mult, batches, d=512, L=12, H=8):
    S.PAD_MULTIPLE = pad_mult
    torch.manual_seed(0)
    m = S.Student(len(VOCAB), d, L, H, 64).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

    def step(lm, kd):
        ids = S.pad_batch(lm, DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            lg = m(ids[:, :-1])
            lmloss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=S.PAD)
            sc = fn(m, VI, kd, DEV)
            K = sc.shape[1]
            t = np.full((len(kd), K), -1e4, dtype=np.float32)
            for i, it in enumerate(kd):
                t[i, :len(it["ts"])] = it["ts"]
            kdl = F.kl_div(F.log_softmax(sc, -1), F.softmax(torch.from_numpy(t).to(DEV), -1), reduction="batchmean")
            loss = lmloss + kdl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for b in batches[:6]:
        step(*b)
    torch.cuda.synchronize()
    t0 = time.time()
    for b in batches[6:]:
        step(*b)
    torch.cuda.synchronize()
    S.PAD_MULTIPLE = 1
    del m, opt
    torch.cuda.empty_cache()
    return (len(batches) - 6) / (time.time() - t0)


def bench():
    if DEV != "cuda":
        print("无显卡，跳过速度对比")
        return
    batches = make_batches(46)       # 每步批次长度都不同，才能体现形状变化的影响
    variants = [("基准（第 1-6 项之后）", S.cand_scores_ref, 1), ("第 7 项：补齐到 8 的倍数", S.cand_scores_ref, 8),
                ("第 8 项：只在候选位置算输出层", S.cand_scores_8, 1), ("第 9 项：候选共享上下文（含第 8 项）", S.cand_scores_9, 1),
                ("第 9 项 + 第 7 项", S.cand_scores_9, 8)]
    base = None
    for name, fn, pm in variants:
        r = bench_variant(fn, pm, batches)
        base = base or r
        print(f"  {name}: {r:.2f} 步/秒（{(r / base - 1) * 100:+.1f}%）")


if __name__ == "__main__":
    test_equivalence()
    print("等价性验证全部通过")
    if "--bench" in sys.argv:
        bench()
