"""拼行训练（DISTILL_PACK）的等价性验证：同一批句子与题目，补齐版与拼行版的语言模型损失、候选分数及其梯度应在浮点误差内相同。
用随机初始化的小模型和编造的数据，不读任何数据目录。用法：python tests/test_pack.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from student_model import BOS, PAD, Student, cand_batch, cand_scores_run, encode, lm_batch_packed, pad_batch  # noqa: E402

CHARS = [chr(0x4e00 + i) for i in range(300)]


def grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def check(dev, dtype, cap, rng):
    torch.manual_seed(0)
    vocab = ["<pad>", "<unk>", "<s>"] + CHARS
    vi = {c: i for i, c in enumerate(vocab)}
    model = Student(len(vocab), 64, 2, 4, 40).to(dev)
    model.train()
    sents = [np.array([BOS] + [rng.randrange(3, len(vocab)) for _ in range(rng.randint(1, 38))], dtype=np.int16) for _ in range(24)]
    items = []
    for _ in range(12):
        ctx = "".join(rng.choice(CHARS) for _ in range(rng.choice([0, 1, 5, 20, 50])))       # empty and over-long contexts included
        items.append({"ctx": ctx, "c": ["".join(rng.choice(CHARS) for _ in range(rng.randint(1, 3))) for _ in range(rng.randint(1, 9))]})
    out = {}
    for mode in ("padded", "packed"):
        model.zero_grad(set_to_none=True)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dtype == "bf16"):
            if mode == "padded":
                ids = pad_batch(sents, dev)
                lg = model(ids[:, :-1])
                lm = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=PAD)
                sc = cand_scores_run(model, cand_batch(model.ctx, vi, items), dev)
            else:
                pb = {k: torch.from_numpy(v).to(dev) for k, v in lm_batch_packed(sents, cap).items()}
                H = model.hidden(pb["ids"], pos=pb["pp"], mask=pb["mk"])
                lm = F.cross_entropy((H[pb["r"], pb["c"]] @ model.emb.weight.T).float(), pb["t"])
                sc = cand_scores_run(model, cand_batch(model.ctx, vi, items, cap), dev)
            valid = sc > -1e3
            loss = lm + (sc * valid).sum() / valid.sum()
        loss.backward()
        out[mode] = (lm.item(), sc.detach().float().cpu(), grads(model).float().cpu())
    d_lm = abs(out["padded"][0] - out["packed"][0])
    d_sc = (out["padded"][1] - out["packed"][1]).abs().max().item()
    d_g = (out["padded"][2] - out["packed"][2]).abs().max().item()
    tol = (2e-2, 5e-2, 5e-2) if dtype == "bf16" else (1e-4, 1e-3, 1e-3)
    ok = d_lm < tol[0] and d_sc < tol[1] and d_g < tol[2]
    print(f"{dev} {dtype} cap={cap}: LM loss diff {d_lm:.2e}, score diff {d_sc:.2e}, gradient diff {d_g:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    rng = random.Random(3)
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    ok = True
    for dev in devs:
        for cap in (0, 40, 96):
            if cap == 0:
                continue
            ok &= check(dev, "fp32", cap, rng)
        if dev == "cuda":
            ok &= check(dev, "bf16", 96, rng)
    print("拼行等价性验证全部通过" if ok else "拼行等价性验证失败")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
