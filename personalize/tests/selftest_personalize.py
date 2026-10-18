"""个人化微调的自测：全部使用合成数据（编造的字与词），不得使用任何真实个人数据。
覆盖：语料解析（revise 丢弃、异常行、分卷与 gz）、隐私守卫（输出或语料在仓库内则拒绝）、
成对微调后留出集命中率提升且公开评测不掉点、导出 GGUF、manifest 不含文字、续跑、验收不过时放弃且不导出。
用法：python personalize/tests/selftest_personalize.py"""
import gzip
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "pipeline"))
import personalize as P  # noqa: E402
import student_model as S  # noqa: E402

rng = np.random.RandomState(5)
ALPHA = [chr(0x4E00 + i * 3) for i in range(60)]
VOCAB = ["<pad>", "<unk>", "<bos>"] + ALPHA
VI = {c: i for i, c in enumerate(VOCAB)}
N_CTX = 20
CTX = [ALPHA[k] + ALPHA[(k + 7) % 20] for k in range(N_CTX)]
WORDS = {}
for k in range(N_CTX):
    while True:
        a = "".join(rng.choice(ALPHA[20:], 2))
        b = "".join(rng.choice(ALPHA[20:], 2))
        if a != b:
            WORDS[k] = (a, b)
            break


def build_base(dst):
    """合成基础模型：每个上文后 85% 接 A、15% 接 B，所以基础模型偏好 A。"""
    torch.manual_seed(0)
    cfg = dict(student_d=64, student_layers=2, student_heads=4, student_ctx=32)
    m = S.Student(len(VOCAB), 64, 2, 4, 32)
    sents = []
    for _ in range(6000):
        k = rng.randint(N_CTX)
        sents.append([S.BOS] + S.encode(VI, CTX[k] + WORDS[k][0 if rng.rand() < 0.85 else 1]))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    m.train()
    for step in range(500):
        ids = S.pad_batch([sents[i] for i in rng.randint(0, len(sents), 64)], "cpu")
        lg = m(ids[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=S.PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(dict(model=m.state_dict(), cfg=cfg, vocab=VOCAB), dst)


def build_corpus(d):
    rows, hard_ids = [], []
    n = 0
    for k in range(5):                       # 个人偏好：这些上文后用户选 B，而模型首位是 A
        for _ in range(24):
            n += 1
            rows.append(f"{n}\t{29000000 + n}\t4\t{CTX[k]}\t{WORDS[k][1]}\t{WORDS[k][0]}\t{WORDS[k][0]}\thard\t12\tq")
            hard_ids.append(str(n))
    for k in range(5, N_CTX):                # 锚点：选了首位 A
        for _ in range(8):
            n += 1
            rows.append(f"{n}\t{29000000 + n}\t4\t{CTX[k]}\t{WORDS[k][0]}\t{WORDS[k][0]}\t{WORDS[k][0]}\tanchor\t9\tq")
    for i in hard_ids[:10]:                  # 10 条 revise：对应条目必须被丢弃
        rows.append(f"{i}\t29000999\t4\t{CTX[0]}\t{WORDS[0][1]}\t{WORDS[0][0]}\t{WORDS[0][0]}\trevise\t2\tq")
    rows += ["garbage line without tabs", "x\t1\t2"]
    half = len(rows) // 2
    header = "id\tts\tn\tctx\tchosen\tfinal\torig\ttype\tsecs\tmodel"
    (d / "select_corpus.tsv").write_text(header + "\n" + "\n".join(rows[:half]) + "\n", encoding="utf-8")
    with gzip.open(d / "select_corpus.001.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(rows[half:]) + "\n")
    return set(hard_ids[:10])


def build_public(path):
    items = []
    for i in range(600):
        k = 5 + rng.randint(N_CTX - 5)       # 公开评测只含与个人偏好无关的上文
        t = 0 if rng.rand() < 0.85 else 1
        items.append(dict(id=i, ctx=CTX[k], c=list(WORDS[k]), f=[10, 5], t=t, s=int(t > 0)))
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n", encoding="utf-8")


def expect_exit(argv, code):
    try:
        P.run(argv)
    except SystemExit as e:
        assert e.code == code, f"退出码应为 {code}，实际 {e.code}"
        return
    raise AssertionError("应当拒绝运行")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="personalize_selftest_"))
    base, corpus, pub = tmp / "base.pt", tmp / "corpus", tmp / "public.jsonl"
    corpus.mkdir()
    build_base(base)
    revised = build_corpus(corpus)
    build_public(pub)
    common = ["--corpus", str(corpus), "--base", str(base), "--public-eval", str(pub), "--device", "cpu"]

    # 1) 语料解析
    recs, st = P.parse_corpus(corpus)
    assert st["revoked_entries"] == 10, st
    assert st["malformed"] >= 2, st
    assert not any(r["id"] in revised for r in recs), "revise 指向的条目必须被丢弃"
    print("解析 ok:", st)

    # 2) 隐私守卫
    expect_exit(common + ["--out", str(P.REPO / "personalize" / "leak_test")], 2)
    expect_exit(["--corpus", str(P.REPO / "README.md"), "--base", str(base), "--public-eval", str(pub),
                 "--out", str(tmp / "o")], 2)
    assert not (P.REPO / "personalize" / "leak_test").exists()
    print("隐私守卫 ok")

    # 3) 正常微调：留出集命中率提升、公开评测不掉点、导出
    out = tmp / "out"
    rc = P.run(common + ["--out", str(out), "--epochs", "10", "--lr", "1e-3", "--w-kl", "0.2"])
    assert rc == 0, "应当通过验收"
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["holdout_top1_tuned"] > man["holdout_top1_base"], man
    assert man["public_acc_base"] - man["public_acc_tuned"] <= 0.01 + 1e-9, man
    g = out / "student_personal.q8_0.gguf"
    assert g.exists() and g.stat().st_size > 10_000 and (out / "student_personal.q8_0.gguf.vocab.txt").exists()
    assert g.read_bytes()[:4] == b"GGUF"
    assert not re.search(r"[一-鿿]", (out / "manifest.json").read_text(encoding="utf-8")), "manifest 不得含文字"
    print("微调 ok:", {k: man[k] for k in ("holdout_top1_base", "holdout_top1_tuned", "public_acc_base", "public_acc_tuned")})

    # 4) 续跑：同一输出目录再跑一次，应从检查点继续且仍通过
    assert P.run(common + ["--out", str(out), "--epochs", "10", "--lr", "1e-3", "--w-kl", "0.2"]) == 0
    print("续跑 ok")

    # 5) 验收不过：学习率过大破坏公开评测，应放弃且不导出
    bad = tmp / "bad"
    rc = P.run(common + ["--out", str(bad), "--epochs", "10", "--lr", "30", "--w-kl", "0.0", "--w-lm", "0.0"])
    assert rc == 1 and (bad / "abandoned.json").exists(), "应当放弃"
    assert not list(bad.glob("*.gguf")) and not (bad / "student_personal.pt").exists(), "放弃时不得导出"
    print("放弃路径 ok")
    print("selftest 通过（合成数据，临时目录 " + str(tmp) + "）")


if __name__ == "__main__":
    main()
