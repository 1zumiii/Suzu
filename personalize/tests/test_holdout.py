"""留出集划分的自测（不需要 torch，合成数据）。用法：python personalize/tests/test_holdout.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import holdout as H  # noqa: E402


def recs(n_hard=200, n_anchor=600, seed=1):
    rnd = random.Random(seed)
    out = []
    for i in range(n_hard + n_anchor):
        out.append(dict(id=f"r{i}", ts=29_000_000 + rnd.randrange(100_000), typ="hard" if i < n_hard else "anchor"))
    return out


def main():
    rs = recs()
    tr, ho, cut = H.split(rs, "time", 0.1)
    hard_ho = [r for r in ho if r["typ"] == "hard"]
    hard_tr = [r for r in tr if r["typ"] == "hard"]
    assert cut is not None
    assert len(tr) + len(ho) == len(rs)
    assert 18 <= len(hard_ho) <= 22, len(hard_ho)                      # 约 10% 的 hard
    assert max(r["ts"] for r in tr) < min(r["ts"] for r in ho)         # 训练全部早于留出
    assert any(r["typ"] == "anchor" for r in ho)                       # 截止之后的 anchor 也不参与训练
    assert hard_tr and all(r["ts"] < cut for r in hard_tr)
    print("按时间留出 ok:", len(hard_tr), len(hard_ho))

    # 时间戳缺失的条目留在训练集
    rs2 = rs + [dict(id="z0", ts=0, typ="hard"), dict(id="z1", ts=0, typ="anchor")]
    tr2, ho2, _ = H.split(rs2, "time", 0.1)
    assert {"z0", "z1"} <= {r["id"] for r in tr2}
    print("缺时间戳 ok")

    # 全部没有时间戳时退回哈希划分
    rs3 = [dict(r, ts=0) for r in rs]
    tr3, ho3, cut3 = H.split(rs3, "time", 0.1)
    assert cut3 is None and ho3 and len(tr3) + len(ho3) == len(rs3)
    assert {r["id"] for r in ho3} == {r["id"] for r in rs3 if H.hash_holdout(r)}
    print("退回哈希 ok")

    # 分钟与秒的换算
    assert H.to_unix(29_880_000) == 29_880_000 * 60
    assert H.to_unix(1_792_743_600) == 1_792_743_600
    print("test_holdout 通过")


if __name__ == "__main__":
    main()
