"""第 10 项（教师打分前缀复用）的小样本一致性验证：同一批真实评测题，旧的逐候选整句打分与新的前缀复用打分对比。
用法：python tests/test_teacher_prefix.py <模型名，如 q35_0p8b> <items_eval.jsonl> [题数，默认 200]
需要先下载好对应模型；只读评测题，不写任何数据目录以外的文件。"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("DISTILL_DATA", tempfile.mkdtemp(prefix="teacherprefix_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import torch  # noqa: E402

import common as C  # noqa: E402
import s2_score  # noqa: E402


def main():
    name, path = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    items = [json.loads(l) for l in open(path, encoding="utf-8")][:n]
    scorer = s2_score.make_scorer(name)
    scorer.max_seqs = C.CFG["score_seqs"]

    def run_old():
        out, i = [], 0
        while i < len(items):
            batch, m = [], 0
            while i < len(items) and (not batch or m + len(items[i]["c"]) <= scorer.max_seqs):
                batch.append(items[i]); m += len(items[i]["c"]); i += 1
            out += scorer.score(batch)
        return out

    run_old()                                   # 预热
    scorer.score_shared(items[:8])
    torch.cuda.synchronize()
    t0 = time.time(); old = run_old(); torch.cuda.synchronize(); t_old = time.time() - t0
    t0 = time.time(); new = scorer.score_shared(items); torch.cuda.synchronize(); t_new = time.time() - t0
    diffs, agree, hit_old, hit_new = [], 0, 0, 0
    for it, a, b in zip(items, old, new):
        assert len(a) == len(b)
        diffs += [abs(x - y) for x, y in zip(a, b)]
        agree += int(max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__))
        hit_old += int(max(range(len(a)), key=a.__getitem__) == it["t"])
        hit_new += int(max(range(len(b)), key=b.__getitem__) == it["t"])
    diffs.sort()
    print(f"{name}，{len(items)} 题，{sum(len(i['c']) for i in items)} 个候选")
    print(f"分数差：平均 {sum(diffs) / len(diffs):.4f}，中位 {diffs[len(diffs) // 2]:.4f}，p99 {diffs[int(len(diffs) * .99)]:.4f}，最大 {diffs[-1]:.4f}")
    print(f"首选一致 {agree}/{len(items)}；对照真实答案：旧 {hit_old}，新 {hit_new}")
    print(f"耗时：旧 {t_old:.2f}s（{len(items) / t_old:.1f} 题/秒），新 {t_new:.2f}s（{len(items) / t_new:.1f} 题/秒），"
          f"新是旧的 {t_old / t_new:.2f} 倍速")


if __name__ == "__main__":
    main()
