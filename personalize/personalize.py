"""个人化微调：在用户自己的设备上，用本机的选词记录（select_corpus.tsv）微调基础学生模型。

隐私约定（务必遵守）：
- 本仓库不含任何个人语料或个人权重；个人语料与微调出的权重只留在本机用户目录，不进任何仓库。
- 语料路径与输出目录都不得位于本仓库目录内，否则脚本拒绝运行；输出默认写到 ~/.ime-personal/。
- 脚本不打印任何语料文字，日志与 manifest 里只有数量和指标。

输入 select_corpus.tsv，10 列（制表符分隔）：
  id、分钟时间戳、输入字母数、上文最后 6 字、所选词、最终首位、原首位、类型（hard|anchor）、距上次上屏秒数、模型。
  revise 行指向可能选错的条目：同 id 的 revise 行（或 `revise<TAB>id`）出现时，该 id 的条目整体丢弃。

流程：丢弃 revise 条目 → hard 条目构造成对样本（所选词为正例，最终首位与原首位为反例），anchor 条目作不变样本
（打分保持与基础模型一致）→ 从基础学生模型出发，用成对排序损失加语言模型与 KL 正则微调（断点续跑）
→ 留出 10% 个人样本验证首位命中率必须高于基础模型，同时公开评测集准确率下降不超过 1 个百分点，
否则放弃（不导出）→ 通过则导出 Q8_0 的 GGUF 与词表。

用法：
  python personalize/personalize.py --corpus <select_corpus.tsv 或其所在目录> --base <student_best.pt> \\
      --public-eval <items_eval.jsonl> [--out <输出目录>]
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(HERE))
import student_model as S  # noqa: E402

PUBLIC_DROP_LIMIT = 0.01   # 公开评测集准确率最多允许下降 1 个百分点


def refuse(msg, code=2):
    print("拒绝运行：" + msg, file=sys.stderr)
    raise SystemExit(code)


def ensure_outside_repo(p, what):
    rp = Path(p).resolve()
    if rp == REPO or REPO in rp.parents:
        refuse(f"{what}位于仓库目录内（{what}不得进入仓库）。请换到用户目录。")


# ---------------------------------------------------------------- 语料解析
def corpus_files(path):
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.glob("select_corpus*") if f.suffix in (".tsv", ".gz"))
    return [p]


def parse_corpus(path):
    """返回 (记录列表, 统计)。统计只含数量。"""
    recs, revoked, stats = {}, set(), dict(rows=0, malformed=0, revise=0)
    for f in corpus_files(path):
        opener = gzip.open if f.suffix == ".gz" else open
        with opener(f, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                cols = line.rstrip("\r\n").split("\t")
                if not cols or not cols[0] or cols[0] == "id":
                    continue
                stats["rows"] += 1
                if cols[0] == "revise" and len(cols) >= 2:
                    revoked.add(cols[1])
                    stats["revise"] += 1
                    continue
                if len(cols) >= 8 and cols[7] == "revise":
                    revoked.add(cols[0])
                    stats["revise"] += 1
                    continue
                if len(cols) < 8 or cols[7] not in ("hard", "anchor") or not cols[3] or not cols[4]:
                    stats["malformed"] += 1
                    continue
                recs[cols[0]] = dict(id=cols[0], ctx=cols[3], chosen=cols[4], final=cols[5], orig=cols[6], typ=cols[7])
    kept = [r for i, r in recs.items() if i not in revoked]
    stats["revoked_entries"] = len(recs) - len(kept)
    return kept, stats


def negatives(r):
    return [x for x in dict.fromkeys([r["final"], r["orig"]]) if x and x != r["chosen"]]


def is_holdout(r):
    return int(hashlib.md5(r["id"].encode("utf-8")).hexdigest()[:8], 16) % 10 == 0


# ---------------------------------------------------------------- 评估
def personal_top1(model, vi, recs, dev):
    """所选词得分严格高于所有反例的比例。"""
    hit, n = 0, 0
    model.eval()
    with torch.no_grad():
        for a in range(0, len(recs), 64):
            part = recs[a:a + 64]
            items = [dict(ctx=r["ctx"], c=[r["chosen"]] + negatives(r)) for r in part]
            sc = S.cand_scores(model, vi, items, dev)
            for k, it in enumerate(items):
                m = len(it["c"])
                if m < 2:
                    continue
                n += 1
                hit += bool(sc[k, 0] > sc[k, 1:m].max())
    return hit / max(n, 1), n


def public_acc(model, vi, items, dev):
    ok = 0
    model.eval()
    with torch.no_grad():
        for a in range(0, len(items), 64):
            part = items[a:a + 64]
            sc = S.cand_scores(model, vi, part, dev).argmax(1).tolist()
            ok += sum(p == it["t"] for p, it in zip(sc, part))
    return ok / max(len(items), 1)


def load_public_items(path, limit, seed=0):
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rng = np.random.RandomState(seed)
    rng.shuffle(items)
    return items[:limit]


# ---------------------------------------------------------------- 微调
def load_student(path, dev):
    ck = torch.load(path, map_location=dev)
    c = ck["cfg"]
    m = S.Student(len(ck["vocab"]), c["student_d"], c["student_layers"], c["student_heads"], c["student_ctx"]).to(dev)
    m.load_state_dict(ck["model"])
    return m, ck["cfg"], ck["vocab"]


def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def finetune(model, base, vi, hard, anchors, args, out, dev):
    n = len(hard) + len(anchors)
    allrecs = hard + anchors
    steps = max(1, math.ceil(n / args.batch)) * args.epochs
    fp = hashlib.md5(("|".join(sorted(r["id"] for r in allrecs)) + f"|{args.lr}|{args.batch}|{args.epochs}").encode()).hexdigest()
    base_anchor = {}
    with torch.no_grad():
        for a in range(0, len(anchors), 64):
            part = anchors[a:a + 64]
            sc = S.cand_scores(base, vi, [dict(ctx=r["ctx"], c=[r["chosen"]]) for r in part], dev)[:, 0]
            for r, v in zip(part, sc.tolist()):
                base_anchor[r["id"]] = v
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step = 0
    last = out / "ft_ckpt.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev)
        if ck.get("fp") == fp:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            step = ck["step"]
            print(f"从第 {step} 步续跑")
    model.train()
    t0 = time.time()
    while step < steps:
        rs = np.random.RandomState(1234 + step)
        batch = [allrecs[i] for i in rs.randint(0, n, args.batch)]
        hb = [r for r in batch if r["typ"] == "hard"]
        ab = [r for r in batch if r["typ"] == "anchor"]
        loss = torch.zeros((), device=dev)
        # 成对排序：所选词应高于反例
        pair_src = [r for r in hb + ab if negatives(r)]
        if pair_src:
            items = [dict(ctx=r["ctx"], c=[r["chosen"]] + negatives(r)) for r in pair_src]
            sc = S.cand_scores(model, vi, items, dev)
            terms = []
            for k, it in enumerate(items):
                for j in range(1, len(it["c"])):
                    terms.append(F.softplus(-(sc[k, 0] - sc[k, j])))
            loss = loss + torch.stack(terms).mean()
        # 锚点：所选词的得分保持与基础模型一致，防遗忘
        if ab:
            sa = S.cand_scores(model, vi, [dict(ctx=r["ctx"], c=[r["chosen"]]) for r in ab], dev)[:, 0]
            tgt = torch.tensor([base_anchor[r["id"]] for r in ab], device=dev)
            loss = loss + args.w_anchor * ((sa - tgt) ** 2).mean()
        # 语言模型交叉熵与对基础模型的 KL 正则（在本批文本上）
        seqs = [[S.BOS] + S.encode(vi, r["ctx"] + r["chosen"])[: model.ctx - 1] for r in batch]
        ids = S.pad_batch(seqs, dev)
        lg = model(ids[:, :-1])
        with torch.no_grad():
            lb = base(ids[:, :-1])
        mask = (ids[:, 1:] != S.PAD).float()
        ce = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=S.PAD)
        kl = (F.kl_div(F.log_softmax(lg.float(), -1), F.softmax(lb.float(), -1), reduction="none").sum(-1) * mask).sum() / mask.sum().clamp(min=1)
        loss = loss + args.w_lm * ce + args.w_kl * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % args.ckpt_every == 0 or step == steps:
            atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, fp=fp), last)
        if step % 50 == 0:
            print(f"step {step}/{steps} loss={loss.item():.4f} ({time.time() - t0:.0f}s)")
    model.eval()
    return steps


# ---------------------------------------------------------------- 主流程
def parse_args(argv):
    ap = argparse.ArgumentParser(description="个人化微调（只在本机运行，个人语料与权重不进仓库）")
    ap.add_argument("--corpus", required=True, help="select_corpus.tsv，或其所在目录（含分卷与 .gz）")
    ap.add_argument("--base", required=True, help="基础学生模型检查点（公开语料训练的 student_best.pt）")
    ap.add_argument("--public-eval", required=True, dest="public_eval", help="公开评测题 items_eval.jsonl，用于检查遗忘")
    ap.add_argument("--out", default=str(Path.home() / ".ime-personal"), help="输出目录（默认 ~/.ime-personal）")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--w-anchor", type=float, default=0.05, dest="w_anchor")
    ap.add_argument("--w-lm", type=float, default=0.05, dest="w_lm")
    ap.add_argument("--w-kl", type=float, default=1.0, dest="w_kl")
    ap.add_argument("--ckpt-every", type=int, default=50, dest="ckpt_every")
    ap.add_argument("--public-limit", type=int, default=3000, dest="public_limit")
    ap.add_argument("--min-holdout", type=int, default=5, dest="min_holdout")
    ap.add_argument("--device", default="auto")
    return ap.parse_args(argv)


def run(argv=None):
    args = parse_args(argv)
    ensure_outside_repo(args.corpus, "语料路径")
    out = Path(args.out)
    ensure_outside_repo(out, "输出目录")
    out.mkdir(parents=True, exist_ok=True)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(0)

    recs, stats = parse_corpus(args.corpus)
    hard_all = [r for r in recs if r["typ"] == "hard" and negatives(r)]
    anc_all = [r for r in recs if r["typ"] == "anchor"]
    hard_tr = [r for r in hard_all if not is_holdout(r)]
    hard_ho = [r for r in hard_all if is_holdout(r)]
    anc_tr = [r for r in anc_all if not is_holdout(r)]
    print(f"语料：{stats['rows']} 行，revise 丢弃 {stats['revoked_entries']} 条，格式异常 {stats['malformed']} 行；"
          f"hard 训练 {len(hard_tr)}、留出 {len(hard_ho)}，anchor 训练 {len(anc_tr)}")
    if len(hard_ho) < args.min_holdout or not hard_tr:
        refuse(f"hard 样本太少（留出 {len(hard_ho)} 条，至少需要 {args.min_holdout} 条），先多积累一些再微调。", 3)

    model, cfg, vocab = load_student(args.base, dev)
    vi = {c: i for i, c in enumerate(vocab)}
    base = copy.deepcopy(model).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    pub = load_public_items(args.public_eval, args.public_limit)
    base_hit, n_ho = personal_top1(base, vi, hard_ho, dev)
    base_pub = public_acc(base, vi, pub, dev)

    steps = finetune(model, base, vi, hard_tr, anc_tr, args, out, dev)
    tuned_hit, _ = personal_top1(model, vi, hard_ho, dev)
    tuned_pub = public_acc(model, vi, pub, dev)
    result = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), steps=steps, hard_train=len(hard_tr), hard_holdout=n_ho,
                  anchor_train=len(anc_tr), revise_dropped=stats["revoked_entries"],
                  holdout_top1_base=round(base_hit, 4), holdout_top1_tuned=round(tuned_hit, 4),
                  public_acc_base=round(base_pub, 4), public_acc_tuned=round(tuned_pub, 4), public_items=len(pub))
    ok_personal = tuned_hit > base_hit
    ok_public = (base_pub - tuned_pub) <= PUBLIC_DROP_LIMIT + 1e-9
    result["accepted"] = bool(ok_personal and ok_public)
    print(json.dumps(result, ensure_ascii=False))
    if not result["accepted"]:
        why = []
        if not ok_personal:
            why.append("个人留出集首位命中率没有高于基础模型")
        if not ok_public:
            why.append(f"公开评测集准确率下降超过 {PUBLIC_DROP_LIMIT * 100:.0f} 个百分点")
        result["abandoned_because"] = why
        (out / "abandoned.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print("放弃（不导出）：" + "；".join(why), file=sys.stderr)
        return 1

    atomic_save(dict(model=model.state_dict(), cfg=cfg, vocab=vocab), out / "student_personal.pt")
    import export_q8
    gguf_path = export_q8.write_gguf(cfg, vocab, model.state_dict(), out / "student_personal.q8_0.gguf", q8=True)
    (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已导出 {gguf_path}（权重只留本机，勿提交）")
    return 0


if __name__ == "__main__":
    sys.exit(run())
