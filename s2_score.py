"""阶段 2：给题目打分。每个「系统」（ngram / 各个 Qwen 模型 / 学生）单独输出 scores/<名字>.jsonl。
一行 {"id": 题号, "s": [各候选的分数]}，分数 = 候选在上文之后的对数概率之和，越大越好。
追加写入，重启后跳过已完成的题（断点续跑）。
"""
import math
import os
import time
from collections import Counter

import common as C
from common import CFG, DATA, SCORES, log


class DummyScorer:
    """仅用于自测：带噪声的「作弊」打分，sigma 越大越不准。"""

    def __init__(self, spec):
        self.sigma, self.max_seqs = spec.get("sigma", 2.0), 30

    def score(self, items):
        import random
        if os.environ.get("DISTILL_SLOW"):
            time.sleep(float(os.environ["DISTILL_SLOW"]))
        out = []
        for it in items:
            r = random.Random(it["id"] * 7919 + int(self.sigma * 100))
            out.append([(2.0 if i == it["t"] else 0.0) + r.gauss(0, self.sigma) for i in range(len(it["c"]))])
        return out


class NgramScorer:
    """字符三元模型（Witten-Bell 平滑），只用训练句。对应输入法里常见的文法模型，作为「不用大模型」的基线。"""

    def __init__(self, spec=None):
        n = CFG["ngram_train_sentences"]
        self.cnt = [None, {}, {}, {}]
        self.ctxc = [None, Counter(), Counter(), Counter()]
        self.typ = [None, Counter(), Counter(), Counter()]
        for si, line in enumerate(open(DATA / "train_sentences.txt", encoding="utf-8")):
            if si >= n:
                break
            if si % 20000 == 0:
                C.check_stop()
            t = "^^" + line.strip() + "$"
            for i in range(2, len(t)):
                w = t[i]
                for k in (1, 2, 3):
                    h = t[i - k + 1:i] if k > 1 else ""
                    d = self.cnt[k]
                    key = h + w
                    if key not in d:
                        self.typ[k][h] += 1
                        d[key] = 0
                    d[key] += 1
                    self.ctxc[k][h] += 1
        self.max_seqs = 10**9

    def prob(self, w, hist):
        p = 1.0 / 20000
        for k in (1, 2, 3):
            h = hist[-(k - 1):] if k > 1 else ""
            c = self.ctxc[k].get(h, 0)
            if c:
                T = self.typ[k][h]
                p = (self.cnt[k].get(h + w, 0) + T * p) / (c + T)
        return p

    def score(self, items):
        out = []
        for it in items:
            base = ("^^" + it["ctx"])[-2:]
            row = []
            for cand in it["c"]:
                h, s = base, 0.0
                for ch in cand:
                    s += math.log(self.prob(ch, h))
                    h = (h + ch)[-2:]
                row.append(s)
            out.append(row)
        return out


class HFScorer:
    def __init__(self, spec):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        mid = spec["hf"]
        self.tok = AutoTokenizer.from_pretrained(mid)
        kw = {"device_map": {"": 0}}
        if spec.get("quant") == "nf4":
            kw["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

        def load(cls):
            try:
                return cls.from_pretrained(mid, dtype=torch.bfloat16, **kw)
            except TypeError:
                return cls.from_pretrained(mid, torch_dtype=torch.bfloat16, **kw)

        try:
            self.model = load(AutoModelForCausalLM)
        except Exception as e:   # Qwen3.5 是多模态架构，纯文本入口加载不了时退回多模态入口
            log(f"AutoModelForCausalLM 加载失败（{type(e).__name__}: {str(e)[:120]}），改用多模态入口")
            cls = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
            self.model = load(cls)
        self.model.eval()
        unk = self.tok.unk_token_id
        start = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.start = self.tok.eos_token_id if start is None or start == unk else start
        self.max_seqs = CFG["score_seqs"]

    def score(self, items):
        torch = self.torch
        try:
            with torch.no_grad():
                return self._score(items)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(items) == 1:
                raise
            self.max_seqs = max(len(items[0]["c"]), self.max_seqs // 2)
            log(f"显存不足，序列批大小降到 {self.max_seqs}")
            h = len(items) // 2
            return self.score(items[:h]) + self.score(items[h:])

    def _score(self, items):
        torch = self.torch
        texts, spans = [], []
        for it in items:
            texts += [it["ctx"] + c for c in it["c"]]
        toks = self.tok(texts, add_special_tokens=False)["input_ids"]
        seqs, lcps, pos = [], [], 0
        for it in items:
            tk = toks[pos:pos + len(it["c"])]
            pos += len(it["c"])
            lcp = 0
            for col in zip(*tk):
                if len(set(col)) > 1:
                    break
                lcp += 1
            lcp = min(lcp, min(len(t) for t in tk) - 1)   # 每个候选至少有一个 token 参与比较
            seqs += [[self.start] + t for t in tk]
            lcps += [lcp] * len(tk)
        L = max(len(s) for s in seqs)
        dev = self.model.device
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        att = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(dev), att.to(dev)
        logits = self.model(input_ids=ids, attention_mask=att).logits
        lens = att.sum(1)
        flat = []
        for a in range(0, len(seqs), 8):
            lg = logits[a:a + 8, :-1].float().log_softmax(-1)
            lp = lg.gather(-1, ids[a:a + 8, 1:, None]).squeeze(-1)
            j = torch.arange(1, L, device=dev)[None, :]
            lo = torch.tensor(lcps[a:a + 8], device=dev)[:, None] + 1
            m = (j >= lo) & (j < lens[a:a + 8, None])
            flat += (lp * m).sum(1).tolist()
            del lg, lp
        out, pos = [], 0
        for it in items:
            out.append(flat[pos:pos + len(it["c"])])
            pos += len(it["c"])
        return out


def _repeat_cache(pkv, r):
    """把缓存的批维度按 r 复制（相邻复制）。Qwen3.5 的线性注意力层没有现成的 batch_repeat_interleave，这里手动处理：
    线性注意力层的卷积状态与循环状态，加上完整注意力层的 key/value。"""
    for layer in pkv.layers:
        if hasattr(layer, "conv_states"):
            for i in layer.conv_states:
                if layer.conv_states[i] is not None:
                    layer.conv_states[i] = layer.conv_states[i].repeat_interleave(r, 0)
                if layer.recurrent_states[i] is not None:
                    layer.recurrent_states[i] = layer.recurrent_states[i].repeat_interleave(r, 0)
        if getattr(layer, "keys", None) is not None and getattr(layer, "values", None) is not None:
            layer.keys = layer.keys.repeat_interleave(r, 0)
            layer.values = layer.values.repeat_interleave(r, 0)


def _shared_prefix_scores(self, items, max_seqs=32):
    """【实验，流水线未使用】教师打分的前缀复用（清单第 10 项）。实测数值正确（首选一致、对真实答案的命中相同），
    但比 _score 慢：0.8B 为 0.4 到 0.55 倍，4B(nf4) 为 0.85 倍，原因是线性注意力层的循环状态复制开销大于省下的重复计算。
    测试见 tests/test_teacher_prefix.py。同一题各候选的公共 token 前缀只算一次并缓存，每个候选只算各自不同的尾部。
    Qwen3.5 含线性注意力层，前缀补齐会破坏循环状态，所以按前缀长度分桶，同长度的题一起算，不补齐前缀。
    与 _score 的口径完全一致：每个候选从第一个不同的 token 起，对数概率之和。"""
    torch = self.torch
    prep, groups = [], {}
    for idx, it in enumerate(items):
        tk = self.tok([it["ctx"] + c for c in it["c"]], add_special_tokens=False)["input_ids"]
        lcp = 0
        for col in zip(*tk):
            if len(set(col)) > 1:
                break
            lcp += 1
        lcp = min(lcp, min(len(t) for t in tk) - 1)
        prep.append((tk, lcp))
        groups.setdefault(lcp, []).append(idx)
    out = [None] * len(items)
    dev = self.model.device
    for lcp, idxs in groups.items():
        km_all = max(len(items[i]["c"]) for i in idxs)
        step = max(1, max_seqs // km_all)
        for a0 in range(0, len(idxs), step):
            chunk = idxs[a0:a0 + step]
            B = len(chunk)
            Km = max(len(items[i]["c"]) for i in chunk)
            prefix = torch.tensor([[self.start] + prep[i][0][0][:lcp] for i in chunk], device=dev)
            first, tails = [], []
            for i in chunk:
                tk, _ = prep[i]
                K = len(tk)
                for k in range(Km):
                    t = tk[k if k < K else 0]
                    first.append(t[lcp])
                    tails.append(t[lcp:])          # 第 0 个由前缀最后位置预测，其余由尾部前向预测
            pre = self.model(input_ids=prefix, use_cache=True, logits_to_keep=1)
            lp0 = pre.logits[:, -1].float().log_softmax(-1)                    # [B,V]
            s0 = lp0[torch.arange(B, device=dev).repeat_interleave(Km), torch.tensor(first, device=dev)]
            tot = s0.clone()
            Tm = max(len(t) for t in tails) - 1
            if Tm > 0:
                inp = torch.zeros(B * Km, Tm, dtype=torch.long)
                for r, t in enumerate(tails):
                    if len(t) > 1:
                        inp[r, :len(t) - 1] = torch.tensor(t[:-1])
                pkv = pre.past_key_values
                _repeat_cache(pkv, Km)
                o = self.model(input_ids=inp.to(dev), past_key_values=pkv, use_cache=True)
                for a1 in range(0, B * Km, 32):
                    lg = o.logits[a1:a1 + 32].float().log_softmax(-1)
                    for r in range(lg.shape[0]):
                        t = tails[a1 + r]
                        if len(t) > 1:
                            tgt = torch.tensor(t[1:], device=dev)
                            tot[a1 + r] += lg[r, torch.arange(len(t) - 1, device=dev), tgt].sum()
            tot = tot.tolist()
            for b, i in enumerate(chunk):
                out[i] = tot[b * Km:b * Km + len(items[i]["c"])]
    return out


HFScorer.score_shared = lambda self, items, **kw: _score_shared_nograd(self, items, **kw)


def _score_shared_nograd(self, items, **kw):
    with self.torch.no_grad():
        return _shared_prefix_scores(self, items, **kw)


class StudentScorer:
    def __init__(self, spec=None):
        import torch
        from student_model import load_student, score_items
        self.torch, self.score_items = torch, score_items
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.vocab = load_student(C.CKPT / "student_best.pt", self.dev)
        self.max_seqs = 512

    def score(self, items):
        with self.torch.no_grad():
            return self.score_items(self.model, self.vocab, items, self.dev)


def make_scorer(name):
    if name == "ngram":
        return NgramScorer()
    if name == "student":
        return StudentScorer()
    spec = C.models()[name]
    if spec.get("backend") == "dummy":
        return DummyScorer(spec)
    return HFScorer(spec)


def score_unit(out_name, items_path, make):
    items = C.read_jsonl(items_path)
    out_path = SCORES / f"{out_name}.jsonl"
    done = C.load_done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    log(f"[{out_name}] 共 {len(items)} 题，已完成 {len(done)}，待做 {len(todo)}")
    if not todo:
        return
    scorer = make()
    app = C.Appender(out_path)
    prog = C.Progress(out_name, len(items), len(done))
    i = 0
    try:
        while i < len(todo):
            C.check_stop()
            batch, n = [], 0
            while i < len(todo) and (not batch or n + len(todo[i]["c"]) <= scorer.max_seqs):
                batch.append(todo[i])
                n += len(todo[i]["c"])
                i += 1
            res = scorer.score(batch)
            for it, s in zip(batch, res):
                app.write(dict(id=it["id"], s=[round(x, 4) if math.isfinite(x) else -1e9 for x in s]))
            app.flush()
            prog.tick(len(batch))
    finally:
        app.close()
    prog.tick(0, force=True)


def run_eval(name, items="items_eval", tag=""):
    score_unit(name + tag, DATA / (items + ".jsonl"), lambda: make_scorer(name))


def run_teacher_train(teacher):
    score_unit(f"{teacher}__train", DATA / "items_train.jsonl", lambda: make_scorer(teacher))
