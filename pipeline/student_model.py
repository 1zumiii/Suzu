"""学生模型：字符级小型 Transformer 语言模型，给「上文 + 候选」打对数概率之和，与教师打分同口径。"""
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, BOS = 0, 1, 2


def build_vocab(lines, size):
    cnt = Counter()
    for l in lines:
        cnt.update(l.strip())
    return ["<pad>", "<unk>", "<bos>"] + [c for c, _ in cnt.most_common(size - 3)]


def _gelu_tanh(x):
    """GPT-2 的 tanh 近似 GELU。与 llama.cpp 的 gpt2 结构数值一致，便于导出 GGUF。"""
    return F.gelu(x, approximate="tanh")


# ---- hashed n-gram embeddings (optional) --------------------------------------------------------------------------------------
# e_t = tok_emb[c_t] + pos_emb[t] + sum_n P_n( sum_h T_n[ hash_h(c_{t-n+1} .. c_t) mod rows_n ] )
# The window holds the vocabulary indices of the n tokens that end at position t on the logical path of that token (PAD = 0 fills
# the missing places, BOS counts as a token).  Hash (int64 arithmetic with wrap-around, same on CPU and GPU):
#   h = FNV_OFFSET + head * GOLDEN;  for each token x in the window, oldest first:  h = (h ^ (x + 1)) * FNV_PRIME
#   h = h ^ ((h >> 32) & 0xffffffff);  row = (h & INT64_MAX) mod rows
FNV_OFFSET = 14695981039346656037 - 2 ** 64
FNV_PRIME = 1099511628211
GOLDEN = 0x9E3779B97F4A7C15 - 2 ** 64
INT64_MAX = 2 ** 63 - 1


def ngram_windows(ids, prev, n):
    """ids [B,T], prev [B,T] (index of the logical predecessor of each token, -1 = none).  Returns [B,T,n] token indices, oldest first."""
    B, T = ids.shape
    cols, cur = [ids], prev
    bidx = torch.arange(B, device=ids.device)[:, None].expand(B, T)
    for _ in range(n - 1):
        valid = cur >= 0
        tok = torch.where(valid, ids[bidx, cur.clamp(min=0)], torch.zeros_like(ids))
        cols.append(tok)
        nxt = prev[bidx, cur.clamp(min=0)]
        cur = torch.where(valid, nxt, torch.full_like(cur, -1))
    return torch.stack(cols[::-1], -1)


def ngram_hash(win, head, rows):
    """win [...,n] int64 -> row indices [...] in [0, rows)."""
    seed = (FNV_OFFSET + head * GOLDEN + 2 ** 63) % 2 ** 64 - 2 ** 63        # wrap to signed 64 bit
    h = torch.full(win.shape[:-1], seed, dtype=torch.int64, device=win.device)
    for j in range(win.shape[-1]):
        h = (h ^ (win[..., j] + 1)) * FNV_PRIME
    h = h ^ ((h >> 32) & 0xFFFFFFFF)
    return (h & INT64_MAX) % rows


class NgramEmbed(nn.Module):
    def __init__(self, d, ns, log_rows, dim, heads):
        super().__init__()
        self.ns, self.rows, self.heads = list(ns), 2 ** log_rows, heads
        self.tables = nn.ModuleList(nn.Embedding(self.rows, dim) for _ in self.ns)
        self.proj = nn.ModuleList(nn.Linear(dim, d, bias=False) for _ in self.ns)
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)
        for p in self.proj:
            nn.init.zeros_(p.weight)          # step 0 output equals the base model
        self.track = False
        for i in range(len(self.ns)):          # rows that were looked up at least once while tracking (not saved in the state dict)
            self.register_buffer(f"touched{i}", torch.zeros(self.rows, dtype=torch.bool), persistent=False)

    def forward(self, ids, prev):
        out = 0
        for i, n in enumerate(self.ns):
            win = ngram_windows(ids, prev, n)
            e = 0
            for h in range(self.heads):
                r = ngram_hash(win, h, self.rows)
                if self.track:
                    getattr(self, f"touched{i}")[r.reshape(-1)] = True
                e = e + self.tables[i](r)
            out = out + self.proj[i](e)
        return out


class Student(nn.Module):
    def __init__(self, vocab, d, layers, heads, ctx, ngram=None):
        super().__init__()
        self.ctx = ctx
        self.ngram_cfg = ngram
        self.ngram = NgramEmbed(d, ngram["ns"], ngram["log_rows"], ngram["dim"], ngram["heads"]) if ngram else None
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation=_gelu_tanh,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)   # 输入输出共用词向量，默认 std=1 会让初始 logits 过大、损失从 20 起步
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()

    def hidden(self, ids, pos=None, mask=None, prev=None):
        """输出层之前的隐状态。mask 为 [B,T,T] 的布尔张量（True 表示不允许注意），pos 为 [B,T] 的位置编号；
        都不给时是普通的因果注意力。"""
        T = ids.shape[1]
        if pos is None:
            pos = torch.arange(T, device=ids.device)[None]
        x = self.emb(ids) + self.pos(pos)
        if self.ngram is not None:
            if prev is None:                   # plain sequence: the predecessor of a token is the one before it
                prev = torch.arange(-1, T - 1, device=ids.device)[None].expand(ids.shape[0], T)
            x = x + self.ngram(ids, prev)
        if mask is None:
            m = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
            x = self.blocks(x, mask=m, is_causal=True)
        else:
            heads = self.blocks.layers[0].self_attn.num_heads
            x = self.blocks(x, mask=mask.repeat_interleave(heads, 0), is_causal=False)
        return self.ln(x)

    def forward(self, ids):
        return self.hidden(ids) @ self.emb.weight.T          # 输入输出共用词向量


def encode(vocab_index, text):
    return [vocab_index.get(c, UNK) for c in text]


PAD_MULTIPLE = 1   # 第 7 项：把批内长度补齐到该数的倍数（默认 1，不补）


def pad_batch(seqs, device):
    L = max(len(s) for s in seqs)
    L = -(-L // PAD_MULTIPLE) * PAD_MULTIPLE
    t = np.zeros((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        t[i, :len(s)] = s                      # 在 numpy 里填充，只传一次显卡；s 可以是列表或 numpy 数组
    return torch.from_numpy(t).to(device)


def cand_scores_ref(model, vocab_index, items, device):
    """参考实现（逐候选各算一整句），仅用于测试对照。返回 [N, K] 的候选分数张量（不足 K 的位置填 -1e4），可反传。"""
    seqs, cmask_spans, owner = [], [], []
    for n, it in enumerate(items):
        ctx = encode(vocab_index, it["ctx"])[-(model.ctx - 6):]
        for k, c in enumerate(it["c"]):
            cs = encode(vocab_index, c)
            seqs.append([BOS] + ctx + cs)
            cmask_spans.append((len(ctx), len(cs)))
            owner.append((n, k))
    ids = pad_batch(seqs, device)
    logits = model(ids[:, :-1])
    lp = F.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)    # [S, L-1]
    j = torch.arange(lp.shape[1], device=device)[None]
    lo = torch.tensor([a for a, _ in cmask_spans], device=device)[:, None]
    hi = torch.tensor([a + b for a, b in cmask_spans], device=device)[:, None]
    tot = (lp * ((j >= lo) & (j < hi))).sum(1)                                           # 目标下标 ctx..ctx+len-1 即候选各字
    K = max(len(it["c"]) for it in items)
    ns = torch.tensor([n for n, _ in owner], device=device)
    ks = torch.tensor([k for _, k in owner], device=device)
    out = torch.full((len(items), K), -1e4, device=device)
    return out.index_put((ns, ks), tot)        # 一次性写入 [N,K]，可反传


def cand_scores_8(model, vocab_index, items, device):
    """第 8 项：布局与 cand_scores 相同，但输出层与 log_softmax 只在候选字符的位置上算。"""
    seqs, spans, owner = [], [], []
    for n, it in enumerate(items):
        ctx = encode(vocab_index, it["ctx"])[-(model.ctx - 6):]
        for k, c in enumerate(it["c"]):
            cs = encode(vocab_index, c)
            seqs.append([BOS] + ctx + cs)
            spans.append((len(ctx), len(cs)))
            owner.append((n, k))
    ids = pad_batch(seqs, device)
    H = model.hidden(ids[:, :-1])
    j = torch.arange(H.shape[1], device=device)[None]
    lo = torch.tensor([a for a, _ in spans], device=device)[:, None]
    hi = torch.tensor([a + b for a, b in spans], device=device)[:, None]
    rows, cols = ((j >= lo) & (j < hi)).nonzero(as_tuple=True)
    logits = H[rows, cols] @ model.emb.weight.T
    lp = F.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:][rows, cols][:, None]).squeeze(-1)
    tot = torch.zeros(len(seqs), device=device, dtype=lp.dtype).index_add(0, rows, lp)
    K = max(len(it["c"]) for it in items)
    ns = torch.tensor([n for n, _ in owner], device=device)
    ks = torch.tensor([k for _, k in owner], device=device)
    return torch.full((len(items), K), -1e4, device=device).index_put((ns, ks), tot)


def cand_scores_9(model, vocab_index, items, device):
    """第 9 项（含第 8 项）：同一题的所有候选共享上下文。每题只有一条序列：[BOS] + 上文 + 各候选依次拼接，
    用自定义注意力掩码让每个候选只看见上文和自己前面的字，位置编号从上文之后各自重新开始，
    所以与逐候选单独算完全等价，但上文只算一次。输出层只在需要的位置上算。"""
    N = len(items)
    K = max(len(it["c"]) for it in items)
    toks, poss, masks, prevs = [], [], [], []
    g_r, g_c, g_t, g_o = [], [], [], []
    for n, it in enumerate(items):
        ctx = [BOS] + encode(vocab_index, it["ctx"])[-(model.ctx - 6):]
        lc = len(ctx)
        seq, pos, starts = list(ctx), list(range(lc)), []
        prev = list(range(-1, lc - 1))
        for c in it["c"]:
            cs = encode(vocab_index, c)
            starts.append((len(seq), cs))
            st0 = len(seq)
            seq += cs
            pos += [lc + j for j in range(len(cs))]
            prev += [lc - 1 if j == 0 else st0 + j - 1 for j in range(len(cs))]   # first char follows the last context token
        T = len(seq)
        m = np.ones((T, T), dtype=bool)                       # True = 不允许注意
        m[:lc, :lc] = np.triu(np.ones((lc, lc), dtype=bool), 1)
        for k, (st, cs) in enumerate(starts):
            e = st + len(cs)
            m[st:e, :lc] = False
            m[st:e, st:e] = np.triu(np.ones((len(cs), len(cs)), dtype=bool), 1)
            for j, t in enumerate(cs):                        # 第 j 个字由前一个位置的隐状态预测
                g_r.append(n)
                g_c.append(lc - 1 if j == 0 else st + j - 1)
                g_t.append(t)
                g_o.append(n * K + k)
        toks.append(seq)
        poss.append(pos)
        masks.append(m)
        prevs.append(prev)
    Tm = max(len(x) for x in toks)
    ids = np.zeros((N, Tm), dtype=np.int64)
    pp = np.zeros((N, Tm), dtype=np.int64)
    pv = np.full((N, Tm), -1, dtype=np.int64)
    mk = np.ones((N, Tm, Tm), dtype=bool)
    for n in range(N):
        T = len(toks[n])
        ids[n, :T], pp[n, :T], mk[n, :T, :T] = toks[n], poss[n], masks[n]
        pv[n, :T] = prevs[n]
        mk[n, T:, 0] = False                                  # 补齐行至少能看第 0 个位置，避免全屏蔽产生 NaN
    H = model.hidden(torch.from_numpy(ids).to(device), pos=torch.from_numpy(pp).to(device),
                     mask=torch.from_numpy(mk).to(device), prev=torch.from_numpy(pv).to(device))
    r = torch.tensor(g_r, device=device)
    c = torch.tensor(g_c, device=device)
    logits = H[r, c] @ model.emb.weight.T
    lp = F.log_softmax(logits.float(), -1).gather(-1, torch.tensor(g_t, device=device)[:, None]).squeeze(-1)
    tot = torch.zeros(N * K, device=device, dtype=lp.dtype).index_add(0, torch.tensor(g_o, device=device), lp)
    ns = torch.tensor([n for n, it in enumerate(items) for _ in it["c"]], device=device)
    ks = torch.tensor([k for it in items for k in range(len(it["c"]))], device=device)
    return torch.full((N, K), -1e4, device=device).index_put((ns, ks), tot[ns * K + ks])


cand_scores = cand_scores_9   # 正式实现：候选共享上下文，与 cand_scores_ref 数值等价（见 tests/test_items_7_9.py）


def score_items(model, vocab, items, device):
    model.eval()
    vi = {c: i for i, c in enumerate(vocab)}
    out = []
    for a in range(0, len(items), 64):
        part = items[a:a + 64]
        sc = cand_scores(model, vi, part, device).tolist()
        out += [sc[i][:len(it["c"])] for i, it in enumerate(part)]
    return out


def load_student(path, device):
    ck = torch.load(path, map_location=device)
    c = ck["cfg"]
    model = Student(len(ck["vocab"]), c["student_d"], c["student_layers"], c["student_heads"], c["student_ctx"], ck.get("ngram")).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck["vocab"]
