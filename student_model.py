"""学生模型：字符级小型 Transformer 语言模型，给「上文 + 候选」打对数概率之和，与教师打分同口径。"""
from collections import Counter

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


class Student(nn.Module):
    def __init__(self, vocab, d, layers, heads, ctx):
        super().__init__()
        self.ctx = ctx
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

    def forward(self, ids):
        T = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        x = self.ln(self.blocks(x, mask=mask, is_causal=True))
        return x @ self.emb.weight.T          # 输入输出共用词向量


def encode(vocab_index, text):
    return [vocab_index.get(c, UNK) for c in text]


def pad_batch(seqs, device):
    L = max(len(s) for s in seqs)
    t = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, :len(s)] = torch.tensor(s)
    return t.to(device)


def cand_scores(model, vocab_index, items, device):
    """返回 [N, K] 的候选分数张量（不足 K 的位置填 -1e4），可反传。"""
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
    out = torch.full((len(items), K), -1e4, device=device)
    for s, (n, k) in enumerate(owner):
        out[n, k] = tot[s]
    return out


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
    model = Student(len(ck["vocab"]), c["student_d"], c["student_layers"], c["student_heads"], c["student_ctx"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck["vocab"]
