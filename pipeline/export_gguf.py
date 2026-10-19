"""把学生检查点导出为 llama.cpp 的 GGUF（gpt2 结构），并给出 PyTorch 参考输出供数值比对。
用法：python export_gguf.py <student_best.pt> <输出.gguf> [参考文本]
词表单独写成 <输出>.vocab.txt（一行一个字符，行号即 token 编号），由 DLL 自己做字符到编号的映射，
GGUF 里的分词器信息只是占位，不会被使用。"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

if os.environ.get("GGUF_PY"):   # llama.cpp 源码里的 gguf-py 目录；未设置则使用 pip 安装的 gguf 包
    sys.path.insert(0, os.environ["GGUF_PY"])
import gguf

import student_model as S


def main():
    src, dst = sys.argv[1], Path(sys.argv[2])
    ck = torch.load(src, map_location="cpu")
    c, vocab, sd = ck["cfg"], ck["vocab"], ck["model"]
    d, L, H, ctx, V = c["student_d"], c["student_layers"], c["student_heads"], c["student_ctx"], len(vocab)
    w = gguf.GGUFWriter(str(dst), "gpt2")
    w.add_name("rime-student")
    w.add_context_length(ctx)
    w.add_embedding_length(d)
    w.add_block_count(L)
    w.add_feed_forward_length(4 * d)
    w.add_head_count(H)
    w.add_layer_norm_eps(1e-5)
    q8 = os.environ.get("SUZU_Q8") == "1"      # Q8_0 for 2-D weights (not the position table), like the released models
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0 if q8 else gguf.LlamaFileType.ALL_F32)
    # 占位分词器（llama.cpp 加载时要求有词表）
    w.add_tokenizer_model("llama")
    w.add_token_list([t if len(t) else "<empty>" for t in vocab])
    w.add_token_scores([0.0] * V)
    w.add_token_types([gguf.TokenType.CONTROL if t.startswith("<") and len(t) > 1 else gguf.TokenType.NORMAL for t in vocab])
    w.add_unk_token_id(S.UNK)
    w.add_bos_token_id(S.BOS)
    w.add_pad_token_id(S.PAD)

    def t(name):
        return sd[name].float().numpy().astype(np.float32)

    def add_tensor(name, arr):
        if q8 and arr.ndim == 2 and name != "position_embd.weight":
            w.add_tensor(name, gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(name, arr)

    add_tensor("token_embd.weight", t("emb.weight"))
    add_tensor("position_embd.weight", t("pos.weight"))
    add_tensor("output_norm.weight", t("ln.weight"))
    add_tensor("output_norm.bias", t("ln.bias"))
    for i in range(L):
        p = f"blocks.layers.{i}."
        add_tensor(f"blk.{i}.attn_norm.weight", t(p + "norm1.weight"))
        add_tensor(f"blk.{i}.attn_norm.bias", t(p + "norm1.bias"))
        add_tensor(f"blk.{i}.attn_qkv.weight", t(p + "self_attn.in_proj_weight"))
        add_tensor(f"blk.{i}.attn_qkv.bias", t(p + "self_attn.in_proj_bias"))
        add_tensor(f"blk.{i}.attn_output.weight", t(p + "self_attn.out_proj.weight"))
        add_tensor(f"blk.{i}.attn_output.bias", t(p + "self_attn.out_proj.bias"))
        add_tensor(f"blk.{i}.ffn_norm.weight", t(p + "norm2.weight"))
        add_tensor(f"blk.{i}.ffn_norm.bias", t(p + "norm2.bias"))
        add_tensor(f"blk.{i}.ffn_up.weight", t(p + "linear1.weight"))
        add_tensor(f"blk.{i}.ffn_up.bias", t(p + "linear1.bias"))
        add_tensor(f"blk.{i}.ffn_down.weight", t(p + "linear2.weight"))
        add_tensor(f"blk.{i}.ffn_down.bias", t(p + "linear2.bias"))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    Path(str(dst) + ".vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    print(f"已写出 {dst}（{dst.stat().st_size/2**20:.1f} MB），词表 {V}，层 {L}，维度 {d}")

    # 参考输出：序列 [BOS]+文本 的逐 token 对数概率之和，以及末位置前 5 个 logits
    text = sys.argv[3] if len(sys.argv) > 3 else "".join(vocab[3:3 + 8])
    vi = {ch: i for i, ch in enumerate(vocab)}
    ids = [S.BOS] + S.encode(vi, text)
    m = S.Student(V, d, L, H, ctx)
    m.load_state_dict(sd)
    m.eval()
    with torch.no_grad():
        lg = m(torch.tensor([ids]))[0].float()
    lp = lg.log_softmax(-1)
    total = sum(float(lp[k, ids[k + 1]]) for k in range(len(ids) - 1))
    print("ids", ",".join(map(str, ids)))
    print(f"REF sum_logprob {total:.5f}")
    print("REF last_logits", " ".join(f"{float(x):.5f}" for x in lg[-1, :5]))


if __name__ == "__main__":
    main()
