"""把学生检查点导出为 gpt2 结构的 GGUF，可选 Q8_0 量化（只量化二维权重，偏置、层归一化、位置向量保持 F32）。
词表另存为 <输出>.vocab.txt（一行一个字符，行号即编号），由推理端自行做字符到编号的映射。
依赖 gguf：pip 安装的 gguf 包，或用环境变量 GGUF_PY 指向 llama.cpp 源码里的 gguf-py 目录。"""
import os
import sys
from pathlib import Path

import numpy as np

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
import gguf  # noqa: E402

PAD, UNK, BOS = 0, 1, 2   # 与 student_model 一致


def write_gguf(cfg, vocab, sd, dst, q8=True):
    dst = Path(dst)
    d, L, H, ctx, V = cfg["student_d"], cfg["student_layers"], cfg["student_heads"], cfg["student_ctx"], len(vocab)
    w = gguf.GGUFWriter(str(dst), "gpt2")
    w.add_name("rime-student")
    w.add_context_length(ctx)
    w.add_embedding_length(d)
    w.add_block_count(L)
    w.add_feed_forward_length(4 * d)
    w.add_head_count(H)
    w.add_layer_norm_eps(1e-5)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0 if q8 else gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model("llama")   # 占位分词器：加载时要求有词表，实际不使用
    w.add_token_list([t if len(t) else "<empty>" for t in vocab])
    w.add_token_scores([0.0] * V)
    w.add_token_types([gguf.TokenType.CONTROL if t.startswith("<") and len(t) > 1 else gguf.TokenType.NORMAL for t in vocab])
    w.add_unk_token_id(UNK)
    w.add_bos_token_id(BOS)
    w.add_pad_token_id(PAD)

    def t(name):
        return sd[name].detach().float().cpu().numpy().astype(np.float32)

    def add(name, arr, quant):
        if quant and q8 and arr.ndim == 2 and arr.shape[1] % 32 == 0:
            w.add_tensor(name, gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0),
                         raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(name, arr)

    add("token_embd.weight", t("emb.weight"), True)
    add("position_embd.weight", t("pos.weight"), False)
    add("output_norm.weight", t("ln.weight"), False)
    add("output_norm.bias", t("ln.bias"), False)
    for i in range(L):
        p = f"blocks.layers.{i}."
        add(f"blk.{i}.attn_norm.weight", t(p + "norm1.weight"), False)
        add(f"blk.{i}.attn_norm.bias", t(p + "norm1.bias"), False)
        add(f"blk.{i}.attn_qkv.weight", t(p + "self_attn.in_proj_weight"), True)
        add(f"blk.{i}.attn_qkv.bias", t(p + "self_attn.in_proj_bias"), False)
        add(f"blk.{i}.attn_output.weight", t(p + "self_attn.out_proj.weight"), True)
        add(f"blk.{i}.attn_output.bias", t(p + "self_attn.out_proj.bias"), False)
        add(f"blk.{i}.ffn_norm.weight", t(p + "norm2.weight"), False)
        add(f"blk.{i}.ffn_norm.bias", t(p + "norm2.bias"), False)
        add(f"blk.{i}.ffn_up.weight", t(p + "linear1.weight"), True)
        add(f"blk.{i}.ffn_up.bias", t(p + "linear1.bias"), False)
        add(f"blk.{i}.ffn_down.weight", t(p + "linear2.weight"), True)
        add(f"blk.{i}.ffn_down.bias", t(p + "linear2.bias"), False)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    Path(str(dst) + ".vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    return dst
