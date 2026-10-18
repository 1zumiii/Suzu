# 数据与产物放置约定

- **外部大体积数据一律放在 `DISTILL_DATA` 目录**（路径由使用者自定，建议放在空间充足的数据盘），不入库：
  维基语料分片和 HF 模型缓存（`hf\`）、句子与题目（`sentences.txt`、`items_*.jsonl`）、各系统打分（`scores\`）、检查点（`ckpt\`）。
  新增的语料也放在 `DISTILL_DATA\corpus\`，一类一个子目录，并在本文件登记来源与下载日期。
- **入库的只有**：代码、报告（`reports\`）和训练好的小模型（`models\<名字>\`，GGUF 加词表文件）。
- 训练检查点（`.pt`，含优化器状态，体积大）不入库，需要时由 `pipeline/export_gguf.py` 导出。

## 登记

| 日期 | 数据 | 来源 | 位置 |
|---|---|---|---|
| 2026-10-16 | 维基百科中文 20231101 第 1 分片（560 MB） | HF 数据集 wikimedia/wikipedia | `DISTILL_DATA\hf\` |
| 2026-10-16 | Qwen3.5-0.8B / 4B / 9B 权重 | HF Qwen/Qwen3.5-* | `DISTILL_DATA\hf\` |
