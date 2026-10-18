# 声明与致谢

## 训练数据

Suzu 的训练和评测题均来自维基百科中文（Hugging Face 数据集 `wikimedia/wikipedia`，快照 20231101，第 1 分片），
文本由维基百科的贡献者撰写，按 CC BY-SA 3.0 与 GFDL 许可发布。仓库不包含语料原文，只包含由语料派生的统计结果和模型。
使用本模型时，请保留对维基百科及其贡献者的署名。用该语料训练得到的模型权重是否适用“相同方式共享”条款，目前没有定论；
发布权重时采用的许可证见 `LICENSE`，并在模型卡中重复本声明。

## 教师与对照模型

教师打分和对照评测使用 Qwen3.5-0.8B、Qwen3.5-4B、Qwen3.5-9B（Qwen 团队，Apache-2.0）。
仓库不包含这些模型的权重；训练标签由它们对候选打分得到，学生模型的权重独立训练。

## 依赖

jieba、pypinyin、opencc-python-reimplemented、transformers、bitsandbytes、flash-linear-attention、triton-windows 等，
各自遵循其许可证，版本见 `requirements.txt` 与 `setup.ps1`。
