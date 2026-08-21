# Learning NanoGPT from 0 to 1 / 手搓大模型：从零构建 NanoGPT

从零开始构建你自己的 GPT：30 天系列课程，面向零基础读者。

Build your own GPT from scratch: a 30-day course series for absolute beginners. Every concept is unpacked layer by layer ("庖丁解牛"), with real code, real experiments, and real outputs — all reproducible on a Mac mini (Apple Silicon).

## 系列简介

- **目标读者**：零基础（会一点 Python 更好）。学完能理解 GPT 原理、看懂 GPT-2 源码、自己训练一个 mini GPT
- **节奏**：每天一课，共 30 课
- **风格**：庖丁解牛——每个概念摊开、一层层讲清楚，代码 + 图例 + 内部流程
- **环境**：Mac mini（Apple Silicon / MPS）+ PyTorch 2.12.1

## 章节结构约定

每个章节一个目录，统一结构：

```
NN-slug/
├── README.md       # 该课完整讲解（中文）
├── code/           # 该课代码（有代码的章节才有）
└── images/         # 该课配图
```

**关于代码**：不是每章都有代码。
- 有代码的章节：`code/` 目录放可运行脚本，README 顶部标注 `本节代码：✅`
- 纯概念章节：README 顶部标注 `本节代码：无（纯概念）`，并尽量附一个可运行的小 demo
- 所有代码在 Mac mini（Apple Silicon + MPS）上实测可运行

## 30 课大纲

| # | 章节 | 档位 | 状态 |
|---|------|------|------|
| 01 | [全景：一次对话背后发生了什么](01-panorama/README.md) | C | ✅ 已发布 |
| 02 | [环境搭建 + 第一个张量程序](02-environment/README.md) | C | ✅ 已发布 |
| 03 | [数学地基上：线代+微积分直觉](03-math-foundations/README.md) | C | ✅ 已发布 |
| 04 | [数学地基下：概率+信息论](04-probability-information/README.md) | C | ✅ 已发布 |
| 05 | [NumPy 手写线性回归+梯度下降](05-linear-regression/README.md) | A | ✅ 已发布 |
| 06 | [感知机→MLP：前向传播](06-mlp-forward/README.md) | A | ✅ 已发布 |
| 07 | [反向传播：从错误反推梯度](07-backpropagation/README.md) | S | ✅ 已发布 |
| 08 | [Loss 与收敛](08-loss-convergence/README.md) | S | ✅ 已发布 |
| 09 | [优化器：SGD/Adam/学习率](09-optimizers/README.md) | A | ✅ 已发布 |
| 10 | [泛化：过拟合与正则化](10-generalization/README.md) | S | ✅ 已发布 |
| 11 | [Embedding：让计算机理解"词"](11-embedding/README.md) | S | ✅ 已发布 |
| 12 | [语言模型：next token prediction](12-next-token/README.md) | A | ✅ 已发布 |
| 13 | [Tokenizer：BPE 原理](13-tokenizer/README.md) | C | ✅ 已发布 |
| 14 | [Attention 直觉](14-attention/README.md) | S | ✅ 已发布 |
| 15 | [QKV 详解](15-qkv/README.md) | S | ✅ 已发布 |
| 16 | [因果掩码 + 多头](16-causal-multihead/README.md) | S | ✅ 已发布 |
| 17 | [Transformer 块](17-transformer-block/README.md) | S | ✅ 已发布 |
| 18 | [GPT 全景：读 model.py](18-model-py/README.md) | A | ✅ 已发布 |
| 19 | [项目骨架：配置/数据/训练循环](19-train-loop/README.md) | B | ✅ 已发布 |
| 20 | [手搓 BPE Tokenizer](20-bpe-tokenizer/README.md) | A | ✅ 已发布 |
| 21 | 手搓 Self-Attention（QKV 落地） | S | 📝 计划中 |
| 22 | 手搓 Transformer Block + 位置编码 | A | 📝 计划中 |
| 23 | 手搓 GPT 模型完整 forward | S | 📝 计划中 |
| 24 | 训练循环：loss 收敛实战 | S | 📝 计划中 |
| 25 | 生成与评估 | B | 📝 计划中 |
| 26 | 里程碑①：训练莎士比亚模型 | S | 📝 计划中 |
| 27 | 换大数据集 | B | 📝 计划中 |
| 28 | 微调对话风格 | B | 📝 计划中 |
| 29 | 视野：Scaling law → GPT-2/3 → RLHF | C | 📝 计划中 |
| 30 | 结业：把模型做成可玩的东西 + 复盘 | B | 📝 计划中 |

> 档位说明：S=解剖核心课（逐层拆解+多图+逐行代码）；A=概念+实验课；B=实操课（代码跑通优先）；C=轻量课（故事化，点到为止）

## 环境要求

```bash
# Mac mini (Apple Silicon) / 任意 macOS
# PyTorch 2.12.1（注意：2.13.0 有 torch.save bug，请勿使用）
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.12.1 numpy tiktoken
```

参考实现：nanoGPT（https://github.com/karpathy/nanoGPT）

## 进度

- 2026-08-03：系列启动，第 1 课发布
- 完整大纲见 [00-outline.md](00-outline.md)
