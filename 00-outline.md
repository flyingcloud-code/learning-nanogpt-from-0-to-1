# 手搓大模型：从零构建 NanoGPT —— 系列大纲（30 课定稿）

## 系列信息

- **名称**：《手搓大模型：从零构建 NanoGPT》
- **定位**：面向零基础读者的 30 天系列课程。学完能理解 GPT 原理、看懂 GPT-2 源码、自己训练一个 mini GPT
- **节奏**：每天 1 篇，连续 30 天，08:00 发布
- **渠道**：微信公众号（走 wechat-publish.sh 管道）
- **风格**：庖丁解牛——每个概念摊开、一层层讲清楚，代码 + 图例 + 内部流程三件套
- **难度曲线**：前 6 课零基础（只要求会 Python 基础语法），第 30 课达到中级（能读懂 nanoGPT 全部源码）
- **档位定义**：S=解剖核心课（必须庖丁解牛，2-4 图，逐行代码）；A=概念+实验课（重点概念拆开，1-2 图+真实输出）；B=实操课（代码跑通优先）；C=轻量课（故事化，1 图，代码点到为止）

## 环境基线（Mac mini 已验证 ✅ 2026-08-03）

| 项 | 值 |
|---|---|
| 硬件 | Apple Silicon Mac mini（MPS） |
| PyTorch | **2.12.1**（⚠️ 2.13.0 有 torch.save bug：`ModuleNotFoundError: torch.utils.serialization`） |
| nanoGPT | ~/projects/main-agent/nanoGPT（venv 已配好） |
| 数据 | tinyshakespeare，1.1MB，65 字符词表，train 1,003,854 / val 111,540 tokens |
| 实测速度 | baby GPT（6层/6头/384维，10.65M 参数）~790ms/step |
| 实测收敛 | 300 步 loss 4.27→1.90；1000 步 → train 1.28 / val 1.52 |
| 完整训练 | 5000 步 ≈ 1.5-2 小时（含 checkpoint 保存开销），Mac mini 后台可跑 |

## 庖丁解牛写作模板（S/A 档每篇必套）

```
0. 一句话直觉 —— 先大白话，让读者"哦"一下
1. 动机：没有它行不行？（从读者会问的问题出发）
2. 第一层解剖：结构长啥样（架构图）
3. 第二层解剖：数据怎么流动（流程图画输入→输出）
4. 第三层解剖：数学细节（每个符号用直觉解释，不堆公式）
5. 代码层：最小可运行代码，逐行注释
6. 实验层：真实运行输出（loss 曲线/生成文本/热力图）
7. 误区与彩蛋
```

## 图例规范

- **架构图**：SVG（baoyu-diagram 深色主题）或手绘风
- **流程图**：数据流动（输入→每层→输出）
- **曲线图**：matplotlib 真实训练输出，不造假
- **数字示例**：表格形式手算（如 QKV 2 个词算一遍）
- **每篇至少 1 张图**，S 档 2-4 张

## 30 课大纲

### 阶段一：地基（数学 + 工具）—— 第 1-5 课

| # | 标题 | 档位 | 核心内容 | 图例需求 |
|---|------|------|----------|----------|
| 1 | 全景：一次对话背后发生了什么 | C | LLM 全流程：输入→token→模型→概率→采样 | 对话流程图 |
| 2 | 环境搭建 + 第一个张量程序 | C | venv/torch 2.12.1 安装、MPS 验证、张量基础、**torch 2.13.0 坑** | 安装流程 |
| 3 | 数学地基上：线代+微积分直觉 | C | 矩阵乘法=批量映射、导数=斜率、链式法则 | 矩阵乘法可视化 |
| 4 | 数学地基下：概率+信息论 | C | 分布、熵=惊讶度、交叉熵=损失源头、KL 散度 | 分布曲线 |
| 5 | NumPy 手写线性回归+梯度下降 | A | 第一次训练、loss 曲线 | loss 下降曲线 |

### 阶段二：神经网络核心 —— 第 6-11 课

| # | 标题 | 档位 | 核心内容 | 图例需求 |
|---|------|------|----------|----------|
| 6 | 感知机→MLP：前向传播 | A | 神经元、激活函数、隐藏层、手写前向 | 网络结构图 |
| 7 | **反向传播：从错误反推梯度** | S | 计算图、链式法则落地、手推 2 层网络、梯度检查 | 计算图（前向+反向） |
| 8 | **Loss 与收敛** | S | 交叉熵/MSE、loss 曲线读法、学习率影响 | 不同 lr 曲线对比（实验） |
| 9 | 优化器：SGD/Adam/学习率 | A | 动量、Adam 直觉 | 优化路径示意 |
| 10 | **泛化：过拟合与正则化** | S | 数据集划分、dropout、weight decay、早停 | 过拟合曲线（实验对比） |
| 11 | **Embedding：让计算机理解"词"** | S | one-hot→查表、nn.Embedding 权重、索引计算、PCA 可视化 | 词表矩阵图 + PCA 散点图（真实） |

### 阶段三：语言模型与 Transformer —— 第 12-18 课

| # | 标题 | 档位 | 核心内容 | 图例需求 |
|---|------|------|----------|----------|
| 12 | 语言模型：next token prediction | A | n-gram→神经网络、训练目标、困惑度 | next-token 示意 |
| 13 | Tokenizer：BPE 原理 | C | 为什么需要、合并过程、词表构建 | BPE 合并演示 |
| 14 | **Attention 直觉** | S | "找相关词"、加权求和、softmax | 注意力权重热力图（真实） |
| 15 | **QKV 详解** | S | Q/K/V 三矩阵、Q·Kᵀ/softmax/·V、2 个词手算 | 矩阵分解图 + 手算表 |
| 16 | **因果掩码 + 多头** | S | 只看左边、mask 矩阵、多头=多视角 | mask 矩阵图 |
| 17 | **Transformer 块** | S | LayerNorm/残差/FFN/位置编码、完整块 | 块结构图 + 位置编码热力图 |
| 18 | GPT 全景：读 model.py | A | token emb + pos emb + blocks + LM head、参数量 | GPT 架构总图 |

### 阶段四：手搓 NanoGPT —— 第 19-28 课

| # | 标题 | 档位 | 核心内容 | 图例需求 |
|---|------|------|----------|----------|
| 19 | 项目骨架：配置/数据/训练循环 | B | 读 train.py、get_batch、AdamW、lr 调度 | 训练循环流程图 |
| 20 | 手搓 BPE Tokenizer | A | 从零实现 BPE | 代码为主 |
| 21 | **手搓 Self-Attention（QKV 落地）** | S | 不用现成 API 实现 attention、与 nn.MultiheadAttention 对拍 | 矩阵流程 |
| 22 | 手搓 Transformer Block + 位置编码 | A | 组装 LayerNorm/残差/FFN | 结构图 |
| 23 | 手搓 GPT 模型完整 forward | S | 拼装全部组件、forward 流程、参数量核对 | GPT 结构图 |
| 24 | **训练循环：loss 收敛实战** | S | 训练微型 GPT、真实 loss 曲线、断点续训 | 真实 loss 曲线 |
| 25 | 生成与评估 | B | temperature/top-k/top-p/perplexity | 不同温度输出对比 |
| 26 | **里程碑①：训练莎士比亚模型** | S | 完整 5000 步训练（Mac 上 1.5-2h）、生成文本展示 | 完整 loss 曲线 + 生成样例 |
| 27 | 换大数据集 | B | char→BPE、更大语料、效果对比 | 两种方案对比 |
| 28 | 微调对话风格 | B | fine-tune 原理、对话数据、微调前后对比 | 前后生成对比 |

### 阶段五：进阶视野 —— 第 29-30 课

| # | 标题 | 档位 | 核心内容 | 图例需求 |
|---|------|------|----------|----------|
| 29 | 视野：Scaling law → GPT-2/3 → RLHF | C | 参数/数据/算力规律、InstructGPT 对齐 | scaling 曲线 |
| 30 | 结业：把模型做成可玩的东西 + 复盘 | B | 封装小工具、系列回顾、进阶路径 | 成果展示 |

## 核心课清单（S 档 11 课，重点投入）

7 反向传播、8 Loss收敛、10 泛化、11 Embedding、14 Attention直觉、15 QKV、16 因果掩码+多头、17 Transformer块、21 手搓Attention、23 手搓GPT、24 训练循环、26 里程碑①

## 发布检查清单（每篇发布前）

- [ ] 概念已按庖丁解牛模板摊开（S/A 档）
- [ ] 至少 1 张图（S 档 2-4 张），图为真实数据/真实结构
- [ ] 代码在本机跑通过（venv: ~/projects/main-agent/nanoGPT/.venv，torch 2.12.1）
- [ ] 实验输出为真实结果（loss 曲线、生成文本），不造假
- [ ] 面向小白：无未解释术语、数学只讲直觉
- [ ] 长文无 AI 味：无 emoji 滥用、无"深入剖析/核心洞察/这意味着/本质上"
- [ ] 不用"你"通指读者
- [ ] 文末必须有 GitHub 公开路径段落：本课 `https://github.com/flyingcloud-code/learning-nanogpt-from-0-to-1/tree/main/NN-slug` + 系列仓库 `https://github.com/flyingcloud-code/learning-nanogpt-from-0-to-1`（md/html/飞书/微信四处一致，NN-slug 与 repo 实际目录一致）
- [ ] 已转 HTML（baoyu-markdown-to-html --theme modern --color blue），走 wechat-publish.sh 发布
- [ ] 微信 draft/add 后必须 draft/get 回读验证（40007 即重推；验证通过才算发布成功，并记录 media_id 到 drafts-state.json）
- [ ] 文件写 /Volumes/External-HD-data/leo-universe/learning/series/hand-made-gpt/，编号 01-30

## 每日 cron 设计（待创建）

- 触发：每天 08:00，持续 30 天（第 1 天从课 01 开始，按大纲顺序推进）
- 任务：读本大纲 + 当天课号 → 跑/复用已验证代码 → 写文章（庖丁解牛模板）→ 生成配图 → 转 HTML → 发公众号
- 交付：origin（本对话）
