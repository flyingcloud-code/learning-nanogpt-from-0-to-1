# 手搓大模型 20：手搓 BPE Tokenizer——一个没有梯度的组件，怎么从零写到对拍

> 本节代码：✅ 见 `code/`（20-bpe-tokenizer.py 一个脚本跑完：手写训练/编码/解码三件套 + 与官方 tiktoken 对拍 + round-trip 验证 + 画图数据；20-make-charts.py 画图）
>
> 本系列《手搓大模型：从零构建 NanoGPT》的第二十课。
> 目标读者：零基础。第 13 课讲了 BPE 的原理：反复数相邻对、把最高频的一对焊死。今天不调现成的 tiktoken，**从头把它写出来**——训练、编码、解码三件套，然后拿去和 GPT-2 官方词表对拍。

先摆一组真实数字（Mac mini 实测，一字未改）：

- **BPE 训练不需要梯度、不需要 loss、不需要反向传播**：20,000 字符莎士比亚文本，纯 Python 训练 300 次合并只花 **0.32 秒**——整个系列里唯一一个"零学习"组件；
- **encode → decode 往返 100% 还原**：6,000 字符测试段，编码成 2,678 个词块再解码，**一字不差**。BPE 是可逆的；
- **同一个词，词表大小决定切法**：手写 358 词块的词表把 `love` 切成 `l`+`ove`，GPT-2 官方 50,257 词块把它整个打包成一个 token——**词表越大，常见词越整**；
- **字节级 BPE 能处理中文**：`你好，世界！` 转成 UTF-8 字节再跑 BPE，编码、解码、还原，全流程畅通——这就是 GPT-2 处理所有语言的真实做法。

**这一课的核心思想只有一句大白话：BPE 的"训练"学的不是权重，而是一张"怎么切"的规则表。模型会背单词，分词器负责把文本切成模型认识的词块——前者靠梯度，后者靠数数。** 今天把这张规则表的三个工序全部手写出来。

## 代码

```bash
# 完整实验（训练/编码/解码三件套 + tiktoken 对拍 + 画图数据）
~/projects/main-agent/nanoGPT/.venv/bin/python code/20-bpe-tokenizer.py

# 画图（压缩比曲线 + 词表对比表）
~/projects/main-agent/nanoGPT/.venv/bin/python code/20-make-charts.py
```

依赖：`tiktoken`（仅对拍验证用；BPE 训练本身只需要标准库 `collections.Counter`）、`numpy`（画图数据）、`matplotlib`（画图）。

## 真实输出速览

```text
训练 300 次合并耗时: 0.316s（纯 Python，20,000 字符）
前 12 次合并: 'e'+' ' 517次 → 'th' 402次 → 't '+' 321次 ...
round-trip 一致性: True（6000 字符 -> 2678 词块 -> 一字不差还原）
手写 358 词块 vs 官方 r50k 50257:
  'the'  -> ['the'] / ['the']          （两边都整块）
  'love' -> ['l','ove'] / ['love']      （词表越大越整）
  'unhappiness' -> ['un','ha','p','p','in','es','s'] / ['un','h','appiness']
字节级 BPE: '你好，世界！' 66 字节 -> 3 词块 -> 往返还原 True
压缩比曲线: 合并 0→1.00x, 300→2.27x, 1500→3.47x, 3000→4.44x（收益递减）
```

## 配图

| 图 | 内容 |
|----|------|
| `images/20-bpe-flow.png` | 三件套流程图（训练/编码/解码） |
| `images/20-vocab-compare.png` | 手写 vs 官方对拍（真实输出） |
| `images/20-compression-ratio.png` | 压缩比随合并次数变化（真实数据） |
| `images/cover-20.png` | 封面 |

## 教学灵魂

1. **BPE 是系列里唯一"零学习"组件**：没有梯度、没有 loss、没有反向传播，纯数数 + 合并，300 次合并 0.32 秒；
2. **可逆性**：encode → decode 往返 100% 还原，BPE 是"换一种更紧凑的表示"不是有损压缩；
3. **词表大小决定"基本粒子"的粒度**：同一个词，358 词表里是碎片，50,257 词表里是整块；
4. **字节级 BPE = GPT-2 处理全世界语言的秘密**：任何语言先转 UTF-8 字节（0-255），再在字节上做合并。

## 系列目录

- 上一课：[19 项目骨架：训练循环、get_batch、AdamW、学习率调度](../19-train-loop/README.md)
- 下一课：[21 手搓 Self-Attention（QKV 落地）](../21-self-attention/README.md)（即将发布）

---

**本课完整代码与全文已开源到 GitHub（public）：**

https://github.com/flyingcloud-code/learning-nanogpt-from-0-to-1/tree/main/20-bpe-tokenizer

**系列仓库（30 课陆续更新中）：**

https://github.com/flyingcloud-code/learning-nanogpt-from-0-to-1
