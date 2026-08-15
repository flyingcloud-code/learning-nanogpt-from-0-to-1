#!/usr/bin/env python3
"""
14-attention-demo.py — 第 14 课《Attention 直觉》最小可运行演示
纯 numpy 实现 self-attention：没有 torch，没有训练，一个函数看懂核心。
依赖: numpy（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 14-attention-demo.py
"""
import numpy as np

# ─────────────────────────────────────────────
# 一个迷你序列：3 个 token，每个 token 用 4 维向量表示
# （实际模型里这来自 embedding 层，第 11 课讲过：一个字符 = 一行数字）
# ─────────────────────────────────────────────
X = np.array([
    [1.0, 0.0, 0.0, 0.0],   # token 0：字符 'A' 的向量
    [0.0, 1.0, 0.0, 0.0],   # token 1：字符 'B' 的向量
    [0.0, 0.0, 1.0, 0.0],   # token 2：字符 'C' 的向量
], dtype=float)
T, d = X.shape  # T=3 个 token，每个 d=4 维
print("输入 X（3 个 token，每个 4 维）：")
print(X)

# 3 个可学习矩阵：Wq（查什么）、Wk（是什么）、Wv（贡献什么）
# 真实模型里这些是训练出来的；这里随手给一组固定值演示计算过程
rng = np.random.default_rng(42)
Wq = rng.normal(0, 1, (d, d))
Wk = rng.normal(0, 1, (d, d))
Wv = rng.normal(0, 1, (d, d))

# ─────────────────────────────────────────────
# self-attention 的全部公式，就这 4 行：
# ─────────────────────────────────────────────
Q = X @ Wq                     # 每个 token 变成“查询向量”：我想找谁？
K = X @ Wk                     # 每个 token 变成“键向量”：我是什么？
V = X @ Wv                     # 每个 token 变成“值向量”：我能提供什么？
scores = Q @ K.T / np.sqrt(d)  # 相关度打分：查询 × 键，除以 sqrt(d) 防止数值爆炸
weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights = weights / weights.sum(axis=-1, keepdims=True)  # softmax：归一成“注意力分配比例”
out = weights @ V              # 加权求和：用比例把各 token 的值混合起来

print("\nQ（查询，每行代表一个 token 在问“谁跟我相关”）：")
print(np.round(Q, 2))
print("\nK（键，每行代表一个 token 在回答“我是谁”）:")
print(np.round(K, 2))
print("\n相关度分数 scores = Q·Kᵀ / √d：")
print(np.round(scores, 2))
print("\nsoftmax 后的注意力权重（每行加起来 = 1）：")
print(np.round(weights, 2))
print("\n输出 out = 权重 × 值 的加权求和：")
print(np.round(out, 2))

# ─────────────────────────────────────────────
# 关键观察：第 i 个 token 的输出，是所有 token 的值的加权混合
# 权重由“相关性”决定 —— 这就是“注意力”三个字的全部含义
# ─────────────────────────────────────────────
print("\n解读：out[0] = %.2f×V[0] + %.2f×V[1] + %.2f×V[2]"
      % tuple(weights[0]))
print("token 0 的输出里，自己占 %.0f%%，token 1 占 %.0f%%，token 2 占 %.0f%%"
      % tuple(weights[0] * 100))
