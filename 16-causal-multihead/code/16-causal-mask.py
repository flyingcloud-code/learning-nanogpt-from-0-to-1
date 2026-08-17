#!/usr/bin/env python3
"""
16-causal-mask.py — 第 16 课《因果掩码 + 多头》概念演示
依赖: numpy（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 16-causal-mask.py

目的：用最小代码回答两个问题——
  1. 因果掩码（causal mask）到底长什么样？为什么说它只是"把右上角涂黑"？
  2. mask 加在 softmax 之前，为什么"未来位置的权重"会精确等于 0？

本文所有数字均为本机真实运行输出，可复现。
"""
import numpy as np

# ─────────────────────────────────────────────
# 1. mask 矩阵本身：一个下三角的 0/1 矩阵
#    第 i 行、第 j 列表示"token i 能否看 token j"（j <= i 才能看）
# ─────────────────────────────────────────────
T = 5
mask = np.tril(np.ones((T, T), dtype=float))
print("=" * 56)
print("mask 矩阵（T=5）：1 = 允许看，0 = 禁止看")
print("=" * 56)
print(mask)
print("注意：第 0 行只有自己（0 个历史），第 4 行能看到 0~4 全部历史。")

# ─────────────────────────────────────────────
# 2. 给一组"分数"（真实模型里来自 Q·Kᵀ/√d）
#    故意让未来位置也有高分——模拟"模型其实想偷看右边"的情况
# ─────────────────────────────────────────────
rng = np.random.default_rng(16)
scores = rng.normal(0, 1, (T, T))          # 打分：谁跟谁相关（未加 mask）
scores[0, 3] = 4.0                          # 故意：token 0 很想看 token 3（未来！）
scores[2, 4] = 3.5                          # 故意：token 2 很想看 token 4（未来！）

print("\n" + "=" * 56)
print("打分矩阵 scores = Q·Kᵀ/√d（未加 mask 前）")
print("=" * 56)
print(np.round(scores, 2))
print("高亮两个\"想偷看未来\"的格子：scores[0,3]=4.0, scores[2,4]=3.5")

# ─────────────────────────────────────────────
# 3. 加 mask：把禁止区填成 -inf（不是 0，是负无穷！）
#    这就是"只看左边"的全部实现——一行代码
# ─────────────────────────────────────────────
masked = np.where(mask.astype(bool), scores, -np.inf)   # 上三角位置换成 -inf
print("\n" + "=" * 56)
print("加 mask 后：np.where(允许, scores, -inf)")
print("=" * 56)
print(masked)

# ─────────────────────────────────────────────
# 4. softmax：-inf 会变成 exp(-inf) = 0
#    所以未来位置权重精确等于 0，每一行权重加起来 = 1
# ─────────────────────────────────────────────
def softmax_row(v):
    e = np.exp(v - v.max())      # 减最大值：防溢出，不影响结果
    return e / e.sum()

weights = np.stack([softmax_row(masked[i]) for i in range(T)])
print("\n" + "=" * 56)
print("softmax 之后：未来位置权重精确为 0")
print("=" * 56)
print(np.round(weights, 4))

future_weight = weights[np.triu_indices(T, k=1)].max()
print(f"\n所有\"未来位置\"（j>i）的最大权重 = {future_weight:.6f}")
print(f"（因为 exp(-inf) = 0，所以是精确的 0，不是\"近似\"）")
print(f"每一行的权重之和 = {weights.sum(axis=1)}（都是 1，softmax 归一）")

# ─────────────────────────────────────────────
# 5. 对照：如果不加 mask，那个"偷看未来"的 4.0 会怎样？
# ─────────────────────────────────────────────
w_nomask = softmax_row(scores[0])
print("\n" + "=" * 56)
print("对照：同一行（token 0）不加 mask 的 softmax 结果")
print("=" * 56)
print(f"token 0 看 token 3（未来）的权重 = {w_nomask[3]:.4f}  ← 不加 mask 它真敢看")
print(f"token 0 看 token 3（未来）的权重 = {weights[0, 3]:.4f}  ← 加了 mask 直接归零")
print(f"同一行其余位置权重对比：不加 mask = {np.round(w_nomask, 4)}")
print(f"                   加了 mask = {np.round(weights[0], 4)}")

# ─────────────────────────────────────────────
# 6. 手写因果注意力完整版：6 行代码，这就是 nanoGPT 里 attention 的心脏
#    （真实模型里 Q/K/V 来自三个投影矩阵，这里用随机数代替，只看结构）
# ─────────────────────────────────────────────
X = rng.normal(0, 1, (T, 8))     # T 个 token，每个 8 维（真实模型来自 embedding）
d = X.shape[1]
Wq = rng.normal(0, 0.5, (d, d))
Wk = rng.normal(0, 0.5, (d, d))
Wv = rng.normal(0, 0.5, (d, d))

Q = X @ Wq                        # 查询：我在找什么
K = X @ Wk                        # 键：我是什么
V = X @ Wv                        # 值：我能提供什么
s = Q @ K.T / np.sqrt(d)          # 打分：谁跟我相关（未加 mask）
s = s + np.where(mask.astype(bool), 0.0, -np.inf)  # ← 因果掩码就插在这里：打分之后、softmax 之前
w = np.stack([softmax_row(s[i]) for i in range(T)])  # 归一：每行和 = 1
out = w @ V                       # 混合：按权重搬信息

print("\n" + "=" * 56)
print("手写因果注意力 6 行（结构版）：")
print("=" * 56)
print("  Q = X @ Wq;  K = X @ Wk;  V = X @ Wv")
print("  s = Q @ K.T / np.sqrt(d)")
print("  s = s + np.where(允许, 0, -inf)     ← 因果掩码，只此一行")
print("  w = softmax(s);   out = w @ V")
print(f"\n最终输出 out 的形状 = {out.shape}（每个 token 只吸收了它左边的信息）")
print("\n实验完成：因果掩码 = 打分后把右上角填 -inf，softmax 让未来位置权重精确归零。")
