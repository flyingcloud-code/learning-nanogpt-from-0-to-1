#!/usr/bin/env python3
"""
15-qkv-handcalc.py — 第 15 课《QKV 详解》实验脚本（2 个 token 手算 QKV 全过程）
依赖: numpy（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 15-qkv-handcalc.py

目的：用 2 个 token 把 Q/K/V 三矩阵、Q·Kᵀ/softmax/·V 的每个数字都算一遍，
     让读者看到 Attention 的"零件"到底在干什么。
"""
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=120)

# ─────────────────────────────────────────────
# 1. 输入：2 个 token 的向量（d=4 维）
#    真实模型里这是 embedding 层的输出（第 11 课），这里直接给固定向量
# ─────────────────────────────────────────────
X = np.array([
    [0.8, 0.2, -0.5, 0.1],   # token 0（比如字符 'h'）
    [0.3, -0.7, 0.9, 0.4],   # token 1（比如字符 'e'）
], dtype=float)
T, d = X.shape               # T=2 个 token，每个 d=4 维
print("输入 X（2 个 token，每个 4 维向量）：")
print(X)
print(f"  形状: ({T}, {d})\n")

# ─────────────────────────────────────────────
# 2. 三个可学习矩阵 Wq / Wk / Wv
#    真实模型里它们随机初始化、训练得到；这里固定种子，保证数字可复现
# ─────────────────────────────────────────────
rng = np.random.default_rng(15)
Wq = rng.normal(0, 1, (d, d))
Wk = rng.normal(0, 1, (d, d))
Wv = rng.normal(0, 1, (d, d))

print("Wq（查询投影矩阵，d×d）：")
print(Wq)
print("Wk（键投影矩阵，d×d）：")
print(Wk)
print("Wv（值投影矩阵，d×d）：")
print(Wv)
print()

# ─────────────────────────────────────────────
# 3. 投影：X @ Wq = Q，X @ Wk = K，X @ Wv = V
#    矩阵乘法 = 批量映射（第 3 课结论）：一次算完所有 token
# ─────────────────────────────────────────────
Q = X @ Wq
K = X @ Wk
V = X @ Wv

print("Q（查询：每个 token 在想找什么）＝ X @ Wq：")
print(Q)
print("K（键：每个 token 是什么）＝ X @ Wk：")
print(K)
print("V（值：每个 token 能提供什么内容）＝ X @ Wv：")
print(V)
print()

# ─────────────────────────────────────────────
# 4. 打分：scores = Q @ Kᵀ / √d
#    Q 的第 i 行 与 K 的第 j 行做点积 → 分数表 (i, j) = "token i 觉得 token j 多相关"
# ─────────────────────────────────────────────
scores_raw = Q @ K.T                    # 不缩放版本（先看看）
scores = scores_raw / np.sqrt(d)        # 除以 √d 后的版本
print("Q @ Kᵀ（不缩放的相关度分数，2×2）：")
print(scores_raw)
print("Q @ Kᵀ / √d（除以 √4=2 后）：")
print(scores)
print()

# ─────────────────────────────────────────────
# 5. softmax 归一：每行变成"加起来=1 的比例"
# ─────────────────────────────────────────────
def softmax_rows(M):
    e = np.exp(M - M.max(axis=-1, keepdims=True))   # 减去每行最大值，数值稳定
    return e / e.sum(axis=-1, keepdims=True)

weights = softmax_rows(scores)
print("softmax(Q·Kᵀ/√d)（每行是注意力分配比例，行和=1）：")
print(weights)
print(f"  行和: {weights.sum(axis=1)}\n")

# ─────────────────────────────────────────────
# 6. 混合：out = weights @ V —— 每个 token 的新表示
# ─────────────────────────────────────────────
out = weights @ V
print("out = weights @ V（每个 token 的新表示＝按相关度加权混合所有 V）：")
print(out)
print()

# ─────────────────────────────────────────────
# 7. 手工展开第一行的计算（把矩阵乘法写成人话）
# ─────────────────────────────────────────────
print("=" * 70)
print("手工展开：out[0] 是怎么来的")
print("=" * 70)
print("步骤 1：token 0 的查询向量 Q[0] =", Q[0])
print("步骤 2：与每个 K 做点积得到原始分数：")
for j in range(T):
    print(f"  score(0→{j}) = Q[0]·K[{j}] = {Q[0] @ K[j]:.2f}")
print("步骤 3：除以 √d 再 softmax，得到权重：")
for j in range(T):
    print(f"  w(0→{j}) = {weights[0][j]:.4f}")
print("步骤 4：加权求和 out[0] = w(0→0)·V[0] + w(0→1)·V[1]：")
for j in range(T):
    print(f"  + {weights[0][j]:.4f} × V[{j}] = {weights[0][j] * V[j]}")
print(f"  = {out[0]}")
print("（第 0 行输出 = 两个 token 信息的加权混合，权重由相关度决定）")
