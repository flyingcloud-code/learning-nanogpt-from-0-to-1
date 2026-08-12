"""
第 11 课最小演示：Embedding 查表 = one-hot 矩阵乘法
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 11-mini.py
依赖: torch（venv 已装）
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

vocab = ['h', 'e', 'l', 'o']          # 词表 4 个字符
emb = nn.Embedding(4, 3)              # 4x3 权重矩阵：4 行词表 × 3 维向量

with torch.no_grad():
    W = emb.weight.numpy()

print("权重矩阵 W = (4 行词表) x (3 维向量)：")
print(W.round(3))

idx = torch.tensor([1])               # 字符 'e' 的索引
vec = emb(idx).detach().numpy()       # 查表：取第 1 行
print("\n查表 emb([1])   =", vec.round(3))

onehot = torch.zeros(1, 4)
onehot[0, 1] = 1.0
vec2 = (onehot @ emb.weight).detach().numpy()   # one-hot 乘权重矩阵
print("one-hot 乘权重 =", vec2.round(3))

diff = abs(vec[0] - vec2[0]).max().item()
print(f"\n最大差异 = {diff:.2e}  → 完全相等，查表只是省略了 one-hot 而已")
