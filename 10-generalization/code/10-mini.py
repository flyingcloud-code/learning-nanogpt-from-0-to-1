# -*- coding: utf-8 -*-
"""
第 10 课最小演示：过拟合 vs L2 正则（40 行）
==============================================
运行命令：
  ~/projects/main-agent/nanoGPT/.venv/bin/python 10-mini.py

依赖：torch 2.12.1 + numpy（venv 已装）
输出：两行对比 —— 无正则 vs 加 L2 惩罚
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
LR, STEPS = 1e-3, 3000

# 1. 数据：sin 的 12 个带噪训练点 + 300 个干净验证点
np.random.seed(0)
x_tr = np.random.uniform(-1.5, 1.5, 12).astype(np.float32)
y_tr = (np.sin(2 * np.pi * x_tr) + 0.3 * np.random.randn(12)).astype(np.float32)
x_va = np.linspace(-1.5, 1.5, 300).astype(np.float32)
y_va = np.sin(2 * np.pi * x_va).astype(np.float32)
xt, yt = torch.tensor(x_tr[:, None]), torch.tensor(y_tr[:, None])
xv, yv = torch.tensor(x_va[:, None]), torch.tensor(y_va[:, None])

def run(lam):
    """lam=0 无正则；lam>0 加 L2 惩罚。返回 (train_loss, val_loss, 权重范数)。"""
    torch.manual_seed(42)                    # 固定起点：两种配置从同一个模型出发
    model = nn.Sequential(                    # 大容量 MLP：1-512-512-1（26 万参数）
        nn.Linear(1, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 1))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        mse = nn.functional.mse_loss(model(xt), yt)
        l2 = sum((p ** 2).sum() for p in model.parameters())  # ① 所有权重平方和
        (mse + lam * l2).backward()          # ② 损失 = 数据误差 + λ×权重平方和
        opt.step()
    with torch.no_grad():
        t = nn.functional.mse_loss(model(xt), yt).item()
        v = nn.functional.mse_loss(model(xv), yv).item()
    w = sum((p ** 2).sum().item() for p in model.parameters())
    return t, v, w

t0, v0, w0 = run(0.0)                        # 无正则
t1, v1, w1 = run(1e-3)                       # L2 正则 λ=1e-3
print(f"无正则   : train={t0:.5f} val={v0:.4f} 权重范数={w0:.1f}  ← 背答案")
print(f"加 L2    : train={t1:.5f} val={v1:.4f} 权重范数={w1:.1f}  ← 学规律")
