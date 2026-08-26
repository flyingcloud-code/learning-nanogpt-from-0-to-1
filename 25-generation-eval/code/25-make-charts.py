#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 25 课画图（真实数据，Mac mini 实测）：
  图 1 25-temperature-dist.png —— temperature 对"下一个字符分布"的影响（2x2：三温度条形图 + 温度-熵曲线）
  图 2 25-ppl-compare.png     —— perplexity 对比（随机模型 / 250 步 / 1000 步）

数据来源：
  - logits 来自 25-generate.py 实验 1 保存的 /tmp/25-ambig-logits.pt（真实模型输出）
  - ppl 来自 25-generate.py 实验 4 与探索脚本（真实 val 集评估）
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

plt.rcParams["font.family"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
os.makedirs(OUT, exist_ok=True)

# ---------- 读真实 logits（25-generate.py 实验 1 保存） ----------
d = torch.load("/tmp/25-ambig-logits.pt", weights_only=False)
logits = d["logits"]          # 65 维：'...SEBAST' 位置模型对下一个字符的分数
itos = d["itos"]

# ---------- 图 1：temperature 对分布的影响 ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
temps = [0.2, 0.8, 1.5]
colors = ["#C44E52", "#4C72B0", "#55A868"]
for ax, T, color in zip(axes[0], temps, colors):
    probs = F.softmax(logits / T, dim=-1)
    H = (-probs * torch.log(probs + 1e-9)).sum().item()
    topk = torch.topk(probs, 12)
    chars = [itos[i] for i in topk.indices.tolist()]
    vals = topk.values.tolist()
    labels = [f"'{c}'" if c not in "\n " else ("'\\n'" if c == "\n" else "' '") for c in chars]
    ax.barh(range(len(vals))[::-1], vals, color=color, alpha=0.85)
    ax.set_yticks(range(len(vals))[::-1])
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("概率", fontsize=9)
    ax.set_title(f"T = {T}    熵 H = {H:.3f}", fontsize=12, fontweight="bold", color=color)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, len(vals) - 1 - i, f"{v:.3f}", va="center", fontsize=9)
axes[0][0].set_xlim(0, 1.05)

# 右下：温度 vs 熵曲线（真实计算）
Ts = np.arange(0.1, 3.01, 0.1)
Hs = []
p1s = []
for T in Ts:
    probs = F.softmax(logits / T, dim=-1)
    Hs.append((-probs * torch.log(probs + 1e-9)).sum().item())
    p1s.append(probs.max().item())
ax = axes[1][0]
ax.plot(Ts, Hs, "-o", color="#DD8452", markersize=3, linewidth=2, label="分布熵 H（越大越混乱）")
ax.set_xlabel("temperature T", fontsize=10)
ax.set_ylabel("熵 H（nats）", fontsize=10)
ax.set_title("T 越大，分布越平缓、熵越高", fontsize=12, fontweight="bold", color="#DD8452")
ax.grid(alpha=0.3)
for T, H in [(0.2, Hs[1]), (0.8, Hs[7]), (1.5, Hs[14])]:
    ax.annotate(f"T={T}\nH={H:.2f}", (T, H), textcoords="offset points",
                xytext=(6, -18), fontsize=9, color="#333333")

ax = axes[1][1]
ax.plot(Ts, p1s, "-o", color="#4C72B0", markersize=3, linewidth=2, label="最大候选概率 p1")
ax.set_xlabel("temperature T", fontsize=10)
ax.set_ylabel("最可能字符的概率", fontsize=10)
ax.set_title("T 越小，'最可能字符'越独裁（p1 越接近 1）", fontsize=12, fontweight="bold", color="#4C72B0")
ax.grid(alpha=0.3)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
for T, p in [(0.2, p1s[1]), (1.5, p1s[14])]:
    ax.annotate(f"T={T}\np1={p:.2f}", (T, p), textcoords="offset points",
                xytext=(6, 10), fontsize=9, color="#333333")

fig.suptitle("第 25 课：temperature 是概率分布的'整形旋钮'（真实模型输出，位置 '…SEBAST' 之后，Mac mini 实测）",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
p1 = os.path.join(OUT, "25-temperature-dist.png")
plt.savefig(p1, bbox_inches="tight")
plt.close()
print("saved", p1)

# ---------- 图 2：perplexity 对比 ----------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
names = ["随机模型\n(65 字符等概率)", "250 步模型\n(val loss 2.07)", "1000 步模型\n(val loss 1.52)"]
ppls = [65.0, 7.93, 4.57]
colors2 = ["#999999", "#DD8452", "#55A868"]
bars = ax.bar(names, ppls, color=colors2, alpha=0.9, width=0.55)
for bar, v in zip(bars, ppls):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"ppl = {v:.2f}",
            ha="center", fontsize=12, fontweight="bold", color="#333333")
ax.set_ylabel("perplexity（平均要猜几个字符）", fontsize=11)
ax.set_ylim(0, 78)
ax.set_title("第 25 课：perplexity = exp(loss)——模型越会猜，ppl 越低（真实 val 集评估，Mac mini 实测）",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.text(2, 70, "ppl=65：完全没学过，65 个字符平均猜\nppl=4.57：平均只拿不准 4.57 个字符",
        fontsize=10, color="#555555", va="top", ha="center")
plt.tight_layout()
p2 = os.path.join(OUT, "25-ppl-compare.png")
plt.savefig(p2, bbox_inches="tight")
plt.close()
print("saved", p2)
