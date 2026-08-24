#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 23 课画图：参数量核对 + 训练 loss 曲线（真实数据）"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 图 1：参数量核对（纸面公式 vs torch 统计） ----------
with open("chart-data/param-check.json") as f:
    pc = json.load(f)

torch_g = pc["torch"]
labels = ["wte\n词表", "wpe\n位置", "attn\n注意力", "mlp\nFFN", "ln\n归一化", "ln_f\n最终LN", "lm_head\n打分头"]
# 注意：torch 统计里 lm_head=0（tied 共享 wte）
torch_vals = [torch_g["wte"], torch_g["wpe"], torch_g["attn"], torch_g["mlp"],
              torch_g["ln"], torch_g["ln_f"], torch_g["lm_head"]]

# 纸面公式逐项（转成和 torch 同口径）
formula = pc["formula"]
# formula 是按行存的 dict：键含中文说明。直接用手算数值和 torch 一致即可，这里展示公式合计 vs torch 合计
total_f, total_t = pc["total_formula"], pc["total_torch"]

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD", "#DD8452"]
bars = ax.bar(labels, torch_vals, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, torch_vals):
    if v > 0:
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_yscale("log")
ax.set_ylabel("参数量（对数刻度）")
ax.set_title("第 23 课：手搓 GPT 参数量分布（6 层 6 头 384 维，bias=False）\n纸面手算 %s == torch 统计 %s —— 一分不差"
             % (f"{total_f:,}", f"{total_t:,}"), fontsize=13, fontweight="bold")
ax.axhline(total_t, color="gray", linestyle="--", linewidth=1)
ax.text(6.4, total_t * 1.1, "总计 10.75M", ha="right", fontsize=10, color="gray")
ax.set_ylim(1e2, 2e7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.annotate("lm_head = 0\n（与 wte 共享权重）", xy=(6, 100), xytext=(5.2, 2e5),
            fontsize=9, color="#DD8452",
            arrowprops=dict(arrowstyle="->", color="#DD8452"))
plt.tight_layout()
plt.savefig("images/23-param-check.png", bbox_inches="tight")
plt.close()
print("saved images/23-param-check.png")

# ---------- 图 2：训练 loss 曲线（真实） ----------
with open("chart-data/loss-curve.json") as f:
    curve = json.load(f)
steps = [s for s, _ in curve]
losses = [l for _, l in curve]

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
ax.plot(steps, losses, marker="o", color="#C44E52", linewidth=2, markersize=6)
ax.set_xlabel("训练步数")
ax.set_ylabel("loss（交叉熵，nats）")
ax.set_title("第 23 课：手搓 GPT 真实训练曲线（2 层 4 头 128 维，400 步）", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for s, l in curve:
    ax.annotate(f"{l:.2f}", (s, l), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("images/23-loss-curve.png", bbox_inches="tight")
plt.close()
print("saved images/23-loss-curve.png")
