#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 24 课画图：真实 loss 曲线（从零 + 断点续训拼接）+ 同步数对比（真实数据）"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 读真实日志 ----------
with open("ckpts/runA-log.json") as f:
    runA = json.load(f)          # [step, train_loss, val_loss, lr]
with open("ckpts/runB-log.json") as f:
    runB = json.load(f)          # step 401..800（从 step400 checkpoint 恢复）
with open("ckpts/fresh200-log.json") as f:
    fresh = json.load(f)         # 从零训练 200 步

# ---------- 图 1：从零 + 断点续训拼接的完整曲线 ----------
steps_a = [s for s, l, v, lr in runA]
loss_a = [l for s, l, v, lr in runA]
val_a_steps = [s for s, l, v, lr in runA if v is not None]
val_a = [v for s, l, v, lr in runA if v is not None]

steps_b = [s for s, l, v, lr in runB]
loss_b = [l for s, l, v, lr in runB]
val_b_steps = [s for s, l, v, lr in runB if v is not None]
val_b = [v for s, l, v, lr in runB if v is not None]

fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
ax.axvspan(0, 400, color="#4C72B0", alpha=0.08)
ax.axvspan(400, 800, color="#55A868", alpha=0.08)
ax.text(200, 1.72, "从零训练 runA", ha="center", fontsize=11, color="#4C72B0", fontweight="bold")
ax.text(600, 1.72, "断点续训 runB", ha="center", fontsize=11, color="#55A868", fontweight="bold")

ax.plot(steps_a, loss_a, marker="o", color="#4C72B0", linewidth=2, markersize=5,
        label="train loss（从零，step 1-400）")
ax.plot(steps_b, loss_b, marker="o", color="#55A868", linewidth=2, markersize=5,
        label="train loss（断点续训，step 401-800）")
ax.plot(val_a_steps, val_a, "D", color="#C44E52", markersize=7,
        label="val loss（从零）")
ax.plot(val_b_steps, val_b, "D", color="#DD8452", markersize=7,
        label="val loss（断点续训）")

ax.axvline(400, color="gray", linestyle="--", linewidth=1.5)
ax.text(404, 2.42, "step 400 checkpoint 保存 → 恢复", fontsize=10, color="gray")

# 标注关键点
for s, l in [(1, 4.2148), (400, 2.0024), (401, 1.9598), (800, 1.8008)]:
    ax.annotate(f"{l:.3f}", (s, l), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9, color="#333333")
ax.annotate(f"val 2.03", (400, 2.0319), textcoords="offset points",
            xytext=(-18, -22), ha="center", fontsize=9, color="#C44E52")
ax.annotate(f"val 1.90", (800, 1.8972), textcoords="offset points",
            xytext=(-18, -24), ha="center", fontsize=9, color="#DD8452")

ax.set_xlabel("训练步数（累计）", fontsize=11)
ax.set_ylabel("loss（交叉熵，nats）", fontsize=11)
ax.set_title("第 24 课：断点续训真实曲线——step 400 保存，恢复后 loss 无缝衔接（2 层 4 头 128 维，Mac mini 实测）",
             fontsize=12.5, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("../../images/24-loss-curve.png", bbox_inches="tight")
plt.close()
print("saved ../../images/24-loss-curve.png")

# ---------- 图 2：同样 200 步：从零 vs 断点续训 ----------
fresh_steps = [s for s, l, v, lr in fresh]
fresh_loss = [l for s, l, v, lr in fresh]
# runB 前 200 步（step 401..600，映射到"本次投入步数"1..200）
resume_steps = [s - 400 for s, l, v, lr in runB if s <= 600]
resume_loss = [l for s, l, v, lr in runB if s <= 600]

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
ax.plot(fresh_steps, fresh_loss, marker="o", color="#C44E52", linewidth=2, markersize=5,
        label="从零重训 200 步")
ax.plot(resume_steps, resume_loss, marker="o", color="#55A868", linewidth=2, markersize=5,
        label="断点续训 200 步（恢复自 step 400 checkpoint）")

# 端点标注
ax.annotate("2.28", (200, 2.2763), textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=10, color="#C44E52", fontweight="bold")
ax.annotate("1.88", (200, 1.8802), textcoords="offset points", xytext=(0, -20),
            ha="center", fontsize=10, color="#55A868", fontweight="bold")
ax.annotate("同一个 200 步，\n续训模型已经累计学完 600 步",
            xy=(200, 1.8802), xytext=(120, 1.62), fontsize=10, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333"))

ax.set_xlabel("本次训练投入的步数", fontsize=11)
ax.set_ylabel("loss（交叉熵，nats）", fontsize=11)
ax.set_title("第 24 课：同样花 200 步，断点续训 vs 从零重训（真实数据，Mac mini 实测）",
             fontsize=12.5, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig("../../images/24-resume-compare.png", bbox_inches="tight")
plt.close()
print("saved ../../images/24-resume-compare.png")
