# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 08-make-charts.py
# 依赖: matplotlib + numpy（venv 已装）；数据由 08-loss-convergence.py 生成（若 npz 不存在会自动先跑）
# 第 8 课配图：3 张真实数据图（学习率对比 / sigmoid 梯度消失 / CE vs MSE 收敛）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys

# 数据文件：脚本在 code/ 下时 → 上级目录；脚本在课程根目录时 → 同级目录
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "code" else os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# 若实验数据不存在，先跑主实验生成（保证开箱即跑）
if not (os.path.exists(os.path.join(BASE, "exp1_hist.npz")) and os.path.exists(os.path.join(BASE, "exp3_hist.npz"))):
    print("实验数据不存在，先运行 08-loss-convergence.py ...")
    subprocess.run([sys.executable, os.path.join(BASE, "08-loss-convergence.py")], cwd=BASE, check=True)
d1 = np.load(os.path.join(BASE, "exp1_hist.npz"))
d3 = np.load(os.path.join(BASE, "exp3_hist.npz"))

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

DARK = "#0f172a"
GRAY = "#64748b"
CYAN = "#22d3ee"
ROSE = "#fb7185"
AMBER = "#fbbf24"
GREEN = "#4ade80"
VIOLET = "#a78bfa"

# ============ 图 1：不同学习率的 loss 曲线 ============
lrs = [0.0005, 0.005, 0.05, 0.5]
colors = {"0.0005": GRAY, "0.005": AMBER, "0.05": GREEN, "0.5": ROSE}
labels = {"0.0005": "lr=0.0005（太小，爬不动）", "0.005": "lr=0.005（偏小）", "0.05": "lr=0.05（合适）", "0.5": "lr=0.5（太大，爆炸）"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
steps = np.arange(2000)

# 左图：前 500 步，线性轴（能看到 lr=0.5 的爆炸）
ax1.set_facecolor(DARK)
for lr in lrs:
    h = d1[f"lr{lr}"]
    ax1.plot(steps[:500], h[:500], color=colors[str(lr)], label=labels[str(lr)], linewidth=2)
ax1.set_yscale("log")
ax1.set_ylim(1e-4, 1e8)
ax1.set_xlabel("训练步数", color="white")
ax1.set_ylabel("loss（对数轴）", color="white")
ax1.set_title("前 500 步：学习率决定命运", fontsize=11, color="white")
ax1.legend(fontsize=9, facecolor=DARK, edgecolor=GRAY, labelcolor="white")
ax1.tick_params(colors="white")
for s in ["top", "right"]:
    ax1.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax1.spines[s].set_color(GRAY)

# 右图：lr=0.5 爆炸的"现场"（前 16 步）
h = d1["lr0.5"]
ax2.set_facecolor(DARK)
ax2.plot(steps[:16], h[:16], color=ROSE, marker="o", markersize=5, linewidth=2)
ax2.set_yscale("log")
ax2.set_xlabel("训练步数", color="white")
ax2.set_ylabel("loss（对数轴）", color="white")
ax2.set_title("lr=0.5：16 步从 4.95 炸到 10²⁸ → NaN", fontsize=11, color="white")
ax2.tick_params(colors="white")
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax2.spines[s].set_color(GRAY)
ax2.annotate("loss 每步 ×40~60", xy=(8, 8.5e16), xytext=(4, 1e12), color=ROSE, fontsize=10,
             arrowprops=dict(arrowstyle="->", color=ROSE))

fig.suptitle("同一个任务、同一个网络、同一起点，只有学习率不同", fontsize=13, color="white", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(IMG_DIR, "08-lr-compare.png"), dpi=150, facecolor=DARK)
plt.close(fig)
print("saved 08-lr-compare.png")

# ============ 图 2：sigmoid 饱和区，CE vs MSE 梯度 ============
yh = np.array([0.5, 0.9, 0.99, 0.999, 0.9999])
ce_g = yh - 0.0
mse_g = 2 * (yh - 0.0) * yh * (1 - yh)

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.set_facecolor(DARK)
ax.plot(yh, ce_g, color=CYAN, marker="o", linewidth=2.5, label="交叉熵梯度 = ŷ − y（错多少，推多狠）")
ax.plot(yh, mse_g, color=ROSE, marker="s", linewidth=2.5, label="MSE梯度 = 2(ŷ−y)·ŷ(1−ŷ)（被压扁）")
ax.set_xlabel("模型输出 ŷ（越接近 1 = 越自信地猜'是'，但答案是 0）", color="white")
ax.set_ylabel("梯度大小（学习信号）", color="white")
ax.set_title("预测越自信、错得越离谱，MSE 的梯度越接近 0 —— 学不动", fontsize=11, color="white")
ax.legend(fontsize=9, facecolor=DARK, edgecolor=GRAY, labelcolor="white")
ax.tick_params(colors="white")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRAY)
ax.annotate("ŷ=0.999 时\nMSE 梯度只剩 0.2%", xy=(0.999, 0.002), xytext=(0.86, 0.35), color=ROSE, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=ROSE))
ax.annotate("交叉熵梯度始终 ≈ 错误大小", xy=(0.99, 0.99), xytext=(0.72, 0.85), color=CYAN, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=CYAN))
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "08-gradient-saturation.png"), dpi=150, facecolor=DARK)
plt.close(fig)
print("saved 08-gradient-saturation.png")

# ============ 图 3：XOR 上 CE vs MSE 收敛对比 ============
ce = d3["ce"]
mse = d3["mse"]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.set_facecolor(DARK)
ax.plot(steps, ce, color=CYAN, linewidth=2, label="交叉熵（CE）")
ax.plot(steps, mse, color=ROSE, linewidth=2, label="MSE")
ax.axhline(0.05, color=GRAY, linestyle="--", linewidth=1)
ax.text(10, 0.055, "loss=0.05（基本学会的线）", color=GRAY, fontsize=9)
ax.set_xlabel("训练步数", color="white")
ax.set_ylabel("loss", color="white")
ax.set_title("XOR 分类，相同初始化：CE 在 976 步学会，MSE 拖到 1188 步", fontsize=11, color="white")
ax.legend(fontsize=10, facecolor=DARK, edgecolor=GRAY, labelcolor="white")
ax.tick_params(colors="white")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRAY)
ax.annotate("CE 976 步", xy=(976, 0.05), xytext=(700, 0.25), color=CYAN, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=CYAN))
ax.annotate("MSE 1188 步", xy=(1188, 0.05), xytext=(1150, 0.35), color=ROSE, fontsize=10,
            arrowprops=dict(arrowstyle="->", color=ROSE))
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "08-ce-vs-mse.png"), dpi=150, facecolor=DARK)
plt.close(fig)
print("saved 08-ce-vs-mse.png")
