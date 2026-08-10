# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 09-make-charts.py
# 依赖: matplotlib + numpy（venv 已装）；数据缺失时自动跑 09-optimizers.py 生成
# 第 9 课图表：左=狭长山谷优化路径（真实轨迹），右=sin 回归 loss 曲线（真实数据）
import os
import sys
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# macOS 中文字体（按名字设，别用 findSystemFonts 探测）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 路径约定：脚本与 npz 都在 code/ 下；图片输出到课程根目录 images/
CODE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(CODE) if os.path.basename(CODE) == "code" else CODE
IMG_DIR = os.path.join(BASE, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# 数据缺失时自动先跑主实验（保证读者 clone repo 后开箱即跑）
if not (os.path.exists(os.path.join(CODE, "exp1_paths.npz")) and os.path.exists(os.path.join(CODE, "exp2_hist.npz"))):
    subprocess.run([sys.executable, os.path.join(CODE, "09-optimizers.py")], cwd=CODE, check=True)

d1 = np.load(os.path.join(CODE, "exp1_paths.npz"))
d2 = np.load(os.path.join(CODE, "exp2_hist.npz"))

# ============ 图 1：狭长山谷等高线 + 4 条优化路径 ============
def quad_loss(x, y):
    return 25.25 * x * x - 49.5 * x * y + 25.25 * y * y

fig, ax1 = plt.subplots(figsize=(8.6, 7.0))
fig.patch.set_facecolor("#0f172a")

# 等高线
gx = np.linspace(-0.6, 1.3, 400)
gy = np.linspace(-0.6, 1.3, 400)
X, Y = np.meshgrid(gx, gy)
Z = quad_loss(X, Y)
levels = np.logspace(-2, 0.4, 10)
cs = ax1.contour(X, Y, Z, levels=levels, colors="#475569", linewidths=0.9)
ax1.clabel(cs, inline=True, fontsize=6, fmt="%.2f", colors="#94a3b8")

# 4 条路径：SGD 0.01（爬）、SGD 0.02（震荡卡死）、Momentum 0.01（冲）、Adam 0.1（直线进）
paths = [
    (d1["sgd"],  "#f87171", "SGD lr=0.01（爬）", 25),
    (d1["osc"],  "#fb923c", "SGD lr=0.02（震荡卡死）", 25),
    (d1["mom"],  "#4ade80", "Momentum lr=0.01（冲）", 25),
    (d1["adam"], "#22d3ee", "Adam lr=0.1（直线进场）", 25),
]
for p, color, label, step in paths:
    ax1.plot(p[::step, 0], p[::step, 1], color=color, linewidth=1.6, label=label)
    ax1.plot(p[::step, 0], p[::step, 1], ".", color=color, markersize=3.5)

ax1.plot(*d1["start"], marker="*", markersize=14, color="#fbbf24", label="起点")
ax1.plot(0, 0, marker="o", markersize=7, color="white", label="谷底 (0,0)")
ax1.set_xlim(-0.6, 1.3); ax1.set_ylim(-0.6, 1.3)
ax1.set_aspect("equal")
ax1.set_xlabel("参数 x"); ax1.set_ylabel("参数 y")
ax1.set_title("狭长山谷：曲率差 100 倍，同一个学习率命运天差地别", fontsize=11, color="#e2e8f0")
ax1.legend(loc="upper right", fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
ax1.tick_params(colors="#94a3b8")
for spine in ax1.spines.values():
    spine.set_color("#334155")

fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "09-optimizer-paths.png"), dpi=150, facecolor="#0f172a")
print("saved 09-optimizer-paths.png")
plt.close(fig)

# ============ 图 2：sin 回归 loss 曲线 ============
fig, ax2 = plt.subplots(figsize=(11, 5.4))
fig.patch.set_facecolor("#0f172a")

steps = np.arange(1, len(d2["sgd"]) + 1)
curves = [
    (d2["sgd"],      "#f87171", "SGD lr=0.05 → 0.000088"),
    (d2["momentum"], "#4ade80", "Momentum lr=0.05 → 0.000006"),
    (d2["adam"],     "#60a5fa", "Adam lr=0.001 → 0.000100"),
    (d2["adam2"],    "#22d3ee", "Adam lr=0.01 → 0.000002"),
]
for hist, color, label in curves:
    ax2.plot(steps, hist, color=color, linewidth=1.6, label=label)

ax2.set_yscale("log")
ax2.set_xlim(0, 2000); ax2.set_ylim(1e-6, 10)
ax2.set_xlabel("训练步数"); ax2.set_ylabel("loss（对数轴）")
ax2.set_title("sin 曲线回归：2000 步，调好的 SGD 输给随手选的 Adam", fontsize=11, color="#e2e8f0")
ax2.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
ax2.tick_params(colors="#94a3b8")
for spine in ax2.spines.values():
    spine.set_color("#334155")

fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "09-sin-loss-curves.png"), dpi=150, facecolor="#0f172a")
print("saved 09-sin-loss-curves.png")
plt.close(fig)
