# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 07-make-charts.py
# 依赖: matplotlib + numpy（venv 已装）；先跑 07-backprop.py 生成 _07_data.npz
# 生成第 7 课配图：
#   左图：手写反向传播 vs autograd 训练的 loss 曲线（逐点重合）
#   右图：梯度检查散点——13 个参数的手写梯度 vs 数值梯度（全落在 y=x 上）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# 输出目录：脚本在 code/ 下时 → 上级 images/；脚本在课程根目录时 → 同级 images/
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "code" else os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# macOS 中文字体
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

d = np.load("_07_data.npz")
losses_manual = d["losses_manual"]
losses_auto = d["losses_auto"]
grad_manual = d["grad_manual"]
grad_numeric = d["grad_numeric"]

# ===== 左图：loss 曲线（真实训练数据） =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

steps = np.arange(len(losses_manual))
ax1.plot(steps, losses_manual, color="#22d3ee", linewidth=2.4, label="手写反向传播（梯度全靠手推公式）")
ax1.plot(steps, losses_auto, color="#fb7185", linewidth=1.8, linestyle="--", label="autograd（PyTorch 自动求导）")
ax1.set_xlabel("训练步数")
ax1.set_ylabel("loss（二分类交叉熵）")
ax1.set_title("两条线逐点重合：手写 BP 和 autograd 训练出完全一样的模型", fontsize=11)
ax1.legend(fontsize=9)
ax1.set_ylim(0, 0.85)
ax1.grid(alpha=0.25, color="#334155")
ax1.annotate("3000 步后\nloss = 0.001762\n准确率 100%", xy=(3000, 0.001762), xytext=(1650, 0.30),
             fontsize=10, color="white",
             arrowprops=dict(arrowstyle="->", color="#94a3b8"))

# ===== 右图：梯度检查散点（13 个参数） =====
ax2.scatter(grad_numeric, grad_manual, s=55, c="#34d399", edgecolors="white", linewidths=0.8, zorder=3, label="13 个参数：手写梯度 vs 数值梯度")
lim = max(np.abs(grad_manual).max(), np.abs(grad_numeric).max()) * 1.15
ax2.plot([-lim, lim], [-lim, lim], color="#fbbf24", linestyle="--", linewidth=1.6, label="y = x（完全一致线）")
ax2.set_xlabel("数值梯度（有限差分，float64）")
ax2.set_ylabel("手写梯度（链式法则公式）")
ax2.set_title("梯度检查：每个点都落在 y=x 上，最大误差 7.2e-09", fontsize=11)
ax2.legend(fontsize=9)
ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.grid(alpha=0.25, color="#334155")

fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "07-loss-gradient.png"), dpi=150, facecolor="#0f172a")
print(f"saved {os.path.join(IMG_DIR, '07-loss-gradient.png')}")
