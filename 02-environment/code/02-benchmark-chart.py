# 手搓大模型 02：CPU vs MPS 基准对比图（真实数据，matplotlib）
# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 02-benchmark-chart.py
# 依赖: matplotlib（venv 已装）+ 02-mps-benchmark.py 的真实输出

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# macOS 中文字体
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 真实数据（来自 02-mps-benchmark.py 在本机运行输出）
sizes = ["1024", "2048", "4096", "8192"]
cpu_times = [0.002, 0.012, 0.114, 0.835]
mps_times = [0.002, 0.007, 0.040, 0.341]
speedup = [1.0, 1.7, 2.8, 2.5]

x = np.arange(len(sizes))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# 左图：耗时对比（对数轴）
bars1 = ax1.bar(x - width/2, cpu_times, width, label="CPU", color="#64748b")
bars2 = ax1.bar(x + width/2, mps_times, width, label="MPS (GPU)", color="#22d3ee")
ax1.set_yscale("log")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{s}x{s}" for s in sizes])
ax1.set_ylabel("耗时（秒，对数轴）")
ax1.set_title("2048x2048 矩阵乘法耗时对比\n（小矩阵 GPU 无优势，大矩阵才加速）", fontsize=10)
ax1.legend()
for b in list(bars1) + list(bars2):
    ax1.annotate(f"{b.get_height():.3f}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

# 右图：加速比
bars3 = ax2.bar(x, speedup, width=0.5, color="#fbbf24")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{s}x{s}" for s in sizes])
ax2.set_ylabel("加速比（CPU 耗时 / MPS 耗时）")
ax2.set_title("MPS 加速比（>1 表示 GPU 更快）", fontsize=10)
ax2.axhline(1.0, color="white", linestyle="--", linewidth=1)
ax2.set_ylim(0, 3.5)
for b in bars3:
    ax2.annotate(f"{b.get_height():.1f}x", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10, color="#fbbf24")

fig.suptitle("手搓大模型 02：MPS 加速实测（Mac mini · torch 2.12.1）", fontsize=12, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("images/02-mps-benchmark.png", dpi=150, facecolor="#0f172a")
print("saved images/02-mps-benchmark.png")
