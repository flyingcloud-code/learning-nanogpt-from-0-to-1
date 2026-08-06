# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 04-make-charts.py
# 依赖: matplotlib + numpy（venv 已装）
# 生成第 4 课配图：左=惊讶度曲线（-log p），右=训练前后 loss（交叉熵）对比
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# macOS 中文字体（直接按名字设，别探测系统字体——会撞损坏字体崩掉）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# ===== 左图：惊讶度曲线 I(p) = -ln(p) = 平均惊讶度 ==========
# 数学函数：概率越小，看到它发生时越惊讶。用自然对数（与 loss 同单位 nait）。
p = np.linspace(0.01, 1.0, 400)
surprise = -np.log(p)
ax1.plot(p, surprise, color="#22d3ee", linewidth=2.5)
# 标注几个直观点
for pp, label in [(0.5, "p=0.5 → 惊讶 0.69"), (0.1, "p=0.1 → 惊讶 2.30"), (0.02, "p=0.02 → 惊讶 3.91")]:
    s = -np.log(pp)
    ax1.scatter([pp], [s], color="#fbbf24", s=40, zorder=5)
    ax1.annotate(label, xy=(pp, s), xytext=(pp + 0.06, s + 0.35),
                 fontsize=9, color="#fbbf24")
ax1.set_xlabel("概率 p（模型认为该字符出现的可能性）")
ax1.set_ylabel("惊讶度 = -ln(p)")
ax1.set_title("惊讶度曲线：概率越小，越惊讶", fontsize=11)
ax1.set_ylim(0, 5.2)
ax1.grid(alpha=0.25, color="#334155")

# ===== 右图：训练让 loss（交叉熵）下降 ==========
# 真实数据（Mac mini 实测 / nanoGPT 莎士比亚数据集，65 字符词表）：
#   瞎猜均匀分布熵 ln(65) = 4.174；1000 步后 train 1.28 / val 1.52
stages = ["瞎猜\n(均匀分布)", "训练后\ntrain", "训练后\nval"]
losses = [4.174, 1.28, 1.52]
colors = ["#64748b", "#22d3ee", "#fbbf24"]
bars = ax2.bar(stages, losses, color=colors, width=0.55)
for b, v in zip(bars, losses):
    ax2.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                 xytext=(0, 4), textcoords="offset points", ha="center", fontsize=12)
ax2.set_ylabel("loss（= 交叉熵，单位 nait）")
ax2.set_title("训练让平均惊讶度从 4.17 降到 1.28", fontsize=11)
ax2.set_ylim(0, 5.0)
ax2.grid(axis="y", alpha=0.25, color="#334155")

fig.suptitle("第 4 课：熵 = 平均惊讶度，loss = 交叉熵", fontsize=12, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("images/04-distribution.png", dpi=150, facecolor="#0f172a")
print("saved images/04-distribution.png")
