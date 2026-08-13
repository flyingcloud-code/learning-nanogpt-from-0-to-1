"""
第 12 课画图：训练 loss 曲线 + 困惑度对比（真实实验数据）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 12-make-charts.py
依赖: torch + numpy + matplotlib（venv 已装）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

d = np.load("exp12.npz")
history = d["history"]

# ============ 图 1：训练 loss 曲线 ============
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
steps = np.arange(len(history))
ax.plot(steps, history, color="#0F4C81", lw=2, label="神经网络 MLP 训练 loss")
ax.axhline(np.log(65), color="#B91C1C", ls="--", lw=1.2, label="随机猜测基线 ln(65) = 4.17")
ax.axhline(2.4875, color="#D97757", ls=":", lw=1.2, label="bigram 计数模型 val loss = 2.49")
ax.set_xlabel("训练步数")
ax.set_ylabel("loss（越小越好）")
ax.set_title("第 12 课：神经网络语言模型 3000 步训练曲线（真实数据，Mac mini 5.7s）")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("12-loss-curve.png", dpi=150)
plt.close()

# ============ 图 2：困惑度对比 ============
names = ["随机猜测", "bigram\n计数模型", "trigram\n计数模型", "神经网络\n语言模型"]
ppls = [float(d["random_ppl"]), float(d["bigram_smooth_ppl"]), float(d["trigram_ppl"]), float(d["neural_ppl"])]
colors = ["#94A3B8", "#FBBF24", "#F59E0B", "#0F4C81"]

fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
bars = ax.bar(names, ppls, color=colors, width=0.6)
for bar, v in zip(bars, ppls):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("困惑度 perplexity（越小越好）")
ax.set_title("第 12 课：四个模型的困惑度对比——\"平均要猜几个字符\"（真实数据）")
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, 70)
plt.tight_layout()
plt.savefig("12-ppl-compare.png", dpi=150)
plt.close()

print("图表已保存: 12-loss-curve.png, 12-ppl-compare.png")
