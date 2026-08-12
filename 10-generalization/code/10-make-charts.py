# -*- coding: utf-8 -*-
"""
第 10 课画图脚本：从 exp10.npz / exp10b.npz 生成三张真实曲线图
运行：~/projects/main-agent/nanoGPT/.venv/bin/python 10-make-charts.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体（macOS PingFang）
for f in ["PingFang HK", "PingFang SC", "Songti SC", "Heiti TC"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f, "Arial Unicode MS", "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

D = np.load("exp10.npz")
B = np.load("exp10b.npz")
steps = D["steps"]

# 颜色与标签
C = {
    "baseline":           ("#e11d48", "baseline 无正则"),
    "l2_1e-3":            ("#10b981", "L2 正则 λ=1e-3"),
    "dropout0.4":         ("#f59e0b", "dropout 0.4"),
    "baseline+earlystop": ("#3b82f6", "baseline + 早停"),
}

# ---------- 图 1：baseline 的过拟合曲线 ----------
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
ax.plot(steps, D["baseline_hist_train"], color="#10b981", lw=2, label="训练集 loss（背答案）")
ax.plot(steps, D["baseline_hist_val"], color="#e11d48", lw=2, label="验证集 loss（考新题）")
ax.axvline(500, color="#64748b", ls="--", lw=1)
ax.annotate("第 500 步：验证集最低点 0.3674\n再往下训练就「考砸」了", xy=(500, 0.37), xytext=(1800, 0.62),
            fontsize=10, color="#e11d48",
            arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1.2))
ax.annotate("训练 loss 一路降到 0.00001\n= 12 个点背得一字不差", xy=(9500, 0.05), xytext=(4500, 0.55),
            fontsize=10, color="#10b981",
            arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.2))
ax.set_xlabel("训练步数", fontsize=11)
ax.set_ylabel("loss（MSE）", fontsize=11)
ax.set_yscale("log")
ax.set_title("过拟合曲线：训练 loss 一直降，验证 loss 先降后升（12 个训练点，噪声 0.3）", fontsize=12)
ax.legend(loc="upper right", fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("images/10-overfit-curves.png")
plt.close(fig)

# ---------- 图 2：四种配置的验证集曲线 ----------
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
for name, (col, lab) in C.items():
    ax.plot(steps, D[f"{name}_hist_val"], color=col, lw=1.8, label=lab)
ax.set_xlabel("训练步数", fontsize=11)
ax.set_ylabel("验证集 loss", fontsize=11)
ax.set_title("同一起点、同一个模型，只换正则化手段（验证集曲线）", fontsize=12)
ax.legend(loc="upper right", fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("images/10-regularization-compare.png")
plt.close(fig)

# ---------- 图 3：拟合曲线（过拟合乱抖 vs L2 平滑） ----------
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
ax.scatter(D["x_train"], D["y_train"], s=45, color="#0f172a", edgecolor="#22d3ee",
           linewidth=1.2, zorder=5, label="训练点（12 个，带噪声）")
ax.plot(D["x_val"], D["y_val"], color="#94a3b8", ls="--", lw=1.6,
        label="真实函数 sin(2πx)")
ax.plot(D["x_val"], D["baseline_pred"], color="#e11d48", lw=1.8,
        label="baseline 预测：背下噪声，点之间乱抖")
ax.plot(D["x_val"], D["l2_1e-3_pred"], color="#10b981", lw=1.8,
        label="L2 预测：压住权重，贴近真实函数")
ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("y", fontsize=11)
ax.set_title("同一个模型，两种结局：背答案 vs 学规律", fontsize=12)
ax.legend(loc="lower center", fontsize=9, ncol=2)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("images/10-fit-comparison.png")
plt.close(fig)

# ---------- 图 4（补充）：数据量的力量 ----------
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
ax.plot(steps, B["n12_hist"], color="#e11d48", lw=1.8, label="12 个训练点 → 验证 loss 回升到 0.48")
ax.plot(steps, B["n200_hist"], color="#10b981", lw=1.8, label="200 个训练点 → 验证 loss 低到 0.03")
ax.set_xlabel("训练步数", fontsize=11)
ax.set_ylabel("验证集 loss", fontsize=11)
ax.set_yscale("log")
ax.set_title("更多数据 = 终极正则化（同样模型，什么都没改）", fontsize=12)
ax.legend(loc="upper right", fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("images/10-data-scale.png")
plt.close(fig)

print("4 张图已生成：10-overfit-curves / 10-regularization-compare / 10-fit-comparison / 10-data-scale")
