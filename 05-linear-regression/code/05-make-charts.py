# 05-make-charts.py
# 手搓大模型 05：画 loss 下降曲线 + 直线拟合过程（全部为真实训练数据）
# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 05-make-charts.py
# 依赖: matplotlib + numpy（venv 已装）
# 输出: images/05-loss-curve.png, images/05-fit-evolution.png
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# macOS 中文字体（直接按名字设，别用 findSystemFonts 探测）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 与主程序完全一致的训练（同种子，结果可复现）----
rng = np.random.default_rng(42)
N = 50
x = rng.uniform(-3, 3, N)
true_w, true_b = 2.0, 1.0
y = true_w * x + true_b + rng.normal(0, 0.5, N)

w, b = 0.0, 0.0
lr, steps = 0.05, 200
losses, ws, bs = [], [], []
for step in range(steps):
    y_hat = w * x + b
    diff = y_hat - y
    loss = np.mean(diff ** 2)
    losses.append(loss)
    ws.append(w)
    bs.append(b)
    w -= lr * np.mean(2 * diff * x)
    b -= lr * np.mean(2 * diff)
ws.append(w)
bs.append(b)

# ---- 图 1：loss 下降曲线（对数纵轴，陡降一目了然）----
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.plot(range(steps), losses, color="#22d3ee", linewidth=2.2)
ax.set_yscale("log")
ax.set_xlabel("训练步数 step", fontsize=11)
ax.set_ylabel("loss（对数轴）", fontsize=11)
ax.set_title("第一次训练：loss 从 13.17 一路降到 0.14", fontsize=13)

# 关键点标注（真实数值）
ax.annotate("起步 13.17\n（模型啥也不会）", xy=(0, losses[0]), xytext=(12, 9),
            arrowprops=dict(arrowstyle="->", color="#94a3b8"),
            color="#e2e8f0", fontsize=10)
ax.annotate("20 步 → 0.15\n（基本学会了）", xy=(20, losses[20]), xytext=(55, 0.9),
            arrowprops=dict(arrowstyle="->", color="#94a3b8"),
            color="#fbbf24", fontsize=10)
ax.annotate("200 步 → 0.14\n（贴住噪声地板）", xy=(199, losses[-1]), xytext=(120, 0.03),
            arrowprops=dict(arrowstyle="->", color="#94a3b8"),
            color="#e2e8f0", fontsize=10)

# 噪声下限参考线（噪声 std=0.5 → 方差 0.25）
ax.axhline(0.25, color="#f87171", linestyle="--", linewidth=1.2)
ax.text(205, 0.27, "噪声地板 ≈ 0.25", color="#f87171", fontsize=9)

ax.set_xlim(-5, 215)
ax.set_ylim(0.02, 30)
fig.tight_layout()
fig.savefig("images/05-loss-curve.png", dpi=150, facecolor="#0f172a")

# ---- 图 2：直线拟合过程（数据散点 + 不同步数的线）----
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.scatter(x, y, s=22, color="#64748b", alpha=0.85, label="数据点（50 个带噪样本）")

xs = np.linspace(-3.2, 3.2, 100)
snapshots = [("初始 w=0, b=0", 0, 0, "#64748b", "--"),
             ("4 步", ws[4], bs[4], "#fbbf24", "-."),
             ("20 步", ws[20], bs[20], "#a78bfa", "--"),
             ("200 步（最终）", ws[200], bs[200], "#22d3ee", "-")]
for label, sw, sb, color, ls in snapshots:
    ax.plot(xs, sw * xs + sb, color=color, linestyle=ls, linewidth=2, label=f"{label}  (w={sw:.2f}, b={sb:.2f})")
ax.plot(xs, true_w * xs + true_b, color="white", linestyle=":", linewidth=2, label="真相 y = 2x + 1")

ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("y", fontsize=11)
ax.set_title("梯度下降在'看'数据：一条直线越拧越准", fontsize=13)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig("images/05-fit-evolution.png", dpi=150, facecolor="#0f172a")

print("saved: images/05-loss-curve.png, images/05-fit-evolution.png")
