# 03-make-charts.py — 第 3 课配图生成（真实数据）
# 运行：~/projects/main-agent/nanoGPT/.venv/bin/python 03-make-charts.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ============ 图 1：矩阵乘法 = 批量映射（格子手算图） ============
A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
x = np.array([1.0, 0.0, -1.0])
b = A @ x

fig, ax = plt.subplots(figsize=(9, 5.2))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")

# 画 A 的格子（2 行 3 列）
a_x0, a_y0, cell = 0.5, 2.0, 1.0
for i in range(2):
    for j in range(3):
        rect = plt.Rectangle((a_x0 + j * cell, a_y0 - (i + 1) * cell), cell, cell,
                             fill=(i + j) % 2 == 0, facecolor="#dbeafe", edgecolor="#1e3a8a", lw=1.5)
        ax.add_patch(rect)
        ax.text(a_x0 + j * cell + cell / 2, a_y0 - i * cell - cell / 2,
                f"{A[i, j]:.0f}", ha="center", va="center", fontsize=13, color="#0f172a")
ax.text(a_x0 + 1.5 * cell, a_y0 - 2 * cell - 0.35, "A  (2×3)  权重矩阵",
        ha="center", fontsize=11, color="#1e3a8a")

# 画 x 的格子（3 行 1 列）
x_x0 = a_x0 + 4 * cell
for i in range(3):
    rect = plt.Rectangle((x_x0, a_y0 - (i + 1) * cell), cell, cell,
                         facecolor="#fef3c7", edgecolor="#92400e", lw=1.5)
    ax.add_patch(rect)
    ax.text(x_x0 + cell / 2, a_y0 - i * cell - cell / 2,
            f"{x[i]:.0f}", ha="center", va="center", fontsize=13, color="#0f172a")
ax.text(x_x0 + cell / 2, a_y0 - 2 * cell - 0.35, "x  (3)  输入",
        ha="center", fontsize=11, color="#92400e")

# 乘法号
ax.text(x_x0 - 1.0, a_y0 - cell, "×", ha="center", va="center", fontsize=22, color="#334155")

# 等号
ax.text(x_x0 + 2.0, a_y0 - cell, "=", ha="center", va="center", fontsize=22, color="#334155")

# 画 b 的格子（2 行 1 列）
b_x0 = x_x0 + 3.2 * cell
for i in range(2):
    rect = plt.Rectangle((b_x0, a_y0 - (i + 1) * cell), cell, cell,
                         facecolor="#dcfce7", edgecolor="#166534", lw=1.5)
    ax.add_patch(rect)
    ax.text(b_x0 + cell / 2, a_y0 - i * cell - cell / 2,
            f"{b[i]:.0f}", ha="center", va="center", fontsize=13, color="#0f172a")
ax.text(b_x0 + cell / 2, a_y0 - 2 * cell - 0.35, "b  (2)  输出",
        ha="center", fontsize=11, color="#166534")

# 手算标注
ax.text(a_x0, -0.35, "b₀ = 1×1 + 2×0 + 3×(-1) = -2", fontsize=11, color="#166534")
ax.text(a_x0, -0.85, "b₁ = 4×1 + 5×0 + 6×(-1) = -2", fontsize=11, color="#166534")

# 批量映射标注
ax.annotate("4 个输入一起算：", xy=(b_x0 + 0.5, a_y0 + 0.55), fontsize=11, color="#334155")
ax.text(a_x0, a_y0 + 1.45, "同一个 A，一次映射一整批 → 批量映射（GPT 每秒做几万亿次）",
        fontsize=12, color="#b91c1c", fontweight="bold")

ax.set_xlim(-0.5, b_x0 + 2.0)
ax.set_ylim(-1.5, 4.3)
ax.axis("off")
ax.set_title("矩阵乘法 = 批量映射", fontsize=15, fontweight="bold", color="#0f172a", pad=14)
plt.tight_layout()
plt.savefig("../images/03-matrix-mult.png", dpi=150, bbox_inches="tight")
plt.close()

# ============ 图 2：导数 = 斜率（y=x² 与切线） ============
xs = np.linspace(-1.0, 3.0, 300)
ys = xs ** 2

x0, y0 = 1.5, 1.5 ** 2
k = 2 * x0          # 解析导数 = 2x
x_tan = np.linspace(0.0, 3.0, 50)
y_tan = k * (x_tan - x0) + y0

fig, ax = plt.subplots(figsize=(9, 5.2))
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")
ax.plot(xs, ys, color="#2563eb", lw=2.5, label="f(x) = x²")
ax.plot(x_tan, y_tan, color="#dc2626", lw=2, ls="--", label=f"切线：斜率 = f'(1.5) = {k}")
ax.plot([x0], [y0], "o", color="#dc2626", ms=8, zorder=5)
ax.annotate(f"f'({x0}) = 2×{x0} = {k}", xy=(x0, y0), xytext=(x0 + 0.25, y0 - 1.8),
            fontsize=12, color="#dc2626",
            arrowprops=dict(arrowstyle="->", color="#dc2626"))
ax.axhline(0, color="#64748b", lw=0.8)
ax.axvline(0, color="#64748b", lw=0.8)
ax.set_xlim(-1.0, 3.2)
ax.set_ylim(-2.5, 9.5)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("f(x)", fontsize=12)
ax.legend(fontsize=11, loc="upper left")
ax.set_title("导数 = 那一点的斜率（切线有多陡）", fontsize=15, fontweight="bold", color="#0f172a", pad=14)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("../images/03-derivative.png", dpi=150, bbox_inches="tight")
plt.close()

print("charts done")
