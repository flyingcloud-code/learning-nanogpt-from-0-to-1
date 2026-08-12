"""
第 11 课画图：PCA 对比 + loss 曲线（真实实验数据）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 11-make-charts.py
依赖: torch + numpy + matplotlib（venv 已装）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# macOS 中文字体（DejaVu Sans 缺 CJK 字形，会导致中文乱码）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

d = np.load("exp11.npz")
proj_init, proj = d["proj_init"], d["proj"]
chars = [c for c in d["chars"]]
history = d["history"]

# 字符分类：大写 / 小写 / 空格换行 / 标点
upper = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
lower = set("abcdefghijklmnopqrstuvwxyz")
punct = set("!$&',-.0123456789:;?")  # 数字也归入"符号"类
cats = []
for c in chars:
    if c in upper:
        cats.append("upper")
    elif c in lower:
        cats.append("lower")
    elif c in (" ", "\n"):
        cats.append("space")
    else:
        cats.append("punct")

cat_color = {"upper": "#E4572E", "lower": "#2E86AB", "space": "#4CAF50", "punct": "#92617E"}
cat_label = {"upper": "大写字母", "lower": "小写字母", "space": "空格/换行", "punct": "标点/数字"}

# ---------- 图 1: 训练前 vs 训练后 PCA 对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, proj_data, title in [
    (axes[0], proj_init, "训练前（随机初始化）：一团乱麻"),
    (axes[1], proj, "训练后（bigram 1000 步）：结构浮现"),
]:
    for cat in ("upper", "lower", "space", "punct"):
        mask = np.array([c == cat for c in cats])
        ax.scatter(proj_data[mask, 0], proj_data[mask, 1],
                   s=40, c=cat_color[cat], label=cat_label[cat], alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=3)
    for i, c in enumerate(chars):
        ax.annotate(c, (proj_data[i, 0], proj_data[i, 1]),
                    fontsize=11, fontweight="bold",
                    xytext=(4, 3), textcoords="offset points", color="#333333")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
plt.suptitle("第 11 课：65 个字符的 embedding 用 PCA 压到 2 维（真实数据）",
             fontsize=14, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("images/11-pca-compare.png", dpi=160, bbox_inches="tight")
plt.close()
print("saved images/11-pca-compare.png")

# ---------- 图 2: 训练后 PCA（只标重点：元音 / 常见辅音 / 标点 / 空格）----------
fig, ax = plt.subplots(figsize=(9.5, 6.6))
vowels = set("aeiou")
common = set("rstlndcmshp")
for cat in ("upper", "lower", "space", "punct"):
    mask = np.array([c == cat for c in cats])
    ax.scatter(proj[mask, 0], proj[mask, 1], s=55,
               c=cat_color[cat], label=cat_label[cat], alpha=0.55,
               edgecolors="white", linewidths=0.5, zorder=2)
# 重点字符放大加深
for i, c in enumerate(chars):
    big = (c in vowels) or (c in common) or c in (" ", "\n") or c in "!?:;,."
    if big:
        ax.annotate(c, (proj[i, 0], proj[i, 1]),
                    fontsize=15, fontweight="bold",
                    xytext=(5, 4), textcoords="offset points",
                    color=cat_color[cats[i]], zorder=5)
# 圈出元音
va = np.array([proj[i] for i, c in enumerate(chars) if c in vowels])
cx, cy = va[:, 0].mean(), va[:, 1].mean()
r = max(np.linalg.norm(va - [cx, cy], axis=1).max() + 0.55, 1.2)
ax.add_patch(plt.Circle((cx, cy), r, fill=False, color="#E4572E",
                        linewidth=2.2, linestyle="--", alpha=0.8, zorder=4))
ax.text(cx, cy + r + 0.15, "元音 a e i o u 聚成一团", ha="center",
        fontsize=12, color="#E4572E", fontweight="bold")
# 圈出标点
vp = np.array([proj[i] for i, c in enumerate(chars) if c in "!?:;,."])
cx2, cy2 = vp[:, 0].mean(), vp[:, 1].mean()
r2 = max(np.linalg.norm(vp - [cx2, cy2], axis=1).max() + 0.55, 1.2)
ax.add_patch(plt.Circle((cx2, cy2), r2, fill=False, color="#92617E",
                        linewidth=2.2, linestyle="--", alpha=0.8, zorder=4))
ax.text(cx2, cy2 - r2 - 0.45, "标点 ！？，。；：聚团", ha="center",
        fontsize=12, color="#92617E", fontweight="bold")
ax.set_title("第 11 课：训练后的字符 embedding——元音与标点各占领地（真实数据）", fontsize=13)
ax.set_xlabel("PC1", fontsize=11)
ax.set_ylabel("PC2", fontsize=11)
ax.grid(True, alpha=0.25, linestyle="--")
ax.set_axisbelow(True)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig("images/11-pca-clusters.png", dpi=160, bbox_inches="tight")
plt.close()
print("saved images/11-pca-clusters.png")

# ---------- 图 3: loss 曲线 ----------
fig, ax = plt.subplots(figsize=(9, 5.2))
steps = history[:, 0]
losses = history[:, 1]
ax.plot(steps, losses, "-o", color="#2E86AB", linewidth=2.2, markersize=5,
        label="训练 loss（每 100 步记录）")
ax.axhline(d["val_final"], color="#E4572E", linestyle="--", linewidth=1.8,
           label=f"最终验证 loss = {d['val_final']:.4f}")
ax.set_title("第 11 课：只有 embedding 的语言模型，1000 步 loss 曲线（真实数据）", fontsize=13)
ax.set_xlabel("训练步数", fontsize=11)
ax.set_ylabel("交叉熵 loss", fontsize=11)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("images/11-loss-curve.png", dpi=160, bbox_inches="tight")
plt.close()
print("saved images/11-loss-curve.png")
