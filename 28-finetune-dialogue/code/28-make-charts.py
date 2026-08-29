#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 28 课画图：
1) 28-loss-curve.png   —— 微调 400 步三条真实曲线：train（背对话）/ dialog_val（过拟合）/ shake_val（灾难性遗忘）
2) 28-before-after.png —— 微调前 vs 微调后，同一句提问的真实生成对比
数据全部来自 28-finetune.py 的真实输出（finetune-history.json + sample-*.txt），不造假。
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT_SERIES = "/Volumes/External-HD-data/leo-universe/learning/series/hand-made-gpt/images"
OUT_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
os.makedirs(OUT_SERIES, exist_ok=True)
os.makedirs(OUT_REPO, exist_ok=True)

hist = json.load(open("finetune-history.json"))
steps = [h[0] for h in hist]
train = [h[1] for h in hist]
ev = [h for h in hist if h[2] is not None]
ev_steps = [h[0] for h in ev]
dv = [h[2] for h in ev]
sv = [h[3] for h in ev]

print("train 首末:", train[0], train[-1])
print("dialog_val:", dv)
print("shake_val:", sv)

# ============================================================
# 图 1：微调 400 步的三条真实曲线
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6.4), dpi=150)

ax.axvspan(0, 50, color="#C44E52", alpha=0.10)
ax.axvspan(50, 400, color="#4C72B0", alpha=0.07)
ax.text(25, 4.35, "前 50 步\n格式塌方", ha="center", fontsize=10, color="#C44E52", fontweight="bold")
ax.text(225, 4.35, "死记硬背 + 遗忘加深", ha="center", fontsize=10, color="#4C72B0", fontweight="bold")

ax.plot(steps, train, color="#4C72B0", linewidth=1.5, label="train loss（对话训练集，每步记录）")
ax.plot(ev_steps, dv, "D", color="#DD8452", markersize=8, label="dialog val loss（10 组没见过的问答）")
ax.plot(ev_steps, sv, "s", color="#C44E52", markersize=8, label="shake val loss（莎士比亚验证集）")

ax.axhline(4.174, color="gray", linestyle=":", linewidth=1.2)
ax.text(318, 4.22, "ln(65) ≈ 4.17 = 纯瞎猜", fontsize=9, color="gray")
ax.axhline(1.5218, color="#2C7FB8", linestyle="--", linewidth=1.2)
ax.text(318, 1.56, "微调前莎士比亚 val ≈ 1.52", fontsize=9, color="#2C7FB8")

# 关键点标注
ax.annotate(f"train {train[0]:.2f}", (steps[0], train[0]), textcoords="offset points",
            xytext=(8, 10), fontsize=9, color="#4C72B0")
ax.annotate(f"train {train[-1]:.3f}", (400, train[-1]), textcoords="offset points",
            xytext=(-70, 10), fontsize=9, color="#4C72B0")
ax.annotate(f"dialog_val {dv[-1]:.2f}", (400, dv[-1]), textcoords="offset points",
            xytext=(-95, -16), fontsize=9, color="#DD8452")
ax.annotate(f"shake_val {sv[-1]:.2f}\n（比瞎猜还差）", (400, sv[-1]), textcoords="offset points",
            xytext=(-95, 8), fontsize=9, color="#C44E52")

ax.set_xlabel("微调步数", fontsize=12)
ax.set_ylabel("loss（交叉熵，nats）", fontsize=12)
ax.set_title("第 28 课：微调 400 步的三条真实曲线——对话在背、验证在涨、莎士比亚在忘\n（基础模型：26 课莎士比亚字符模型；80 组海盗问答；Mac mini MPS 实测，约 4 分钟）",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, 420)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
p1 = os.path.join(OUT_SERIES, "28-loss-curve.png")
plt.savefig(p1, bbox_inches="tight")
plt.savefig(os.path.join(OUT_REPO, "28-loss-curve.png"), bbox_inches="tight")
plt.close()
print("saved", p1)

# ============================================================
# 图 2：微调前 vs 微调后，同一句提问
# ============================================================
before = open("sample-before-p0.txt", encoding="utf-8").read()
after = open("sample-after-p0.txt", encoding="utf-8").read()

def wrap(text, width=52):
    lines = []
    for raw in text.splitlines():
        while len(raw) > width:
            lines.append(raw[:width])
            raw = raw[width:]
        lines.append(raw)
    return "\n".join(lines)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), dpi=150)
fig.suptitle("第 28 课：同一句提问，微调前 vs 微调后（真实生成，T=0.8）", fontsize=14, fontweight="bold")

axes[0].set_facecolor("#2B2B2B")
axes[0].set_title("微调前：莎士比亚模型", fontsize=13, color="white", fontweight="bold", pad=12)
axes[0].text(0.5, 0.94, wrap(before, 42), transform=axes[0].transAxes, ha="center", va="top",
             fontfamily="monospace", fontsize=11, color="#FFD700", linespacing=1.6)
axes[0].text(0.5, 0.06, "只会接莎士比亚的茬：'Come, by the airs of yours; but we know...'",
             transform=axes[0].transAxes, ha="center", va="bottom", fontsize=10, color="#CCCCCC")
axes[0].set_xticks([]); axes[0].set_yticks([])
for s in axes[0].spines.values():
    s.set_color("#555555")

axes[1].set_facecolor("#0F2B3A")
axes[1].set_title("微调后：海盗管家", fontsize=13, color="white", fontweight="bold", pad=12)
axes[1].text(0.5, 0.94, wrap(after, 42), transform=axes[1].transAxes, ha="center", va="top",
             fontfamily="monospace", fontsize=11, color="#FFD700", linespacing=1.6)
axes[1].text(0.5, 0.06, "用海盗腔回答，还自己接上了下一轮对话",
             transform=axes[1].transAxes, ha="center", va="bottom", fontsize=10, color="#CCCCCC")
axes[1].set_xticks([]); axes[1].set_yticks([])
for s in axes[1].spines.values():
    s.set_color("#555555")

plt.tight_layout(rect=[0, 0, 1, 0.95])
p2 = os.path.join(OUT_SERIES, "28-before-after.png")
plt.savefig(p2, bbox_inches="tight")
plt.savefig(os.path.join(OUT_REPO, "28-before-after.png"), bbox_inches="tight")
plt.close()
print("saved", p2)
