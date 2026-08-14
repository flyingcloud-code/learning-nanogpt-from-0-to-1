#!/usr/bin/env python3
"""13-make-charts.py — 第 13 课图表（真实数据）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 13-make-charts.py
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 脚本所在目录：从任何 cwd 运行都能找到数据/输出
BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# 中文字体
cand = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]
chosen = None
for f in cand:
    if os.path.exists(f):
        chosen = f
        break
if chosen:
    fm.fontManager.addfont(chosen)
    name = fm.FontProperties(fname=chosen).get_name()
    plt.rcParams["font.family"] = name
plt.rcParams["axes.unicode_minus"] = False

with open(os.path.join(BASE, "chart-data.json")) as f:
    d = json.load(f)

# ── 图 1：字符 vs BPE token 数对比（压缩比） ──
fig, ax1 = plt.subplots(figsize=(9, 5.2), dpi=150)
lens = d["lens"]
char_counts = d["char"]
bpe_counts = d["bpe"]

x = range(len(lens))
w = 0.38
b1 = ax1.bar([i - w/2 for i in x], char_counts, width=w, color="#64748b", label="字符级 token 数")
b2 = ax1.bar([i + w/2 for i in x], bpe_counts, width=w, color="#22d3ee", label="BPE 词块数")
for i in x:
    ax1.text(i - w/2, char_counts[i] + 500, f"{char_counts[i]:,}", ha="center", va="bottom", fontsize=9, color="#94a3b8")
    ax1.text(i + w/2, bpe_counts[i] + 500, f"{bpe_counts[i]:,}", ha="center", va="bottom", fontsize=9, color="#22d3ee")
    ratio = char_counts[i] / bpe_counts[i]
    ax1.text(i, max(char_counts[i], bpe_counts[i]) + 2200, f"{ratio:.2f}x", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#fbbf24")
ax1.set_xticks(list(x))
ax1.set_xticklabels([f"{n:,}" for n in lens])
ax1.set_xlabel("文本长度（字符）")
ax1.set_ylabel("token 数")
ax1.set_title("同一段莎士比亚文本：字符级 vs BPE 词块级 token 数（GPT-2 r50k 词表，真实数据）")
ax1.legend(loc="upper left")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_facecolor("#f8fafc")
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "13-token-compare.png"), facecolor="white")
plt.close(fig)
print("saved images/13-token-compare.png")

# ── 图 2：BPE 合并频次条形图（莎士比亚前 12 次合并真实数据） ──
merges = [
    ("'e'+' '", 1347), ("'t'+'h'", 1064), ("'s'+' '", 734), ("'t'+' '", 716),
    ("'o'+'u'", 645), ("','+' '", 576), ("'d'+' '", 572), ("'e'+'r'", 537),
    ("'i'+'n'", 449), ("'a'+'n'", 446), ("' '+'th'", 420), ("'e'+'n'", 379),
]
labels = [m[0] for m in merges]
counts = [m[1] for m in merges]

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
bars = ax.barh(range(len(labels))[::-1], counts, color="#0f766e")
for i, c in enumerate(counts):
    ax.text(c + 20, len(labels) - 1 - i, f"{c}", va="center", fontsize=9, color="#0f766e")
ax.set_yticks(range(len(labels))[::-1])
ax.set_yticklabels(labels)
ax.set_xlabel("出现频次（5 万字符内）")
ax.set_title("BPE 在莎士比亚文本上的前 12 次合并（真实统计：最高频相邻字符对）")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_facecolor("#f8fafc")
fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "13-merge-freq.png"), facecolor="white")
plt.close(fig)
print("saved images/13-merge-freq.png")
