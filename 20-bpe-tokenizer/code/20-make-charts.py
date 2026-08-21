#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 20 课画图：BPE 词表大小 vs 压缩比（真实数据）
运行：
    ~/projects/main-agent/nanoGPT/.venv/bin/python 20-make-charts.py
"""
import os
import json
import sys
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# ---- 中文显示：显式候选路径注册（macOS 26 姿势，第 19 课验证）----
CAND_FONTS = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]
for f in CAND_FONTS:
    if os.path.exists(f):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "STHeiti", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    # 数据由 20-bpe-tokenizer.py 生成；缺失则自动跑主实验
    data_path = os.path.join(BASE, "chart-data.json")
    if not os.path.exists(data_path):
        subprocess.run([sys.executable, os.path.join(BASE, "20-bpe-tokenizer.py")], check=True)
    chart_data = json.load(open(data_path))

    merges = [d["merges"] for d in chart_data]
    ratios = [d["ratio"] for d in chart_data]
    vocabs = [d["vocab"] for d in chart_data]

    # 图 1：压缩比 vs 合并次数（真实数据）
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor("#14161a")
    ax.set_facecolor("#14161a")

    ax.plot(merges, ratios, marker="o", markersize=6, linewidth=2.2,
            color="#55C9EA", label="手写 BPE（5 万字符莎士比亚）")
    # tiktoken r50k 参考线
    ax.axhline(3.30, color="#FA5151", linestyle="--", linewidth=1.8,
               label="tiktoken r50k（GPT-2 官方词表 50,257）")
    ax.axhline(1.0, color="#888888", linestyle=":", linewidth=1.2, label="字符级（无合并）")

    ax.set_xlabel("合并次数（词表大小 = 初始字符 58 + 合并次数）", color="#dddddd", fontsize=12)
    ax.set_ylabel("压缩比（字符数 / 词块数）", color="#dddddd", fontsize=12)
    ax.set_title("BPE 词表越大，压缩比越高——但收益递减", color="#ffffff", fontsize=14, pad=14)
    ax.tick_params(colors="#bbbbbb")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    # 标注关键点
    ax.annotate("合并 300 次\n词表 358，2.27x", xy=(300, 2.27), xytext=(650, 1.85),
                color="#55C9EA", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#55C9EA"))
    ax.annotate("合并 3000 次\n词表 3059，4.44x", xy=(3000, 4.44), xytext=(1700, 4.05),
                color="#55C9EA", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#55C9EA"))
    ax.legend(facecolor="#1d2026", edgecolor="#444444", labelcolor="#dddddd", fontsize=10)

    fig.tight_layout()
    out1 = os.path.join(IMG_DIR, "20-compression-ratio.png")
    fig.savefig(out1, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("图 1 已保存:", out1)

    # 图 2：手写 vs 官方对拍示例（同一批单词的切分对比）
    words = [
        ("the", "['the']", "['the']"),
        ("love", "['l','ove']", "['love']"),
        ("Shakespeare", "['S','ha','k','es','pe','ar','e']", "['Sh','akespeare']"),
        ("unhappiness", "['un','ha','p','p','in','es','s']", "['un','h','appiness']"),
        ("lowest", "['l','ow','es','t']", "['low','est']"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
    fig.patch.set_facecolor("#14161a")
    ax.set_facecolor("#14161a")
    ax.axis("off")
    ax.set_title("同一个词，词表大小决定切法：手写 358 词块 vs GPT-2 官方 50,257 词块",
                 color="#ffffff", fontsize=13, pad=12)

    y = 1.0
    for word, mine, official in words:
        ax.text(0.01, y, word, color="#FECE00", fontsize=14, fontweight="bold",
                family="monospace", va="center")
        ax.text(0.28, y, mine, color="#55C9EA", fontsize=12,
                family="monospace", va="center")
        ax.text(0.62, y, official, color="#7ED6A0", fontsize=12,
                family="monospace", va="center")
        y -= 0.19
    ax.text(0.01, y - 0.05, "小词表：拆得碎（每个词块短）    大词表：常见词整块打包（一眼看全）",
            color="#aaaaaa", fontsize=11)
    ax.text(0.01, y - 0.22, "数据来源：Mac mini 实测，手写 BPE 与 tiktoken r50k_base 各自编码",
            color="#666666", fontsize=9)

    fig.tight_layout()
    out2 = os.path.join(IMG_DIR, "20-vocab-compare.png")
    fig.savefig(out2, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("图 2 已保存:", out2)


if __name__ == "__main__":
    main()
