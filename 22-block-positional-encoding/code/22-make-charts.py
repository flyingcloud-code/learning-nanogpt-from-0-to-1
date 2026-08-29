#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 22 课画图：训练 loss 曲线 + 位置编码外推对比（真实数据）
运行：
    ~/projects/main-agent/nanoGPT/.venv/bin/python 22-make-charts.py
前置：先跑 22-block.py 生成 chart-data.json
"""
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(os.path.dirname(BASE), "images")
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

BG = "#14161a"
FG = "#dddddd"


def make_loss_curve():
    data = json.load(open(os.path.join(BASE, "chart-data.json")))
    ll = data["loss_learned"]
    ls = data["loss_sin"]
    xs = [p["step"] for p in ll]
    yl = [p["loss"] for p in ll]
    ys = [p["loss"] for p in ls]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(xs, yl, linewidth=2.4, color="#55C9EA", label="学习式位置编码（查表）")
    ax.plot(xs, ys, linewidth=2.4, color="#FA5151", linestyle="--", label="正弦式位置编码（公式）")

    ax.set_xlabel("训练步数", color=FG, fontsize=12)
    ax.set_ylabel("交叉熵 loss（越低越好）", color=FG, fontsize=12)
    ax.set_title("同一个手搓 GPT（2 层 4 头 128 维）：两种位置编码 600 步训练", color="#ffffff", fontsize=14, pad=14)
    ax.tick_params(colors="#bbbbbb")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    ax.annotate(f"正弦式 {ys[-1]:.3f}", xy=(xs[-1], ys[-1]), xytext=(430, 3.05),
                color="#FA5151", fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#FA5151"))
    ax.annotate(f"学习式 {yl[-1]:.3f}", xy=(xs[-1], yl[-1]), xytext=(400, 2.62),
                color="#55C9EA", fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#55C9EA"))
    ax.legend(loc="upper right", facecolor="#1e2126", edgecolor="#444444",
              labelcolor=FG, fontsize=11)
    ax.set_xlim(0, 620)
    ax.set_ylim(2.0, 4.6)
    ax.grid(alpha=0.15, color="#888888")

    out = os.path.join(IMG_DIR, "22-loss-curve.png")
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", out)


def make_extrapolate():
    data = json.load(open(os.path.join(BASE, "chart-data.json")))
    el = data["extrapolate_learned"]
    es = data["extrapolate_sin"]
    # 学习式：None = 越界
    xs_l = []
    ys_l = []
    for k, v in el.items():
        if v is not None:
            xs_l.append(int(k))
            ys_l.append(v)
    xs_s = [int(k) for k in es]
    ys_s = [es[k] for k in es]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # 越界阴影区：x > 64
    ax.axvspan(64, 260, color="#FA5151", alpha=0.08)
    ax.text(150, 3.30, "查表越界\n（位置 64+ 无表项）", color="#FA5151", fontsize=11,
            ha="center", va="center")

    ax.plot(xs_s, ys_s, linewidth=2.4, color="#FA5151", marker="o", markersize=5,
            label="正弦式位置编码（公式：任意位置都能算）")
    ax.plot(xs_l, ys_l, linewidth=2.4, color="#55C9EA", marker="s", markersize=6,
            label="学习式位置编码（查表：到 64 戛然而止）")

    ax.set_xlabel("测试序列长度（训练时只见过 64）", color=FG, fontsize=12)
    ax.set_ylabel("验证集 loss（越低越好）", color=FG, fontsize=12)
    ax.set_title("位置编码外推实验：训练 block_size=64，测更长的序列", color="#ffffff", fontsize=14, pad=14)
    ax.tick_params(colors="#bbbbbb")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    ax.axvline(64, color="#888888", linestyle=":", linewidth=1.2)
    ax.text(66, 2.05, "训练长度 64", color="#888888", fontsize=10)
    ax.legend(loc="upper left", facecolor="#1e2126", edgecolor="#444444",
              labelcolor=FG, fontsize=10.5)
    ax.set_xlim(0, 272)
    ax.set_ylim(2.0, 3.5)
    ax.grid(alpha=0.15, color="#888888")

    out = os.path.join(IMG_DIR, "22-extrapolate.png")
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    make_loss_curve()
    make_extrapolate()
