#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 21 课画图：loss 对比曲线 + 注意力热力图（真实数据）
运行：
    ~/projects/main-agent/nanoGPT/.venv/bin/python 21-make-charts.py
"""
import os
import json

import numpy as np
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


def make_loss_compare():
    data = json.load(open(os.path.join(BASE, "chart-loss.json")))
    steps = data["steps"]
    losses_mine = data["losses_mine"]
    losses_mha = data["losses_mha"]
    xs = list(range(1, steps + 1))

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor("#14161a")
    ax.set_facecolor("#14161a")

    ax.plot(xs, losses_mine, linewidth=2.2, color="#55C9EA", label="手写 attention 版（本次实现）")
    ax.plot(xs, losses_mha, linewidth=2.2, color="#FA5151", linestyle="--", label="官方 nn.MultiheadAttention 版")

    ax.set_xlabel("训练步数", color="#dddddd", fontsize=12)
    ax.set_ylabel("交叉熵 loss（越低越好）", color="#dddddd", fontsize=12)
    ax.set_title("同一个 GPT：手写 attention vs 官方 API，400 步 loss 几乎重合", color="#ffffff", fontsize=14, pad=14)
    ax.tick_params(colors="#bbbbbb")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    ax.annotate(f"手写版 {losses_mine[-1]:.3f}", xy=(steps, losses_mine[-1]), xytext=(260, 3.35),
                color="#55C9EA", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#55C9EA"))
    ax.annotate(f"官方版 {losses_mha[-1]:.3f}", xy=(steps, losses_mha[-1]), xytext=(300, 2.70),
                color="#FA5151", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#FA5151"))
    ax.legend(facecolor="#1d2026", edgecolor="#444444", labelcolor="#dddddd", fontsize=10)

    fig.tight_layout()
    out = os.path.join(IMG_DIR, "21-loss-compare.png")
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("已保存", out)


def make_heatmap():
    att0 = np.load(os.path.join(BASE, "chart-att.npy"))  # (40, 40) 训练后的真实注意力
    fig, ax = plt.subplots(figsize=(8, 6.8), dpi=150)
    fig.patch.set_facecolor("#14161a")
    ax.set_facecolor("#14161a")

    im = ax.imshow(att0, cmap="viridis", vmin=0, vmax=att0.max())
    ax.set_xlabel("被关注的 token 位置 j", color="#dddddd", fontsize=12)
    ax.set_ylabel("正在做决定的 token 位置 i", color="#dddddd", fontsize=12)
    ax.set_title("训练 400 步后，第 0 层第 0 个头的注意力热力图（真实）", color="#ffffff", fontsize=13, pad=12)
    ax.tick_params(colors="#bbbbbb")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#bbbbbb")
    cbar.outline.set_edgecolor("#444444")
    cbar.set_label("注意力权重", color="#dddddd", fontsize=11)

    # 标注因果掩码：右上角全 0
    ax.annotate("右上角全 0：\n看不见未来（因果掩码）", xy=(33, 6), xytext=(12, 30),
                color="#ffffff", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#ffffff"))
    ax.axhline(y=34.5, xmin=0.5, color="#ffffff", linewidth=0.8, alpha=0.35, linestyle=":")

    fig.tight_layout()
    out = os.path.join(IMG_DIR, "21-attn-heatmap.png")
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("已保存", out)


if __name__ == "__main__":
    make_loss_compare()
    make_heatmap()
    print("图表完成 ✅")
