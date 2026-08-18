#!/usr/bin/env python3
"""
17-make-charts.py — 第 17 课《Transformer 块》图表生成
依赖: torch + numpy + matplotlib（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 17-make-charts.py

输出:
  images/17-pos-embedding.png   位置编码热力图（真实 wpe + 正弦对照）
  images/17-ablation.png        消融实验 loss 曲线（真实训练数据）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "images")
os.makedirs(IMG, exist_ok=True)

NANOGPT = os.path.expanduser("~/projects/main-agent/nanoGPT")
CKPT = os.path.join(NANOGPT, "out-shakespeare-char/ckpt.pt")


# ─────────────────────────────────────────────
# 图 1：位置编码热力图（真实学习到的 wpe + 正弦公式对照）
# ─────────────────────────────────────────────
def make_pos_embedding_chart():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    wpe = ckpt["model"]["transformer.wpe.weight"].numpy()  # (256, 384)
    print("wpe shape:", wpe.shape)

    # 真实学习到的位置编码：取前 64 维（384 维画不下，截前 64 维足够看结构）
    T, C = 128, 64
    real = wpe[:T, :C]

    # 正弦位置编码（《Attention Is All You Need》公式）
    pe = np.zeros((T, C))
    for pos in range(T):
        for i in range(C):
            if i % 2 == 0:
                pe[pos, i] = np.sin(pos / 10000 ** (i / C))
            else:
                pe[pos, i] = np.cos(pos / 10000 ** ((i - 1) / C))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    vmax = max(np.abs(real).max(), np.abs(pe).max())
    for ax, data, title in [
        (axes[0], real, "真实学习到的位置编码 wpe（前 64 维）"),
        (axes[1], pe, "正弦位置编码（Attention Is All You Need）"),
    ]:
        im = ax.imshow(data.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_xlabel("位置（第几个 token）")
        ax.set_ylabel("embedding 维度")
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("每个位置都有自己独一无二的位置向量：位置 0 和位置 100 长得不一样", fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(IMG, "17-pos-embedding.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存:", out)

    # 顺便输出两个位置的余弦相似度，作为"位置向量各不相同"的数字证据
    sim = np.dot(wpe[10], wpe[200]) / (np.linalg.norm(wpe[10]) * np.linalg.norm(wpe[200]))
    sim_adj = np.dot(wpe[10], wpe[11]) / (np.linalg.norm(wpe[10]) * np.linalg.norm(wpe[11]))
    print(f"位置 10 与位置 200 的余弦相似度: {sim:.4f}")
    print(f"位置 10 与位置 11（相邻）的余弦相似度: {sim_adj:.4f}")


# ─────────────────────────────────────────────
# 图 2：消融实验 loss 曲线（真实数据）
# ─────────────────────────────────────────────
def make_ablation_chart():
    with open(os.path.join(BASE, "17-ablation-loss.json")) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {
        "full":   dict(label="完整块（LayerNorm + 残差）", color="#0F4C81", lw=2.5),
        "no_ln":  dict(label="拆掉 LayerNorm", color="#D97757", lw=2.0, ls="--"),
        "no_res": dict(label="拆掉残差连接", color="#A93226", lw=2.0, ls=":"),
    }
    for name, r in results.items():
        steps = [p["step"] for p in r["losses"]]
        losses = [p["train_loss"] for p in r["losses"]]
        ax.plot(steps, losses, **styles[name])

    ax.set_xlabel("训练步数")
    ax.set_ylabel("训练 loss（交叉熵，越小越好）")
    ax.set_title("3 层微型 GPT：拆掉一个组件，训练曲线变成什么样？（Mac mini 实测）", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    # 标注最终 val
    for name, r in results.items():
        ax.annotate(f"val {r['val_loss']:.2f}", xy=(r["losses"][-1]["step"], r["losses"][-1]["train_loss"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=9,
                    color=styles[name]["color"])
    fig.tight_layout()
    out = os.path.join(IMG, "17-ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存:", out)


if __name__ == "__main__":
    make_pos_embedding_chart()
    make_ablation_chart()
    print("全部图表完成 ✅")
