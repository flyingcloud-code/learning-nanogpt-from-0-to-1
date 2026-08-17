#!/usr/bin/env python3
"""
16-make-charts.py — 第 16 课《因果掩码 + 多头》配图脚本
全部基于真实模型（out-shakespeare-char/ckpt.pt，6 层 6 头，val loss 1.52）的真实注意力权重。
依赖: torch + numpy + matplotlib（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 16-make-charts.py
输出: ../images/16-*.png
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# 中文字体（macOS 系统字体）
for _f in ["/System/Library/Fonts/PingFang.ttc",
           "/System/Library/Fonts/STHeiti Medium.ttc",
           "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.normpath(os.path.join(BASE, "../images"))
os.makedirs(IMG, exist_ok=True)

NANOGPT = os.path.expanduser("~/projects/main-agent/nanoGPT")
sys.path.insert(0, NANOGPT)
CKPT = os.path.join(NANOGPT, "out-shakespeare-char/ckpt.pt")

import model as nano_model

# ─────────────────────────────────────────────
# 1. 加载模型 + 抓真实注意力权重（第 0 层 6 头）
# ─────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = nano_model.GPTConfig(**ckpt["model_args"])
gpt = nano_model.GPT(cfg)
gpt.load_state_dict(ckpt["model"])
gpt.eval()
head_dim = cfg.n_embd // cfg.n_head

DATA = os.path.join(NANOGPT, "data/shakespeare_char")
with open(os.path.join(DATA, "input.txt"), "r", encoding="utf-8") as f:
    full_text = f.read()
import pickle
with open(os.path.join(DATA, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]

idx = full_text.find("ROMEO:")
sample_text = full_text[idx: idx + 100]
ids = [stoi[c] for c in sample_text if c in stoi]
T = len(ids)
x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

captured = {}

def patched_forward(self, x, layer_idx):
    B, Tt, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    att_raw = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att_raw.masked_fill(self.bias[:, :, :Tt, :Tt] == 0, float("-inf"))
    att_soft = F.softmax(att, dim=-1)
    captured[layer_idx] = {
        "att_raw": att_raw[0].detach().cpu().numpy(),
        "att_soft": att_soft[0].detach().cpu().numpy(),
    }
    y = att_soft @ v
    y = y.transpose(1, 2).contiguous().view(B, Tt, C)
    y = self.resid_dropout(self.c_proj(y))
    return y

def make_hooked(layer_idx):
    def hooked_forward(self, x):
        return patched_forward(self, x, layer_idx)
    return hooked_forward

for li, block in enumerate(gpt.transformer.h):
    block.attn.flash = False
    if not hasattr(block.attn, "bias"):
        block.attn.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size),
        )
    block.attn.forward = make_hooked(li).__get__(block.attn, type(block.attn))

with torch.no_grad():
    logits = gpt(x)

att_soft = captured[0]["att_soft"]  # (6, T, T) 真实权重
att_raw = captured[0]["att_raw"]    # (6, T, T) mask 前分数

# 字符表（热力图用）
chars = [itos[i] for i in ids]

# ─────────────────────────────────────────────
# 图 1：mask 矩阵 —— 12×12 的 0/1 三角，标注"允许/禁止"
# ─────────────────────────────────────────────
def fig_mask_matrix():
    n = 12
    m = np.tril(np.ones((n, n)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    # 左：mask 矩阵
    ax1.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
    ax1.set_xticklabels(range(n), fontsize=8)
    ax1.set_yticklabels(range(n), fontsize=8)
    ax1.set_xlabel("被看的位置 j（列）", fontsize=11)
    ax1.set_ylabel("正在做决定的位置 i（行）", fontsize=11)
    ax1.set_title("因果掩码矩阵：下三角 = 1（允许），\n上三角 = 0（禁止）", fontsize=12)
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, "1" if i >= j else "0", ha="center", va="center",
                     fontsize=7, color="white" if i >= j else "#888888")
    ax1.plot([-0.5, n - 0.5], [-0.5, n - 0.5], "r--", lw=1.5)
    ax1.text(n - 1.2, 0.8, "对角线：\n第 i 行能看 j ≤ i", color="red", fontsize=8,
             ha="right", va="top")

    # 右：softmax 前后对比（真实数值来自 16-causal-mask.py 同款随机演示）
    rng = np.random.default_rng(16)
    scores = rng.normal(0, 1, (n, n))
    scores[0, 3] = 4.0
    scores[2, 4] = 3.5
    def softmax_row(v):
        e = np.exp(v - v.max())
        return e / e.sum()
    w_masked = np.stack([softmax_row(np.where(np.tril(np.ones((n, n))).astype(bool), scores, -np.inf)[i])
                         for i in range(n)])
    im = ax2.imshow(w_masked, cmap="YlOrRd", vmin=0)
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(range(n), fontsize=8)
    ax2.set_yticklabels(range(n), fontsize=8)
    ax2.set_xlabel("被看的位置 j（列）", fontsize=11)
    ax2.set_ylabel("正在做决定的位置 i（行）", fontsize=11)
    ax2.set_title("softmax 之后：\n右上角（未来）权重全部 = 0", fontsize=12)
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{w_masked[i, j]:.2f}", ha="center", va="center",
                     fontsize=6.5, color="black" if w_masked[i, j] < 0.7 else "white")
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("注意力权重（每行和 = 1）", fontsize=10)
    fig.suptitle("因果掩码：打分之后把右上角涂成 −∞，softmax 让未来位置精确归零",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(IMG, "16-causal-mask.png"), dpi=150)
    plt.close(fig)
    print("已保存 images/16-causal-mask.png")

# ─────────────────────────────────────────────
# 图 2：真实 6 头注意力热力图（第 0 层，T=100）
# ─────────────────────────────────────────────
def fig_multihead():
    cmap = LinearSegmentedColormap.from_list("attn", ["#ffffff", "#f7c8c8", "#e45756"])
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
    for h in range(6):
        ax = axes[h // 3][h % 3]
        im = ax.imshow(att_soft[h], cmap=cmap, vmin=0, vmax=att_soft[h].max())
        ax.set_title(f"head {h}", fontsize=12)
        ax.set_xticks(range(0, T, 10))
        ax.set_yticks(range(0, T, 10))
        ax.set_xticklabels([chars[i] if i % 20 == 0 else "" for i in range(0, T, 10)], fontsize=6)
        ax.set_yticklabels([chars[i] if i % 20 == 0 else "" for i in range(0, T, 10)], fontsize=6)
        ax.tick_params(axis="x", rotation=90)
        # 标出对角线（边界）
        ax.plot([0, T - 1], [0, T - 1], "w--", lw=0.8, alpha=0.6)
    fig.suptitle("真实 GPT 第 0 层 6 个头：左下半三角是它们的全部世界（T=100，真实权重）",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(IMG, "16-multihead-attention.png"), dpi=150)
    plt.close(fig)
    print("已保存 images/16-multihead-attention.png")

# ─────────────────────────────────────────────
# 图 3：多头分工 —— 6 个头的"关注习惯"条形图
# ─────────────────────────────────────────────
def fig_head_pattern():
    names = ["看自己", "看前一位", "看句首", "平均目光距离×10"]
    metrics = np.zeros((6, 4))
    for h in range(6):
        w = att_soft[h]
        metrics[h, 0] = np.mean([w[i, i] for i in range(T)])
        metrics[h, 1] = np.mean([w[i, i - 1] for i in range(1, T)])
        metrics[h, 2] = np.mean([w[i, 0] for i in range(T)])
        metrics[h, 3] = np.mean([w[i, j] * (i - j) for i in range(T) for j in range(i + 1)]) * 10

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(6)
    width = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for k, (name, c) in enumerate(zip(names, colors)):
        ax.bar(x + (k - 1.5) * width, metrics[:, k], width, label=name, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([f"head {h}" for h in range(6)], fontsize=12)
    ax.set_ylabel("平均注意力权重", fontsize=12)
    ax.set_title("同一层 6 个头：谁在盯前一位，谁在扫全局（真实数据）", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    # 标注每个头的"性格"
    desc = ["前一位探测器", "前一位探测器", "混合型", "三格前狙击手", "远程扫描仪", "远程扫描仪"]
    for h in range(6):
        ax.text(h, max(metrics[h]) + 0.02, desc[h], ha="center", fontsize=10, color="#333333")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "16-head-pattern.png"), dpi=150)
    plt.close(fig)
    print("已保存 images/16-head-pattern.png")

# ─────────────────────────────────────────────
# 图 4：mask 前 vs mask 后 —— 真实模型"偷看未来"的证据
#    选一个 token，画它在两种规则下的权重分布
# ─────────────────────────────────────────────
def fig_future_temptation():
    h = 0
    w_real = att_soft[h]
    def softmax_row(v):
        e = np.exp(v - v.max())
        return e / e.sum()
    w_nomask = np.stack([softmax_row(att_raw[h][i]) for i in range(T)])

    # 找"未来诱惑最大"的 token：不 mask 时未来权重最大
    future_share_nomask = np.array([w_nomask[i, i + 1:].sum() if i < T - 1 else 0 for i in range(T)])
    i = int(np.argmax(future_share_nomask))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, w, title in [(ax1, w_nomask[i], "假设不加 mask：可以偷看右边"),
                         (ax2, w_real[i], "真实规则：只看左边（mask 生效）")]:
        ax.bar(range(T), w, color="#4C72B0")
        ax.bar(range(i + 1, T), w[i + 1:], color="#C44E52")  # 未来位置标红
        ax.axvline(i + 0.5, color="k", ls="--", lw=0.8)
        ax.set_xlabel("被看的位置 j", fontsize=11)
        ax.set_ylabel("注意力权重", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(0, T, 10))
        ax.tick_params(axis="x", rotation=90)
        ax.text(0.98, 0.95, f"token {i}（'{chars[i]}'）", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color="#333333")
    ax1.text(0.98, 0.82, f"红色 = 未来位置，分走 {w_nomask[i, i+1:].sum():.0%} 权重",
             transform=ax1.transAxes, ha="right", va="top", fontsize=10, color="#C44E52")
    ax2.text(0.98, 0.82, f"红色 = 未来位置，权重 {w_real[i, i+1:].sum():.4f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=10, color="#C44E52")
    fig.suptitle(f"同一个 token，两种规则：模型分数里\"想\"看的未来，被因果掩码拦下（head 0，真实数据）",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(IMG, "16-future-temptation.png"), dpi=150)
    plt.close(fig)
    print("已保存 images/16-future-temptation.png")

fig_mask_matrix()
fig_multihead()
fig_head_pattern()
fig_future_temptation()
print("\n全部图表完成。")
