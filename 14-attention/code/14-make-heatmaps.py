#!/usr/bin/env python3
"""
14-make-heatmaps.py — 第 14 课《Attention 直觉》真实注意力热力图
从训练好的 checkpoint（out-shakespeare-char/ckpt.pt，1000 步，val loss 1.52）
提取真实注意力权重并画热力图。全部是真实数据，不是示意图。
依赖: torch + numpy + matplotlib（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 14-make-heatmaps.py
输出: ../images/14-attn-*.png
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

# 中文字体（macOS 系统字体，避免豆腐块）
for _f in ["/System/Library/Fonts/PingFang.ttc",
           "/System/Library/Fonts/STHeiti Medium.ttc",
           "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(BASE), "images")
os.makedirs(OUTDIR, exist_ok=True)
NANOGPT = os.path.expanduser("~/projects/main-agent/nanoGPT")
sys.path.insert(0, NANOGPT)
CKPT = os.path.join(NANOGPT, "out-shakespeare-char/ckpt.pt")

import model as nano_model

# 1. 加载模型
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = nano_model.GPTConfig(**ckpt["model_args"])
gpt = nano_model.GPT(cfg)
gpt.load_state_dict(ckpt["model"])
gpt.eval()

# 2. 字符映射
import pickle
with open(os.path.join(NANOGPT, "data/shakespeare_char/meta.pkl"), "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]

# 3. 探测文本
sample_text = (
    "ROMEO: But soft, what light through yonder window breaks? "
    "It is the east, and Juliet is the sun."
)
ids = [stoi[c] for c in sample_text if c in stoi]
T = len(ids)
x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
chars = [itos[i] for i in ids]

# 4. 抓注意力权重
captured = {}
def make_hooked(layer_idx):
    def hooked_forward(self, x):
        B, Tt, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :Tt, :Tt] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        captured[layer_idx] = att[0].detach().cpu().numpy()
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, Tt, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
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
    gpt(x)

# 5. 配色：白→橙→深红（看得清“聚焦”）
cmap = LinearSegmentedColormap.from_list("attn", ["#ffffff", "#ffd29d", "#ff8c42", "#d62728", "#7a0c0c"])

def plot_heatmap(att, fname, title, start=48, end=None, annotate=False):
    """att: (T,T) 注意力权重矩阵，画 [start:end]×[start:end] 区域"""
    end = end or T
    w = att[start:end, :end]
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(w, cmap=cmap, vmin=0, vmax=w.max() if w.max() > 0 else 1)
    ax.set_xticks(range(end - start))
    ax.set_xticklabels(chars[start:end], fontsize=9)
    ax.set_yticks(range(end - start))
    ax.set_yticklabels([f"{chars[i]}" for i in range(start, end)], fontsize=9)
    ax.set_xlabel("被关注的 token（键 K）", fontsize=12)
    ax.set_ylabel("正在做决定的 token（查询 Q）", fontsize=12)
    ax.set_title(title, fontsize=14)
    if annotate:
        for ii in range(end - start):
            for jj in range(end - start):
                val = w[ii, jj]
                if val > 0.05:
                    ax.text(jj, ii, f"{val:.1f}", ha="center", va="center", fontsize=7, color="#333")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("注意力权重（越大越被关注）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已生成", fname)

# 图 1：第 0 层 head 0 —— 盯着“前一个字符”的经典头
w = captured[0][0]  # (96,96)
# 找一个权重最大处附近展示，更直观：取中间一段 40 个字符
plot_heatmap(w, "14-attn-layer0-head0.png",
             "真实注意力：第 0 层 head 0（模型学会盯前一个字符）", start=30, end=66)

# 图 2：第 0 层全部 6 个 head 对比 —— 每个头各看各的
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for h, ax in enumerate(axes.flat):
    w = captured[0][h][30:66, :66]
    im = ax.imshow(w, cmap=cmap, vmin=0, vmax=w.max() if w.max() > 0 else 1)
    ax.set_xticks(range(36))
    ax.set_xticklabels(chars[30:66], fontsize=7)
    ax.set_yticks(range(36))
    ax.set_yticklabels(chars[30:66], fontsize=7)
    ax.set_title(f"head {h}", fontsize=13)
    ax.set_xlabel("被关注的 token", fontsize=9)
    ax.set_ylabel("查询 token", fontsize=9)
fig.suptitle("真实注意力：第 0 层 6 个头在同一段文本上的分工（每个头各看各的）", fontsize=15)
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "14-attn-layer0-multihead.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("已生成 14-attn-layer0-multihead.png")

# 图 3：逐层统计 —— 浅层看局部，深层看全局
entropies, distances = [], []
for li in range(cfg.n_layer):
    att = captured[li]
    ents, dists = [], []
    for h in range(cfg.n_head):
        w = att[h]
        for i in range(1, T):
            p = w[i, :i+1]
            ents.append(-np.sum(p * np.log(p + 1e-12)))
            dists.append(np.sum(p * (i - np.arange(i+1))))
    entropies.append(np.mean(ents))
    distances.append(np.mean(dists))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
layers = range(cfg.n_layer)
axes[0].bar(layers, entropies, color="#d62728")
axes[0].set_xticks(list(layers))
axes[0].set_xlabel("层（0 最浅，5 最深）", fontsize=11)
axes[0].set_ylabel("平均注意力熵", fontsize=11)
axes[0].set_title("熵：越低 = 注意力越集中（只盯少数位置）", fontsize=12)
axes[1].bar(layers, distances, color="#ff8c42")
axes[1].set_xticks(list(layers))
axes[1].set_xlabel("层（0 最浅，5 最深）", fontsize=11)
axes[1].set_ylabel("平均注意力距离（字符）", fontsize=11)
axes[1].set_title("距离：越小 = 只看附近，越大 = 放眼全文", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "14-attn-by-layer.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("已生成 14-attn-by-layer.png")
print("层统计：熵", np.round(entropies, 3), "距离", np.round(distances, 2))
