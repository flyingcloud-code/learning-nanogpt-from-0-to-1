#!/usr/bin/env python3
"""
15-make-charts.py — 第 15 课《QKV 详解》图表（全部真实数据，不造假）
依赖: torch + numpy + matplotlib（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 15-make-charts.py
输出: ../images/15-qkv-handcalc.png  ../images/15-qkv-score-dist.png  ../images/15-qkv-dim-std.png
"""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体（Mac 系统字体）
for cand in [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]:
    if os.path.exists(cand):
        fm.fontManager.addfont(cand)
        name = fm.FontProperties(fname=cand).get_name()
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "..", "images")
os.makedirs(IMG, exist_ok=True)

NANOGPT = os.path.expanduser("~/projects/main-agent/nanoGPT")
sys.path.insert(0, NANOGPT)
CKPT = os.path.join(NANOGPT, "out-shakespeare-char/ckpt.pt")
import model as nano_model
import pickle

# ─────────────────────────────────────────────
# 0. 真实模型探测（与 15-qkv-probe.py 同逻辑，抓第 0 层真实 Q/K）
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
with open(os.path.join(DATA, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]

sample_text = (
    "ROMEO: But soft, what light through yonder window breaks? "
    "It is the east, and Juliet is the sun."
)
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
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :Tt, :Tt] == 0, float("-inf"))
    att_soft = F.softmax(att, dim=-1)
    captured[layer_idx] = {"q": q[0].detach().cpu().numpy(), "k": k[0].detach().cpu().numpy()}
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

q = captured[0]["q"]  # (6, T, 64)
k = captured[0]["k"]
mask = np.tril(np.ones((T, T), dtype=bool))
scores_raw = np.einsum("hij,hkj->hik", q, k)
scores_scaled = scores_raw / math.sqrt(head_dim)

# 维度实验：用前 d_test 维重算
dim_std = []
for d_test in [8, 16, 32, 64]:
    qs, ks = q[:, :, :d_test], k[:, :, :d_test]
    sr = np.einsum("hij,hkj->hik", qs, ks)
    dim_std.append(float(sr[:, mask].std()))

# ─────────────────────────────────────────────
# 图 1：2 个 token 手算全过程（真实数字，seed=15，与 15-qkv-handcalc.py 一致）
# ─────────────────────────────────────────────
rng = np.random.default_rng(15)
X = np.array([[0.8, 0.2, -0.5, 0.1], [0.3, -0.7, 0.9, 0.4]])
Wq, Wk, Wv = rng.normal(0, 1, (4, 4)), rng.normal(0, 1, (4, 4)), rng.normal(0, 1, (4, 4))
Q, K, V = X @ Wq, X @ Wk, X @ Wv
scores_raw2 = Q @ K.T
scores2 = scores_raw2 / np.sqrt(4)
def softmax_rows(M):
    e = np.exp(M - M.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
W2 = softmax_rows(scores2)
out2 = W2 @ V

BG = "#0f172a"
PANEL = "#1e293b"

def draw_matrix(ax, mat, title, colors, fmt="{:.2f}", fontsize=13, vmin=None, vmax=None):
    """mat: (rows, cols) 数字矩阵 → 深色格子 + 数字"""
    mat = np.asarray(mat, dtype=float)
    rows, cols = mat.shape
    im = ax.imshow(mat, cmap=colors, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([f"d{i}" for i in range(cols)], fontsize=10, color="#94a3b8")
    ax.set_yticklabels([f"tok{i}" for i in range(rows)], fontsize=10, color="#94a3b8")
    ax.tick_params(length=0)
    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, fmt.format(mat[r, c]), ha="center", va="center",
                    fontsize=fontsize, color="white", fontfamily="monospace")
    ax.set_title(title, fontsize=13, color="white", pad=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

fig = plt.figure(figsize=(17, 8.6), facecolor=BG)
fig.patch.set_facecolor(BG)

# 区块 1：投影 Q/K/V
ax1 = fig.add_axes([0.025, 0.62, 0.20, 0.28]); draw_matrix(ax1, Q, "① 查询 Q = X@Wq", "Blues")
ax2 = fig.add_axes([0.245, 0.62, 0.20, 0.28]); draw_matrix(ax2, K, "② 键 K = X@Wk", "Purples")
ax3 = fig.add_axes([0.465, 0.62, 0.20, 0.28]); draw_matrix(ax3, V, "③ 值 V = X@Wv", "Greens")
fig.text(0.69, 0.80, "每个 token 的\n同一个向量\n乘三个不同矩阵\n→ 三个角色", fontsize=12, color="#94a3b8", va="center")
fig.text(0.025, 0.925, "投影：X (2×4) 乘 Wq / Wk / Wv，得到 Q / K / V", fontsize=14, color="#22d3ee", fontweight="bold")

# 区块 2：打分 + 归一
ax4 = fig.add_axes([0.025, 0.28, 0.16, 0.24]); draw_matrix(ax4, scores_raw2, "④ 原始分数 Q·K^T", "YlOrRd", fmt="{:+.2f}", fontsize=15)
ax5 = fig.add_axes([0.22, 0.28, 0.16, 0.24]); draw_matrix(ax5, scores2, "⑤ ÷√d=2 后", "YlOrRd", fmt="{:+.2f}", fontsize=15)
ax6 = fig.add_axes([0.415, 0.28, 0.16, 0.24]); draw_matrix(ax6, W2, "⑥ softmax 权重", "Reds", fmt="{:.2f}", fontsize=15)
fig.text(0.62, 0.40, "行方向归一\n每行加起来 = 1", fontsize=12, color="#94a3b8", va="center")
fig.text(0.025, 0.545, "打分与归一：相关度分数 → 比例（行和=1）", fontsize=14, color="#fbbf24", fontweight="bold")

# 区块 3：混合输出
ax7 = fig.add_axes([0.68, 0.28, 0.20, 0.28]); draw_matrix(ax7, out2, "⑦ 输出 out = 权重@V", "Greens", fmt="{:+.2f}", fontsize=13)
fig.text(0.62, 0.62, "每个 token 的新表示\n= 所有 V 的加权混合", fontsize=12, color="#94a3b8", va="center")
fig.text(0.025, 0.185, "混合：权重 @ V → 输出（第 0 行 = 0.61×V[0] + 0.39×V[1]）", fontsize=14, color="#34d399", fontweight="bold")

fig.text(0.025, 0.05, "手算全过程（d=4，固定随机种子 15，与 15-qkv-handcalc.py 输出一致）", fontsize=11, color="#64748b")
plt.savefig(os.path.join(IMG, "15-qkv-handcalc.png"), dpi=150, facecolor=BG)
plt.close(fig)
print("图 1 已保存：15-qkv-handcalc.png")

# ─────────────────────────────────────────────
# 图 2：真实模型 Q·Kᵀ 分数分布（缩放前 vs 缩放后）
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
raw_vis = scores_raw[:, mask].flatten()
scaled_vis = scores_scaled[:, mask].flatten()

ax = axes[0]
ax.hist(raw_vis, bins=80, color="#fb7185", alpha=0.85)
ax.axvline(0, color="white", lw=0.8, ls="--")
ax.set_title(f"不缩放：Q·K^T 原始分数\nstd={raw_vis.std():.2f}  范围 [{raw_vis.min():.0f}, {raw_vis.max():.0f}]", fontsize=13, color="white")
ax.set_xlabel("分数值", fontsize=11, color="#94a3b8")
ax.set_ylabel("出现次数", fontsize=11, color="#94a3b8")
ax.tick_params(colors="#94a3b8")
for s in ax.spines.values(): s.set_color("#334155")
ax.text(0.98, 0.95, "动辄 ±100\nsoftmax 直接饱和", transform=ax.transAxes, ha="right", va="top",
        fontsize=12, color="#fda4af")

ax = axes[1]
ax.hist(scaled_vis, bins=80, color="#22d3ee", alpha=0.85)
ax.axvline(0, color="white", lw=0.8, ls="--")
ax.set_title(f"缩放后：Q·K^T / √{head_dim}\nstd={scaled_vis.std():.2f}  范围 [{scaled_vis.min():.0f}, {scaled_vis.max():.0f}]", fontsize=13, color="white")
ax.set_xlabel("分数值", fontsize=11, color="#94a3b8")
ax.set_ylabel("出现次数", fontsize=11, color="#94a3b8")
ax.tick_params(colors="#94a3b8")
for s in ax.spines.values(): s.set_color("#334155")
ax.text(0.98, 0.95, "回到 ±3 附近\nsoftmax 有中间地带\n梯度才能流动", transform=ax.transAxes, ha="right", va="top",
        fontsize=12, color="#67e8f9")

fig.suptitle("真实 GPT 第 0 层全部 6 个头、96 个字符的 Q·K^T 分数分布（同一批数据）",
             fontsize=15, color="white", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(IMG, "15-qkv-score-dist.png"), dpi=150, facecolor=BG)
plt.close(fig)
print("图 2 已保存：15-qkv-score-dist.png")

# ─────────────────────────────────────────────
# 图 3：head_dim 越大，分数 std 越大 → 为什么必须缩放
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=BG)
fig.patch.set_facecolor(BG)

ds = np.array([8, 16, 32, 64])
ax.plot(ds, dim_std, "o-", color="#fb7185", lw=2.5, ms=9, label="真实模型分数 std（实测）")
ax.plot(ds, np.sqrt(ds), "s--", color="#94a3b8", lw=1.8, label="理论基准 √d（元素独立时）")
ax.plot(ds, np.array(dim_std) / np.sqrt(ds), "^-", color="#34d399", lw=2, ms=8, label="缩放后 std（除以 √d）")

for i, d in enumerate(ds):
    ax.annotate(f"{dim_std[i]:.1f}", (d, dim_std[i]), textcoords="offset points",
                xytext=(6, 6), fontsize=11, color="#fda4af", fontweight="bold")
    ax.annotate(f"{dim_std[i]/math.sqrt(d):.1f}", (d, dim_std[i]/math.sqrt(d)), textcoords="offset points",
                xytext=(6, -14), fontsize=10, color="#6ee7b7")

ax.set_xticks(ds)
ax.set_xticklabels([f"head_dim={d}" for d in ds], fontsize=11, color="#94a3b8")
ax.set_ylabel("Q·K^T 分数的标准差", fontsize=12, color="white")
ax.set_title("维度越大，相关度分数越膨胀——这就是必须除以 √d 的原因", fontsize=14, color="white")
ax.legend(fontsize=11, facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
ax.tick_params(colors="#94a3b8")
for s in ax.spines.values(): s.set_color("#334155")
ax.grid(alpha=0.2, color="#334155")

plt.tight_layout()
plt.savefig(os.path.join(IMG, "15-qkv-dim-std.png"), dpi=150, facecolor=BG)
plt.close(fig)
print("图 3 已保存：15-qkv-dim-std.png")
print("全部图表完成（真实数据）。")
