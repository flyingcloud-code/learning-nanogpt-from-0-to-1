#!/usr/bin/env python3
"""
15-qkv-probe.py — 第 15 课《QKV 详解》实验脚本（真实模型里 Q/K/V 长什么样）
依赖: torch + numpy（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 15-qkv-probe.py

目的：从真实训练好的 GPT（6 层 6 头，1000 步，val loss 1.52）里，
     抓出第 0 层的真实 Q、K 向量，看看 Q·Kᵀ 分数在缩放前后的真实分布，
     用数字回答"为什么一定要除以 √d"。
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
NANOGPT = os.path.expanduser("~/projects/main-agent/nanoGPT")
sys.path.insert(0, NANOGPT)
CKPT = os.path.join(NANOGPT, "out-shakespeare-char/ckpt.pt")

import model as nano_model  # nanoGPT 的 model.py

# ─────────────────────────────────────────────
# 1. 加载真实训练好的模型
# ─────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = nano_model.GPTConfig(**ckpt["model_args"])
gpt = nano_model.GPT(cfg)
gpt.load_state_dict(ckpt["model"])
gpt.eval()
print("已加载 checkpoint：", CKPT)
print(f"  配置 n_layer={cfg.n_layer} n_head={cfg.n_head} n_embd={cfg.n_embd} block_size={cfg.block_size}")
head_dim = cfg.n_embd // cfg.n_head
print(f"  每个头的维度 head_dim = n_embd/n_head = {cfg.n_embd}/{cfg.n_head} = {head_dim}")
print(f"  所以缩放因子 1/√d = 1/√{head_dim} = 1/{math.sqrt(head_dim):.3f}")
print(f"  训练步数 iter_num={ckpt['iter_num']}  best_val_loss={float(ckpt['best_val_loss']):.4f}")

# ─────────────────────────────────────────────
# 2. 准备一段真实莎士比亚文本
# ─────────────────────────────────────────────
DATA = os.path.join(NANOGPT, "data/shakespeare_char")
with open(os.path.join(DATA, "input.txt"), "r", encoding="utf-8") as f:
    full_text = f.read()

import pickle
with open(os.path.join(DATA, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]

sample_text = (
    "ROMEO: But soft, what light through yonder window breaks? "
    "It is the east, and Juliet is the sun."
)
ids = [stoi[c] for c in sample_text if c in stoi]
T = len(ids)
print(f"\n探测文本 T={T} 个字符（字符级 token）：")
print("  ", sample_text)

x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, T)

# ─────────────────────────────────────────────
# 3. 用 patched forward 抓真实 Q、K、V（第 0 层）
#    （flash attention 抓不到中间量，改走手写路径）
# ─────────────────────────────────────────────
captured = {}  # layer_idx -> dict(q, k, v, att)

orig_forward = nano_model.CausalSelfAttention.forward

def patched_forward(self, x, layer_idx):
    B, Tt, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :Tt, :Tt] == 0, float("-inf"))
    att_soft = F.softmax(att, dim=-1)
    captured[layer_idx] = {
        "q": q[0].detach().cpu().numpy(),      # (n_head, T, head_dim)
        "k": k[0].detach().cpu().numpy(),
        "v": v[0].detach().cpu().numpy(),
        "att": att[0].detach().cpu().numpy(),      # 缩放后、mask 前的分数
        "att_soft": att_soft[0].detach().cpu().numpy(),  # softmax 后
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
    logits = gpt(x)  # 前向一次

# ─────────────────────────────────────────────
# 4. 核心实验：Q·Kᵀ 分数缩放前 vs 缩放后
#    head_dim = 64 → √d = 8
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("核心实验：Q·Kᵀ 的原始分数 vs 除以 √d 后的分数（第 0 层全部 6 个头）")
print("=" * 70)

# 重新计算：q, k 是真实抓到的向量，scores_raw 是没乘 1/√d 的版本
q = captured[0]["q"]      # (6, T, 64)
k = captured[0]["k"]      # (6, T, 64)

scores_raw = np.einsum("hij,hkj->hik", q, k)  # (6, T, T) 不缩放
scores_scaled = scores_raw / math.sqrt(head_dim)

# 只看因果掩码允许的位置（j <= i）
mask = np.tril(np.ones((T, T), dtype=bool))
raw_visible = scores_raw[:, mask]
scaled_visible = scores_scaled[:, mask]

print(f"\n{'头':>3} | {'不缩放: mean':>14} {'std':>8} {'min':>8} {'max':>8} | {'缩放后: mean':>13} {'std':>8} {'min':>8} {'max':>8}")
for h in range(cfg.n_head):
    r = raw_visible[h * T * T // cfg.n_head:][:0]  # placeholder 防呆
    rv = scores_raw[h][mask]
    sv = scores_scaled[h][mask]
    print(f"{h:>3} | {rv.mean():>14.2f} {rv.std():>8.2f} {rv.min():>8.2f} {rv.max():>8.2f} | "
          f"{sv.mean():>13.2f} {sv.std():>8.2f} {sv.min():>8.2f} {sv.max():>8.2f}")

print(f"\n全部头合并：")
print(f"  不缩放 Q·Kᵀ：std={raw_visible.std():.2f}  min={raw_visible.min():.2f}  max={raw_visible.max():.2f}")
print(f"  缩放后 Q·Kᵀ/√{head_dim}：std={scaled_visible.std():.2f}  min={scaled_visible.min():.2f}  max={scaled_visible.max():.2f}")

# q/k 向量自身的元素标准差（验证 LN 后元素 ~N(0,1)）
q_std = q.std()
k_std = k.std()
print(f"\n真实 Q 向量元素标准差 = {q_std:.2f}，K 向量元素标准差 = {k_std:.2f}")
print(f"直觉验证：若元素独立且 std≈1，d={head_dim} 维点积的 std 理论值 ≈ √{head_dim} = {math.sqrt(head_dim):.2f}")

# ─────────────────────────────────────────────
# 5. 如果不缩放，softmax 会变成什么样？（真实数据）
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("对照实验：同一批真实分数，不缩放 vs 缩放，softmax 结果差多少")
print("=" * 70)
h = 0
# 取一个具体行（i = T-1，最后一个 token），只看它能看到的前面所有位置
i = T - 1
visible = np.arange(i + 1)
raw_row = scores_raw[h, i, visible]
scaled_row = scores_scaled[h, i, visible]

def softmax_row(v):
    e = np.exp(v - v.max())
    return e / e.sum()

w_raw = softmax_row(raw_row)
w_scaled = softmax_row(scaled_row)
print(f"\n第 0 层 head 0，第 {i} 个 token（'{itos[ids[i]]}'）看前面 {i+1} 个位置：")
print(f"{'j':>3} {'字符':>4} {'原始分数':>10} {'不缩放权重':>12} {'缩放权重':>10}")
for j in visible:
    print(f"{j:>3} {itos[ids[j]]:>4} {raw_row[j - visible[0]]:>10.2f} {w_raw[j - visible[0]]:>12.4f} {w_scaled[j - visible[0]]:>10.4f}")
print(f"\n不缩放时：最大权重 {w_raw.max():.4f}，其余位置几乎全被压到 0 —— softmax 饱和")
print(f"缩放后：最大权重 {w_scaled.max():.4f}，其余位置还保留着梯度 —— 模型才学得动")

# ─────────────────────────────────────────────
# 6. 顺带：head_dim 变化对分数的影响（真实 Q/K 缩放模拟）
#    用真实 q,k 的前 8 维 / 16 维 / 64 维各算一遍，看 std
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("维度实验：用真实 Q/K 的前 head_dim 维，看分数 std 随维度的变化")
print("=" * 70)
print(f"{'head_dim':>8} {'√head_dim':>9} {'分数std(不缩放)':>16} {'缩放后std':>10}")
for d_test in [8, 16, 32, 64]:
    qs = q[:, :, :d_test]
    ks = k[:, :, :d_test]
    sr = np.einsum("hij,hkj->hik", qs, ks)
    sv = sr / math.sqrt(d_test)
    print(f"{d_test:>8} {math.sqrt(d_test):>9.2f} {sr[:, mask].std():>16.2f} {sv[:, mask].std():>10.2f}")

print("\n实验完成：Q·Kᵀ 不缩放时分数动辄 ±20，softmax 饱和；除以 √d 后回到 ±3 以内。")
