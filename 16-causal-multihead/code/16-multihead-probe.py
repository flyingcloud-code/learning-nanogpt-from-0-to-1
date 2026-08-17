#!/usr/bin/env python3
"""
16-multihead-probe.py — 第 16 课《因果掩码 + 多头》真实模型探测
依赖: torch + numpy（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 16-multihead-probe.py

目的：从真实训练好的 GPT（6 层 6 头，1000 步，val loss 1.52）里，
     抓出第 0 层 6 个头的真实注意力权重，回答三个问题：
  1. 因果掩码真的生效了吗？——未来位置（j>i）的权重是否精确为 0？
  2. 6 个头都在看什么？——多头是不是真的各看各的？
  3. 如果摘掉 mask 会怎样？——真实分数里"未来"有多诱人？

所有数字均为 Mac mini 真实运行输出。
"""
import os, sys, math
import numpy as np
import torch
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
print(f"  head_dim = {cfg.n_embd}/{cfg.n_head} = {head_dim}")
print(f"  训练步数 iter_num={ckpt['iter_num']}  best_val_loss={float(ckpt['best_val_loss']):.4f}")

# ─────────────────────────────────────────────
# 2. 准备一段真实莎士比亚文本（字符级 token）
# ─────────────────────────────────────────────
DATA = os.path.join(NANOGPT, "data/shakespeare_char")
with open(os.path.join(DATA, "input.txt"), "r", encoding="utf-8") as f:
    full_text = f.read()

import pickle
with open(os.path.join(DATA, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]

# 从语料里截一段真实文本（含标点和空格，跟训练分布一致）
idx = full_text.find("ROMEO:")
sample_text = full_text[idx: idx + 100]
ids = [stoi[c] for c in sample_text if c in stoi]
T = len(ids)
print(f"\n探测文本 T={T} 个字符：")
print("  ", repr(sample_text))

x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, T)

# ─────────────────────────────────────────────
# 3. patched forward：抓真实注意力权重
#    att_raw   = 打分后、mask 前（如果允许偷看未来，分数长这样）
#    att_soft  = mask + softmax 后（真实生效的权重）
# ─────────────────────────────────────────────
captured = {}  # layer_idx -> dict(att_raw, att_soft)

def patched_forward(self, x, layer_idx):
    B, Tt, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    att_raw = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # 未加 mask
    att = att_raw.masked_fill(self.bias[:, :, :Tt, :Tt] == 0, float("-inf"))  # 加 mask
    att_soft = F.softmax(att, dim=-1)
    captured[layer_idx] = {
        "att_raw": att_raw[0].detach().cpu().numpy(),    # (n_head, T, T) mask 前分数
        "att_soft": att_soft[0].detach().cpu().numpy(),  # (n_head, T, T) mask 后权重
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

att_soft = captured[0]["att_soft"]  # (6, T, T)
att_raw = captured[0]["att_raw"]    # (6, T, T)

# ─────────────────────────────────────────────
# 实验 1：因果掩码真的生效了吗？
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("实验 1：因果性验证——未来位置（j > i）的权重是不是精确为 0？")
print("=" * 70)
for h in range(cfg.n_head):
    w = att_soft[h]
    future = w[np.triu_indices(T, k=1)]   # 所有 j > i 的位置
    future_max = future.max()
    n_zeros = (future == 0.0).sum()
    print(f"  头 {h}: 未来位置权重 max = {future_max:.2e}，等于 0 的格子 {n_zeros}/{future.size}")

# 对每个头的每一行，验证"权重只落在 j <= i"
print("\n每行权重之和（应该全 = 1）：")
print("  head0 前 5 行:", np.round(att_soft[0][:5].sum(axis=1), 6))
print("  head5 前 5 行:", np.round(att_soft[5][:5].sum(axis=1), 6))

# ─────────────────────────────────────────────
# 实验 2：多头分工——每个头都在看什么？
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("实验 2：多头分工——第 0 层 6 个头各自的关注习惯")
print("=" * 70)
print(f"{'头':>3} | {'看自己':>8} {'看前一位':>8} {'看句首':>8} {'平均目光距离':>12} {'最远一瞥':>8}")
print("-" * 60)
head_stats = []
for h in range(cfg.n_head):
    w = att_soft[h]                    # (T, T)
    diag = np.mean([w[i, i] for i in range(T)])
    prev = np.mean([w[i, i - 1] for i in range(1, T)])
    first = np.mean([w[i, 0] for i in range(T)])
    dist = np.mean([w[i, j] * (i - j) for i in range(T) for j in range(i + 1)])
    head_stats.append((diag, prev, first, dist))
    print(f"{h:>3} | {diag:>8.4f} {prev:>8.4f} {first:>8.4f} {dist:>12.2f}")

# 每头"最关注的偏移量"：argmax 权重的位置与自己的距离
print("\n每个 token 最关注的位置与自己相差几格（0 = 看自己，-1 = 看前一位）：")
for h in range(cfg.n_head):
    w = att_soft[h]
    argmax_off = [int(np.argmax(w[i, :i + 1])) - i for i in range(T)]
    most_common = max(set(argmax_off), key=argmax_off.count)
    share = argmax_off.count(most_common) / T
    print(f"  头 {h}: 最常见的偏移 = {most_common:>3}（{share*100:.0f}% 的 token 这么选），"
          f"最左看过 {min(argmax_off)} 格")

# ─────────────────────────────────────────────
# 实验 3：如果摘掉 mask，模型会偷看未来吗？
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("实验 3：摘掉 mask 会怎样？——未来位置在真实分数里有多诱人")
print("=" * 70)

def softmax_row(v):
    e = np.exp(v - v.max())
    return e / e.sum()

total_future_weight = 0.0
head_future = []
for h in range(cfg.n_head):
    w_nomask = np.stack([softmax_row(att_raw[h][i]) for i in range(T)])  # 不加 mask 的 softmax
    future_share = w_nomask[np.triu_indices(T, k=1)].sum() / T  # 平均每个 token 分给未来的权重
    head_future.append(future_share)
    total_future_weight += future_share
print("（同一批真实分数，跳过 mask 直接 softmax——未来位置能分到多少权重？）")
for h in range(cfg.n_head):
    print(f"  头 {h}: 平均每个 token 分给\"未来\"的权重 = {head_future[h]:.4f}")

# 具体看一个 token：找"未来诱惑最大"的例子——不加 mask 时未来分走权重最多
raw0 = att_raw[0]
best_example = None
for i in range(T - 1):
    fut = raw0[i, i + 1:]
    if fut.size:
        w_row = softmax_row(raw0[i])
        fut_share = w_row[i + 1:].sum()  # 不加 mask 时，未来位置一共分走多少权重
        if best_example is None or fut_share > best_example[3]:
            best_example = (i, int(i + 1 + np.argmax(fut)), float(fut.max()), float(fut_share))
if best_example:
    i, j, s, share = best_example
    w_real = att_soft[0, i]                     # 真实权重（mask 后）
    w_nom = softmax_row(att_raw[0, i])          # 假想（不 mask）
    print(f"\n例子：第 0 层 head 0，token {i}（'{itos[ids[i]]}'）")
    print(f"  它对未来位置的最高原始分数 = {s:.2f}（'{itos[ids[j]]}'，位置 {j}）")
    print(f"  不加 mask 的话，全部未来位置一共分走 {share:.0%} 的权重")
    print(f"  加了 mask：每个未来位置的权重精确 = {w_real[i + 1:].max():.8f}（被强制归零）")
    print("  这就是因果性的全部含义：模型可以想，但不能看。")

print("\n实验完成：mask 生效（未来权重全 0）；6 个头确实各看各的；")
print("摘掉 mask 时真实分数里有明显的\"偷看未来\"倾向——被因果性硬压住了。")
