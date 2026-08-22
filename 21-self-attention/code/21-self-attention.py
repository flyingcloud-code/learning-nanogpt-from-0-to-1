#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 21 课：手搓 Self-Attention（QKV 落地）
===========================================
不用 nn.MultiheadAttention / F.scaled_dot_product_attention，
用纯矩阵乘法把 attention 写出来，然后和官方 API 对拍验证。

运行（Mac mini / Apple Silicon，torch 2.12.1）：
    ~/projects/main-agent/nanoGPT/.venv/bin/python 21-self-attention.py

依赖：torch, numpy（venv 已装）
"""
import os
import math
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
np.random.seed(1337)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"设备: {DEVICE}  torch {torch.__version__}")

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(os.path.dirname(BASE), "images")
os.makedirs(IMG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Part 1：手写【单头】Self-Attention——不用任何现成 API
# 唯一用的"现成"东西是 torch 的矩阵乘法 @、softmax、masked_fill
# 输入 x: (T, C)，T=token 数，C=每个 token 的维度
# ---------------------------------------------------------------------------

def hand_attention(x, Wq, Wk, Wv, Wo, causal=True):
    """
    手写单头 self-attention 的完整前向。
    返回每一步的中间结果，方便逐层解剖。
    x: (T, C)
    Wq, Wk, Wv, Wo: (C, C)
    """
    T, C = x.shape
    # ① 投影：同一个 x，乘三个矩阵，变成三个角色
    q = x @ Wq          # (T, C) 查询：我在找什么
    k = x @ Wk          # (T, C) 键：我是什么
    v = x @ Wv          # (T, C) 值：我能提供什么

    # ② 打分：Q 的每一行和 K 的每一行做点积 → 相关度矩阵 (T, T)
    scores = q @ k.T / math.sqrt(C)   # 除以 √d 防 softmax 饱和（第 15 课）

    # ③ 因果掩码：预测第 t 个 token 时，只能看 0..t-1，不能看未来
    if causal:
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))

    # ④ 归一：softmax 把每一行变成加起来 = 1 的权重
    att = F.softmax(scores, dim=-1)   # (T, T)

    # ⑤ 混合：按权重加权 V 的所有行 → 每个 token 的新表示
    out = att @ v       # (T, C)
    out = out @ Wo      # 输出投影（把多头拼回来的信息再整合一次）

    return {"q": q, "k": k, "v": v, "scores": scores, "att": att, "out": out}


def part1_hand_single_head():
    print("\n" + "=" * 60)
    print("Part 1：手写单头 Self-Attention（4 个 token，C=8）")
    print("=" * 60)
    T, C = 4, 8
    torch.manual_seed(42)
    x = torch.randn(T, C)
    Wq = torch.randn(C, C) * 0.1
    Wk = torch.randn(C, C) * 0.1
    Wv = torch.randn(C, C) * 0.1
    Wo = torch.randn(C, C) * 0.1

    r = hand_attention(x, Wq, Wk, Wv, Wo, causal=True)
    print(f"输入 x 形状: {tuple(x.shape)}  (4 个 token，每个 8 维)")
    print(f"Q 形状: {tuple(r['q'].shape)}   K 形状: {tuple(r['k'].shape)}   V 形状: {tuple(r['v'].shape)}")
    print(f"打分矩阵 scores ({T}x{T}，左下角是过去，右上角被掩码成 -inf):")
    for row in r["scores"].tolist():
        print("   " + " ".join(f"{v:8.3f}" for v in row))
    print(f"注意力权重 att（每行加起来 = 1，右上角 = 0）:")
    for row in r["att"].tolist():
        print("   " + " ".join(f"{v:6.3f}" for v in row))
    # 验证掩码：att 的右上三角必须全是 0
    triu = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert (r["att"][triu].abs() < 1e-8).all(), "因果掩码失效！右上角应该有 0"
    print(f"✅ 因果掩码验证通过：右上三角注意力权重全为 0")
    return r


# ---------------------------------------------------------------------------
# Part 2：手写【多头因果】Self-Attention（nanoGPT 同款结构）
# 不用 nn.MultiheadAttention，不用 F.scaled_dot_product_attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """手写多头因果 Self-Attention，结构对齐 nanoGPT 的 CausalSelfAttention。"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # 一个矩阵同时算 QKV（nanoGPT 的 c_attn 就是这么干的）
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)  # 输出投影

        # 因果掩码：下三角全 1，缓存起来不用每次重建（nanoGPT 同款 trick）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(1, 1, 1024, 1024)).view(1, 1, 1024, 1024),
        )

    def forward(self, x):
        B, T, C = x.shape  # B=batch, T=token 数, C=维度

        # ① 投影：一个线性层算 QKV（把 3C 维输出切成三段）
        qkv = self.c_attn(x)             # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)    # 各 (B, T, C)

        # ② 拆多头：C -> (n_head, head_dim)，并转置让 head 维到前面
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ③ 打分 + 缩放：每个 head 独立算 Q·Kᵀ / √head_dim
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        #   (B, H, T, T)

        # ④ 因果掩码：只看左边（未来是 -inf，softmax 后变 0）
        mask = self.bias[:, :, :T, :T]  # (1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))

        # ⑤ softmax 归一
        att = F.softmax(att, dim=-1)

        # ⑥ 混合：att @ V，再把多头拼回 (B, T, C)
        y = att @ v                      # (B, H, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 拼回多头

        # ⑦ 输出投影
        y = self.c_proj(y)               # (B, T, C)
        return y, att


# ---------------------------------------------------------------------------
# Part 3：对拍——手写版 vs nn.MultiheadAttention（官方 API）
# 把同一份权重装进官方 MHA，相同输入，比较输出误差
# ---------------------------------------------------------------------------

def part3_compare_mha():
    print("\n" + "=" * 60)
    print("Part 3：对拍——手写版 vs nn.MultiheadAttention")
    print("=" * 60)
    torch.manual_seed(7)
    B, T, C, H = 2, 12, 64, 4

    x = torch.randn(B, T, C).to(DEVICE)

    mine = CausalSelfAttention(C, H).to(DEVICE)
    mha = nn.MultiheadAttention(C, H, batch_first=True, bias=False).to(DEVICE)

    # 权重拷贝：官方 MHA 的 in_proj_weight 形状 (3C, C)，恰好就是
    # [Wq; Wk; Wv] 拼在一起，和我们的 c_attn.weight 布局一致
    with torch.no_grad():
        mha.in_proj_weight.copy_(mine.c_attn.weight)
        mha.out_proj.weight.copy_(mine.c_proj.weight)

    mine.eval(); mha.eval()
    # 因果掩码：加法掩码（additive mask），保留位置加 0，屏蔽位置加 -inf
    # 注意：这个 torch 版本里 MHA 的 bool mask 语义与 SDPA 不同（会反转），
    # 用 float 加法掩码最稳——正好演示"掩码就是往分数里加 -inf"
    mask = torch.zeros(T, T, dtype=torch.float32, device=DEVICE)
    mask.masked_fill_(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEVICE)), float("-inf"))
    with torch.no_grad():
        y_mine, att_mine = mine(x)
        y_mha, _ = mha(x, x, x, attn_mask=mask, need_weights=False)

    diff = (y_mine - y_mha).abs().max().item()
    print(f"输入 x: {tuple(x.shape)}")
    print(f"手写版输出: {y_mine[0, 0, :4].tolist()}")
    print(f"官方版输出: {y_mha[0, 0, :4].tolist()}")
    print(f"最大绝对误差: {diff:.3e}")
    assert diff < 1e-5, "对拍失败：手写版和官方版差异过大"
    print("✅ 对拍通过：手写 attention 与 nn.MultiheadAttention 输出一致（误差 < 1e-5）")
    return diff


# ---------------------------------------------------------------------------
# Part 4：完整训练对比——手写 attention 版 GPT vs 官方 MHA 版 GPT
# 同样数据、同样优化器，训练 400 步，看 loss 是否都能降下去
# ---------------------------------------------------------------------------

def load_shakespeare():
    """加载 shakespeare_char 预处理好的 bin（nanoGPT 的 data 格式）"""
    data_dir = os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char")
    train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    print(f"数据: train {len(train)} tokens, val {len(val)} tokens")
    return train, val


class MiniGPT(nn.Module):
    """微型 GPT：token emb + pos emb + N 个手写 attention 块 + LM head"""

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, use_mha=False):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.use_mha = use_mha

        if use_mha:
            # 对照组：官方 nn.MultiheadAttention
            self.attns = nn.ModuleList(
                [nn.MultiheadAttention(n_embd, n_head, batch_first=True, bias=False)
                 for _ in range(n_layer)]
            )
        else:
            # 实验组：手写 attention
            self.attns = nn.ModuleList(
                [CausalSelfAttention(n_embd, n_head) for _ in range(n_layer)]
            )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # 输出层与 token emb 共享权重（nanoGPT 也这么干）
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.ones_(m.weight)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)                       # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, C)
        x = tok + pos

        for i, attn in enumerate(self.attns):
            if self.use_mha:
                # 加法因果掩码（float）：保留加 0，屏蔽加 -inf
                # ⚠️ 这个 torch 版本里 MHA 的 bool mask 语义会反转，必须用 float
                mask = torch.zeros(T, T, dtype=torch.float32, device=idx.device)
                mask.masked_fill_(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device)), float("-inf"))
                y, _ = attn(x, x, x, attn_mask=mask, need_weights=False)
                x = x + y  # 残差
            else:
                y, _ = attn(x)
                x = x + y  # 残差

        x = self.ln_f(x)
        logits = self.lm_head(x)                        # (B, T, vocab)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_batch(train, val, block_size, batch_size, device):
    """随机取一个 batch：nanoGPT get_batch 的简化版"""
    ix = torch.randint(len(train) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((train[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((train[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def train_one(model, train, val, steps=400, block_size=64, batch_size=32, lr=3e-4):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(train, val, block_size, batch_size, DEVICE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    dt = time.time() - t0
    print(f"  完成 {steps} 步，用时 {dt:.1f}s ({dt/steps*1000:.0f} ms/step)")
    return losses


def part4_train_compare():
    print("\n" + "=" * 60)
    print("Part 4：训练对比——手写 attention 版 GPT vs 官方 MHA 版 GPT")
    print("=" * 60)
    train, val = load_shakespeare()

    vocab_size = 65
    block_size = 64
    n_layer, n_head, n_embd = 2, 4, 128   # 微型配置，Mac mini 上 1 分钟以内
    steps = 400

    torch.manual_seed(1337)
    model_mine = MiniGPT(vocab_size, block_size, n_layer, n_head, n_embd, use_mha=False)
    model_mha = MiniGPT(vocab_size, block_size, n_layer, n_head, n_embd, use_mha=True)

    print("\n--- 训练【手写 attention 版】 ---")
    losses_mine = train_one(model_mine, train, val, steps=steps,
                            block_size=block_size, batch_size=32)

    print("\n--- 训练【官方 MHA 版】 ---")
    losses_mha = train_one(model_mha, train, val, steps=steps,
                           block_size=block_size, batch_size=32)

    # 验证：手写版和官方版 loss 都收敛，且两者曲线接近（差值小）
    final_mine, final_mha = losses_mine[-1], losses_mha[-1]
    print(f"\n手写版最终 loss: {final_mine:.4f}   官方版最终 loss: {final_mha:.4f}")
    print(f"两者差值: {abs(final_mine - final_mha):.4f}")
    assert final_mine < 3.3, f"手写版 loss 没有收敛（{final_mine:.4f}），attention 实现有问题"
    assert abs(final_mine - final_mha) < 0.2, f"手写版与官方版差异过大（{abs(final_mine - final_mha):.4f}）"
    print("✅ 手写 attention 版 GPT 训练收敛，和官方 API 曲线几乎重合")

    # 保存 loss 数据给画图脚本
    data = {
        "steps": steps,
        "losses_mine": losses_mine,
        "losses_mha": losses_mha,
    }
    with open(os.path.join(BASE, "chart-loss.json"), "w") as f:
        json.dump(data, f)
    print("loss 数据已存 chart-loss.json")

    # 保存一个手写模型的注意力权重（供热力图）
    torch.manual_seed(1337)
    x_sample, _ = get_batch(train, val, block_size, 1, DEVICE)
    model_mine.eval()
    with torch.no_grad():
        # 走 forward 拿 att
        tok = model_mine.token_emb(x_sample)
        pos = model_mine.pos_emb(torch.arange(block_size, device=DEVICE))
        h = tok + pos
        att_maps = []
        for attn in model_mine.attns:
            y, att = attn(h)
            att_maps.append(att[0].detach().cpu().numpy())  # (H, T, T)
            h = h + y
    # 取第 0 层第 0 个头，前 40 个 token 的注意力
    att0 = att_maps[0][0][:40, :40]
    np.save(os.path.join(BASE, "chart-att.npy"), att0)
    print(f"注意力热力图数据已存 chart-att.npy（{att0.shape}）")

    return data


if __name__ == "__main__":
    part1_hand_single_head()
    diff = part3_compare_mha()
    data = part4_train_compare()
    print("\n全部完成 ✅")
