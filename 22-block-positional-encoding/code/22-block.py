#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 22 课：手搓 Transformer Block + 位置编码
=============================================
不用 nn.LayerNorm / nn.MultiheadAttention / nn.TransformerEncoderLayer，
把 LayerNorm、FFN、位置编码、残差全部手写出来，组装成 Transformer Block，
再和官方 API 对拍，最后训练两个 mini GPT（学习式 vs 正弦式位置编码），
做"位置编码能不能外推"的真实实验。

运行（Mac mini / Apple Silicon，torch 2.12.1）：
    ~/projects/main-agent/nanoGPT/.venv/bin/python 22-block.py

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

# ---------------------------------------------------------------------------
# Part 0：四个手搓零件
# ---------------------------------------------------------------------------

# ---------- 零件 1：LayerNorm（第 17 课讲过：对齐刻度） ----------
class MyLayerNorm(nn.Module):
    """对一个 token 的整个向量做归一化：减均值、除标准差，再学回来的缩放平移。"""

    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))   # 可学习的缩放（初始 1 = 不缩放）
        self.bias = nn.Parameter(torch.zeros(ndim))    # 可学习的平移（初始 0 = 不平移）
        self.eps = eps                                 # 防除 0 的小数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)                            # 每个 token 自己的均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)              # 方差
        x_norm = (x - mean) / torch.sqrt(var + self.eps)               # 对齐刻度
        return self.weight * x_norm + self.bias                        # 微调回来


# ---------- 零件 2：因果多头注意力（第 21 课手搓版，直接搬来用） ----------
class MyCausalSelfAttention(nn.Module):
    def __init__(self, ndim, n_head, max_block):
        super().__init__()
        assert ndim % n_head == 0
        self.n_head = n_head
        self.head_dim = ndim // n_head
        self.c_attn = nn.Linear(ndim, 3 * ndim, bias=False)   # QKV 合并投影
        self.c_proj = nn.Linear(ndim, ndim, bias=False)       # 输出投影
        # 因果掩码：下三角，只允许看左边（第 16 课主角）
        self.register_buffer("mask", torch.tril(torch.ones(max_block, max_block))
                             .view(1, 1, max_block, max_block))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)              # (B,T,3C) 切成 Q/K/V
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,T,hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)     # 打分
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # 掩码
        att = F.softmax(att, dim=-1)                          # 归一
        y = att @ v                                           # 混合
        y = y.transpose(1, 2).contiguous().view(B, T, C)      # 拼回
        return self.c_proj(y)


# ---------- 零件 3：FFN（第 17 课讲过：先放大 4 倍想，再缩回来） ----------
class MyMLP(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        # 注：nanoGPT 默认开 bias；这里为了和官方 API 对拍时参数一一对应，统一用 bias=False
        self.c_fc = nn.Linear(ndim, 4 * ndim, bias=False)      # 放大 4 倍
        self.gelu = nn.GELU()                                  # 非线性激活
        self.c_proj = nn.Linear(4 * ndim, ndim, bias=False)    # 缩回原维度

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


# ---------- 零件 4a：学习式位置编码（GPT 用的就是查表） ----------
class LearnedPosEmbedding(nn.Module):
    """查表式位置编码。表的大小 = 训练时的 block_size（64 行）。
    位置 64 之后没有表项——想外推？查表越界。
    注意：MPS 上 nn.Embedding 越界不报错、静默返回全 0 向量（CPU 会抛 IndexError），
    所以这里显式检查，否则模型会在"位置失忆"的状态下偷偷跑下去。"""

    def __init__(self, block_size, ndim):
        super().__init__()
        self.wpe = nn.Embedding(block_size, ndim)     # 每个位置一个向量，只有 64 行

    def forward(self, pos):
        if pos.max() >= self.wpe.num_embeddings:
            raise IndexError(
                f"位置 {pos.max().item()} 超过训练长度 {self.wpe.num_embeddings}，查表越界！")
        return self.wpe(pos)


# ---------- 零件 4b：正弦式位置编码（论文《Attention Is All You Need》公式） ----------
class SinusoidalPosEmbedding(nn.Module):
    """固定公式生成位置向量，不需要训练，任意位置都能算。"""

    def __init__(self, ndim, max_pos=256):
        super().__init__()
        pe = torch.zeros(max_pos, ndim)
        position = torch.arange(0, max_pos).unsqueeze(1).float()      # (max_pos, 1)
        div = torch.exp(torch.arange(0, ndim, 2).float()
                        * (-math.log(10000.0) / ndim))                # 频率衰减
        pe[:, 0::2] = torch.sin(position * div)   # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div)   # 奇数维用 cos
        self.register_buffer("pe", pe)            # buffer：随模型保存但不更新

    def forward(self, pos):
        return self.pe[pos]


# ---------- 零件 5：Block = 开会 + 干活 + 留后路 + 对齐刻度 ----------
class MyBlock(nn.Module):
    """nanoGPT 的 Block 全部逻辑就下面 4 行。"""

    def __init__(self, ndim, n_head, max_block):
        super().__init__()
        self.ln_1 = MyLayerNorm(ndim)                         # 对齐刻度①
        self.attn = MyCausalSelfAttention(ndim, n_head, max_block)  # 开会（注意力）
        self.ln_2 = MyLayerNorm(ndim)                         # 对齐刻度②
        self.mlp = MyMLP(ndim)                                # 干活（FFN）

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))    # 第一段：对齐 → 开会 → 加回原输入
        x = x + self.mlp(self.ln_2(x))     # 第二段：对齐 → 干活 → 加回原输入
        return x


# ---------------------------------------------------------------------------
# Part 0.5：手搓 mini GPT（把 5 个零件装起来）
# ---------------------------------------------------------------------------

class MyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd,
                 pos_kind="learned", max_pos=256):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)        # 词嵌入（第 11 课）
        if pos_kind == "learned":
            self.pos_emb = LearnedPosEmbedding(block_size, n_embd)
        else:
            self.pos_emb = SinusoidalPosEmbedding(n_embd, max_pos)
        self.blocks = nn.ModuleList(
            [MyBlock(n_embd, n_head, max_pos) for _ in range(n_layer)])
        self.ln_f = MyLayerNorm(n_embd)                        # 最后的对齐
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # 输出词表概率

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok = self.tok_emb(idx)                                # (B,T,C) 查词表
        pos = torch.arange(T, device=idx.device)               # 位置 0,1,2,...
        x = tok + self.pos_emb(pos)                            # 词 + 位置（第 17 课）
        for block in self.blocks:
            x = block(x)                                       # 一层层过车间
        x = self.ln_f(x)
        logits = self.lm_head(x)                               # (B,T,vocab)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """逐 token 自回归生成（第 12 课：next token prediction）"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)                          # 一次前向
            logits = logits[:, -1, :] / temperature             # 只看最后一个位置
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率采样
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# 数据加载（和 nanoGPT 相同的预处理格式）
# ---------------------------------------------------------------------------

def load_shakespeare():
    data_dir = os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char")
    train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    meta = np.load(os.path.join(data_dir, "meta.pkl"), allow_pickle=True)
    if isinstance(meta, dict):
        vocab_size = meta["vocab_size"]
        itos = meta["itos"]
    else:
        vocab_size = int(meta)
        itos = None
    return train, val, vocab_size, itos


def get_batch(train, val, block_size, batch_size, device, split="train"):
    data = train if split == "train" else val
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64))
                     for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
                     for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Part 1：对拍①——手搓 LayerNorm vs nn.LayerNorm
# ---------------------------------------------------------------------------

def part1_compare_layernorm():
    print("\n" + "=" * 60)
    print("Part 1：手搓 LayerNorm vs nn.LayerNorm")
    print("=" * 60)
    torch.manual_seed(0)
    x = torch.randn(2, 8, 128, device=DEVICE)
    mine = MyLayerNorm(128).to(DEVICE)
    official = nn.LayerNorm(128).to(DEVICE)
    official.load_state_dict(mine.state_dict())   # 复制同一份参数
    with torch.no_grad():
        y_mine = mine(x)
        y_off = official(x)
    diff = (y_mine - y_off).abs().max().item()
    print(f"输入: {tuple(x.shape)}  输出: {tuple(y_mine.shape)}")
    print(f"最大绝对误差: {diff:.3e}")
    assert diff < 1e-5, "LayerNorm 对拍失败"
    print("✅ 对拍通过：手搓 LayerNorm 与 nn.LayerNorm 一致")
    return diff


# ---------------------------------------------------------------------------
# Part 2：对拍②——手搓 Block vs nn.TransformerEncoderLayer（官方块）
# ---------------------------------------------------------------------------

def part2_compare_block():
    print("\n" + "=" * 60)
    print("Part 2：手搓 Block vs nn.TransformerEncoderLayer")
    print("=" * 60)
    torch.manual_seed(1)
    ndim, n_head, T = 128, 4, 16
    x = torch.randn(2, T, ndim, device=DEVICE)

    my_block = MyBlock(ndim, n_head, max_block=256).to(DEVICE)
    # 官方块：norm_first=True 就是 pre-LN 结构（第 17 课彩蛋）
    off_block = nn.TransformerEncoderLayer(
        d_model=ndim, nhead=n_head, dim_feedforward=4 * ndim,
        dropout=0.0, activation=F.gelu, layer_norm_eps=1e-5,
        batch_first=True, norm_first=True).to(DEVICE)

    # 把手搓参数复制给官方层（官方 attention/mlp 的 bias 置 0，因为手搓无 bias）
    with torch.no_grad():
        off_block.norm1.load_state_dict(my_block.ln_1.state_dict())
        off_block.norm2.load_state_dict(my_block.ln_2.state_dict())
        off_block.self_attn.in_proj_weight.copy_(my_block.attn.c_attn.weight)
        off_block.self_attn.in_proj_bias.zero_()
        off_block.self_attn.out_proj.weight.copy_(my_block.attn.c_proj.weight)
        off_block.self_attn.out_proj.bias.zero_()
        off_block.linear1.weight.copy_(my_block.mlp.c_fc.weight)
        off_block.linear1.bias.zero_()
        off_block.linear2.weight.copy_(my_block.mlp.c_proj.weight)
        off_block.linear2.bias.zero_()

    with torch.no_grad():
        y_mine = my_block(x)
        # 官方层要传加法掩码（float 版，第 21 课踩过的坑）
        mask = torch.zeros(T, T, dtype=torch.float32, device=DEVICE)
        mask.masked_fill_(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEVICE)),
                          float("-inf"))
        y_off = off_block(x, src_mask=mask)

    diff = (y_mine - y_off).abs().max().item()
    print(f"输入: {tuple(x.shape)}  输出: {tuple(y_mine.shape)}")
    print(f"最大绝对误差: {diff:.3e}")
    assert diff < 1e-4, "Block 对拍失败"
    print("✅ 对拍通过：手搓 Block 与 nn.TransformerEncoderLayer 一致")
    return diff


# ---------------------------------------------------------------------------
# Part 3：训练两个 mini GPT——学习式 vs 正弦式位置编码
# ---------------------------------------------------------------------------

def train_model(pos_kind, train, val, vocab_size, steps=600, log_every=100,
                block_size=64, batch_size=16, n_layer=2, n_head=4, n_embd=128):
    print(f"\n--- 训练【{pos_kind} 位置编码】 ---")
    model = MyGPT(vocab_size, block_size, n_layer, n_head, n_embd,
                  pos_kind=pos_kind).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(train, val, block_size, batch_size, DEVICE, "train")
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == 1:
            losses.append({"step": step, "loss": round(loss.item(), 4)})
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    dt = time.time() - t0
    print(f"  用时 {dt:.1f}s  ({dt / steps * 1000:.1f} ms/step)")
    return model, losses


def evaluate_length(model, val, vocab_size, lengths, device,
                    batch_size=8, block_size=64):
    """在不同序列长度下评估 val loss——外推实验的核心。
    学习式模型查表越界时捕获异常，返回 None（表示物理上无法外推）。"""
    model.eval()
    results = {}
    with torch.no_grad():
        for L in lengths:
            try:
                x, y = get_batch(train=None, val=val, block_size=L,
                                 batch_size=batch_size, device=device, split="val")
                logits, loss = model(x, y)
                results[L] = round(loss.item(), 4)
            except (IndexError, RuntimeError) as e:
                results[L] = None          # 查表越界：长度超过训练长度
                print(f"   长度 {L}: 学习式位置编码查表越界 ({type(e).__name__})")
    model.train()
    return results


# ---------------------------------------------------------------------------
# Part 4：生成文本（用训练好的模型）
# ---------------------------------------------------------------------------

def generate_text(model, meta_chars, start="KING ", n=300, temperature=0.8):
    stoi = {ch: i for i, ch in enumerate(meta_chars)}
    idx = torch.tensor([[stoi[c] for c in start]], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, n, temperature=temperature)[0].tolist()
    return start + "".join(meta_chars[i] for i in out[len(start):])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    train, val, vocab_size, itos = load_shakespeare()
    print(f"数据: train {len(train)} tokens, val {len(val)} tokens, 词表 {vocab_size}")

    # 字符表：从 meta 的 itos 取（第 13 课：tokenizer 输出）
    if itos is None:
        chars = "".join(chr(i) for i in range(vocab_size))
    else:
        chars = "".join(itos[i] for i in range(vocab_size))

    # 对拍
    diff_ln = part1_compare_layernorm()
    diff_block = part2_compare_block()

    # 训练两个模型（同一个种子、同一份数据顺序，只有位置编码不同）
    steps = 600
    model_learned, loss_learned = train_model(
        "learned", train, val, vocab_size, steps=steps)
    model_sin, loss_sin = train_model(
        "sinusoidal", train, val, vocab_size, steps=steps)

    # 外推评估：训练只见过 64 个位置，测更长的序列会怎样？
    print("\n--- 外推评估：训练 block_size=64，测更长序列 ---")
    lengths = [16, 32, 48, 64, 80, 96, 128, 192, 256]
    ev_learned = evaluate_length(model_learned, val, vocab_size, lengths, DEVICE)
    ev_sin = evaluate_length(model_sin, val, vocab_size, lengths, DEVICE)
    print(f"{'长度':>6} {'学习式':>10} {'正弦式':>8}")
    for L in lengths:
        lv = ev_learned[L]
        lv_s = f"{lv:.3f}" if lv is not None else "越界!"
        print(f"{L:>6} {lv_s:>10} {ev_sin[L]:>8.3f}")

    # 生成文本（学习式模型）
    text = generate_text(model_learned, chars)
    print("\n--- 学习式模型生成（600 步后） ---")
    print(text[:400])

    # 保存数据供画图
    data = {
        "diff_layernorm": diff_ln,
        "diff_block": diff_block,
        "loss_learned": loss_learned,
        "loss_sin": loss_sin,
        "extrapolate_learned": {str(k): v for k, v in ev_learned.items()},
        "extrapolate_sin": {str(k): v for k, v in ev_sin.items()},
        "generated": text,
        "steps": steps,
    }
    out = os.path.join(BASE, "chart-data.json")
    with open(out, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n数据已保存: {out}")

    # 打印对拍结论汇总
    print("\n" + "=" * 60)
    print("结论汇总")
    print("=" * 60)
    print(f"LayerNorm 对拍最大误差: {diff_ln:.3e}")
    print(f"Block 对拍最大误差:    {diff_block:.3e}")
    print(f"学习式 600 步最终 loss: {loss_learned[-1]['loss']}")
    print(f"正弦式 600 步最终 loss: {loss_sin[-1]['loss']}")
    l_256 = ev_learned[256]
    print(f"外推 64→256 学习式: {ev_learned[64]} → {'查表越界' if l_256 is None else l_256}")
    print(f"外推 64→256 正弦式: {ev_sin[64]} → {ev_sin[256]}")


if __name__ == "__main__":
    main()
