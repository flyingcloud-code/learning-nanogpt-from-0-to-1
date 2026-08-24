#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 23 课：手搓 GPT 模型完整 forward
====================================
不用任何 nn.Transformer 高级 API，只用 nn.Linear / nn.Embedding / nn.GELU
这些最基础算子，把前 21/22 课手搓的零件（LayerNorm、注意力、MLP、Block）
正式拼装成完整 GPT：wte + wpe + blocks + ln_f + lm_head，一个不落。

三件事：
  1. forward 形状流 —— 打印每一步张量形状 (B,T) -> (B,T,384) -> ... -> (B,T,65)
  2. 参数量核对 —— 纸面上手算公式 vs torch 数出来，逐项对比（nanoGPT 配置 6/6/384）
  3. 训练 + 生成 —— 400 步训练微型 GPT，loss 真实下降，采样莎士比亚腔文本

运行（Mac mini / Apple Silicon，torch 2.12.1）：
    ~/projects/main-agent/nanoGPT/.venv/bin/python 23-gpt.py

依赖：torch, numpy（venv 已装）
数据：data/shakespeare_char（本系列第 19 课已备好，train 1,003,854 / val 111,540 tokens）
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
print("=" * 70)

# ============================================================
# 第 1 部分：手搓零件（第 21/22 课已逐个验证，这里原样复用）
# ============================================================

class MyLayerNorm(nn.Module):
    """手搓 LayerNorm：对齐刻度。weight 可学习缩放，bias 可学习平移。"""

    def __init__(self, ndim, bias=True, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))   # 初始 1 = 不缩放
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # 初始 0 = 不平移
        self.eps = eps                                 # 防除 0

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)                       # 每个 token 自己的均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)         # 总体方差
        x = (x - mean) / torch.sqrt(var + self.eps)               # 对齐刻度
        if self.bias is not None:
            x = x + self.bias
        return self.weight * x                                    # 微调回来


class MyCausalSelfAttention(nn.Module):
    """手搓多头因果注意力：投影 -> 打分 -> 掩码 -> softmax -> 混合（第 21 课）。"""

    def __init__(self, n_embd, n_head, max_block=1024):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)   # QKV 合并投影
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)       # 输出投影
        self.register_buffer("bias", torch.tril(torch.ones(1, 1, max_block, max_block)))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)                                      # (B,T,3C)
        q, k, v = qkv.split(C, dim=2)                             # 切三份
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,T,hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # 打分
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # 只看左边
        att = F.softmax(att, dim=-1)                              # 归一
        y = att @ v                                               # 加权混合
        y = y.transpose(1, 2).contiguous().view(B, T, C)          # 拼回头
        return self.c_proj(y)


class MyMLP(nn.Module):
    """手搓 FFN：放大 4 倍干活，再缩回原维度（第 22 课）。"""

    def __init__(self, n_embd, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)      # 384 -> 1536
        self.gelu = nn.GELU()                                     # 非线性
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)    # 1536 -> 384

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class MyBlock(nn.Module):
    """手搓 Transformer Block：开会 + 干活，各配一条残差旁路（第 22 课）。"""

    def __init__(self, n_embd, n_head, bias=True, max_block=1024):
        super().__init__()
        self.ln_1 = MyLayerNorm(n_embd, bias=bias)                # 对齐刻度①
        self.attn = MyCausalSelfAttention(n_embd, n_head, max_block)  # 开会
        self.ln_2 = MyLayerNorm(n_embd, bias=bias)                # 对齐刻度②
        self.mlp = MyMLP(n_embd, bias=bias)                       # 干活

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # 开会，残差加回原输入
        x = x + self.mlp(self.ln_2(x))    # 干活，残差加回原输入
        return x


# ============================================================
# 第 2 部分：今天的主角 —— 完整 GPT（5 块积木，一个不落）
# ============================================================

class MyGPT(nn.Module):
    """手搓 GPT：wte(查词表) + wpe(查位置) + blocks(开会干活) + ln_f(对齐) + lm_head(打分)。

    forward 的全部核心只有 6 行：
        tok_emb = self.wte(idx)                       # 1. 查表：字符 -> 向量
        pos_emb = self.wpe(pos)                       # 2. 查表：位置 -> 向量
        x = tok_emb + pos_emb                         # 3. 相加：词义 + 位置
        for block in self.blocks: x = block(x)        # 4. 穿过 N 个块
        x = self.ln_f(x)                              # 5. 最后对齐一次刻度
        logits = self.lm_head(x)                      # 6. 打分：每个字符的分数
    """

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd,
                 bias=False, tie_weights=True, max_block=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.wte = nn.Embedding(vocab_size, n_embd)          # 词嵌入表：65 行
        self.wpe = nn.Embedding(block_size, n_embd)          # 位置嵌入表：256 行
        self.blocks = nn.ModuleList([                        # N 个手搓块
            MyBlock(n_embd, n_head, bias=bias, max_block=max_block)
            for _ in range(n_layer)
        ])
        self.ln_f = MyLayerNorm(n_embd, bias=bias)           # 最终归一化
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # 打分头

        if tie_weights:
            # weight tying：打分矩阵和词表矩阵共享同一份权重（第 18 课）
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)                       # 初始化

    def _init_weights(self, module):
        """标准初始化：Linear/Embedding 用正态分布，残差投影按 1/sqrt(2N) 缩放。"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # 残差层的输出投影按 1/sqrt(2*n_layer) 缩小，训练更稳
        for name, p in module.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

    def forward(self, idx, targets=None):
        """idx: (B,T) 整型字符 id；targets: (B,T) 下一个字符 id。返回 logits 或 loss。"""
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度 {T} 超过训练长度 {self.block_size}"

        tok_emb = self.wte(idx)                              # (B,T,C) 查词表
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)                              # (T,C) 查位置表
        x = tok_emb + pos_emb                                # 词义 + 位置
        for block in self.blocks:                            # 穿过每个块
            x = block(x)
        x = self.ln_f(x)                                     # 最后对齐
        logits = self.lm_head(x)                             # (B,T,vocab) 打分

        loss = None
        if targets is not None:
            # 交叉熵：把 (B,T,vocab) 拉平 -> 每个位置都是"65 选 1"的分类问题
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """自回归采样：把已生成的 token 拼回输入，继续预测下一个。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]             # 只看最近 block_size 个
            logits, _ = self.forward(idx_cond)               # (B,T,vocab)
            logits = logits[:, -1, :] / temperature          # 取最后一个位置
            probs = F.softmax(logits, dim=-1)                # 变成概率
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率抽样
            idx = torch.cat((idx, idx_next), dim=1)          # 拼回去
        return idx

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 数据：莎士比亚字符级（第 19 课备好）
# ============================================================

def load_shakespeare(data_dir):
    with open(os.path.join(data_dir, "input.txt"), "r", encoding="utf-8") as f:
        data = f.read()
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    n = int(0.9 * len(data))
    train = torch.tensor([stoi[c] for c in data[:n]], dtype=torch.long)
    val = torch.tensor([stoi[c] for c in data[n:]], dtype=torch.long)
    return train, val, vocab_size, itos


def get_batch(train, val, block_size, batch_size, device, split="train"):
    data = train if split == "train" else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# Part 1：forward 形状流 —— 每一步张量长什么样？
# ============================================================

def part1_shape_flow():
    print("\n[Part 1] forward 形状流（完整配置 6 层 6 头 384 维）")
    vocab_size, block_size = 65, 256
    cfg = dict(vocab_size=vocab_size, block_size=block_size,
               n_layer=6, n_head=6, n_embd=384)
    model = MyGPT(**cfg).to(DEVICE)

    x = torch.randint(0, vocab_size, (4, 64), device=DEVICE)   # 模拟 batch=4, 64 个字符
    B, T = x.shape
    tok_emb = model.wte(x)
    print(f"  idx            : {tuple(x.shape)}   整型字符 id")
    print(f"  wte(idx)       : {tuple(tok_emb.shape)}   查词表")
    pos = torch.arange(0, T, dtype=torch.long, device=DEVICE)
    pos_emb = model.wpe(pos)
    print(f"  wpe(pos)       : {tuple(pos_emb.shape)}   查位置表")
    h = tok_emb + pos_emb
    print(f"  tok+pos        : {tuple(h.shape)}   相加")
    for i, blk in enumerate(model.blocks):
        h = blk(h)
        if i in (0, 2, 5):
            print(f"  block[{i}]      : {tuple(h.shape)}   穿过块（形状不变才能叠）")
    h = model.ln_f(h)
    print(f"  ln_f           : {tuple(h.shape)}   最后对齐")
    logits = model.lm_head(h)
    print(f"  lm_head        : {tuple(logits.shape)}   打分 65 个字符")
    print("  一句话：形状从 (B,T) 进，变 (B,T,384) 一路不变，最后变成 (B,T,65)。")


# ============================================================
# Part 2：参数量核对 —— 纸面公式 vs torch 数出来
# ============================================================

def part2_param_check():
    print("\n[Part 2] 参数量核对（nanoGPT 完整配置 6/6/384, block_size=256, vocab=65, bias=False）")
    vocab_size, block_size, n_layer, n_head, n_embd = 65, 256, 6, 6, 384
    model = MyGPT(vocab_size=vocab_size, block_size=block_size,
                  n_layer=n_layer, n_head=n_head, n_embd=n_embd).to(DEVICE)

    # ---- 纸面手算公式 ----
    def form(n_in, n_out):
        return n_in * n_out

    wte = form(vocab_size, n_embd)
    wpe = form(block_size, n_embd)
    per_attn = form(n_embd, 3 * n_embd) + form(n_embd, n_embd)     # c_attn + c_proj
    per_mlp = form(n_embd, 4 * n_embd) + form(4 * n_embd, n_embd)  # c_fc + c_proj
    per_ln = n_embd                                                 # LayerNorm: bias=False 只有 weight(384)
    per_block = per_attn + per_mlp + 2 * per_ln
    total_formula = wte + wpe + n_layer * per_block + n_embd        # + ln_f(384)；lm_head tied 共享 wte 不重复计

    rows = [
        ("wte（词表）", wte),
        ("wpe（位置）", wpe),
        (f"{n_layer} 个块 × c_attn", n_layer * form(n_embd, 3 * n_embd)),
        (f"{n_layer} 个块 × c_proj(attn)", n_layer * form(n_embd, n_embd)),
        (f"{n_layer} 个块 × c_fc", n_layer * form(n_embd, 4 * n_embd)),
        (f"{n_layer} 个块 × c_proj(mlp)", n_layer * form(4 * n_embd, n_embd)),
        (f"{n_layer} 个块 × ln_1/ln_2", n_layer * 2 * n_embd),
        ("ln_f（最终）", n_embd),
        ("lm_head（tied 共享）", 0),
    ]
    print("\n  纸面手算（公式 = 输入维 × 输出维）：")
    total_calc = 0
    for name, n in rows:
        total_calc += n
        print(f"    {name:<28} = {n:>10,}")
    print(f"    {'合计':<28} = {total_calc:>10,}")
    print(f"    （检查：纸面合计 {total_calc:,} vs 公式 {total_formula:,}，应相等）")

    # ---- torch 数出来（named_parameters 名称如 blocks.0.attn.c_attn.weight） ----
    groups = {"wte": 0, "wpe": 0, "attn": 0, "mlp": 0, "ln": 0, "ln_f": 0, "lm_head": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        parts = name.split(".")
        if parts[0] == "wte":
            groups["wte"] += n
        elif parts[0] == "wpe":
            groups["wpe"] += n
        elif parts[0] == "blocks":
            if any("attn" in part for part in parts):
                groups["attn"] += n
            elif any("mlp" in part for part in parts):
                groups["mlp"] += n
            elif any("ln" in part for part in parts):
                groups["ln"] += n
        elif parts[0] == "ln_f":
            groups["ln_f"] += n
        elif parts[0] == "lm_head":
            groups["lm_head"] += n
    total_torch = sum(groups.values())

    print("\n  torch 统计（model.named_parameters() 按组件分组）：")
    for k, v in groups.items():
        print(f"    {k:<12} = {v:>10,}")
    print(f"    {'合计':<12} = {total_torch:>10,}")

    print(f"\n  ✅ 纸面手算 {total_calc:,} == torch 统计 {total_torch:,}  {'一致！' if total_calc == total_torch else '不一致！！'}")
    # 存数据画图
    os.makedirs("chart-data", exist_ok=True)
    with open("chart-data/param-check.json", "w") as f:
        json.dump({"formula": {k: v for k, v in rows}, "torch": groups,
                   "total_formula": total_calc, "total_torch": total_torch}, f, indent=2)
    print("  chart-data/param-check.json 已保存")


# ============================================================
# Part 3：训练 + 生成 —— 手搓 GPT 真的能学吗？
# ============================================================

def part3_train_generate():
    print("\n[Part 3] 训练微型 GPT（2 层 4 头 128 维，400 步）+ 生成")
    data_dir = os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char")
    train, val, vocab_size, itos = load_shakespeare(data_dir)
    print(f"  数据: train {len(train):,} tokens, val {len(val):,} tokens, vocab {vocab_size}")

    cfg = dict(vocab_size=vocab_size, block_size=64,
               n_layer=2, n_head=4, n_embd=128)
    model = MyGPT(**cfg).to(DEVICE)
    n_params = model.count_parameters()
    print(f"  微型 GPT 参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    steps, log_every, batch_size = 400, 50, 32
    history = []
    t0 = time.time()

    for step in range(1, steps + 1):
        x, y = get_batch(train, val, cfg["block_size"], batch_size, DEVICE, "train")
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0:
            history.append((step, round(loss.item(), 4)))
            print(f"  step {step:>4}  loss {loss.item():.4f}  ({time.time()-t0:.1f}s)")
            t0 = time.time()

    with open("chart-data/loss-curve.json", "w") as f:
        json.dump(history, f)
    print("  chart-data/loss-curve.json 已保存")

    # ---- 生成 ----
    model.eval()
    stoi = {ch: i for i, ch in itos.items()}   # 反查表：字符 -> id
    start = torch.tensor([[stoi["K"], stoi["I"], stoi["N"], stoi["G"]]],
                         dtype=torch.long, device=DEVICE)
    out = model.generate(start, max_new_tokens=300, temperature=0.8)[0].tolist()
    text = "".join(itos[i] for i in out)
    print("\n  生成文本（从 'KING' 开始，temperature=0.8，300 字符）：")
    print("  " + text.replace("\n", "\\n"))


if __name__ == "__main__":
    part1_shape_flow()
    part2_param_check()
    part3_train_generate()
