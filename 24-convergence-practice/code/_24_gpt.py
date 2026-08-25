#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 24 课共享模块：手搓 GPT（第 23 课拼装的 5 块积木）+ 数据加载。
被 24-train.py 和 24-model-only-resume.py 共同 import，保证两个实验用同一个模型。
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLayerNorm(nn.Module):
    """手搓 LayerNorm：对齐刻度。weight 可学习缩放，bias 可学习平移。"""
    def __init__(self, ndim, bias=True, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is not None:
            x = x + self.bias
        return self.weight * x


class MyCausalSelfAttention(nn.Module):
    """手搓多头因果注意力：投影 -> 打分 -> 掩码 -> softmax -> 混合（第 21 课）。"""
    def __init__(self, n_embd, n_head, max_block=1024):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(1, 1, max_block, max_block)))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MyMLP(nn.Module):
    """手搓 FFN：放大 4 倍干活，再缩回原维度（第 22 课）。"""
    def __init__(self, n_embd, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class MyBlock(nn.Module):
    """手搓 Transformer Block：开会 + 干活，各配一条残差旁路（第 22 课）。"""
    def __init__(self, n_embd, n_head, bias=True, max_block=1024):
        super().__init__()
        self.ln_1 = MyLayerNorm(n_embd, bias=bias)
        self.attn = MyCausalSelfAttention(n_embd, n_head, max_block)
        self.ln_2 = MyLayerNorm(n_embd, bias=bias)
        self.mlp = MyMLP(n_embd, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MyGPT(nn.Module):
    """手搓 GPT：wte + wpe + blocks + ln_f + lm_head（第 23 课拼装）。"""

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd,
                 bias=False, tie_weights=True, max_block=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            MyBlock(n_embd, n_head, bias=bias, max_block=max_block)
            for _ in range(n_layer)
        ])
        self.ln_f = MyLayerNorm(n_embd, bias=bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        for name, p in module.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度 {T} 超过训练长度 {self.block_size}"
        tok_emb = self.wte(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


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
