#!/usr/bin/env python3
"""
17-ablation.py — 第 17 课《Transformer 块》消融实验
依赖: torch + numpy（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 17-ablation.py

目的：同样一个微型 GPT（1 层块），分别拆掉两个组件，看训练曲线：
  A. full      ：完整块（LayerNorm + 残差 + 注意力 + FFN）
  B. no_ln     ：把 LayerNorm 换成恒等（其他不变）
  C. no_res    ：把残差连接去掉（其他不变）
在莎士比亚字符数据上各训 300 步，比较 train loss 曲线。
所有输出均为 Mac mini（MPS）真实运行结果。
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(17)
np.random.seed(17)

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
VOCAB = 65            # shakespeare_char 字符表大小
N_EMBD = 128          # embedding 维度
N_HEAD = 4            # 头数
N_LAYER = 3           # 块数量（3 层才能放大"没有残差/LN 会怎样"）
BLOCK = 256           # 上下文长度
BATCH = 32            # batch size
STEPS = 300           # 训练步数
LR = 3e-4             # 学习率
EVAL_EVERY = 20       # 每 20 步记录一次 loss
OUT_JSON = "17-ablation-loss.json"

NANOGPT = os.path.expanduser("~/projects/main-agent/nanoGPT")
DATA = os.path.join(NANOGPT, "data/shakespeare_char")


def load_data():
    with open(os.path.join(DATA, "input.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    # 65 字符：直接按 meta.pkl 的顺序建索引（与 nanoGPT 一致）
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], stoi


train_ids, val_ids, _ = load_data()
print(f"数据：train {len(train_ids):,} tokens / val {len(val_ids):,} tokens")


def get_batch(ids, device):
    ix = torch.randint(len(ids) - BLOCK - 1, (BATCH,))
    x = torch.stack([torch.from_numpy(ids[i:i + BLOCK]) for i in ix])
    y = torch.stack([torch.from_numpy(ids[i + 1:i + BLOCK + 1]) for i in ix])
    return x.to(device), y.to(device)


# ─────────────────────────────────────────────
# 微型 GPT：N 层块，可开关 LayerNorm / 残差
# ─────────────────────────────────────────────
class MiniBlock(nn.Module):
    """单个 Transformer 块（带开关）——本课主角"""
    def __init__(self, use_ln=True, use_res=True):
        super().__init__()
        self.use_ln = use_ln
        self.use_res = use_res
        if use_ln:
            self.ln_1 = nn.LayerNorm(N_EMBD)
            self.ln_2 = nn.LayerNorm(N_EMBD)
        else:
            self.ln_1 = nn.Identity()   # 拆掉 LayerNorm
            self.ln_2 = nn.Identity()
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, batch_first=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )
        # 手动因果 mask：上三角 -inf（禁止看未来）
        self.register_buffer("attn_mask", torch.triu(torch.full((BLOCK, BLOCK), float("-inf")), diagonal=1))

    def forward(self, x):
        B, T = x.size(0), x.size(1)
        a = self.ln_1(x)
        a, _ = self.attn(a, a, a, attn_mask=self.attn_mask[:T, :T], need_weights=False)
        x = x + a if self.use_res else a          # 残差 1

        m = self.ln_2(x)
        m = self.mlp(m)
        x = x + m if self.use_res else m          # 残差 2
        return x


class MiniGPT(nn.Module):
    def __init__(self, use_ln=True, use_res=True):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, N_EMBD)
        self.wpe = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.ModuleList([MiniBlock(use_ln=use_ln, use_res=use_res) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        tok = self.wte(idx)                       # (B, T, C)
        pos = self.wpe(torch.arange(T, device=idx.device))  # (T, C)
        x = tok + pos
        for block in self.blocks:                 # N 个块依次加工
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def run_variant(name, use_ln, use_res, device):
    torch.manual_seed(17)
    model = MiniGPT(use_ln=use_ln, use_res=use_res).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    losses = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        xb, yb = get_batch(train_ids, device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, VOCAB), yb.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            losses.append({"step": step, "train_loss": round(float(loss.item()), 4)})
            dt = time.time() - t0
            print(f"[{name}] step {step:3d}  train_loss {float(loss.item()):.4f}  ({dt:.0f}s elapsed)")
    # 最终 val loss（3 个 batch 平均）
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(3):
            xv, yv = get_batch(val_ids, device)
            logits = model(xv)
            val_losses.append(F.cross_entropy(logits.view(-1, VOCAB), yv.view(-1)).item())
    val = float(np.mean(val_losses))
    print(f"[{name}] 最终 val_loss {val:.4f}")
    return losses, val


def main():
    print("=" * 60)
    print("第 17 课消融实验：拆掉 LayerNorm / 残差，会发生什么？")
    print(f"配置：{N_LAYER} 层块, emb={N_EMBD}, head={N_HEAD}, block={BLOCK}, batch={BATCH}, steps={STEPS}")
    print("=" * 60)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"设备：{device}\n")

    results = {}
    for name, use_ln, use_res in [
        ("full",   True,  True),   # 完整块
        ("no_ln",  False, True),   # 拆 LayerNorm
        ("no_res", True,  False),  # 拆残差
    ]:
        losses, val = run_variant(name, use_ln, use_res, device)
        results[name] = {"losses": losses, "val_loss": round(val, 4)}

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {OUT_JSON}")
    for name, r in results.items():
        print(f"  {name:8s} 最终 train_loss {r['losses'][-1]['train_loss']:.4f}  val_loss {r['val_loss']:.4f}")


if __name__ == "__main__":
    main()
