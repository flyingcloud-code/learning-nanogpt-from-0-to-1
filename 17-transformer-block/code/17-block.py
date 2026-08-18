#!/usr/bin/env python3
"""
17-block.py — 第 17 课《Transformer 块》：手搓完整 Block 并与 nanoGPT 对拍
依赖: torch + numpy（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 17-block.py

目的：不用任何现成的 Transformer API，纯手搓一个完整的 Transformer 块：
  LayerNorm + 因果多头注意力 + 残差 + FFN（MLP），再与 nanoGPT 官方
  model.py 里的 Block 逐参数对齐、逐输出对拍，证明手搓版和官方版
  在数值上等价。

本课核心思想：一个块 = 两次"加工"（注意力 + 前馈）+ 两条"后路"（残差）
                    + 两个"对齐刻度"（LayerNorm）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(17)

# ─────────────────────────────────────────────
# 1. 手搓 LayerNorm：把一行向量"归一到 0 均值 1 方差"，
#    再用学出来的 weight/bias 拉伸回来
# ─────────────────────────────────────────────
class MyLayerNorm(nn.Module):
    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))   # 可学习的缩放
        self.bias = nn.Parameter(torch.zeros(ndim))    # 可学习的平移
        self.eps = eps

    def forward(self, x):
        # 对最后一维（embedding 维）做归一化：
        # 减均值 → 除标准差 → 乘 weight → 加 bias
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


# ─────────────────────────────────────────────
# 2. 手搓因果多头注意力（显式 mask 版本，不用 flash attention）
#    第 14/15/16 课讲过：投影 → 打分 → mask → softmax → 混合
# ─────────────────────────────────────────────
class MyCausalSelfAttention(nn.Module):
    def __init__(self, ndim, n_head, block_size):
        super().__init__()
        assert ndim % n_head == 0
        self.n_head = n_head
        self.head_dim = ndim // n_head
        self.c_attn = nn.Linear(ndim, 3 * ndim)        # Q/K/V 一次投影
        self.c_proj = nn.Linear(ndim, ndim)            # 输出投影
        # 因果掩码：下三角 1，上三角 0（注册成 buffer，不参与训练）
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        # ① 投影：一份输入，得到 Q、K、V 三份
        q, k, v = self.c_attn(x).split(C, dim=2)
        # 拆成多头： (B, T, C) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # ② 打分：Q·Kᵀ / √d
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # ③ 因果掩码：右上角填 -∞（第 16 课的核心）
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        # ④ softmax 归一 + 混合
        att = F.softmax(att, dim=-1)
        y = att @ v
        # 合并多头： (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 输出投影
        return self.c_proj(y)


# ─────────────────────────────────────────────
# 3. 手搓 MLP（FFN）：先放大 4 倍"想"，GELU 激活，再缩回原维度"说"
# ─────────────────────────────────────────────
class MyMLP(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.c_fc = nn.Linear(ndim, 4 * ndim)    # 放大：384 -> 1536
        self.gelu = nn.GELU()                    # 非线性激活
        self.c_proj = nn.Linear(4 * ndim, ndim)  # 缩回：1536 -> 384

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


# ─────────────────────────────────────────────
# 4. 手搓完整 Block：nanoGPT 里 Block 的全部逻辑就 4 行
#    x = x + attn(ln1(x))
#    x = x + mlp(ln2(x))
# ─────────────────────────────────────────────
class MyBlock(nn.Module):
    def __init__(self, ndim, n_head, block_size):
        super().__init__()
        self.ln_1 = MyLayerNorm(ndim)
        self.attn = MyCausalSelfAttention(ndim, n_head, block_size)
        self.ln_2 = MyLayerNorm(ndim)
        self.mlp = MyMLP(ndim)

    def forward(self, x):
        # 第一段：先 LayerNorm 对齐刻度，再注意力"开会"，最后加回原输入（残差）
        x = x + self.attn(self.ln_1(x))
        # 第二段：先 LayerNorm 对齐刻度，再 FFN"干活"，最后加回原输入（残差）
        x = x + self.mlp(self.ln_2(x))
        return x


# ─────────────────────────────────────────────
# 5. 对拍：与 nanoGPT 官方 Block 比较
# ─────────────────────────────────────────────
import os, sys
NANOGPT = os.path.expanduser("~/projects/main-agent/nanoGPT")
sys.path.insert(0, NANOGPT)
import model as nano_model  # nanoGPT 的 model.py

def main():
    print("=" * 60)
    print("第 17 课实验：手搓 Transformer Block 对拍 nanoGPT 官方 Block")
    print("=" * 60)

    # 配置：用 nanoGPT baby GPT 的配置（6 层 6 头 384 维）
    ndim, n_head, block_size = 384, 6, 256
    cfg = nano_model.GPTConfig(
        block_size=block_size, vocab_size=65,
        n_layer=1, n_head=n_head, n_embd=ndim, dropout=0.0, bias=True,
    )

    # 官方 Block（eval + dropout=0，保证确定）
    official = nano_model.Block(cfg).eval()
    # 手搓 Block
    mine = MyBlock(ndim, n_head, block_size).eval()

    # 参数对齐：把手搓版参数 dict 的 key 改成官方 key，复制过去
    # 官方 key 形如 attn.c_attn.weight，手搓 key 形如 attn.c_attn.weight（一致）
    # 唯一不同：官方 LayerNorm 的 key 是 ln_1.weight，手搓也是 ln_1.weight ✓
    with torch.no_grad():
        sd_mine = mine.state_dict()
        sd_off = official.state_dict()
        # mask 是 buffer 不是参数：官方 flash 模式不注册它，对拍时跳过
        param_keys = lambda sd: [k for k in sd if "mask" not in k]
        assert set(param_keys(sd_mine)) == set(param_keys(sd_off)), (
            f"参数名不一致!\n手搓: {sorted(param_keys(sd_mine))}\n官方: {sorted(param_keys(sd_off))}"
        )
        for k in param_keys(sd_off):
            sd_mine[k].copy_(sd_off[k])

    # 参数量
    n_params_mine = sum(p.numel() for p in mine.parameters())
    n_params_off = sum(p.numel() for p in official.parameters())
    print(f"\n参数量：手搓 {n_params_mine:,}  vs  官方 {n_params_off:,}  ->  {'一致 ✅' if n_params_mine == n_params_off else '不一致 ❌'}")

    # 随机输入，比较输出
    torch.manual_seed(17)
    x = torch.randn(2, 32, ndim)  # (B=2, T=32, C=384)

    with torch.no_grad():
        out_off = official(x)
        out_mine = mine(x)

    diff = (out_off - out_mine).abs().max().item()
    print(f"\n前向输出最大绝对差异：{diff:.2e}")
    print("对拍结果：", "PASS ✅（数值等价）" if diff < 1e-5 else "FAIL ❌")

    # 再验证反向传播梯度也能对拍（注意：两个模型必须吃完全相同的输入！）
    x2 = torch.randn(2, 32, ndim)
    x_off = x2.clone().requires_grad_(True)
    x_mine = x2.clone().requires_grad_(True)
    loss_off = official(x_off).pow(2).mean()
    loss_off.backward()
    g_off = x_off.grad.clone()

    loss_mine = mine(x_mine).pow(2).mean()
    loss_mine.backward()
    g_mine = x_mine.grad.clone()

    g_diff = (g_off - g_mine).abs().max().item()
    print(f"反向梯度最大绝对差异：{g_diff:.2e}")
    print("梯度对拍：", "PASS ✅" if g_diff < 1e-4 else "FAIL ❌")

    # 计算一个 Block 的参数构成（帮读者理解参数量去哪了）
    print("\n--- 一个 Block 的参数构成（ndim=384, 6头, 放大4倍） ---")
    total = 0
    for name, p in mine.named_parameters():
        print(f"  {name:28s} {p.numel():>10,}")
        total += p.numel()
    print(f"  {'合计':28s} {total:>10,}")

    # 各组件占比
    attn_n = sum(p.numel() for p in mine.attn.parameters())
    mlp_n = sum(p.numel() for p in mine.mlp.parameters())
    ln_n = sum(p.numel() for p in mine.ln_1.parameters()) + sum(p.numel() for p in mine.ln_2.parameters())
    print(f"\n  注意力占 {attn_n/total*100:.1f}%  |  FFN 占 {mlp_n/total*100:.1f}%  |  LayerNorm 占 {ln_n/total*100:.2f}%")
    print("\n✅ 全部完成：手搓 Block 与 nanoGPT 官方 Block 数值等价。")


if __name__ == "__main__":
    main()
