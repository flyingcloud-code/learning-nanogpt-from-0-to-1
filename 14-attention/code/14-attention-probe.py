#!/usr/bin/env python3
"""
14-attention-probe.py — 第 14 课《Attention 直觉》实验脚本（探测真实模型注意力）
依赖: torch + numpy + matplotlib（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 14-attention-probe.py
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
# 1. 加载真实训练好的模型（1000 步，val loss 1.52）
# ─────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = nano_model.GPTConfig(**ckpt["model_args"])
gpt = nano_model.GPT(cfg)
gpt.load_state_dict(ckpt["model"])
gpt.eval()
print("已加载 checkpoint：", CKPT)
print(f"  配置 n_layer={cfg.n_layer} n_head={cfg.n_head} n_embd={cfg.n_embd} block_size={cfg.block_size}")
print(f"  训练步数 iter_num={ckpt['iter_num']}  best_val_loss={float(ckpt['best_val_loss']):.4f}")

# ─────────────────────────────────────────────
# 2. 准备一段真实莎士比亚文本（与训练同分布）
# ─────────────────────────────────────────────
DATA = os.path.join(NANOGPT, "data/shakespeare_char")
with open(os.path.join(DATA, "input.txt"), "r", encoding="utf-8") as f:
    full_text = f.read()

# 从数据目录读 meta.pkl 拿到 字符→编号 映射
import pickle
with open(os.path.join(DATA, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]

sample_text = (
    "ROMEO: But soft, what light through yonder window breaks? "
    "It is the east, and Juliet is the sun."
)
print("\n探测文本（真实莎士比亚台词，模型见过同分布文本）：")
print("  ", sample_text)

# 取一段字符，截到 block_size 以内
ids = [stoi[c] for c in sample_text if c in stoi]
T = len(ids)
print(f"  token 数 T={T}（字符级）")

x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, T)

# ─────────────────────────────────────────────
# 3. 用 hook 抓每一层每一头的注意力权重
#    nanoGPT 的 CausalSelfAttention 用 flash attention，抓不到 att
#    这里临时把 flash 关掉，走手写路径，把 softmax 后的 att 存下来
# ─────────────────────────────────────────────
captured = {}  # layer_idx -> (heads, T, T)

orig_forward = nano_model.CausalSelfAttention.forward

def patched_forward(self, x, layer_idx):
    B, Tt, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, Tt, self.n_head, C // self.n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :Tt, :Tt] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    captured[layer_idx] = att[0].detach().cpu().numpy()  # (n_head, T, T)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, Tt, C)
    y = self.resid_dropout(self.c_proj(y))
    return y

# 给每个 CausalSelfAttention 绑定独立的 patched forward，记录自己的层号
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
    logits = gpt(x)  # 前向一次，顺便验证能出预测

# ─────────────────────────────────────────────
# 4. 分析：每层的注意力“聚焦程度”
#    熵（entropy）：权重越集中在少数位置 → 熵越低
#    平均距离：注意力投向前方多远（局部 vs 全局）
# ─────────────────────────────────────────────
print("\n" + "=" * 66)
print("每一层注意力的统计（真实权重，来自上面那段文本）")
print("=" * 66)
print(f"{'层':>3} {'头数':>4} {'平均熵':>10} {'平均距离':>10}  解读")
for li in range(cfg.n_layer):
    att = captured[li]  # (n_head, T, T)
    entropies = []
    distances = []
    for h in range(cfg.n_head):
        w = att[h]  # (T, T)
        for i in range(1, T):
            p = w[i, :i+1]
            entropies.append(-np.sum(p * np.log(p + 1e-12)))
            dist = np.sum(p * (np.arange(i+1))[::-1] * 0 + 0)  # placeholder
        # 距离：当前位置 i 到被关注位置 j 的距离 i - j，按权重加权
        for i in range(1, T):
            p = w[i, :i+1]
            d = np.sum(p * (i - np.arange(i+1)))
            distances.append(d)
    avg_ent = np.mean(entropies)
    avg_dist = np.mean(distances)
    hint = ""
    if avg_ent < 0.8:
        hint = "← 注意力非常集中"
    elif avg_ent > 2.5:
        hint = "← 注意力分散（平均主义）"
    print(f"{li:>3} {cfg.n_head:>4} {avg_ent:>10.3f} {avg_dist:>10.2f}  {hint}")

# ─────────────────────────────────────────────
# 5. 打印一个具体 head 的注意力权重（第 0 层 head 0，最后 6 个 token 看谁）
# ─────────────────────────────────────────────
print("\n" + "=" * 66)
print("第 0 层 head 0：每个 token 把注意力分给谁（只显示最近 8 个 token）")
print("=" * 66)
att = captured[0][0]
chars = [itos[i] for i in ids]
print("  token 列表：" + "".join(chars))
for i in range(T - 8, T):
    p = att[i, :i+1]
    top3 = np.argsort(p)[::-1][:3]
    s = "  ".join(f"'{chars[j]}'={p[j]:.2f}" for j in top3)
    print(f"  token[{i:2d}] '{chars[i]}' 最关注: {s}")

# ─────────────────────────────────────────────
# 6. 生成一段文本，证明模型真的会“接话”
# ─────────────────────────────────────────────
print("\n" + "=" * 66)
print("模型生成演示（temperature=0.8，前 12 个字符为种子）")
print("=" * 66)
seed = "ROMEO: But "
seed_ids = [stoi[c] for c in seed if c in stoi]
with torch.no_grad():
    idx = torch.tensor(seed_ids, dtype=torch.long).unsqueeze(0)
    for _ in range(60):
        idx_cond = idx if idx.size(1) <= cfg.block_size else idx[:, -cfg.block_size:]
        logits, _ = gpt(idx_cond)  # 推理模式返回 (logits, loss) 元组
        logits = logits[:, -1, :] / 0.8
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
gen = "".join(itos[i] for i in idx[0].tolist())
print("  种子:", seed)
print("  生成:", gen)
print("\n实验完成。heatmap 生成见 14-make-heatmaps.py")
