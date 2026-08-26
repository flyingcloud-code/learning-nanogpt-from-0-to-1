#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 25 课：生成与评估——temperature / top-k / top-p / perplexity

运行命令（二选一）：
  # 方式 A：加载已训练 checkpoint（本课文章用的 1000 步莎士比亚模型）
  python 25-generate.py --ckpt /path/to/ckpt.pt

  # 方式 B：没有 checkpoint？快速训 400 步小模型再玩采样（约 1 分钟）
  python 25-generate.py --quick-train

依赖：torch / numpy / matplotlib（本系列 venv 已装，torch 2.12.1）
数据：shakespeare_char（65 字符词表），--data-dir 可改
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 模型：本系列手搓 GPT（第 21/22/23 课拼装） ----------------

class MyLayerNorm(nn.Module):
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
    def __init__(self, n_embd, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class MyBlock(nn.Module):
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
    """wte + wpe + blocks + ln_f + lm_head，第 23 课拼装。"""

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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ---------------- 数据 ----------------

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


# ---------------- checkpoint ----------------

def remap_nanogpt_keys(sd):
    """nanoGPT 官方 model.py 的 key（transformer.h.0.xx）-> 本系列 MyGPT 的 key（blocks.0.xx）。
    attn.bias 是随 block_size 变大小的缓冲区，跳过，用模型自带的。"""
    out = {}
    for k, v in sd.items():
        if k.endswith("attn.bias"):
            continue
        k2 = k[len("transformer."):] if k.startswith("transformer.") else k
        if k2.startswith("h."):
            k2 = "blocks." + k2[2:]
        out[k2] = v
    return out


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["model_args"]
    model = MyGPT(vocab_size=args["vocab_size"], block_size=args["block_size"],
                  n_layer=args["n_layer"], n_head=args["n_head"], n_embd=args["n_embd"],
                  bias=args.get("bias", False), tie_weights=True).to(device)
    model.load_state_dict(remap_nanogpt_keys(ckpt["model"]), strict=False)
    model.eval()
    return model, ckpt


# ---------------- 本课核心：采样三件套 ----------------

def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    """给定模型对下一个字符的分数 logits（形状 [vocab]），按采样策略抽一个字符 id。
    三个旋钮按顺序工作：temperature 缩放 -> top-k 过滤 -> top-p 过滤 -> softmax -> 抽签。"""
    logits = logits / temperature                      # ① temperature：拧分布形状
    if top_k is not None:                              # ② top-k：只留概率最高的 k 个
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = float("-inf")         # 其余候选直接判死刑（-inf）
    if top_p is not None:                              # ③ top-p：累计概率到 p 为止
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum - F.softmax(sorted_logits, dim=-1) > top_p   # 去掉"加上它才超 p"的
        sorted_logits[mask] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)                 # ④ 变成概率分布
    return torch.multinomial(probs, num_samples=1)    # ⑤ 按概率抽签（不是取最大）


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    """自回归生成：每步把已生成的 token 喂回模型，用 sample_next 抽下一个。"""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]         # 只留最近 block_size 个（窗口）
        logits, _ = model(idx_cond)                   # 前向，拿最后位置的分数
        idx_next = sample_next(logits[0, -1, :], temperature, top_k, top_p)
        idx = torch.cat((idx, idx_next.unsqueeze(0)), dim=1)   # 拼到序列后面
    return idx


# ---------------- 评估：perplexity ----------------

@torch.no_grad()
def estimate_loss(model, val, block_size, batch_size, eval_iters=100, device="cpu"):
    """在 val 集上抽 eval_iters 批，平均交叉熵 loss。"""
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(None, val, block_size, batch_size, device, "val")
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.mean(losses))


# ---------------- 快速训练（没有 checkpoint 时的兜底） ----------------

def quick_train(train, val, vocab_size, device, steps=400, seed=1337):
    """训练一个 2 层 4 头 128 维小模型（410,368 参数），约 1 分钟，够玩采样。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MyGPT(vocab_size=vocab_size, block_size=64, n_layer=2, n_head=4,
                  n_embd=128, bias=False, tie_weights=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    block_size, batch_size = 64, 32
    for step in range(1, steps + 1):
        x, y = get_batch(train, val, block_size, batch_size, device, "train")
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    model.eval()
    return model


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="训练好的 checkpoint 路径（nanoGPT 格式）")
    ap.add_argument("--quick-train", action="store_true", help="没有 checkpoint 时快速训 400 步")
    ap.add_argument("--data-dir", default=os.path.expanduser(
        "~/projects/main-agent/nanoGPT/data/shakespeare_char"), help="数据目录")
    ap.add_argument("--seed", type=int, default=42, help="采样随机种子")
    ap.add_argument("--n-gen", type=int, default=300, help="每段生成字符数")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"设备: {device}  torch {torch.__version__}")

    train, val, vocab_size, itos = load_shakespeare(args.data_dir)
    print(f"数据: train {len(train)} tokens, val {len(val)} tokens, vocab {vocab_size}")
    stoi = {ch: i for i, ch in itos.items()}

    # 模型：checkpoint 优先，否则快速训练
    if args.ckpt:
        model, ckpt = load_checkpoint(args.ckpt, device)
        print(f"加载 checkpoint: {args.ckpt}")
        print(f"  模型: {ckpt['model_args']}")
        print(f"  参数量: {model.count_parameters():,}  iter={ckpt.get('iter_num')}  "
              f"best_val_loss={float(ckpt['best_val_loss']):.4f}")
    elif args.quick_train:
        print("快速训练 400 步小模型（410,368 参数）...")
        model = quick_train(train, val, vocab_size, device, steps=400)
    else:
        raise SystemExit("请用 --ckpt 指定 checkpoint，或用 --quick-train 快速训练")

    # ---- 实验 1：temperature 对"下一个字符分布"的影响（画图用真实数据）----
    print("\n[实验 1] temperature 对概率分布的影响（真实文本位置：'...SEBAST' 之后）")
    ctx_ids = torch.tensor([[stoi[c] for c in "and.\n\nALONSO:\nNo, no, he's gone.\n\nSEBAST"]],
                           dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(ctx_ids)
    logits_last = logits[0, -1, :].float().cpu()
    torch.save({"logits": logits_last, "itos": itos}, "/tmp/25-ambig-logits.pt")
    for T in [0.2, 0.8, 1.5]:
        probs = F.softmax(logits_last / T, dim=-1)
        H = (-probs * torch.log(probs + 1e-9)).sum().item()
        top5 = torch.topk(probs, 5)
        chars = [itos[i] for i in top5.indices.tolist()]
        print(f"  T={T}: 熵H={H:.4f}  top5={[(c, round(p, 3)) for c, p in zip(chars, top5.values.tolist())]}")

    # ---- 实验 2：不同 temperature 生成对比 ----
    def gen_text(prompt, temperature=1.0, top_k=None, top_p=None, n=None, seed=None):
        torch.manual_seed(seed if seed is not None else args.seed)
        idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
        out = generate(model, idx, n or args.n_gen, temperature=temperature,
                       top_k=top_k, top_p=top_p)
        return "".join(itos[i] for i in out[0].tolist())

    prompt = "KING "
    print(f"\n[实验 2] 同一个 prompt {prompt!r}，只拧 temperature 旋钮：")
    print("\n--- temperature = 0.2（保守）---")
    print(gen_text(prompt, temperature=0.2))
    print("\n--- temperature = 0.8（平衡）---")
    print(gen_text(prompt, temperature=0.8))
    print("\n--- temperature = 1.5（疯狂）---")
    print(gen_text(prompt, temperature=1.5))

    print(f"\n[实验 3] top-k / top-p 候选门卫：")
    print("\n--- temperature=1.0 + top_k=10 ---")
    print(gen_text(prompt, temperature=1.0, top_k=10))
    print("\n--- temperature=1.0 + top_p=0.9 ---")
    print(gen_text(prompt, temperature=1.0, top_p=0.9))
    print("\n--- temperature=1.0 + top_p=0.1（极保守）---")
    print(gen_text(prompt, temperature=1.0, top_p=0.1))

    # ---- 实验 4：perplexity 评估 ----
    print("\n[实验 4] perplexity（在 val 集上评估，eval_iters=100）")
    loss_val = estimate_loss(model, val, model.block_size, 32, eval_iters=100, device=device)
    ppl = math.exp(loss_val)
    print(f"  本模型 val loss {loss_val:.4f}  ->  perplexity {ppl:.2f}")
    print(f"  对照：随机模型 ppl = {vocab_size}（{vocab_size} 个字符等概率，每次平均要猜 {vocab_size} 个）")


if __name__ == "__main__":
    main()
