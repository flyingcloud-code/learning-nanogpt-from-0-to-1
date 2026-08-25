#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 24 课：训练循环——loss 收敛实战
====================================
主角不是模型（第 23 课已经手搓完），而是"训练"本身：
  1. 训练循环的核心三行：算 loss -> backward -> optimizer.step()
  2. 真实 loss 曲线：训练 400 步，观察 loss 从 ln(65) 一路下降
  3. 断点续训：checkpoint 保存/恢复，恢复后 loss 无缝衔接

运行（Mac mini / Apple Silicon，torch 2.12.1）：
    python 24-train.py --max-steps 400 --ckpt-every 200 --tag runA --seed 1337
    python 24-train.py --resume ckpts/runA-step400.pt --max-steps 400 --tag runB
    python 24-train.py --max-steps 200 --tag fresh200 --seed 1337

依赖：torch, numpy（venv 已装）
数据：data/shakespeare_char（train 1,003,854 / val 111,540 tokens，65 字符词表）
"""
import os
import sys
import math
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# 第 23 课手搓的 GPT（原样复用，一行不改；抽到 _24_gpt.py 供两个脚本共享）
# ------------------------------------------------------------
from _24_gpt import MyGPT, load_shakespeare, get_batch


# ------------------------------------------------------------
# 训练循环的核心：4 个函数 + 1 个 while
# ------------------------------------------------------------

@torch.no_grad()
def estimate_val_loss(model, train, val, block_size, batch_size, device, n_batch=20):
    """验证 loss：模型不更新，只在 val 集上算平均交叉熵（n_batch 批的平均）。"""
    model.eval()
    total = 0.0
    for _ in range(n_batch):
        x, y = get_batch(train, val, block_size, batch_size, device, "val")
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / n_batch


def get_lr(step, max_steps, warmup, max_lr, min_lr):
    """nanoGPT 同款学习率调度：先热身爬坡，再余弦衰减。"""
    if step < warmup:
        return max_lr * (step + 1) / warmup          # 线性热身
    if step > max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)  # 0 -> 1
    coef = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0 余弦
    return min_lr + coef * (max_lr - min_lr)


def save_checkpoint(path, model, optimizer, step, cfg, best_val_loss):
    """checkpoint 时间胶囊：模型 + 优化器 + 步数 + 配置 + 随机状态，全装进一个 .pt。"""
    ckpt = {
        "model": model.state_dict(),                     # 模型权重（记忆本体）
        "optimizer": optimizer.state_dict(),             # 优化器状态（动量/二阶矩）
        "model_args": {                                  # 重建模型所需的配置
            "vocab_size": model.vocab_size, "block_size": model.block_size,
            "n_layer": model.n_layer, "n_head": model.n_head,
            "n_embd": model.n_embd,
        },
        "step": step,                                    # 训练到第几步
        "best_val_loss": best_val_loss,
        "config": cfg,
        "rng_torch": torch.random.get_rng_state(),       # 随机状态（保证续训可复现）
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    torch.save(ckpt, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [ckpt] step {step} 已保存 -> {path} ({size_mb:.2f} MB)")
    return size_mb


def load_checkpoint(path, device):
    """打开时间胶囊：把模型/优化器/步数/随机状态全部还原。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint 路径")
    parser.add_argument("--tag", type=str, default="run")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", type=str, default="ckpts")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--max-lr", type=float, default=3e-3)
    parser.add_argument("--min-lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--data-dir", type=str,
                        default=os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char"))
    args = parser.parse_args()

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"设备: {DEVICE}  torch {torch.__version__}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train, val, vocab_size, itos = load_shakespeare(args.data_dir)
    print(f"数据: train {len(train):,} tokens, val {len(val):,} tokens, vocab {vocab_size}")

    # ---------- 新建 or 恢复 ----------
    model_args = dict(vocab_size=vocab_size, block_size=args.block_size,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"恢复 checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, DEVICE)
        model_args = ckpt["model_args"]
        model = MyGPT(**model_args).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        # 还原随机状态：续训和连续训练跑出同一条曲线
        torch.random.set_rng_state(ckpt["rng_torch"].cpu())
        np.random.set_state(ckpt["rng_numpy"])
        random.setstate(ckpt["rng_python"])
        print(f"  从 step {start_step} 继续，历史 best_val_loss={best_val_loss:.4f}")
    else:
        model = MyGPT(**model_args).to(DEVICE)
        print(f"新建模型: {model_args}  参数量 {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=0.1)
    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer"])

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"{args.tag}-log.json")
    history = []
    if args.resume and os.path.exists(log_path):
        history = json.load(open(log_path))
        history = [h for h in history if h[0] <= start_step]  # 只保留 <= 恢复点

    cfg = dict(model_args, batch_size=args.batch_size, max_lr=args.max_lr,
               min_lr=args.min_lr, warmup=args.warmup, seed=args.seed,
               tag=args.tag, resume=args.resume, max_steps=args.max_steps)

    # ---------- 训练循环：就是这一个 while ----------
    t0 = time.time()
    step = start_step
    total_steps = start_step + args.max_steps
    print(f"训练: step {start_step + 1} -> {total_steps}（本次 {args.max_steps} 步）")
    model.train()
    while step < total_steps:
        step += 1
        x, y = get_batch(train, val, model.block_size, args.batch_size, DEVICE, "train")
        lr = get_lr(step, total_steps, args.warmup, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        logits, loss = model(x, y)      # ① 前向：算 loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                 # ② 反向：算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                # ③ 更新：参数走一步

        if step == start_step + 1 or step % args.log_every == 0:
            val_loss = None
            if step % args.eval_every == 0 or step == total_steps:
                val_loss = estimate_val_loss(model, train, val, model.block_size,
                                             args.batch_size, DEVICE)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            history.append([step, round(loss.item(), 4),
                            None if val_loss is None else round(val_loss, 4),
                            round(lr, 6)])
            json.dump(history, open(log_path, "w"))
            vtxt = f"  val {val_loss:.4f}" if val_loss is not None else ""
            print(f"  step {step:>5}  loss {loss.item():.4f}  lr {lr:.2e}{vtxt}  ({time.time()-t0:.1f}s)")
            t0 = time.time()

        if step % args.ckpt_every == 0:
            save_checkpoint(os.path.join(args.out_dir, f"ckpt-{args.tag}-step{step}.pt"),
                            model, optimizer, step, cfg, best_val_loss)

    # ---------- 生成：从模型当前状态采样 ----------
    model.eval()
    stoi = {ch: i for i, ch in itos.items()}
    start = torch.tensor([[stoi["K"], stoi["I"], stoi["N"], stoi["G"]]],
                         dtype=torch.long, device=DEVICE)
    out = model.generate(start, max_new_tokens=300, temperature=0.8)[0].tolist()
    text = "".join(itos[i] for i in out)
    sample_path = os.path.join(args.out_dir, f"{args.tag}-sample.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[生成] 从 'KING' 开始，temperature=0.8，300 字符（已存 {sample_path}）：")
    print("  " + text.replace("\n", "\\n"))
    print(f"\n日志: {log_path}  共 {len(history)} 条")
    print("完成 ✅")


if __name__ == "__main__":
    main()
