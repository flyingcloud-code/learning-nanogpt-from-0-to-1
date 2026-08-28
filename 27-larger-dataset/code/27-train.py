#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 27 课：BPE 子词级训练 —— 把 26 课的模型原封不动搬过来，只换词表（字符->BPE）+ 换语料。

用法:
  python 27-train.py --data shakespeare --steps 1000 --tag bpe-shakes --seed 1337
  python 27-train.py --data novels      --steps 1000 --tag bpe-novels  --seed 1337

依赖: torch / numpy / tiktoken（venv: ~/projects/main-agent/nanoGPT/.venv）
模型: _27_gpt.py 里的 MyGPT（第 23 课手搓，26 课加了 dropout），词表换 50257。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import tiktoken

from _27_gpt import MyGPT

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DATA_FILES = {
    "shakespeare": "/Users/openclaw-master/projects/main-agent/nanoGPT/data/shakespeare_char/input.txt",
    "novels": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "novels.txt"),
}
PROMTS = {
    "shakespeare": "KING HENRY VI:\n",
    "novels": "CHAPTER I.\n\nIt was a bright cold day in April,\n",
}


def load_bpe_data(name):
    """读文本 -> tiktoken(gpt2) 编码 -> train/val 90:10 切分。"""
    path = DATA_FILES[name]
    text = open(path, encoding="utf-8").read()
    enc = tiktoken.get_encoding("gpt2")
    ids = np.array(enc.encode(text), dtype=np.int64)
    n = int(0.9 * len(ids))
    train = torch.from_numpy(ids[:n])
    val = torch.from_numpy(ids[n:])
    return train, val, enc


def get_batch(train, val, block_size, batch_size, device, split="train"):
    data = train if split == "train" else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step, max_steps, warmup, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    coef = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coef * (max_lr - min_lr)


@torch.no_grad()
def estimate_val_loss(model, train, val, block_size, batch_size, device):
    model.eval()
    losses = []
    for _ in range(40):
        x, y = get_batch(train, val, block_size, batch_size, device, "val")
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def generate_text(model, enc, prompt, max_new_tokens=300, temperature=0.8):
    model.eval()
    idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=DEVICE)
    idx = model.generate(idx, max_new_tokens, temperature=temperature)
    return prompt + enc.decode(idx[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["shakespeare", "novels"], required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    log_path = f"run-{args.tag}.log"
    logf = open(log_path, "w", encoding="utf-8")
    print(f"设备: {DEVICE}  torch {torch.__version__}")

    train, val, enc = load_bpe_data(args.data)
    print(f"数据 {args.data}: train {len(train):,} tokens, val {len(val):,} tokens, 词表 {enc.n_vocab:,}")

    model = MyGPT(vocab_size=enc.n_vocab, block_size=args.block_size,
                  n_layer=6, n_head=6, n_embd=384, dropout=0.2)
    model.to(DEVICE)
    nparam = model.count_parameters()
    print(f"参数量: {nparam:,}")
    logf.write(json.dumps({"event": "init", "data": args.data, "tokens": len(train) + len(val),
                           "params": nparam, "vocab": enc.n_vocab}) + "\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = get_lr(step, args.steps, args.warmup, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = get_batch(train, val, args.block_size, args.batch_size, DEVICE, "train")
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0:
            elapsed = time.time() - t0
            line = {"step": step, "loss": round(loss.item(), 4), "lr": round(lr, 6), "elapsed_s": round(elapsed, 1)}
            logf.write(json.dumps(line) + "\n")
            logf.flush()
            print(f"  step {step:5d}  loss {loss.item():.4f}  lr {lr:.2e}  ({elapsed:.0f}s)", flush=True)

        if step % args.eval_every == 0:
            vl = estimate_val_loss(model, train, val, args.block_size, args.batch_size, DEVICE)
            logf.write(json.dumps({"step": step, "val": round(vl, 4)}) + "\n")
            logf.flush()
            print(f"  [val] step {step}  val loss {vl:.4f}  ppl {math.exp(vl):.2f}", flush=True)

    # 最终评估 + 保存 + 生成
    vl = estimate_val_loss(model, train, val, args.block_size, args.batch_size, DEVICE)
    print(f"最终 val loss {vl:.4f}  ppl {math.exp(vl):.2f}")
    ckpt = {
        "model": model.state_dict(),
        "config": {"vocab_size": enc.n_vocab, "block_size": args.block_size,
                   "n_layer": 6, "n_head": 6, "n_embd": 384, "dropout": 0.2},
        "step": args.steps, "val_loss": vl, "data": args.data, "tokenizer": "gpt2",
    }
    ckpt_path = f"ckpt-{args.tag}-step{args.steps}.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[ckpt] 已保存 -> {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.0f} MB)")
    logf.write(json.dumps({"event": "final", "val": round(vl, 4), "ckpt": ckpt_path}) + "\n")

    text = generate_text(model, enc, PROMTS[args.data], max_new_tokens=300, temperature=0.8)
    print("\n===== 生成样本 (T=0.8) =====")
    print(text)
    logf.write(json.dumps({"event": "sample", "text": text}) + "\n")
    logf.close()


if __name__ == "__main__":
    main()
