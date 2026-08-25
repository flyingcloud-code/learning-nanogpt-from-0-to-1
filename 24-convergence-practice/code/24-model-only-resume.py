#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 24 课补充实验：只恢复模型权重、不恢复优化器状态，会怎样？
====================================
从 step 400 的 checkpoint 里只取出 model 权重，配一个全新的 AdamW，
再训 100 步。对比正常断点续训（模型 + 优化器一起恢复）：
新优化器的"动量/二阶矩"是空的，恢复后最初几步会像新手一样乱闯。

运行（Mac mini / Apple Silicon，torch 2.12.1）：
    python 24-model-only-resume.py

依赖：torch, numpy；复用 24-train.py 的 MyGPT（import 同目录脚本）
"""
import os
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _24_gpt import MyGPT, load_shakespeare, get_batch  # noqa: E402

torch.manual_seed(1337)
np.random.seed(1337)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"设备: {DEVICE}  torch {torch.__version__}")

data_dir = os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char")
train, val, vocab_size, itos = load_shakespeare(data_dir)

ckpt = torch.load("ckpts/ckpt-runA-step400.pt", map_location=DEVICE, weights_only=False)
model = MyGPT(**ckpt["model_args"]).to(DEVICE)
model.load_state_dict(ckpt["model"])
print(f"已加载模型权重（step {ckpt['step']} 的 checkpoint），但 optimizer 是全新的")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
# 注意：没有 optimizer.load_state_dict(ckpt['optimizer']) —— 动量/二阶矩全部清零

model.train()
history = []
t0 = time.time()
for step in range(1, 101):
    x, y = get_batch(train, val, model.block_size, 32, DEVICE, "train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step == 1 or step % 10 == 0:
        history.append((step, round(loss.item(), 4)))
        print(f"  step {step:>4}  loss {loss.item():.4f}  ({time.time()-t0:.1f}s)")
        t0 = time.time()

with open("ckpts/model-only-log.json", "w") as f:
    json.dump(history, f)
print("ckpts/model-only-log.json 已保存")
