# 19-train-loop.py — 第 19 课实验：训练循环骨架（真实跑 300 步）
# 运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 19-train-loop.py
# 依赖: torch 2.12.1 + numpy（venv 已装）+ nanoGPT 仓库（model.py / data/shakespeare_char）
import os, sys, math, time
import numpy as np
import torch
import importlib.util

NANOGPT = os.path.expanduser("~/projects/main-agent/nanoGPT")
spec = importlib.util.spec_from_file_location("nanogpt_model", os.path.join(NANOGPT, "model.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["nanogpt_model"] = mod
spec.loader.exec_module(mod)
GPT, GPTConfig = mod.GPT, mod.GPTConfig

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- 配置（与 nanoGPT config/train_shakespeare_char.py 一致）----
dataset = "shakespeare_char"
data_dir = os.path.join(NANOGPT, "data", dataset)
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
n_layer, n_head, n_embd = 6, 6, 384
dropout = 0.2
learning_rate = 1e-3
max_iters = 300        # 本实验只跑 300 步（完整 5000 步见第 26 课）
warmup_iters = 100
lr_decay_iters = 5000  # 调度按完整训练设计，本实验只取前 300 步
min_lr = 1e-4
beta1, beta2 = 0.9, 0.99
weight_decay = 1e-1
grad_clip = 1.0
log_interval = 10

# ---- get_batch：与 train.py 完全一致（懒加载 + memmap + 随机起点）----
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ---- 模型（系列 baby GPT）----
config = GPTConfig(vocab_size=65, block_size=block_size, n_layer=n_layer, n_head=n_head,
                   n_embd=n_embd, dropout=dropout, bias=False)
model = GPT(config)
model.to(device)

# ---- AdamW：configure_optimizers 自带分组（decay vs no-decay）----
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), 'cpu')
decay_p = sum(p.numel() for pg in optimizer.param_groups if pg['weight_decay'] > 0 for p in pg['params'])
nodecay_p = sum(p.numel() for pg in optimizer.param_groups if pg['weight_decay'] == 0 for p in pg['params'])
print(f"AdamW 分组: decay={decay_p:,} / no-decay={nodecay_p:,} / 合计={decay_p+nodecay_p:,}")

# ---- lr 调度：与 train.py get_lr 完全一致（warmup + cosine decay）----
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ---- 训练循环（迷你版：forward → backward → clip → step）----
X, Y = get_batch('train')
t0 = time.time()
iters, losses, lrs = [], [], []
for iter_num in range(max_iters):
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    logits, loss = model(X, Y)                    # forward + loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                                # backward
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # grad clip
    optimizer.step()                               # AdamW 更新
    X, Y = get_batch('train')                      # 预取下一批
    if iter_num % log_interval == 0 or iter_num == max_iters - 1:
        iters.append(iter_num)
        losses.append(loss.item())
        lrs.append(lr)
        print(f"iter {iter_num:4d}: loss {loss.item():.4f}, lr {lr:.2e}, {time.time()-t0:5.1f}s", flush=True)
t1 = time.time()
print(f"total {max_iters} iters in {t1-t0:.1f}s -> {1000*(t1-t0)/max_iters:.0f} ms/iter")

# ---- 保存数据供画图 ----
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lesson19_training.npz"),
         iters=np.array(iters), losses=np.array(losses), lrs=np.array(lrs))
print("saved lesson19_training.npz")
