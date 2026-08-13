"""
第 12 课实验 1：n-gram 计数语言模型（纯 numpy，零训练，纯数数）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 12-ngram.py
依赖: numpy（venv 已装）
数据: ~/projects/main-agent/nanoGPT/data/shakespeare_char/{train,val}.bin
"""
import os
import pickle
import numpy as np

DATA = os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char")

train = np.fromfile(os.path.join(DATA, "train.bin"), dtype=np.uint16).astype(np.int64)
val = np.fromfile(os.path.join(DATA, "val.bin"), dtype=np.uint16).astype(np.int64)
with open(os.path.join(DATA, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
itos = meta["itos"]
V = meta["vocab_size"]

print(f"词表大小 V = {V}，train {len(train):,} tokens，val {len(val):,} tokens\n")

# ============ bigram 计数 ============
# N[i][j] = 字符 i 后面跟着字符 j 的次数
N = np.zeros((V, V), dtype=np.float64)
for a, b in zip(train[:-1], train[1:]):
    N[a, b] += 1

# 归一化：P(下一个 = j | 当前 = i) = N[i][j] / sum(N[i])
P = N / (N.sum(axis=1, keepdims=True) + 1e-8)

def bigram_loss(ids):
    a, b = ids[:-1], ids[1:]
    return -np.log(P[a, b]).mean()

tl, vl = bigram_loss(train), bigram_loss(val)
print(f"bigram 计数模型（{V*V} 个格子，零训练）")
print(f"  不做平滑: train loss = {tl:.4f}   val loss = {vl:.4f}   val perplexity = {np.exp(vl):.2f}")

# 平滑版：每个格子 +0.01，避免"训练集里没见过 → 概率 0 → log(0) = -inf"
P_s = (N + 0.01) / (N.sum(axis=1, keepdims=True) + 0.01 * V)

def bigram_loss_s(ids):
    a, b = ids[:-1], ids[1:]
    return -np.log(P_s[a, b]).mean()

tls, vls = bigram_loss_s(train), bigram_loss_s(val)
print(f"  加 0.01 平滑: train loss = {tls:.4f}   val loss = {vls:.4f}   val perplexity = {np.exp(vls):.2f}")

def generate_bigram(seed, length=400, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    out = [seed]
    cur = seed
    for _ in range(length):
        p = P[cur]
        cur = int(rng.choice(V, p=p))
        out.append(cur)
    return "".join(itos[i] for i in out)

print("\n=== bigram 计数模型生成（每步只看前 1 个字符）===")
print(generate_bigram(meta["stoi"]["\n"]))

# ============ trigram 计数 ============
N3 = np.zeros((V, V, V), dtype=np.float64)
for a, b, c in zip(train[:-2], train[1:-1], train[2:]):
    N3[a, b, c] += 1

# 稀疏度：出现过多少种组合 / 所有可能组合
seen = (N3 > 0).sum()
total = V * V * V
print(f"\ntrigram 计数：共 {seen:,} 种组合出现过，占总空间 {total:,} 的 {seen/total*100:.3f}%")
print(f"  → 剩下 {100 - seen/total*100:.3f}% 的组合在训练集里一次都没出现")

# 平滑：给每个格子 +0.01（拉普拉斯平滑），否则未出现的组合概率是 0，log(0) 直接崩溃
P3 = (N3 + 0.01) / (N3.sum(axis=2, keepdims=True) + 0.01 * V)

def trigram_loss(ids):
    a, b, c = ids[:-2], ids[1:-1], ids[2:]
    return -np.log(P3[a, b, c]).mean()

tl3, vl3 = trigram_loss(train), trigram_loss(val)
print(f"\ntrigram 计数模型（含 +0.01 平滑）")
print(f"  train loss = {tl3:.4f}   val loss = {vl3:.4f}   val perplexity = {np.exp(vl3):.2f}")

def generate_trigram(seed2, length=400, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    out = list(seed2)
    a, b = seed2
    for _ in range(length):
        p = P3[a, b]
        c = int(rng.choice(V, p=p))
        out.append(c)
        a, b = b, c
    return "".join(itos[i] for i in out)

print("\n=== trigram 计数模型生成（每步看前 2 个字符）===")
nl = meta["stoi"]["\n"]
sp = meta["stoi"][" "]
print(generate_trigram([nl, sp]))

# 随机猜测基线
print(f"\n随机猜测基线: loss = {np.log(V):.4f}   perplexity = {V}")
