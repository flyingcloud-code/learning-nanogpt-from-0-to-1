"""
第 11 课：Embedding 主实验 —— 词向量到底学到了什么
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 11-embedding.py
依赖: torch 2.12.1 + numpy（venv 已装）
数据: ~/projects/main-agent/nanoGPT/data/shakespeare_char/{train,val}.bin + meta.pkl
"""
import os, pickle, time
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

DATA = os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char")

# ---------- Part 1: 查表 = one-hot 矩阵乘法（等价性证明） ----------
print("=" * 60)
print("Part 1: nn.Embedding 查表 = one-hot × 权重矩阵")
print("=" * 60)

vocab = ['h', 'e', 'l', 'o']          # 微型词表 4 个字符
n_embd = 3                            # 每个字符用一个 3 维向量表示
emb = nn.Embedding(len(vocab), n_embd)
print("权重矩阵 W (4x3)，第 i 行就是第 i 个字符的向量：")
with torch.no_grad():
    W = emb.weight.numpy()
print(W.round(3))
print()

# 查表：把字符 'e'（索引 1）变成向量 = 直接取 W 的第 1 行
idx = torch.tensor([1])               # 'e' 的索引
lookup = emb(idx).detach().numpy()
print("查表 emb([1]) =", lookup.round(3), "← 直接取 W 的第 1 行")

# one-hot 路线：索引 1 → [0,1,0,0]，再乘 W
onehot = np.zeros(4); onehot[1] = 1.0
via_onehot = onehot @ W
print("one-hot 路线 [0,1,0,0] @ W =", via_onehot.round(3))
diff = np.abs(lookup[0] - via_onehot).max()
print(f"两条路线的最大差异 = {diff:.2e}  ← 完全相等，查表只是省略了 one-hot")

# 手算数字示例：查 3 个字符
batch = torch.tensor([0, 2, 3])       # h, l, o
print("\n批量查表 emb([0,2,3]) = (3x3) 矩阵：")
print(emb(batch).detach().numpy().round(3))

# ---------- Part 2: 训练字符级 bigram 模型 ----------
print("\n" + "=" * 60)
print("Part 2: 训练一个只有 embedding 的语言模型（bigram）")
print("=" * 60)

# 读数据
meta = pickle.load(open(os.path.join(DATA, "meta.pkl"), "rb"))
stoi, itos = meta["stoi"], meta["itos"]
vocab_size = len(itos)
print(f"词表大小 = {vocab_size}（莎士比亚全部字符）")

def load_bin(name):
    raw = np.fromfile(os.path.join(DATA, name), dtype=np.uint16)
    return torch.from_numpy(raw.astype(np.int64))

train_data = load_bin("train.bin")
val_data = load_bin("val.bin")
print(f"train tokens = {len(train_data):,}  val tokens = {len(val_data):,}")

# 模型：token embedding → 线性层 → 每个位置预测下一个字符
# 这是 nanoGPT 最简形态：没有 attention，只看"前一个字符"的共现统计
class BigramEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx):
        # idx: (B, T) 字符索引 → (B, T, n_embd) → (B, T, vocab)
        return self.head(self.emb(idx))

N_EMBD = 64
BLOCK = 64          # 序列长度 T
BATCH = 256         # 每批序列数
STEPS = 1000
model = BigramEmbeddingModel(vocab_size, N_EMBD)
nparams = sum(p.numel() for p in model.parameters())
print(f"模型参数量 = {nparams:,}（其中 embedding 占 {vocab_size * N_EMBD:,}）")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK, (BATCH,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    return x, y

def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(50):
            x, y = get_batch(split)
            with torch.no_grad():
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1))
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out

t0 = time.time()
history = []   # (step, train_loss)
for step in range(STEPS):
    x, y = get_batch("train")
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == STEPS - 1:
        history.append((step, loss.item()))
        print(f"step {step:4d}  loss {loss.item():.4f}   ({time.time()-t0:.0f}s)")
print(f"训练完成，用时 {time.time()-t0:.0f}s")

losses = estimate_loss()
print(f"最终  train loss = {losses['train']:.4f}  val loss = {losses['val']:.4f}")

# ---------- Part 3: PCA 投影 + 保存 ----------
print("\n" + "=" * 60)
print("Part 3: 训练后的 embedding 用 PCA 降到 2 维")
print("=" * 60)

with torch.no_grad():
    W_final = model.emb.weight.numpy()          # (65, 64)

# 训练前的 embedding（随机初始化）也投影一次，做对比
torch.manual_seed(0)
model0 = BigramEmbeddingModel(vocab_size, N_EMBD)
with torch.no_grad():
    W_init = model0.emb.weight.numpy()

def pca2(vectors):
    """手写 PCA：中心化 → SVD → 取前两个奇异向量（不依赖 sklearn）"""
    X = vectors - vectors.mean(axis=0)          # 中心化：把原点挪到数据重心
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T                         # 投影到前两个主成分

proj_init = pca2(W_init)
proj = pca2(W_final)
# 打印每个字符的 2D 坐标
print(f"{'char':>5} {'idx':>3} {'x':>10} {'y':>10}")
for i, c in itos.items():
    print(f"{c!r:>5} {i:>3} {proj[i,0]:>10.3f} {proj[i,1]:>10.3f}")

# 顺便看几个"近邻"：用余弦相似度（衡量方向是否一致，比欧氏距离更标准）
def nearest(query_idx, k=5):
    q = W_final[query_idx]
    sims = (W_final @ q) / (np.linalg.norm(W_final, axis=1) * np.linalg.norm(q) + 1e-9)
    order = np.argsort(-sims)
    return [(itos[j], round(float(sims[j]), 3)) for j in order[1:k + 1]]

print("\n与 'a' 最近的字符(余弦):", nearest(stoi['a']))
print("与 'e' 最近的字符(余弦):", nearest(stoi['e']))
print("与 ' ' 最近的字符(余弦):", nearest(stoi[' ']))
print("与 ':' 最近的字符(余弦):", nearest(stoi[':']))

# 保存供画图
np.savez_compressed("exp11.npz",
    history=np.array(history, dtype=float),
    proj=proj, proj_init=proj_init,
    chars=np.array([itos[i] for i in range(vocab_size)]),
    train_final=losses['train'], val_final=losses['val'],
    W_final=W_final,
    nparams=nparams)
print("\n已保存 exp11.npz（loss 历史 + PCA 坐标 + 字符表）")
