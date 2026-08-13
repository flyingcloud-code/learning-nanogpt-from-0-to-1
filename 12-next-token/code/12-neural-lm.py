"""
第 12 课实验 2：神经网络语言模型（真训练，next token prediction）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 12-neural-lm.py
依赖: torch 2.12.1 + numpy（venv 已装）
数据: ~/projects/main-agent/nanoGPT/data/shakespeare_char/{train,val}.bin
"""
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
DATA = os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char")

train = np.fromfile(os.path.join(DATA, "train.bin"), dtype=np.uint16).astype(np.int64)
val = np.fromfile(os.path.join(DATA, "val.bin"), dtype=np.uint16).astype(np.int64)
with open(os.path.join(DATA, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
itos = meta["itos"]
V = meta["vocab_size"]

BLOCK = 8        # 每次看前 8 个字符，预测第 9 个
BATCH = 256
STEPS = 3000
LR = 3e-4

train_t = torch.from_numpy(train)
val_t = torch.from_numpy(val)

def get_batch(split):
    data = train_t if split == "train" else val_t
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + BLOCK] for i in ix])
    return x, y

class NextTokenMLP(nn.Module):
    def __init__(self, vocab_size, block, n_embd, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)      # 每个字符一张 64 维身份证
        self.net = nn.Sequential(                        # 两层 MLP
            nn.Linear(block * n_embd, hidden),           # 8 个字符拼起来 → 256 维
            nn.Tanh(),
            nn.Linear(hidden, vocab_size),               # 256 维 → 65 个候选分数
        )
    def forward(self, idx):
        e = self.emb(idx)                                # (B, 8, 64)
        return self.net(e.view(e.shape[0], -1))          # (B, 65) 每个候选字符的分数

model = NextTokenMLP(V, BLOCK, 64, 256)
nparams = sum(p.numel() for p in model.parameters())
print(f"神经网络语言模型参数量: {nparams:,}")
opt = torch.optim.AdamW(model.parameters(), lr=LR)
history = []
t0 = time.time()

for step in range(STEPS):
    x, y = get_batch("train")
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)   # 训练目标：让真实下一个字符的分数最高
    opt.zero_grad()
    loss.backward()
    opt.step()
    history.append(loss.item())
    if step % 300 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

# 验证集评估
model.eval()
with torch.no_grad():
    losses = []
    for _ in range(200):
        x, y = get_batch("val")
        logits = model(x)
        losses.append(nn.functional.cross_entropy(logits, y).item())
    val_loss = float(np.mean(losses))
print(f"\n训练用时 {time.time()-t0:.1f}s")
print(f"最终 train loss = {history[-1]:.4f}  val loss = {val_loss:.4f}")
print(f"val perplexity  = {np.exp(val_loss):.2f}")

# 生成：给定开头，逐字符预测
def generate(model, prompt, length=400):
    model.eval()
    ids = [meta["stoi"][c] for c in prompt]
    out = list(prompt)
    with torch.no_grad():
        for _ in range(length):
            ctx = ids[-BLOCK:]
            if len(ctx) < BLOCK:                 # prompt 太短时左边补 '\n'(索引0)
                ctx = [0] * (BLOCK - len(ctx)) + ctx
            x = torch.tensor([ctx])
            logits = model(x)[0]
            probs = torch.softmax(logits, dim=0).numpy()
            nxt = int(np.random.choice(V, p=probs))
            out.append(itos[nxt])
            ids.append(nxt)
    return "".join(out)

print("\n=== 神经网络生成（prompt: 'ROMEO:') ===")
print(generate(model, "ROMEO:"))
print("\n=== 神经网络生成（prompt: 'To be, or not to be,') ===")
print(generate(model, "To be, or not to be,"))

# 保存曲线数据 + 关键数字，供画图脚本用
# 12-ngram.py 实测: bigram 平滑 val ppl=12.03, trigram 平滑 val ppl=7.89
np.savez("exp12.npz",
         history=np.array(history),
         bigram_smooth_ppl=12.03,
         trigram_ppl=7.89,
         neural_ppl=np.exp(val_loss),
         neural_val_loss=val_loss,
         random_ppl=V)
print("\n已保存 exp12.npz")
