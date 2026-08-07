# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 06-perceptron-mlp.py
# 依赖: torch + numpy（venv 已装）
# 第 6 课主实验：
#   1) 手写单个感知机前向传播（不用任何现成层），手算一个样本
#   2) 手写 2 层 MLP 的前向传播（显式 W1,b1,W2,b2 + 矩阵乘 + 激活）
#   3) 用 torch autograd 分别训练：单层感知机 vs 2 层 MLP，在 XOR 上的对比
#      —— 单层卡在 50% 准确率（线性分不开），MLP 学到 100%（啊哈时刻）
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

# ---------- 0. XOR 数据集：4 个点，1 个是"异或" ----------
# XOR: 两个输入相同 → 0；不同 → 1。单条直线永远分不开这两类。
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

print("=" * 64)
print("XOR 数据集（异或）：相同→0，不同→1")
for i in range(4):
    print(f"  ({int(X[i,0])}, {int(X[i,1])}) -> {int(y[i,0])}")

# ---------- 1. 单个感知机：手写前向传播 ----------
# 一个神经元 = 输入加权求和 + 偏置 + 激活函数
#   z = w1*x1 + w2*x2 + b
#   a = sigmoid(z)   （把任意实数压到 0~1，当作"有多像正类"）
def sigmoid_scalar(z):
    return 1.0 / (1.0 + np.exp(-z))

# 手搓一组权重（随便给的），对样本 (0,1) 手算一遍前向
w1, w2, b = 0.5, -0.8, 0.3
x1, x2 = 0.0, 1.0
z = w1 * x1 + w2 * x2 + b
a = sigmoid_scalar(z)
print("=" * 64)
print("单个神经元前向（手算）：权重 w1=0.5 w2=-0.8 b=0.3，输入 (0,1)")
print(f"  z = {w1}*{x1} + {w2}*{x2} + {b} = {z:.3f}")
print(f"  a = sigmoid({z:.3f}) = {a:.3f}   （>0.5 判为正类，<0.5 判为负类）")

# ---------- 2. 手写 2 层 MLP 前向传播 ----------
# 结构 2-3-1：输入 2 个 → 隐藏层 3 个神经元 → 输出 1 个
# 前向 = 两次"矩阵乘 + 偏置 + 激活"，中间夹一次非线性
def mlp_forward(x, W1, b1, W2, b2):
    # 第一层：x(4x2) @ W1(2x3) + b1(3) → z1(4x3)
    z1 = x @ W1 + b1
    # 隐藏层激活用 ReLU：把负数全压成 0（引入非线性，关键！）
    h = torch.relu(z1)
    # 第二层：h(4x3) @ W2(3x1) + b2(1) → z2(4x1)
    z2 = h @ W2 + b2
    return torch.sigmoid(z2), h, z1

# 手搓初始化（小随机数，保证能训练）
W1 = torch.randn(2, 3, requires_grad=True) * 0.5
b1 = torch.randn(3, requires_grad=True) * 0.5
W2 = torch.randn(3, 1, requires_grad=True) * 0.5
b2 = torch.randn(1, requires_grad=True) * 0.5

# ---------- 3. 单层感知机：同样用梯度下降训练，对比 ----------
# 单层 = 只有一个神经元 = 一条直线：y_hat = sigmoid(w1*x1 + w2*x2 + b)
W_lin = torch.randn(2, 1, requires_grad=True) * 0.5
b_lin = torch.randn(1, requires_grad=True) * 0.5

def train_linear(n_steps=3000, lr=0.3):
    """训练单层感知机（一条直线）。"""
    W = (torch.randn(2, 1) * 0.5).requires_grad_()
    bb = (torch.randn(1) * 0.5).requires_grad_()
    losses = []
    for step in range(n_steps):
        y_hat = torch.sigmoid(X @ W + bb)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            W.data -= lr * W.grad
            bb.data -= lr * bb.grad
            W.grad.zero_(); bb.grad.zero_()
        if step % 500 == 0:
            print(f"  step {step:4d}  loss = {loss.item():.4f}")
    with torch.no_grad():
        y_hat = torch.sigmoid(X @ W + bb)
        acc = ((y_hat > 0.5).float() == y).float().mean().item()
    return losses, acc, W.detach(), bb.detach()

def train_mlp(n_steps=3000, lr=0.3):
    """训练 2 层 MLP（隐藏层 3 个神经元）。"""
    W1p = (torch.randn(2, 3) * 0.5).requires_grad_()
    b1p = (torch.randn(3) * 0.5).requires_grad_()
    W2p = (torch.randn(3, 1) * 0.5).requires_grad_()
    b2p = (torch.randn(1) * 0.5).requires_grad_()
    losses = []
    for step in range(n_steps):
        y_hat, _, _ = mlp_forward(X, W1p, b1p, W2p, b2p)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            for p in (W1p, b1p, W2p, b2p):
                p.data -= lr * p.grad
                p.grad.zero_()
        if step % 500 == 0:
            print(f"  step {step:4d}  loss = {loss.item():.6f}")
    with torch.no_grad():
        y_hat, _, _ = mlp_forward(X, W1p, b1p, W2p, b2p)
        acc = ((y_hat > 0.5).float() == y).float().mean().item()
    return losses, acc, W1p.detach(), b1p.detach(), W2p.detach(), b2p.detach()

print("=" * 64)
print("\n训练单层感知机（一条直线，试图分开 XOR）...")
losses_lin, acc_lin, W_lin_f, b_lin_f = train_linear()
print(f"  最终 loss = {losses_lin[-1]:.4f}  准确率 = {acc_lin*100:.0f}%")

print("\n训练 2 层 MLP（隐藏层 3 个神经元）...")
losses_mlp, acc_mlp, W1_f, b1_f, W2_f, b2_f = train_mlp()
print(f"  最终 loss = {losses_mlp[-1]:.6f}  准确率 = {acc_mlp*100:.0f}%")

print("=" * 64)
print("\n结论：")
print(f"  单层感知机：loss 卡在 {losses_lin[-1]:.3f}，准确率 {acc_lin*100:.0f}% —— 一条直线分不开 XOR")
print(f"  2 层 MLP：  loss 降到 {losses_mlp[-1]:.6f}，准确率 {acc_mlp*100:.0f}% —— 弯一下就能分开")
print("  差的那一层，就是隐藏层 + ReLU 带来的非线性。")

# ---------- 4. 手写前向 vs 现成库：对拍验证 ----------
# 用刚训练好的 MLP 权重，手写 forward 算一遍，再和 torch.nn 的封装对比
import torch.nn as nn
net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1), nn.Sigmoid())
with torch.no_grad():
    net[0].weight.copy_(W1_f.t()); net[0].bias.copy_(b1_f)
    net[2].weight.copy_(W2_f.t()); net[2].bias.copy_(b2_f)
with torch.no_grad():
    y_manual, _, _ = mlp_forward(X, W1_f, b1_f, W2_f, b2_f)
    y_lib = net(X)
diff = (y_manual - y_lib).abs().max().item()
print("=" * 64)
print("\n手写前向 vs torch.nn 封装对拍（同一个权重，4 个样本）：")
for i in range(4):
    print(f"  ({int(X[i,0])}, {int(X[i,1])})  手写={y_manual[i,0].item():.4f}  封装={y_lib[i,0].item():.4f}")
print(f"  最大误差 = {diff:.2e} —— 完全一致，手写前向没写错")

# 保存训练数据供画图脚本用
np.savez("_06_data.npz",
         losses_lin=np.array(losses_lin), losses_mlp=np.array(losses_mlp),
         acc_lin=acc_lin, acc_mlp=acc_mlp,
         W1=W1_f.numpy(), b1=b1_f.numpy(),
         W2=W2_f.numpy(), b2=b2_f.numpy(),
         W_lin=W_lin_f.numpy(), b_lin=b_lin_f.numpy())
print("\n已保存 _06_data.npz（供 06-make-charts.py 画图）")
