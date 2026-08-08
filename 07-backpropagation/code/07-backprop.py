# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 07-backprop.py
# 依赖: torch + numpy（venv 已装）
# 第 7 课主实验：手写反向传播（完全不用 autograd）训练 2 层 MLP 学会 XOR
# 三个验证：
#   1) 手写梯度 vs autograd 梯度（逐参数对拍）
#   2) 手写梯度 vs 数值梯度（有限差分，梯度检查的标准做法）
#   3) 手写 BP 训练 vs autograd 训练（相同初始化、相同 lr、相同步数）
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ---------- 0. XOR 数据集 ----------
# 相同→0，不同→1。4 个点，一条直线永远分不开（第 6 课的主角）。
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

print("=" * 68)
print("第 7 课：手写反向传播 —— 训练 2 层 MLP（结构 2-3-1）学会 XOR")
print("=" * 68)

# ---------- 1. 前向 + 手写反向（全程不用 autograd） ----------
def forward_manual(X, y, W1, b1, W2, b2):
    """前向传播：和 06 课一模一样。返回 loss 和中间量（反向要用）。"""
    z1 = X @ W1 + b1              # 第一层总分（4×3）
    h = torch.relu(z1)            # 激活：负数归零（4×3）
    z2 = h @ W2 + b2              # 第二层总分（4×1）
    y_hat = torch.sigmoid(z2)     # 输出压到 0~1（4×1）
    loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
    return loss, (z1, h, z2, y_hat)

def backward_manual(X, y, W1, b1, W2, b2, cache):
    """手写反向传播：从 loss 出发，沿前向的路径倒着走，逐层算梯度。

    核心公式（推导见文章正文）：
      dz2 = (y_hat - y) / N                    # sigmoid + 交叉熵 的梯度化简成这么一行
                                               # ⚠️ 除以 N=4：因为 loss 取的是 4 个样本的平均
      dW2 = hᵀ @ dz2 ;  db2 = dz2 按列求和
      dh  = dz2 @ W2ᵀ
      dz1 = dh * (z1 > 0)                      # ReLU 的导数：正数处为 1，负数处为 0
      dW1 = Xᵀ @ dz1 ;  db1 = dz1 按列求和
    """
    z1, h, z2, y_hat = cache
    N = X.shape[0]                                    # 样本数 4（loss 取平均，梯度要除以 N）
    dz2 = (y_hat - y) / N                             # (4×1) loss 对 z2 的梯度
    dW2 = h.T @ dz2                                   # (3×1)
    db2 = dz2.sum(0)                                  # (1,)
    dh = dz2 @ W2.T                                   # (4×3) loss 对 h 的梯度
    dz1 = dh * (z1 > 0).float()                       # (4×3) ReLU 导数筛选
    dW1 = X.T @ dz1                                   # (2×3)
    db1 = dz1.sum(0)                                  # (3,)
    return dW1, db1, dW2, db2

# ---------- 2. autograd 参考实现（PyTorch 自动算梯度） ----------
def forward_backward_autograd(X, y, W1, b1, W2, b2):
    """同一网络，梯度交给 autograd 算。用来对拍手写结果。"""
    W1g, b1g, W2g, b2g = W1.clone(), b1.clone(), W2.clone(), b2.clone()
    for p in (W1g, b1g, W2g, b2g):
        p.requires_grad_()
    z1 = X @ W1g + b1g
    h = torch.relu(z1)
    z2 = h @ W2g + b2g
    y_hat = torch.sigmoid(z2)
    loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
    loss.backward()
    return loss.detach(), (W1g.grad, b1g.grad, W2g.grad, b2g.grad)

# ---------- 3. 数值梯度（有限差分）——梯度检查的金标准 ----------
def numerical_gradients(X, y, W1, b1, W2, b2, eps=1e-6):
    """对每个参数，把它的值拨动 ±eps，看 loss 变化多少，斜率=导数。

    这是"用定义算导数"，不依赖任何求导规则——专门用来证明手写公式没错。
    ⚠️ 用 float64 计算：float32 下 loss 的舍入误差会污染差分结果。
    """
    X64, y64 = X.double(), y.double()
    params64 = [W1.double(), b1.double(), W2.double(), b2.double()]
    grads = []
    for P in params64:
        Pg = torch.zeros_like(P)
        it = np.ndindex(*P.shape)
        for idx in it:
            orig = P[idx].item()
            P[idx] = orig + eps
            loss_p, _ = forward_manual(X64, y64, *params64)
            P[idx] = orig - eps
            loss_m, _ = forward_manual(X64, y64, *params64)
            P[idx] = orig
            Pg[idx] = (loss_p - loss_m) / (2 * eps)
        grads.append(Pg)
    return grads

# ---------- 4. 初始化一组权重，做三路梯度对拍 ----------
print("\n[对拍 1] 手写梯度 vs autograd 梯度（同一组随机权重）")
W1 = (torch.randn(2, 3) * 0.5).requires_grad_(False)
b1 = (torch.randn(3) * 0.5).requires_grad_(False)
W2 = (torch.randn(3, 1) * 0.5).requires_grad_(False)
b2 = (torch.randn(1) * 0.5).requires_grad_(False)

loss, cache = forward_manual(X, y, W1, b1, W2, b2)
g_manual = backward_manual(X, y, W1, b1, W2, b2, cache)
_, g_auto = forward_backward_autograd(X, y, W1, b1, W2, b2)

names = ["dW1", "db1", "dW2", "db2"]
for n, gm, ga in zip(names, g_manual, g_auto):
    print(f"  {n}: 手写 max={gm.abs().max().item():.6f}  autograd max={ga.abs().max().item():.6f}"
          f"  最大差={ (gm - ga).abs().max().item():.2e}")

print("\n[对拍 2] 手写梯度 vs 数值梯度（有限差分 float64，把每个参数拨动 ±1e-6）")
g_num = numerical_gradients(X, y, W1, b1, W2, b2)
for n, gm, gn in zip(names, g_manual, g_num):
    print(f"  {n}: 手写 max={gm.abs().max().item():.6f}  数值 max={gn.abs().max().item():.6f}"
          f"  最大差={ (gm.double() - gn).abs().max().item():.2e}")

# ---------- 5. 手写 BP 训练 vs autograd 训练（相同初始化、相同 lr） ----------
def train_manual(n_steps=3000, lr=0.3, init=None):
    if init is None:
        init = ((torch.randn(2, 3) * 0.5), (torch.randn(3) * 0.5),
                (torch.randn(3, 1) * 0.5), (torch.randn(1) * 0.5))
    W1, b1, W2, b2 = [t.clone() for t in init]
    losses = []
    for step in range(n_steps):
        loss, cache = forward_manual(X, y, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_manual(X, y, W1, b1, W2, b2, cache)
        losses.append(loss.item())
        with torch.no_grad():
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
        if step % 500 == 0:
            print(f"  step {step:4d}  loss = {loss.item():.6f}")
    with torch.no_grad():
        _, (z1, h, z2, y_hat) = forward_manual(X, y, W1, b1, W2, b2)
        acc = ((y_hat > 0.5).float() == y).float().mean().item()
    return losses, acc

def train_autograd(n_steps=3000, lr=0.3, init=None):
    if init is None:
        init = ((torch.randn(2, 3) * 0.5), (torch.randn(3) * 0.5),
                (torch.randn(3, 1) * 0.5), (torch.randn(1) * 0.5))
    W1, b1, W2, b2 = [t.clone().requires_grad_() for t in init]
    losses = []
    for step in range(n_steps):
        z1 = X @ W1 + b1
        h = torch.relu(z1)
        z2 = h @ W2 + b2
        y_hat = torch.sigmoid(z2)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            for p in (W1, b1, W2, b2):
                p.data -= lr * p.grad
                p.grad.zero_()
        if step % 500 == 0:
            print(f"  step {step:4d}  loss = {loss.item():.6f}")
    with torch.no_grad():
        y_hat = torch.sigmoid(X @ W1 + b1)
        y_hat = torch.sigmoid(torch.relu(X @ W1 + b1) @ W2 + b2)
        acc = ((y_hat > 0.5).float() == y).float().mean().item()
    return losses, acc

print("\n[训练 A] 手写反向传播训练（梯度全靠手推公式）")
init = ((torch.randn(2, 3) * 0.5), (torch.randn(3) * 0.5),
        (torch.randn(3, 1) * 0.5), (torch.randn(1) * 0.5))
losses_manual, acc_manual = train_manual(init=init)
print(f"  最终 loss = {losses_manual[-1]:.6f}  准确率 = {acc_manual*100:.0f}%")

print("\n[训练 B] autograd 训练（相同初始化，梯度交给 PyTorch）")
losses_auto, acc_auto = train_autograd(init=init)
print(f"  最终 loss = {losses_auto[-1]:.6f}  准确率 = {acc_auto*100:.0f}%")

# ---------- 6. 保存真实数据给画图脚本 ----------
# 梯度检查散点：13 个参数（W1 6 + b1 3 + W2 3 + b2 1）的手写 vs 数值梯度
flat_manual = torch.cat([g.reshape(-1) for g in g_manual]).numpy()
flat_num = torch.cat([g.reshape(-1) for g in g_num]).numpy()
np.savez("_07_data.npz",
         losses_manual=np.array(losses_manual),
         losses_auto=np.array(losses_auto),
         acc_manual=acc_manual, acc_auto=acc_auto,
         grad_manual=flat_manual, grad_numeric=flat_num)
print("\n已保存 _07_data.npz（真实训练数据，供 07-make-charts.py 画图）")
