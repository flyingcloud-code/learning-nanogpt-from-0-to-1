# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 08-loss-convergence.py
# 依赖: torch + numpy（venv 已装）
# 第 8 课主实验：Loss 与收敛
#   实验 1：sin 曲线回归，4 个学习率（0.0005 / 0.005 / 0.05 / 0.5）
#           —— 学习率太小走不动、合适快速收敛、太大直接发散（NaN）
#   实验 2：sigmoid 输出层的梯度对比（交叉熵 vs MSE）
#           —— 预测"很自信但错了"时，MSE 梯度被 ŷ(1-ŷ) 压扁，几乎学不动
#   实验 3：XOR 分类，相同初始化下交叉熵 vs MSE 谁先收敛
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ================= 实验 1：sin 曲线回归 =================
# 任务：用 1-32-32-1 的 MLP 拟合 y = sin(x)，x 取 [-3, 3] 均匀 60 个点
# 三个学习率的命运完全不同：0.0005 慢慢爬、0.05 快速落地、0.5 直接爆炸

def make_reg_data():
    xs = torch.linspace(-3, 3, 60).view(-1, 1)
    ys = torch.sin(xs)
    return xs, ys

def train_reg(lr, steps=2000, seed=0):
    """训练 sin 回归 MLP，返回每个 step 的 loss 序列。"""
    torch.manual_seed(seed); np.random.seed(seed)
    xs, ys = make_reg_data()
    # 1-32-32-1 网络（隐藏层用 tanh 激活，输出是线性——回归任务不需要 sigmoid）
    W1 = (torch.randn(1, 32) * 0.5).requires_grad_(); b1 = (torch.randn(32) * 0.5).requires_grad_()
    W2 = (torch.randn(32, 32) * 0.5).requires_grad_(); b2 = (torch.randn(32) * 0.5).requires_grad_()
    W3 = (torch.randn(32, 1) * 0.5).requires_grad_(); b3 = (torch.randn(1) * 0.5).requires_grad_()
    params = [W1, b1, W2, b2, W3, b3]
    hist = []
    for step in range(steps):
        h1 = torch.tanh(xs @ W1 + b1)      # 第一隐藏层
        h2 = torch.tanh(h1 @ W2 + b2)      # 第二隐藏层
        out = h2 @ W3 + b3                 # 输出层（回归任务输出不加激活）
        loss = torch.nn.functional.mse_loss(out, ys)   # 回归用 MSE
        loss.backward()
        with torch.no_grad():
            for p in params:
                p.data -= lr * p.grad      # 每个参数沿负梯度走一步
        for p in params:
            p.grad.zero_()
        v = loss.item()
        hist.append(v if np.isfinite(v) else float('nan'))
    return hist

print("=" * 68)
print("第 8 课实验 1：学习率对训练的影响 —— sin 曲线回归（2000 步）")
print("=" * 68)
lrs = [0.0005, 0.005, 0.05, 0.5]
all_hist = {}
for lr in lrs:
    hist = train_reg(lr)
    all_hist[lr] = hist
    valid = [v for v in hist if not np.isfinite(v) == False and np.isfinite(v)]
    valid = [v for v in hist if np.isfinite(v)]
    nan = sum(1 for v in hist if np.isnan(v))
    last = hist[-1]
    last_s = f"{last:.6f}" if np.isfinite(last) else "NaN（训练爆炸）"
    print(f"lr={lr:<7} 最终loss={last_s:<16} NaN步数={nan:<5} "
          f"关键点: { {s: round(hist[s],4) if np.isfinite(hist[s]) else 'NaN' for s in [0,100,300,600,1000,1500,1999]} }")
np.savez("exp1_hist.npz", **{f"lr{lr}": np.array(all_hist[lr]) for lr in lrs})

# ================= 实验 2：sigmoid 输出层梯度对比 =================
# 直觉：模型输出 ŷ 越接近 1（很自信），但正确答案是 0（错了）时：
#   交叉熵梯度 = ŷ - y      → 错多少，梯度就多大，永远有学习信号
#   MSE 梯度 = 2(ŷ-y)ŷ(1-ŷ) → 被 ŷ(1-ŷ) 压扁，自信时几乎为 0，学不动

print()
print("=" * 68)
print("第 8 课实验 2：sigmoid 输出层 —— 交叉熵 vs MSE 的梯度（y=0）")
print("=" * 68)
print(f"{'模型输出 ŷ':>10} {'交叉熵梯度 ŷ-y':>16} {'MSE梯度 2(ŷ-y)ŷ(1-ŷ)':>24} {'MSE/CE 比值':>12}")
ratio_data = []
for yh in [0.5, 0.9, 0.99, 0.999, 0.9999]:
    ce_g = yh - 0.0
    mse_g = 2 * (yh - 0.0) * yh * (1 - yh)
    ratio_data.append((yh, ce_g, mse_g, mse_g / ce_g))
    print(f"{yh:>10.4f} {ce_g:>16.4f} {mse_g:>24.6f} {mse_g/ce_g:>12.4f}")

# ================= 实验 3：XOR 分类，CE vs MSE =================
# 相同初始化、相同 lr，唯一区别是 loss 函数。谁先学会？
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

def train_xor(lr, steps, loss_fn, seed):
    """XOR 分类训练，返回 loss 序列 + 达到 100% 准确率的步数。"""
    torch.manual_seed(seed); np.random.seed(seed)
    W1 = (torch.randn(2, 3) * 0.5).requires_grad_(); b1 = (torch.randn(3) * 0.5).requires_grad_()
    W2 = (torch.randn(3, 1) * 0.5).requires_grad_(); b2 = (torch.randn(1) * 0.5).requires_grad_()
    params = [W1, b1, W2, b2]
    hist = []
    for step in range(steps):
        z1 = X @ W1 + b1
        h = torch.relu(z1)
        z2 = h @ W2 + b2
        yh = torch.sigmoid(z2)                    # 分类任务输出要压到 0~1
        if loss_fn == 'ce':
            loss = torch.nn.functional.binary_cross_entropy(yh, y)
        else:
            loss = torch.nn.functional.mse_loss(yh, y)
        loss.backward()
        with torch.no_grad():
            for p in params:
                p.data -= lr * p.grad
        for p in params:
            p.grad.zero_()
        v = loss.item()
        hist.append(v if np.isfinite(v) else float('nan'))
    with torch.no_grad():
        yh = torch.sigmoid(torch.relu(X @ W1 + b1) @ W2 + b2)
        acc = ((yh > 0.5).float() == y).float().mean().item()
    return hist, acc

print()
print("=" * 68)
print("第 8 课实验 3：XOR 分类 —— 交叉熵 vs MSE（相同初始化，lr=0.1）")
print("=" * 68)
for seed in [0, 3, 6]:
    h_ce, acc_ce = train_xor(0.1, 2000, 'ce', seed)
    h_mse, acc_mse = train_xor(0.1, 2000, 'mse', seed)
    ce_step = next((i for i, v in enumerate(h_ce) if v < 0.05), None)
    mse_step = next((i for i, v in enumerate(h_mse) if v < 0.05), None)
    print(f"seed={seed}: CE 在 step {ce_step} loss<0.05 (acc={acc_ce:.0%}), "
          f"MSE 在 step {mse_step} loss<0.05 (acc={acc_mse:.0%})")

# 存 seed=0 的详细曲线供画图
h_ce0, _ = train_xor(0.1, 2000, 'ce', 0)
h_mse0, _ = train_xor(0.1, 2000, 'mse', 0)
np.savez("exp3_hist.npz", ce=np.array(h_ce0), mse=np.array(h_mse0))
print()
print("实验数据已保存: exp1_hist.npz（学习率对比）、exp3_hist.npz（CE vs MSE）")
