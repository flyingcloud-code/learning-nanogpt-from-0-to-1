# 05-linear-regression.py
# 手搓大模型 05：NumPy 手写线性回归 + 梯度下降
# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 05-linear-regression.py
# 依赖: numpy（venv 已装）
#
# 这一课只做一件事：从零实现"训练"本身。
# 不用 torch，不用 sklearn，只用 numpy 做小学算数级别的运算，
# 让模型自己从数据里"学"出 y = 2x + 1 这条规律。

import numpy as np

# ---------- 1. 造一份"真实世界"的数据 ----------
# 假设世界的真相是 y = 2x + 1，但观测有噪声（现实世界总是有噪声的）
rng = np.random.default_rng(42)          # 固定随机种子，结果可复现
N = 50                                   # 50 个样本
x = rng.uniform(-3, 3, N)                # 输入 x：-3 到 3 均匀分布
true_w, true_b = 2.0, 1.0                # 世界的真实参数（读者知道，模型不知道）
y = true_w * x + true_b + rng.normal(0, 0.5, N)   # y = 2x+1 + 噪声

# ---------- 2. 模型与参数 ----------
# 线性模型：y_hat = w * x + b
# 训练前参数全猜 0（模型对世界一无所知）
w = 0.0
b = 0.0
lr = 0.05                                # 学习率：每次下山迈多大的步子
steps = 200

print("=" * 66)
print("数据：50 个样本，真相是 y = 2x + 1，观测带噪声（std=0.5）")
print(f"初始参数: w = {w}, b = {b}  （模型猜的，啥也不知道）")
print(f"初始 loss = {np.mean((w * x + b - y) ** 2):.4f}")
print("=" * 66)

# ---------- 3. 训练循环（核心就 6 行） ----------
losses = []                              # 记录每一步的 loss，画图用
for step in range(steps):
    y_hat = w * x + b                    # 前向：用当前参数算预测值
    diff = y_hat - y                     # 误差：预测 - 真实
    loss = np.mean(diff ** 2)            # loss = 平均误差平方（MSE）
    losses.append(loss)

    grad_w = np.mean(2 * diff * x)       # loss 对 w 的梯度：误差乘 x 再平均
    grad_b = np.mean(2 * diff)           # loss 对 b 的梯度：误差平均

    w -= lr * grad_w                     # 往梯度反方向走一步（下山）
    b -= lr * grad_b

    if step < 5 or step % 20 == 0:
        print(f"step {step:3d}  loss = {loss:.4f}  w = {w:+.4f}  b = {b:+.4f}")

# ---------- 4. 结果 ----------
print("=" * 66)
print(f"真实参数: w = {true_w}, b = {true_b}")
print(f"学到的:   w = {w:.4f}, b = {b:.4f}")
print(f"最终 loss = {losses[-1]:.4f}（噪声 std=0.5、方差 0.25，loss 已贴住噪声地板）")

# 泛化彩蛋：x = 10 从未出现在训练数据里（数据只有 -3 到 3）
y_predict = w * 10 + b
print(f"预测 x = 10 → y ≈ {y_predict:.2f}（训练数据里根本没有 x=10，模型靠规律外推）")
print(f"如果知道真相：2*10+1 = {true_w * 10 + true_b}")
