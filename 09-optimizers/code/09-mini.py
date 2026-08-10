# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 09-mini.py
# 依赖: numpy（venv 已装）
# 第 9 课最小演示：手搓三个优化器，在"狭长山谷"上跑 300 步看结局
import numpy as np

def grad(x, y):            # 山谷的梯度（解析式，手算即可，不用 autograd）
    return np.array([50.5 * x - 49.5 * y, -49.5 * x + 50.5 * y])

def loss(x, y):            # 山谷的 loss：两个方向曲率差 100 倍
    return 25.25 * x * x - 49.5 * x * y + 25.25 * y * y

def run(name, lr, steps=300, beta=0.9):
    pos = np.array([1.0, 0.8])     # 起点
    vel = np.zeros(2)              # 动量速度
    m = np.zeros(2); v = np.zeros(2)   # Adam 的一阶矩、二阶矩
    for t in range(1, steps + 1):
        g = grad(*pos)
        if name == "sgd":                 # 1. 裸 SGD：只看当前梯度
            step = g
        elif name == "momentum":          # 2. 动量：速度 = 0.9×旧速度 + 当前梯度
            vel = beta * vel + g
            step = vel
        else:                             # 3. Adam：每个方向有自己的"自动步长"
            m = 0.9 * m + 0.1 * g         # 一阶矩：梯度的指数平均（方向）
            v = 0.999 * v + 0.001 * g * g # 二阶矩：梯度平方的指数平均（大小）
            m_hat = m / (1 - 0.9 ** t)    # 偏差修正：把前几步被压小的值拉回来
            v_hat = v / (1 - 0.999 ** t)
            step = m_hat / (np.sqrt(v_hat) + 1e-8)   # 归一化：每步约走 lr
        pos = pos - lr * step
    return loss(*pos)

for name, lr in [("sgd", 0.01), ("momentum", 0.01), ("adam", 0.1)]:
    print(f"{name:8s} lr={lr:<4} 300 步后 loss = {run(name, lr):.3e}")
