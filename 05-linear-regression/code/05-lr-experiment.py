# 05-lr-experiment.py
# 验证：学习率过大时梯度下降会震荡甚至发散（同一份数据，只改 lr）
# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 05-lr-experiment.py
import numpy as np

rng = np.random.default_rng(42)
N = 50
x = rng.uniform(-3, 3, N)
true_w, true_b = 2.0, 1.0
y = true_w * x + true_b + rng.normal(0, 0.5, N)

def train(lr, steps=60):
    w, b = 0.0, 0.0
    losses = []
    for _ in range(steps):
        y_hat = w * x + b
        diff = y_hat - y
        loss = np.mean(diff ** 2)
        losses.append(loss)
        w -= lr * np.mean(2 * diff * x)
        b -= lr * np.mean(2 * diff)
    return w, b, losses

for lr in [0.05, 0.2, 1.0]:
    w, b, losses = train(lr)
    print(f"lr={lr:<5} 最终 w={w:+.4f} b={b:+.4f}  loss 序列前6: " +
          " ".join(f"{v:.2f}" for v in losses[:6]) + " ... 最后: " + " ".join(f"{v:.2f}" for v in losses[-4:]))
