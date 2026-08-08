# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 07-handcalc.py
# 依赖: torch（venv 已装）
# 复现文章里的"手算链式法则"数字示例：样本 (0,0)、seed=42 初始权重
import torch

torch.manual_seed(42)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# 与 07-backprop.py 相同的初始化
W1 = (torch.randn(2, 3) * 0.5)
b1 = (torch.randn(3) * 0.5)
W2 = (torch.randn(3, 1) * 0.5)
b2 = (torch.randn(1) * 0.5)

print("初始权重（seed=42）：")
print("W1 =", W1.tolist())
print("b1 =", b1.tolist())
print("W2 =", W2.tolist())
print("b2 =", b2.tolist())
print()

# 取第 1 个样本 (0,0)
x0 = X[0:1]  # (1,2)
z1 = x0 @ W1 + b1
h = torch.relu(z1)
z2 = h @ W2 + b2
y_hat = torch.sigmoid(z2)
print("样本 (0,0)，y = 0：")
print(f"  z1 = {z1.tolist()}")
print(f"  h  = ReLU(z1) = {h.tolist()}")
print(f"  z2 = {z2.item():.6f}")
print(f"  ŷ  = σ(z2) = {y_hat.item():.6f}")
print()

# 反向（这一课的手写公式，N=4）
N = 4
dz2 = (y_hat - y[0:1]) / N
dh = dz2 @ W2.T
dz1 = dh * (z1 > 0).float()
dW1 = x0.T @ dz1
db1 = dz1.sum(0)
dW2 = h.T @ dz2
db2 = dz2.sum(0)
print("反向（手写公式）：")
print(f"  dz2 = (ŷ−y)/4 = {dz2.item():.6f}")
print(f"  dh  = dz2·W2ᵀ = {dh.tolist()}")
print(f"  dz1 = dh·(z1>0) = {dz1.tolist()}")
print(f"  dW1 = xᵀ·dz1 = {dW1.tolist()}")
print(f"  db1 = Σdz1 = {db1.tolist()}")
print(f"  dW2 = hᵀ·dz2 = {dW2.tolist()}")
print(f"  db2 = Σdz2 = {db2.tolist()}")
print()

# 验证：dz1 里死掉的神经元（z1<=0 的位置）梯度为 0
print(f"  z1 <= 0 的位置（ReLU 已杀死）：{(z1 <= 0).tolist()}")
