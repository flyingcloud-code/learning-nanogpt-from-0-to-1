# 03-math-intuition.py
# 《手搓大模型》第 3 课：线代 + 微积分直觉
# 运行：python 03-math-intuition.py   （依赖：torch 2.12.1 / numpy）
import torch

# ========== 1. 矩阵乘法 = 批量映射 ==========
print("=" * 60)
print("1. 矩阵乘法 = 批量映射")
print("=" * 60)

# 一个 2x3 的矩阵 A，作用在一个 3 维输入 x 上，得到一个 2 维输出
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
x = torch.tensor([1.0, 0.0, -1.0])
print("A 的形状:", tuple(A.shape), " x 的形状:", tuple(x.shape))
print("A @ x =", (A @ x).tolist())
# 手算第一行：1*1 + 2*0 + 3*(-1) = -2 ；第二行：4*1 + 5*0 + 6*(-1) = -2

# 关键直觉：矩阵乘法一次能同时处理"一整批"输入，这就是批量映射
X = torch.stack([x, x * 2, x - 1, -x])   # 4 个不同的输入
print("批量输入 X 的形状:", tuple(X.shape), "（4 个样本，每个 3 维）")
Y = X @ A.T                              # 每个样本都被同一个 A 映射
print("批量输出 Y 的形状:", tuple(Y.shape), "（4 个样本，每个 2 维）")
print("第 1 个输入映射结果:", Y[0].tolist())

# GPT 视角：每个 token 向量都要过一遍权重矩阵，矩阵乘法就是"一次算完全部 token"
tokens = torch.randn(4, 8, 384)          # 第 2 课见过的形状 [batch, seq, dim]
W = torch.randn(384, 128)                # 一个权重矩阵：384 维 -> 128 维
out = tokens @ W
print("GPT 输入形状:", tuple(tokens.shape), "-> 乘权重后:", tuple(out.shape))

# ========== 2. 导数 = 斜率 ==========
print("=" * 60)
print("2. 导数 = 斜率")
print("=" * 60)

def f(x):
    return x ** 2

# 数值求导：用"附近两点连线斜率"近似"那一点的切线斜率"
def derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

for x in [1.0, 2.0, 3.0]:
    print(f"x={x:>3}: f(x)={f(x):>5.1f}  数值导数≈{derivative(f, x):.4f}  解析 2x={2 * x:.1f}")

# ========== 3. 链式法则 = 拆开一层层求 ==========
print("=" * 60)
print("3. 链式法则：复合函数一层层拆")
print("=" * 60)

def g(x):
    return 3 * x + 2          # 内层：直线

def h(u):
    return u ** 2             # 外层：平方

def f_comp(x):
    return h(g(x))            # 复合：f(x) = (3x+2)^2

# 解析答案：df/dx = 2*(3x+2)*3 = 6*(3x+2)
for x in [0.5, 1.5]:
    num = derivative(f_comp, x)
    analytic = 6 * (3 * x + 2)
    print(f"x={x}: 数值≈{num:.4f}  链式法则公式={analytic:.4f}  完全一致={abs(num - analytic) < 1e-4}")

# 神经网络版：y = (3x+2)^2 就是"两层网络"的最简形式
# 第 1 层：u = 3x + 2 （一个神经元，权重 3，偏置 2）
# 第 2 层：y = u^2   （一个非线性激活）
# 反向传播就是链式法则：从 y 出发，把梯度一层层往回传
print("\n彩蛋：两层网络 = 链式法则")
u_val = g(torch.tensor(0.5))
print("x=0.5 时，第一层输出 u =", float(u_val))
print("y = u^2 =", float(h(u_val)))
