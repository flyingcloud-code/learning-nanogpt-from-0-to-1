# 手搓大模型 02：第一个张量程序
# 运行命令: python 02-tensor-basics.py
# 依赖: torch (venv 已装 2.12.1)
# 内容: 张量创建 / 形状 / 设备 / 基本运算 / MPS 加速验证

import torch
import time

print("=" * 60)
print("1. 张量是什么：一个多维数字盒子")
print("=" * 60)

# 标量（0 维）：一个数
scalar = torch.tensor(7)
print("标量 tensor(7):", scalar, "| 维度:", scalar.dim(), "| 形状:", scalar.shape)

# 向量（1 维）：一排数
vector = torch.tensor([1, 2, 3])
print("向量 tensor([1,2,3]):", vector, "| 维度:", vector.dim(), "| 形状:", vector.shape)

# 矩阵（2 维）：一个表格
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("矩阵 shape:", matrix.shape)

# 3 维张量：一摞表格（后面 GPT 的输入就是 3 维的：batch x 序列长度 x 向量维度）
tensor_3d = torch.zeros(4, 8, 384)
print("3 维张量 shape (batch=4, seq_len=8, dim=384):", tensor_3d.shape)

print()
print("=" * 60)
print("2. 设备：代码在哪里计算？")
print("=" * 60)

print("MPS (Apple 芯片 GPU) 可用:", torch.backends.mps.is_available())

# 在 CPU 上创建一个张量
cpu_tensor = torch.randn(1000, 1000)
print("CPU 张量设备:", cpu_tensor.device)

# 把它搬到 MPS（GPU）上
if torch.backends.mps.is_available():
    mps_tensor = cpu_tensor.to("mps")
    print("MPS 张量设备:", mps_tensor.device)
    print("两者相加（先搬回同一设备）:", (mps_tensor + mps_tensor).shape, "✓ 计算成功")

print()
print("=" * 60)
print("3. 第一个真实计算：大矩阵乘法（CPU vs MPS 对比）")
print("=" * 60)

size = 2048
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

# CPU 计时
t0 = time.time()
_ = a_cpu @ b_cpu
cpu_time = time.time() - t0
print(f"CPU  矩阵乘法 {size}x{size}: {cpu_time:.3f} 秒")

# MPS 计时
if torch.backends.mps.is_available():
    a_mps = a_cpu.to("mps")
    b_mps = b_cpu.to("mps")
    torch.mps.synchronize()  # 确保 GPU 任务真正执行完
    t0 = time.time()
    _ = a_mps @ b_mps
    torch.mps.synchronize()
    mps_time = time.time() - t0
    print(f"MPS  矩阵乘法 {size}x{size}: {mps_time:.3f} 秒")
    print(f"加速比: {cpu_time / mps_time:.1f}x")

print()
print("=" * 60)
print("4. 张量基础运算（后面训练全用这些）")
print("=" * 60)

# 随机张量 + 数学运算
x = torch.randn(2, 3)
print("随机张量 x (randn 标准正态分布):")
print(x)
print("逐元素乘 2:", (x * 2).shape, "| 求和:", x.sum().item(), "| 均值:", x.mean().item())

# 索引和切片（和 Python 列表一样）
print("x[0]:", x[0])       # 第一行
print("x[:, 1]:", x[:, 1]) # 第二列

# reshape：改变形状但不改变数据
flat = torch.arange(6)
print("arange(6):", flat, "→ reshape(2,3):")
print(flat.reshape(2, 3))

# 转置
m = torch.tensor([[1, 2], [3, 4]])
print("转置:")
print(m)
print(m.T)

print()
print("=" * 60)
print("5. 版本验证（防止踩坑）")
print("=" * 60)
print("torch 版本:", torch.__version__)
assert torch.__version__ == "2.12.1", f"版本不对！当前 {torch.__version__}，需要 2.12.1"
print("版本检查通过 ✅（2.13.0 有 torch.save bug，请勿使用）")

print()
print("全部运行成功！第一个张量程序完成 🎉")
