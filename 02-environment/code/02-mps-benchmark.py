# 手搓大模型 02：CPU vs MPS 大矩阵乘法对比（真实数据）
# 运行命令: python 02-mps-benchmark.py
# 依赖: torch (2.12.1)
# 内容: 不同规模矩阵乘法下 CPU vs MPS 的耗时对比，验证 GPU 加速

import torch
import time

def bench_cpu(size):
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    # warmup
    _ = a @ b
    t0 = time.time()
    _ = a @ b
    return time.time() - t0

def bench_mps(size):
    a = torch.randn(size, size, device="mps")
    b = torch.randn(size, size, device="mps")
    _ = a @ b
    torch.mps.synchronize()
    t0 = time.time()
    _ = a @ b
    torch.mps.synchronize()
    return time.time() - t0

print(f"{'规模':>12} | {'CPU(s)':>10} | {'MPS(s)':>10} | {'加速比':>8}")
print("-" * 50)
for size in [1024, 2048, 4096, 8192]:
    t_cpu = bench_cpu(size)
    t_mps = bench_mps(size)
    ratio = t_cpu / t_mps
    print(f"{size:>6}x{size:<6} | {t_cpu:>10.3f} | {t_mps:>10.3f} | {ratio:>7.1f}x")
