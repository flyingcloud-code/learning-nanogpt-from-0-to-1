# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 04-prob-info.py
# 依赖: numpy（venv 已装）
# 第 4 课 demo：熵、交叉熵、KL 散度的直觉计算，全部真实输出
import numpy as np

print("=" * 62)
print("1. 熵 H = 平均惊讶度（一个分布本身有多不确定）")
print("=" * 62)

# 一个只有 2 个字符的世界（简化版莎士比亚：'a' 和 'b'）
def entropy(p):
    """p 是概率数组，返回熵（nait，自然对数单位）。
    约定 0*ln(0) = 0：概率为 0 的事件从不发生，不贡献惊讶度。"""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]                      # 丢掉概率为 0 的项（0*ln0 无意义）
    h = -np.sum(p * np.log(p))
    return h + 0.0                    # 熵 ≥ 0，+0.0 把 -0.0 归正（max 对负零无效）

# 分布 1：一半一半 → 完全猜不到 → 熵最大
p_even = np.array([0.5, 0.5])
# 分布 2：九成是 'a' → 基本能猜到 → 熵小
p_skew = np.array([0.9, 0.1])
# 分布 3：永远是 'a' → 毫无悬念 → 熵为 0
p_certain = np.array([1.0, 0.0])

print(f"p=[0.5, 0.5]   熵 H = {entropy(p_even):.4f}   （越平均越难猜）")
print(f"p=[0.9, 0.1]   熵 H = {entropy(p_skew):.4f}   （基本猜得到，惊讶少）")
print(f"p=[1.0, 0.0]   熵 H = {entropy(p_certain):.4f}  （永远不变，零惊讶）")

print()
print("=" * 62)
print("2. 交叉熵 H(p,q) = 用模型 q 猜真实世界 p 的平均惊讶度")
print("=" * 62)

# 真实世界 p：'a' 出现 90%
p_true = np.array([0.9, 0.1])
# 两个候选模型 q
q_good = np.array([0.85, 0.15])   # 猜得挺准
q_bad = np.array([0.1, 0.9])      # 完全猜反

def cross_entropy(p, q):
    return -np.sum(p * np.log(q))

print(f"q 猜得准 (0.85, 0.15): 交叉熵 = {cross_entropy(p_true, q_good):.4f}")
print(f"q 猜反了   (0.10, 0.90): 交叉熵 = {cross_entropy(p_true, q_bad):.4f}")
print(f"真实世界自身 p 的熵:      H(p)   = {entropy(p_true):.4f}   ← 交叉熵的下限")

print()
print("=" * 62)
print("3. KL 散度 = 交叉熵 - 熵 = 因为猜错多付的惊讶")
print("=" * 62)

for name, q in [("猜得准", q_good), ("猜反了", q_bad)]:
    kl = cross_entropy(p_true, q) - entropy(p_true)
    print(f"{name} (q={q.tolist()}): KL = {kl:.4f}  （交叉熵比下限多出的部分）")

print()
print("=" * 62)
print("4. 回到 GPT：loss 就是交叉熵（nait 单位）")
print("=" * 62)

V = 65  # 莎士比亚数据集词表大小（第 1 课验证过）
print(f"词表大小 V = {V}")
print(f"瞎猜的熵   = ln({V}) = {np.log(V):.4f}  （均匀分布，loss 起点）")
print(f"训练 300 步 loss ≈ 4.27（比瞎猜还高，说明一开始连均匀都不如）")
print(f"训练 1000 步 loss = 1.28 → 平均惊讶度 e^1.28 = {np.exp(1.28):.2f}")
print(f"含义：模型平均只需在 {np.exp(1.28):.1f} 个字符里纠结，而不是 65 个")
print()
print("loss 数值越小 = 平均惊讶度越小 = 模型越会猜下一个字符")
