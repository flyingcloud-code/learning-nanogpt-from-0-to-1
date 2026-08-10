# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 09-optimizers.py
# 依赖: torch + numpy（venv 已装）
# 第 9 课《优化器：SGD/Adam/学习率》主实验
# 实验 1: 狭长山谷（旋转二次型）——SGD 之字形 vs Momentum 阻尼 vs Adam 自适应
# 实验 2: sin 曲线回归——三种优化器收敛对比 + 手搓 Adam 与 torch.optim.Adam 对拍
# 实验 3: 动量对震荡梯度的阻尼效果（手算表）+ Adam 首步偏差修正（手算表）
import torch
import numpy as np

torch.manual_seed(42); np.random.seed(42)

# ============================================================
# 实验 1：狭长山谷 f(x,y) = 0.5*u² + 50*v²，u=(x+y)/√2, v=(x−y)/√2
# 展开后: f = 25.25x² − 49.5xy + 25.25y²，Hessian 特征值 1 和 100（曲率差 100 倍）
# 山谷沿 x=y 方向（缓），垂直方向（陡）梯度是 100 倍
# ============================================================
print("=" * 70)
print("实验 1：狭长山谷（曲率差 100 倍）—— 同一个学习率，三种优化器走出的路径")
print("=" * 70)

def quad_grad(x, y):
    """旋转二次型的解析梯度（手算，不需要 autograd）"""
    return np.array([50.5 * x - 49.5 * y, -49.5 * x + 50.5 * y])

def quad_loss(x, y):
    return 25.25 * x * x - 49.5 * x * y + 25.25 * y * y

def run_path_sgd(start, lr, steps, beta=0.0, use_momentum=False):
    """SGD（beta=0）或 Momentum（use_momentum=True, beta=0.9）的路径"""
    pos = np.array(start, dtype=float)
    vel = np.zeros(2)
    path = [pos.copy()]
    for _ in range(steps):
        g = quad_grad(pos[0], pos[1])
        if use_momentum:
            vel = beta * vel + g
            step = vel
        else:
            step = g
        pos = pos - lr * step
        path.append(pos.copy())
    return np.array(path)

def run_path_adam(start, lr, steps, beta1=0.9, beta2=0.999, eps=1e-8):
    """手搓 Adam 的路径（每个坐标有自己的有效步长）"""
    pos = np.array(start, dtype=float)
    m = np.zeros(2); v = np.zeros(2)
    path = [pos.copy()]
    for t in range(1, steps + 1):
        g = quad_grad(pos[0], pos[1])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)          # 偏差修正：前几步 m 被 (1-β1) 压小了
        v_hat = v / (1 - beta2 ** t)
        pos = pos - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(pos.copy())
    return np.array(path)

start = (1.0, 0.8)
steps = 500
sgd_path   = run_path_sgd(start, lr=0.01, steps=steps)                      # 保守学习率
mom_path   = run_path_sgd(start, lr=0.01, steps=steps, use_momentum=True, beta=0.9)  # 动量 0.9
adam_path  = run_path_adam(start, lr=0.1, steps=steps)                      # Adam 敢用大步

print(f"起点 {start}，跑 {steps} 步，终点 loss：")
print(f"  SGD      lr=0.01 : loss = {quad_loss(sgd_path[-1][0], sgd_path[-1][1]):.6f}   位置 = ({sgd_path[-1][0]:+.3f}, {sgd_path[-1][1]:+.3f})")
print(f"  Momentum lr=0.01 : loss = {quad_loss(mom_path[-1][0], mom_path[-1][1]):.6f}   位置 = ({mom_path[-1][0]:+.3f}, {mom_path[-1][1]:+.3f})")
print(f"  Adam     lr=0.1  : loss = {quad_loss(adam_path[-1][0], adam_path[-1][1]):.6f}   位置 = ({adam_path[-1][0]:+.3f}, {adam_path[-1][1]:+.3f})")

# 对照：SGD 把学习率翻倍到 0.02（窄方向更新因子 1−100×0.02 = −1）
# → v 方向（窄方向）永远震荡不衰减，loss 卡死；0.03 则真发散（因子 −2）
osc_path = run_path_sgd(start, lr=0.02, steps=500)
print(f"  对照：SGD lr=0.02（翻倍）→ 500 步后 loss 仍 = {quad_loss(osc_path[-1][0], osc_path[-1][1]):.4f}"
      f"（窄方向更新因子 1−100×0.02 = −1，永远震荡，卡死在半路）")
div_path = run_path_sgd(start, lr=0.03, steps=20)
div_losses = [quad_loss(p[0], p[1]) for p in div_path]
print(f"  对照：SGD lr=0.03 → 第 20 步 loss = {div_losses[-1]:.3e}（因子 −2，真发散）")

np.savez("exp1_paths.npz",
         sgd=sgd_path, mom=mom_path, adam=adam_path, osc=osc_path,
         start=np.array(start), steps=steps)

# ============================================================
# 实验 2：sin 曲线回归（复用第 8 课任务）—— 三种优化器 + 手搓 Adam 对拍官方
# ============================================================
print()
print("=" * 70)
print("实验 2：sin 曲线回归（1-32-32-1 MLP, MSE, 2000 步）")
print("=" * 70)

xs = torch.linspace(-3, 3, 60).view(-1, 1)
ys = torch.sin(xs)

def make_net(seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    W1 = (torch.randn(1, 32) * 0.5).requires_grad_(); b1 = (torch.randn(32) * 0.5).requires_grad_()
    W2 = (torch.randn(32, 32) * 0.5).requires_grad_(); b2 = (torch.randn(32) * 0.5).requires_grad_()
    W3 = (torch.randn(32, 1) * 0.5).requires_grad_(); b3 = (torch.randn(1) * 0.5).requires_grad_()
    return [W1, b1, W2, b2, W3, b3]

def forward(params, x):
    W1, b1, W2, b2, W3, b3 = params
    h1 = torch.tanh(x @ W1 + b1)
    h2 = torch.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3

def train_opt(opt_name, lr, steps=2000, seed=0):
    """手搓三种优化器；'adam-torch' 用官方 torch.optim.Adam 对拍"""
    params = make_net(seed)
    hist = []
    # 优化器状态
    vel = [torch.zeros_like(p) for p in params]      # 动量速度
    m_t = [torch.zeros_like(p) for p in params]      # Adam 一阶矩
    v_t = [torch.zeros_like(p) for p in params]      # Adam 二阶矩
    if opt_name == "adam-torch":
        opt = torch.optim.Adam(params, lr=lr)
    for step in range(steps):
        loss = torch.nn.functional.mse_loss(forward(params, xs), ys)
        loss.backward()
        with torch.no_grad():
            if opt_name == "sgd":
                for p in params:
                    p.data -= lr * p.grad
            elif opt_name == "momentum":
                for i, p in enumerate(params):
                    vel[i] = 0.9 * vel[i] + p.grad
                    p.data -= lr * vel[i]
            elif opt_name in ("adam", "adam2"):
                for i, p in enumerate(params):
                    t = step + 1
                    m_t[i] = 0.9 * m_t[i] + (1 - 0.9) * p.grad
                    v_t[i] = 0.999 * v_t[i] + (1 - 0.999) * p.grad * p.grad
                    m_hat = m_t[i] / (1 - 0.9 ** t)
                    v_hat = v_t[i] / (1 - 0.999 ** t)
                    p.data -= lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
            elif opt_name == "adam-torch":
                opt.step()
        for p in params:
            p.grad.zero_()
        v = loss.item()
        hist.append(v if np.isfinite(v) else float('nan'))
    return hist

configs = [
    ("SGD        lr=0.05 ", "sgd", 0.05),
    ("Momentum   lr=0.05 ", "momentum", 0.05),
    ("手搓Adam   lr=0.001", "adam", 0.001),
    ("手搓Adam   lr=0.01 ", "adam2", 0.01),
    ("torch.Adam lr=0.001", "adam-torch", 0.001),
]
all_hist = {}
for name, opt, lr in configs:
    h = train_opt(opt, lr)
    all_hist[opt] = h
    print(f"  {name} : 最终 loss = {h[-1]:.6f}")

# 对拍：手搓 Adam vs 官方 Adam 的 loss 曲线逐点差
diff = max(abs(a - b) for a, b in zip(all_hist["adam"], all_hist["adam-torch"]))
print(f"\n对拍验证：手搓 Adam 与 torch.optim.Adam 的 loss 曲线最大逐点差 = {diff:.2e}")
print("（< 1e-5 即两者逐点一致，说明公式实现正确）")

np.savez("exp2_hist.npz",
         sgd=np.array(all_hist["sgd"]), momentum=np.array(all_hist["momentum"]),
         adam=np.array(all_hist["adam"]), adam2=np.array(all_hist["adam2"]),
         adam_torch=np.array(all_hist["adam-torch"]))

# ============================================================
# 实验 3：动量阻尼震荡梯度（手算表）+ Adam 偏差修正（手算表）
# ============================================================
print()
print("=" * 70)
print("实验 3：动量如何压住震荡（梯度交替 +10/-10，β=0.9）")
print("=" * 70)
g_seq = [10.0, -10.0, 10.0, -10.0, 10.0, -10.0]
vel = 0.0
print("  步  梯度 g    速度 v=0.9v+g    更新方向")
for i, g in enumerate(g_seq, 1):
    vel = 0.9 * vel + g
    print(f"  {i}   {g:+6.1f}   {vel:+8.2f}     {'向右' if vel > 0 else '向左'}")

print()
print("梯度连续同向（+10 十次，β=0.9）——速度会累积：")
vel = 0.0
for i in range(1, 11):
    vel = 0.9 * vel + 10.0
    if i in (1, 2, 3, 5, 10):
        print(f"  第{i:2d}步: v = {vel:+7.2f}（≈10 × {1 - 0.9**i:5.3f}）")

print()
print("Adam 首步偏差修正（梯度恒为 g，β1=0.9, β2=0.999）：")
m, v = 0.0, 0.0
for t in range(1, 6):
    g = 2.0
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    m_hat = m / (1 - 0.9 ** t)
    v_hat = v / (1 - 0.999 ** t)
    print(f"  第{t}步: m={m:.4f} → m̂={m_hat:.4f}, v={v:.4f} → v̂={v_hat:.4f}, 步长=m̂/(√v̂+ε)={m_hat/(np.sqrt(v_hat)+1e-8):.4f}")

print()
print("全部实验完成 ✅")
