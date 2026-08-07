# 运行命令: ~/projects/main-agent/nanoGPT/.venv/bin/python 06-make-charts.py
# 依赖: matplotlib + numpy（venv 已装）；先跑 06-perceptron-mlp.py 生成 _06_data.npz
# 生成第 6 课配图：左=单层 vs MLP 的 loss 对比曲线，右=XOR 决策边界热力图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# 输出目录：脚本在 code/ 下时 → 上级 images/；脚本在课程根目录时 → 同级 images/
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "code" else os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# macOS 中文字体
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

d = np.load("_06_data.npz")
losses_lin = d["losses_lin"]
losses_mlp = d["losses_mlp"]
W1, b1, W2, b2 = d["W1"], d["b1"], d["W2"], d["b2"]
W_lin, b_lin = d["W_lin"], d["b_lin"]

# ===== 左图：loss 对比曲线（真实训练数据） =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

steps = np.arange(len(losses_lin))
ax1.plot(steps, losses_lin, color="#fb7185", linewidth=2.2, label="单层感知机（1 个神经元）")
ax1.plot(steps, losses_mlp, color="#22d3ee", linewidth=2.2, label="2 层 MLP（隐藏层 3 个神经元）")
ax1.axhline(np.log(2), color="#64748b", linestyle="--", linewidth=1.2)
ax1.text(100, np.log(2) + 0.02, "ln(2) = 0.693（瞎猜线：输出恒为 0.5 的 loss）", color="#94a3b8", fontsize=9)
ax1.set_xlabel("训练步数")
ax1.set_ylabel("loss（二分类交叉熵）")
ax1.set_title("同样训练 3000 步：单层卡死，MLP 一路下降", fontsize=11)
ax1.legend(fontsize=9)
ax1.set_ylim(0, 0.85)
ax1.grid(alpha=0.25, color="#334155")

# ===== 右图：XOR 决策边界热力图 =====
# 在 [−0.3, 1.3]² 网格上采样，看模型把每个点判成 0 还是 1
def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-z))

def mlp_predict_np(xx):
    z1 = xx @ W1 + b1
    h = np.maximum(z1, 0)  # ReLU
    return sigmoid_np(h @ W2 + b2).reshape(xx.shape[0])

def lin_predict_np(xx):
    return sigmoid_np(xx @ W_lin + b_lin).reshape(xx.shape[0])

gx = np.linspace(-0.3, 1.3, 300)
gy = np.linspace(-0.3, 1.3, 300)
GX, GY = np.meshgrid(gx, gy)
pts = np.stack([GX.ravel(), GY.ravel()], axis=1)

# 单层：分界是一条直线（线性），MLP：分界弯起来了
P_lin = lin_predict_np(pts).reshape(GX.shape)
P_mlp = mlp_predict_np(pts).reshape(GX.shape)

X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_data = np.array([0, 1, 1, 0])

# 两个子图：左=单层边界，右=MLP 边界
for ax, P, title, color in [
    (ax1, None, None, None),  # placeholder, will skip
]:
    pass

# 单独建第二个 figure 的左右两半
fig2, (axl, axr) = plt.subplots(1, 2, figsize=(13.5, 5.6))
for ax, P, title in [
    (axl, P_lin, "单层感知机：一条直线，怎么都分不开"),
    (axr, P_mlp, "2 层 MLP：边界弯了一下，四个点全部分开"),
]:
    cs = ax.contourf(GX, GY, P, levels=np.linspace(0, 1, 21), cmap="RdYlBu_r", alpha=0.85)
    ax.contour(GX, GY, P, levels=[0.5], colors="#fbbf24", linewidths=2.2)
    # XOR 四个点
    colors_pt = ["#0ea5e9" if yy == 0 else "#f43f5e" for yy in y_data]
    for (xx, yy), cc in zip(X_data, colors_pt):
        ax.scatter(xx, yy, s=160, c=cc, edgecolors="white", linewidths=1.8, zorder=5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2, color="#334155")

fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(os.path.join(IMG_DIR, "06-loss-comparison.png"), dpi=150, facecolor="#0f172a")
fig2.tight_layout()
fig2.savefig(os.path.join(IMG_DIR, "06-xor-boundary.png"), dpi=150, facecolor="#0f172a")
print(f"saved {os.path.join(IMG_DIR, '06-loss-comparison.png')}")
print(f"saved {os.path.join(IMG_DIR, '06-xor-boundary.png')}")
