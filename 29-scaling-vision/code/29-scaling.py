#!/usr/bin/env python3
"""
29-scaling.py — Scaling Law 幂律拟合（真实公开数据）
数据来源：GPT-2 论文 "Language Models are Unsupervised Multitask Learners" (2019)
Table 3 / Figure 4：WebText 测试集困惑度（PPL）
  117M  -> 35.76
  345M  -> 22.76
  762M  -> 19.41
  1542M -> 17.48

运行：python 29-scaling.py  （依赖 numpy + matplotlib，系列 venv 已装）
输出：29-scaling-curve.png（对数-对数坐标曲线）+ 拟合参数 + 外推预测
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
# macOS 中文字体（PingFang SC），避免方块字
for f in ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 公开数据点（GPT-2 论文，一字不改） ----------
params = np.array([117e6, 345e6, 762e6, 1542e6])   # 参数量 N
ppl    = np.array([35.76, 22.76, 19.41, 17.48])    # WebText 测试困惑度

# ---------- 2. 幂律拟合：PPL = a * N^(-b) ----------
# 取对数变成直线：ln(PPL) = ln(a) - b * ln(N)
# numpy.polyfit 做一次多项式（直线）拟合，返回 [斜率, 截距]
log_N = np.log(params)
log_P = np.log(ppl)
slope, intercept = np.polyfit(log_N, log_P, 1)
# slope 就是 -b，intercept 就是 ln(a)
b = -slope
a = np.exp(intercept)
print(f"[fit] PPL = {a:.2f} * N^(-{b:.4f})")
print(f"[fit] 对数坐标斜率 = {slope:.4f}（每翻一倍参数，ln(PPL) 下降 {(-slope*np.log(2)):.4f}）")

# ---------- 3. 外推：假如继续放大，PPL 会到多少 ----------
for N_future in [10e9, 175e9]:
    pred = a * N_future ** (-b)
    print(f"[extrap] N={N_future/1e9:.0f}B 参数 -> 预测 PPL ≈ {pred:.2f}")

# ---------- 4. 画图：对数-对数坐标 ----------
xs = np.linspace(params.min() * 0.6, params.max() * 2.2, 200)
ys = a * xs ** (-b)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(xs, ys, color="#0F4C81", lw=2.2, label=f"幂律拟合 PPL = {a:.0f}·N$^{{-{b:.3f}}}$")
ax.scatter(params, ppl, color="#FA5151", s=90, zorder=5, label="GPT-2 论文真实数据点")
for N, P in zip(params, ppl):
    ax.annotate(f"{int(N/1e6)}M", (N, P), textcoords="offset points",
                xytext=(8, -10), fontsize=11, color="#333333", fontweight="bold")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("参数量 N（log 坐标）", fontsize=12)
ax.set_ylabel("WebText 测试困惑度 PPL（log 坐标）", fontsize=12)
ax.set_title("Scaling Law：参数越大，困惑度平滑下降（GPT-2 论文公开数据）", fontsize=13)
ax.grid(True, which="both", ls="--", alpha=0.35)
ax.legend(fontsize=11, loc="upper right")
plt.tight_layout()
plt.savefig("29-scaling-curve.png", dpi=150)
print("[chart] saved 29-scaling-curve.png")
