# -*- coding: utf-8 -*-
"""
第 10 课主实验：泛化 —— 过拟合与正则化
========================================
一句话：训练 loss 越低 ≠ 考得越好。模型会把训练集里的噪声也背下来，
val loss 先降后升的"过拟合曲线"就是证据。

运行命令：
  ~/projects/main-agent/nanoGPT/.venv/bin/python 10-overfit.py

依赖：torch 2.12.1 + numpy（venv 已装）
输出：
  - 终端打印四种配置的 train/val loss、val 最优点、权重范数
  - exp10.npz（供 10-make-charts.py 画图）
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
LR = 1e-3
STEPS = 10000            # 全量梯度下降步数

# ---------- 数据：sin 曲线的 12 个带噪训练点 + 300 个干净验证点 ----------
np.random.seed(0)
x_train = np.random.uniform(-1.5, 1.5, 12).astype(np.float32)
y_train = (np.sin(2 * np.pi * x_train) + 0.3 * np.random.randn(12)).astype(np.float32)
x_val = np.linspace(-1.5, 1.5, 300).astype(np.float32)
y_val = np.sin(2 * np.pi * x_val).astype(np.float32)   # 验证集无噪声 = "真实函数"

xt = torch.tensor(x_train[:, None])
yt = torch.tensor(y_train[:, None])
xv = torch.tensor(x_val[:, None])
yv = torch.tensor(y_val[:, None])


def make_model(use_dropout=False):
    """大容量 MLP：1-512-512-1，约 26 万参数，12 个点根本喂不饱它 -> 必过拟合。
    固定初始种子：四种配置从同一个起点出发，对比才公平。"""
    torch.manual_seed(42)
    if use_dropout:
        return nn.Sequential(
            nn.Linear(1, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 1),
        )
    return nn.Sequential(
        nn.Linear(1, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 1),
    )


def train(name, lam=0.0, use_dropout=False, early_stop=False):
    """训练一个配置。lam 是 L2 正则系数（loss = MSE + lam * 权重平方和）。"""
    model = make_model(use_dropout)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    hist_train, hist_val = [], []
    best_val, best_step, best_state = float("inf"), 0, None

    for step in range(1, STEPS + 1):
        model.train()
        opt.zero_grad()
        mse = loss_fn(model(xt), yt)
        l2 = sum((p ** 2).sum() for p in model.parameters())   # 所有权重平方和
        loss = mse + lam * l2                                   # 岭回归式惩罚
        loss.backward()
        opt.step()

        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                t_loss = loss_fn(model(xt), yt).item()
                v_loss = loss_fn(model(xv), yv).item()
            hist_train.append(t_loss)
            hist_val.append(v_loss)
            if v_loss < best_val:
                best_val, best_step = v_loss, step
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 最终模型在验证网格上的预测（画拟合曲线用）
    model.eval()
    with torch.no_grad():
        final_train = loss_fn(model(xt), yt).item()
        final_val = loss_fn(model(xv), yv).item()
        pred = model(xv).numpy().ravel()

    w_norm = sum((p ** 2).sum().item() for p in model.parameters())

    if early_stop:   # 早停：把参数恢复回 val loss 最低的那一步
        model.load_state_dict(best_state)
        with torch.no_grad():
            es_train = loss_fn(model(xt), yt).item()
            es_val = loss_fn(model(xv), yv).item()
            es_pred = model(xv).numpy().ravel()
    else:
        es_train = es_val = es_pred = None

    return dict(name=name, hist_train=hist_train, hist_val=hist_val,
                final_train=final_train, final_val=final_val,
                best_val=best_val, best_step=best_step,
                w_norm=w_norm, pred=pred,
                es_train=es_train, es_val=es_val, es_pred=es_pred)


def main():
    configs = [
        ("baseline",           dict(lam=0.0,    use_dropout=False, early_stop=False)),
        ("l2_1e-3",            dict(lam=1e-3,   use_dropout=False, early_stop=False)),
        ("dropout0.4",         dict(lam=0.0,    use_dropout=True,  early_stop=False)),
        ("baseline+earlystop", dict(lam=0.0,    use_dropout=False, early_stop=True)),
    ]
    results = {}
    for name, kw in configs:
        print(f"训练 {name} ...", flush=True)
        results[name] = train(name, **kw)

    print("\n" + "=" * 86)
    print(f"{'配置':<20}{'最终train':>12}{'最终val':>12}{'val最低':>12}{'最低步数':>10}{'权重范数':>12}")
    print("-" * 86)
    for name, r in results.items():
        print(f"{name:<20}{r['final_train']:>12.5f}{r['final_val']:>12.4f}"
              f"{r['best_val']:>12.4f}{r['best_step']:>10d}{r['w_norm']:>12.1f}")
    print("-" * 86)
    es = results["baseline+earlystop"]
    print("baseline+earlystop 恢复最优点后: train = %.5f, val = %.4f（省下 %d 步）"
          % (es["es_train"], es["es_val"], STEPS - es["best_step"]))

    # 过拟合分叉点：train 与 val 首次明显拉开差距的步数
    bt = np.array(results["baseline"]["hist_train"])
    bv = np.array(results["baseline"]["hist_val"])
    gap = bv - bt
    fork = 100 * (np.argmax(gap > 0.1) + 1) if np.any(gap > 0.1) else STEPS
    print(f"\nbaseline 过拟合分叉点：第 {fork} 步后 val loss 明显高于 train loss（val 开始回升）")

    np.savez("exp10.npz",
             x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
             steps=np.arange(100, STEPS + 1, 100),
             **{f"{k}_hist_train": v["hist_train"] for k, v in results.items()},
             **{f"{k}_hist_val": v["hist_val"] for k, v in results.items()},
             **{f"{k}_pred": v["pred"] for k, v in results.items()},
             **{f"{k}_es_pred": v["es_pred"] for k, v in results.items() if v["es_pred"] is not None})


if __name__ == "__main__":
    main()
