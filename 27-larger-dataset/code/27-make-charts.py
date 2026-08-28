#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 27 课：用真实运行数据画对比图。
用法: python 27-make-charts.py
依赖: numpy + matplotlib（venv 已装）
输入: run-bpe-shakes.log / run-bpe-novels.log / 26 课 train.log（字符模型）
输出: images/27-loss-compare.png（左右两联：per-token + per-char）
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# CJK 字体（macOS）：否则中文标题渲染成方块
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC", "PingFang SC", "Hiragino Sans GB"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images")
os.makedirs(OUT, exist_ok=True)

# 各自语料的真实压缩比（chars / bpe tokens）
def compute_ratio(path):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    text = open(path, encoding="utf-8").read()
    return len(text) / len(enc.encode(text))

RATIO_SHAKES = compute_ratio("/Users/openclaw-master/projects/main-agent/nanoGPT/data/shakespeare_char/input.txt")
RATIO_NOVELS = compute_ratio(os.path.join(HERE, "..", "novels.txt"))
print(f"压缩比: 莎翁 {RATIO_SHAKES:.2f}  小说集 {RATIO_NOVELS:.2f}")


def parse_log(path, key="loss"):
    steps, vals = [], []
    if not os.path.exists(path):
        return steps, vals
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if key in d and "step" in d:
            steps.append(d["step"])
            vals.append(d[key])
    return steps, vals


def parse_char_log(path):
    """26 课 train.log 是文本行: '  step  1000  loss 1.3748 ... val 1.5274'"""
    steps, losses, vals = [], [], []
    for line in open(path):
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "step" and parts[2] == "loss":
            steps.append(int(parts[1]))
            losses.append(float(parts[3]))
            for i, p in enumerate(parts):
                if p == "val":
                    vals.append((int(parts[1]), float(parts[i + 1])))
    return steps, losses, vals


def main():
    # ---- 数据 ----
    s_steps, s_loss = parse_log(os.path.join(HERE, "run-bpe-shakes.log"))
    n_steps, n_loss = parse_log(os.path.join(HERE, "run-bpe-novels.log"))
    s_val_steps, s_val = parse_log(os.path.join(HERE, "run-bpe-shakes.log"), key="val")
    n_val_steps, n_val = parse_log(os.path.join(HERE, "run-bpe-novels.log"), key="val")
    c_steps, c_loss, c_vals = parse_char_log(os.path.join(HERE, "..", "..", "26-milestone-train", "code", "train.log"))

    # per-char 换算：BPE loss ÷ 各自压缩比（一个 token ≈ 3.3/3.7 个字符）；字符模型 1 token = 1 字符
    s_loss_char = [v / RATIO_SHAKES for v in s_loss]
    n_loss_char = [v / RATIO_NOVELS for v in n_loss]
    s_val_char = [v / RATIO_SHAKES for v in s_val]
    n_val_char = [v / RATIO_NOVELS for v in n_val]

    # ---- 左图：per-token loss（词表不同，绝对值不可直接比）----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    ax1.plot(s_steps, s_loss, label="BPE@莎翁 (train)", color="#0F4C81")
    ax1.plot(n_steps, n_loss, label="BPE@小说集 (train)", color="#D97757")
    ax1.scatter(s_val_steps, s_val, label="BPE@莎翁 (val)", color="#0F4C81", marker="o", s=30)
    ax1.scatter(n_val_steps, n_val, label="BPE@小说集 (val)", color="#D97757", marker="o", s=30)
    ax1.set_title("按 token 算（词表不同，绝对值不可比）")
    ax1.set_xlabel("训练步数")
    ax1.set_ylabel("loss（nats/token）")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ---- 右图：per-char loss（换算到每个字符，公平对比）----
    ax2.plot(s_steps, s_loss_char, label="BPE@莎翁 (train)", color="#0F4C81")
    ax2.plot(n_steps, n_loss_char, label="BPE@小说集 (train)", color="#D97757")
    ax2.plot(c_steps, c_loss, label="字符@莎翁 (train, 26课)", color="#A9A9A9", alpha=0.85)
    ax2.scatter(s_val_steps, s_val_char, label="BPE@莎翁 (val)", color="#0F4C81", marker="o", s=30)
    ax2.scatter(n_val_steps, n_val_char, label="BPE@小说集 (val)", color="#D97757", marker="o", s=30)
    cv_steps = [v[0] for v in c_vals]
    cv_vals = [v[1] for v in c_vals]
    ax2.scatter(cv_steps, cv_vals, label="字符@莎翁 (val, 26课)", color="#A9A9A9", marker="x", s=30)
    ax2.set_title("按字符算（÷压缩比 3.30，公平对比）")
    ax2.set_xlabel("训练步数")
    ax2.set_ylabel("loss（nats/字符）")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("第 27 课：换词表 + 换语料 —— 三种模型真实训练曲线（Mac mini MPS）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "27-loss-compare.png")
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
