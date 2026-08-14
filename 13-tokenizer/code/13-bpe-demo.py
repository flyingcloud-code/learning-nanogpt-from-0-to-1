#!/usr/bin/env python3
"""
13-bpe-demo.py — 第 13 课《Tokenizer：BPE 原理》实验脚本
依赖: tiktoken + numpy（venv 已装）
运行: ~/projects/main-agent/nanoGPT/.venv/bin/python 13-bpe-demo.py
"""
import os
import re
import json
from collections import Counter, defaultdict

# ─────────────────────────────────────────────
# 0. 数据
# ─────────────────────────────────────────────
DATA = os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char")
with open(os.path.join(DATA, "input.txt"), "r", encoding="utf-8") as f:
    full_text = f.read()
text = full_text[:50000]  # 演示用前 5 万字符

print("=" * 62)
print("第 13 课实验：字符级 vs BPE 词块级")
print("=" * 62)
print(f"演示文本：莎士比亚 input.txt 前 {len(text):,} 字符")

# ─────────────────────────────────────────────
# 1. 字符级（前 12 课的做法）：一个字符 = 一个 token
# ─────────────────────────────────────────────
char_tokens = len(text)
print(f"\n[1] 字符级分词：1 个字符 = 1 个 token")
print(f"    token 数 = {char_tokens:,}")

# ─────────────────────────────────────────────
# 2. 手写一个简化版 BPE：只做“统计相邻对 + 合并”
#    输入：一串字符；输出：合并了 num_merges 次后的词块序列
# ─────────────────────────────────────────────

def train_simple_bpe(tokens, num_merges):
    """最简 BPE：反复找最高频相邻对，合并成一个新 token。
    tokens 是 list[str]（每个元素一个 token/词块）。
    返回 (最终 token 序列, 合并记录列表)。
    """
    merges = []
    next_id = 0
    # 给初始字符分配 id（用负数偏移避免与真实 id 冲突，这里直接用字符当 token）
    cur = tokens[:]
    for step in range(num_merges):
        # 统计相邻对频率
        pairs = Counter(zip(cur, cur[1:]))
        if not pairs:
            break
        (a, b), cnt = pairs.most_common(1)[0]
        new_tok = f"{a}{b}"  # 合并成新词块（真实实现会给新 id，这里用字符串拼接演示）
        # 扫描合并
        new_cur = []
        i = 0
        while i < len(cur):
            if i + 1 < len(cur) and cur[i] == a and cur[i + 1] == b:
                new_cur.append(new_tok)
                i += 2
            else:
                new_cur.append(cur[i])
                i += 1
        merges.append((a, b, cnt, len(new_cur)))
        cur = new_cur
    return cur, merges


print("\n[2] 经典例子：'low low low low low lower lowest newest'")
sample = "low low low low low lower lowest newest"
sample_tokens = list(sample)
_, merges = train_simple_bpe(sample_tokens, 5)
for i, (a, b, cnt, remain) in enumerate(merges, 1):
    print(f"    合并 {i}: '{a}'+'{b}' -> '{a}{b}'   出现 {cnt} 次   剩余词块 {remain}")

print("\n[3] 莎士比亚 5 万字符上的真实合并过程（前 12 次）")
sh_tokens = list(text)
_, merges = train_simple_bpe(sh_tokens, 12)
for i, (a, b, cnt, remain) in enumerate(merges, 1):
    display_a = a.replace("\n", "\\n")
    display_b = b.replace("\n", "\\n")
    print(f"    合并 {i:2d}: {display_a!r:8s} + {display_b!r:8s} -> {display_a!r}{display_b!r:10s}  频次 {cnt:6d}")

# ─────────────────────────────────────────────
# 3. tiktoken：真实 GPT-2 词块分词（r50k_base，GPT-2 用的就是它）
# ─────────────────────────────────────────────
print("\n[4] tiktoken 真实分词对比（r50k_base = GPT-2 用的 BPE 词表，50257 个词块）")
import tiktoken
enc = tiktoken.get_encoding("r50k_base")
bpe_ids = enc.encode(text)
print(f"    同一段 5 万字符：")
print(f"      字符级 token 数 = {char_tokens:,}")
print(f"      BPE 词块数     = {len(bpe_ids):,}")
print(f"      压缩比 = {char_tokens / len(bpe_ids):.2f}x")

# 常用词怎么切
print("\n    几个词的 BPE 切法（GPT-2 真实词表）：")
for w in ["Romeo", "Juliet", "love", "the", " Shakespeare", "low", "lowest", "hello", " world"]:
    ids = enc.encode(w)
    pieces = [enc.decode([i]) for i in ids]
    print(f"      {w!r:18s} -> {pieces}")

# 生僻/英文词根
print("\n    词根/词缀真实切法：")
for w in ["unhappiness", "incredibly", "tokenization", "Tokenizing"]:
    ids = enc.encode(w)
    pieces = [enc.decode([i]) for i in ids]
    print(f"      {w!r:18s} -> {pieces}")

# ─────────────────────────────────────────────
# 4. 图表数据：字符 vs BPE 的 token 数（不同文本长度下）
# ─────────────────────────────────────────────
print("\n[5] 生成图表数据 chart-data.json")
lens = [1000, 5000, 10000, 25000, 50000]
char_counts, bpe_counts = [], []
for n in lens:
    seg = full_text[:n]
    char_counts.append(len(seg))
    bpe_counts.append(len(enc.encode(seg)))
    print(f"    {n:>6,} 字符 -> BPE {len(enc.encode(seg)):>6,} 词块  ({n/len(enc.encode(seg)):.2f}x)")
with open("chart-data.json", "w") as f:
    json.dump({"lens": lens, "char": char_counts, "bpe": bpe_counts}, f)

print("\n完成 ✅")
