#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 27 课：字符切分 vs BPE 子词切分 —— 同一个文本，两种 token 化。
用法: python 27-tokenize.py
依赖: tiktoken（venv 已装）。输出: tokenize-stats.json + 终端对比表
"""
import json

import tiktoken

DATA = "/Users/openclaw-master/projects/main-agent/nanoGPT/data/shakespeare_char/input.txt"
SAMPLE = "To be, or not to be: that is the question:\n"


def char_tokenize(text):
    """字符级切分：每个字符一个 token（第 1-26 课的做法）。"""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}  # 65 个字符 -> 编号
    return [stoi[c] for c in text], len(chars)


def bpe_tokenize(text, enc):
    """BPE 子词切分：tiktoken 的 gpt2 词表（50257 个子词，第 13 课的原理）。"""
    return enc.encode(text)


def main():
    text = open(DATA, encoding="utf-8").read()
    enc = tiktoken.get_encoding("gpt2")

    char_ids, char_vocab = char_tokenize(text)
    bpe_ids = bpe_tokenize(text, enc)

    print(f"文本总字符数: {len(text):,}")
    print(f"字符级: 词表 {char_vocab} 个, token 数 {len(char_ids):,}")
    print(f"BPE 级: 词表 {enc.n_vocab:,} 个, token 数 {len(bpe_ids):,}")
    ratio = len(char_ids) / len(bpe_ids)
    print(f"压缩比: {ratio:.2f}x（BPE 一个 token 平均吃掉 {ratio:.2f} 个字符）")

    print("\n同一句话的两种切法：")
    print(f"原文: {SAMPLE.strip()}")
    char_parts = [repr(c) for c in SAMPLE]
    print(f"字符: {char_parts[:49]}  ... 共 {len(SAMPLE)} 个 token")
    sample_ids = enc.encode(SAMPLE)   # 注意：样例重新编码，不要覆盖上面的全文本 bpe_ids
    print(f"BPE : {[enc.decode_single_token_bytes(i).decode('utf-8', errors='replace') for i in sample_ids]}")
    print(f"      ... 共 {len(sample_ids)} 个 token")

    stats = {
        "chars": len(text),
        "char_vocab": char_vocab,
        "char_tokens": len(char_ids),
        "bpe_vocab": enc.n_vocab,
        "bpe_tokens": len(bpe_ids),
        "ratio": round(ratio, 3),
    }
    json.dump(stats, open("tokenize-stats.json", "w"), ensure_ascii=False, indent=2)
    print("\n已保存 tokenize-stats.json")


if __name__ == "__main__":
    main()
