#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 20 课实验：手搓 BPE Tokenizer
================================
从零实现完整的 BPE（训练/编码/解码三件套），并与官方 tiktoken r50k_base 对拍。

运行：
    ~/projects/main-agent/nanoGPT/.venv/bin/python 20-bpe-tokenizer.py

依赖：numpy（可选，仅用于画图数据）、tiktoken（仅用于对拍验证；训练不需要）
"""
import os
import json
import time
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char/input.txt")
IMG_DIR = os.path.join(BASE, "images")
os.makedirs(IMG_DIR, exist_ok=True)


# ============================================================
# 一、手写 BPE：训练（学合并规则）
# ============================================================
class HandBPETrainer:
    """纯字符级 BPE 训练器：输入是字符列表，输出一张"合并规则表"。

    规则表：每一条规则 = (两个词块 -> 一个新词块)。
    BPE 的"训练" = 数相邻对出现次数，把最高频的一对焊死，重复 N 次。
    """

    def __init__(self):
        self.merges = {}          # {(a, b): new}  合并规则
        self.vocab = None         # 训练完生成的词表

    def train(self, chars, num_merges, verbose=True):
        """chars: list[str]（一个字符一个元素）；num_merges: 合并次数"""
        tokens = list(chars)  # 初始：每个字符是一个 token
        vocab = {i: ch for i, ch in enumerate(sorted(set(chars)))}  # 初始词表
        # 初始词表长度：有多少不同字符就有多少个
        next_id = len(vocab)

        # 让 tokens 变成整数 id 列表（算法更快，且和真实实现一致）
        stoi = {ch: i for i, ch in vocab.items()}
        ids = [stoi[ch] for ch in tokens]

        merge_log = []
        for step in range(num_merges):
            # ① 统计相邻对
            pairs = Counter(zip(ids, ids[1:]))
            # ② 最高频的一对
            (a, b), cnt = pairs.most_common(1)[0]
            # ③ 焊死：新词块拿一个新 id
            new_id = next_id
            next_id += 1
            self.merges[(a, b)] = new_id
            vocab[new_id] = vocab[a] + vocab[b]

            # ④ 全序列替换
            new_ids = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

            if verbose:
                merge_log.append((step + 1, vocab[a], vocab[b], vocab[new_id], cnt))

        self.vocab = vocab
        self.itos = {i: s for i, s in vocab.items()}  # 反查表：id -> 字符串
        return merge_log, ids


# ============================================================
# 二、手写 BPE：编码（应用规则）
# ============================================================
def encode_with_merges(ids, merges):
    """把 id 列表按合并规则编码成更短的 id 列表。

    贪心：重复找"rank 最小的一对"合并。rank 即合并发生的时间，
    越早合并的 rank 越小，优先级越高。
    """
    rank = {pair: i for i, pair in enumerate(merges.keys())}

    while True:
        # 找所有相邻对里 rank 最小的那个
        pairs = [(rank.get((ids[i], ids[i + 1]), float("inf")), i)
                 for i in range(len(ids) - 1)]
        if not pairs:
            break
        min_rank, min_idx = min(pairs, key=lambda x: x[0])
        if min_rank == float("inf"):
            break  # 没有可合并的了
        a, b = ids[min_idx], ids[min_idx + 1]
        new_id = merges[(a, b)]
        ids = ids[:min_idx] + [new_id] + ids[min_idx + 2:]
    return ids


def decode_ids(ids, itos):
    """反查表解码：id 列表 -> 字符串（与训练时第 13 课的字符串拼接一致）"""
    return "".join(itos[i] for i in ids)


# ============================================================
# 三、手写字节级 BPE（处理中文/emoji/任意 Unicode）
# ============================================================
class HandByteBPE:
    """字节级 BPE：先把文本编码成 UTF-8 字节（0-255），再在字节上做 BPE。

    这就是 GPT-2 真实的做法：任何语言都能转成字节，字节就 256 种，
    永远不会遇到"词表里没有的字符"。
    """

    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.merges = {}          # {(a,b): new_id}
        self.vocab = None
        self.bytes_to_unicode = None  # 占位，下面构建

    def _build_byte_decoder(self):
        """构造 0-255 字节 -> 可见 Unicode 字符的映射（GPT-2 同款逻辑）。"""
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def train(self, text):
        """text: 原始字符串；训练出合并规则。"""
        # 文本 -> 字节列表
        raw_bytes = list(text.encode("utf-8"))
        self.byte_decoder = self._build_byte_decoder()
        self.byte_encoder = {v: k for k, v in self.byte_decoder.items()}

        # 字节 -> 可见字符
        chars = [self.byte_decoder[b] for b in raw_bytes]
        # 字符 -> id
        vocab = {i: ch for i, ch in enumerate(sorted(set(chars)))}
        stoi = {ch: i for i, ch in vocab.items()}
        ids = [stoi[ch] for ch in chars]
        next_id = len(vocab)

        merge_log = []
        for step in range(self.num_merges):
            pairs = Counter(zip(ids, ids[1:]))
            (a, b), cnt = pairs.most_common(1)[0]
            new_id = next_id
            next_id += 1
            self.merges[(a, b)] = new_id
            vocab[new_id] = vocab[a] + vocab[b]
            new_ids = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            if step < 20 or step % 50 == 49:
                merge_log.append((step + 1, vocab[a], vocab[b], vocab[new_id], cnt))

        self.vocab = vocab
        self.itos = {i: s for i, s in vocab.items()}
        self.stoi = {s: i for i, s in vocab.items()}
        return ids, merge_log

    def encode(self, text):
        """编码：文本 -> id 列表（与 GPT-2 同思路的贪心合并）"""
        raw_bytes = list(text.encode("utf-8"))
        chars = [self.byte_decoder[b] for b in raw_bytes]
        ids = [self.stoi[ch] for ch in chars]
        return encode_with_merges(ids, self.merges)

    def decode(self, ids):
        """解码：id 列表 -> 原始文本"""
        # id -> token 字符串（可能是多字节合并体）-> 逐字符转回字节 -> 原始文本
        out_bytes = bytearray()
        for i in ids:
            for ch in self.itos[i]:
                out_bytes.append(self.byte_encoder[ch])
        return bytes(out_bytes).decode("utf-8", errors="replace")


# ============================================================
# 四、实验入口
# ============================================================
def main():
    print("=" * 70)
    print("第 20 课实验：手搓 BPE Tokenizer（Mac mini 实测）")
    print("=" * 70)

    # ---- 读取莎士比亚数据 ----
    text = open(DATA, encoding="utf-8").read()
    print(f"\n数据: shakespeare_char/input.txt, 总字符数 {len(text)}")

    # ---- 实验 1：经典 low 例子（GPT-2 论文原例）----
    print("\n" + "=" * 70)
    print("实验 1：经典 low 例子（GPT-2 论文原例）")
    print("=" * 70)
    low_text = "low low low low low lower lowest newest"
    chars = list(low_text)
    trainer = HandBPETrainer()
    merge_log, final_ids = trainer.train(chars, 5, verbose=True)
    print(f"\n原文本: {low_text!r}（{len(chars)} 字符）")
    for step, a, b, new_tok, cnt in merge_log:
        print(f"合并 {step}: {a!r} + {b!r} -> {new_tok!r}  频次 {cnt}")
    final_toks = [trainer.vocab[i] for i in final_ids]
    print(f"\n5 轮合并后: {len(chars)} 字符 -> {len(final_ids)} 词块")
    print(f"词块序列: {final_toks}")

    # ---- 实验 2：在莎士比亚上训练 300 词块，看真实合并 ----
    print("\n" + "=" * 70)
    print("实验 2：莎士比亚文本训练 300 词块（真实合并规则）")
    print("=" * 70)
    sample = text[:20000]
    trainer2 = HandBPETrainer()
    t0 = time.time()
    merge_log2, ids2 = trainer2.train(list(sample), 300, verbose=True)
    t1 = time.time()
    print(f"\n训练 300 次合并耗时: {t1 - t0:.3f}s（纯 Python，20,000 字符）")
    print("\n前 12 次真实合并（GPT 系列实现与 tiktoken 同款逻辑）:")
    for step, a, b, new_tok, cnt in merge_log2[:12]:
        print(f"合并 {step:3d}: {a!r} + {b!r} -> {new_tok!r}  频次 {cnt}")
    print("\n第 100-105 次合并:")
    for step, a, b, new_tok, cnt in merge_log2[99:105]:
        print(f"合并 {step:3d}: {a!r} + {b!r} -> {new_tok!r}  频次 {cnt}")
    print("\n第 250-255 次合并:")
    for step, a, b, new_tok, cnt in merge_log2[249:255]:
        print(f"合并 {step:3d}: {a!r} + {b!r} -> {new_tok!r}  频次 {cnt}")
    print(f"\n词表大小 = 初始 {len(trainer2.vocab) - 300} + 合并 300 = {len(trainer2.vocab)}")
    print(f"20,000 字符 -> {len(ids2)} 词块，压缩比 {20000 / len(ids2):.2f}x")

    # ---- 实验 3：encode/decode 往返 + 与 tiktoken 对拍 ----
    print("\n" + "=" * 70)
    print("实验 3：encode/decode 往返 + 与官方 tiktoken 对拍")
    print("=" * 70)
    test_text = text[12345:18345]  # 6000 字符测试段
    # 用训练好的规则表编码
    stoi2 = {s: i for i, s in trainer2.vocab.items()}
    test_ids_char = [stoi2[c] for c in list(test_text)]
    encoded_ids = encode_with_merges(test_ids_char, trainer2.merges)
    decoded = decode_ids(encoded_ids, trainer2.itos)
    print(f"round-trip 一致性: {decoded == test_text}（encode -> decode 完整还原）")
    print(f"测试段 {len(test_text)} 字符 -> {len(encoded_ids)} 词块，压缩比 {len(test_text) / len(encoded_ids):.2f}x")

    # 官方 tiktoken 对拍
    import tiktoken
    enc = tiktoken.get_encoding("r50k_base")
    t0 = time.time()
    bpe_ids = enc.encode(test_text)
    t1 = time.time()
    pieces = [enc.decode([i]) for i in bpe_ids[:40]]
    print(f"\ntiktoken r50k_base（GPT-2 官方词表 50,257）:")
    print(f"同一段 {len(test_text)} 字符 -> {len(bpe_ids)} 词块，压缩比 {len(test_text) / len(bpe_ids):.2f}x, 耗时 {t1 - t0:.4f}s")
    print(f"前 40 个词块: {pieces}")

    # 对拍：手写 vs 官方对同一批单词的切分
    words = ["Shakespeare", "Romeo", "love", "the", "unhappiness", "lowest", "hello", "world", "tokenization", "incredibly"]
    print("\n手写 300 词块 vs 官方 r50k_base 对拍（同一批单词）:")
    print(f"{'单词':<16}{'手写(300词块)':<28}{'官方 r50k':<28}")
    for w in words:
        mine = encode_with_merges([stoi2[c] for c in list(w)], trainer2.merges)
        mine_s = [trainer2.vocab[i] for i in mine]
        off_s = [enc.decode([i]) for i in enc.encode(w)]
        print(f"{w:<16}{str(mine_s):<28}{str(off_s):<28}")

    # ---- 实验 4：字节级 BPE 处理中文 ----
    print("\n" + "=" * 70)
    print("实验 4：字节级 BPE 处理中文（GPT-2 真实做法）")
    print("=" * 70)
    zh_text = "你好，世界！你好，大模型。Hello world! 你好呀 🐎"
    print(f"原文本: {zh_text}")
    print(f"UTF-8 字节数: {len(zh_text.encode('utf-8'))}，字符数（Python len）: {len(zh_text)}")
    bb = HandByteBPE(num_merges=50)
    ids_bb, log_bb = bb.train(zh_text + " " * 4 + zh_text + " " * 3 + zh_text)  # 重复几次让合并有意义
    enc_zh = bb.encode(zh_text)
    dec_zh = bb.decode(enc_zh)
    print(f"字节级 BPE 编码: {len(enc_zh)} 个词块")
    print(f"往返还原: {dec_zh == zh_text} -> {dec_zh!r}")
    print(f"前 10 个合并:")
    for step, a, b, new_tok, cnt in log_bb[:10]:
        print(f"  {step:3d}: {a!r} + {b!r} -> {new_tok!r}  频次 {cnt}")

    # ---- 实验 5：压缩比随词表大小变化（画图数据）----
    print("\n" + "=" * 70)
    print("实验 5：词表大小 vs 压缩比（画图数据）")
    print("=" * 70)
    sample2 = text[:50000]
    chart_data = []
    for merges_n in [0, 10, 50, 100, 200, 400, 800, 1500, 3000]:
        tr = HandBPETrainer()
        _, ids_n = tr.train(list(sample2), merges_n, verbose=False)
        ratio = len(sample2) / len(ids_n) if ids_n else 0.0
        chart_data.append({"merges": merges_n, "vocab": len(tr.vocab), "ratio": round(ratio, 3)})
        print(f"合并 {merges_n:5d} -> 词表 {len(tr.vocab):5d}, {len(sample2)} 字符 -> {len(ids_n):6d} 词块, 压缩比 {ratio:.2f}x")
    json.dump(chart_data, open(os.path.join(BASE, "chart-data.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n画图数据已存: {os.path.join(BASE, 'chart-data.json')}")

    # tiktoken 压缩比对比（同 5 万字符）
    t0 = time.time()
    bpe_50k = enc.encode(sample2)
    t1 = time.time()
    print(f"\ntiktoken r50k 同 5 万字符: {len(bpe_50k)} 词块, 压缩比 {50000 / len(bpe_50k):.2f}x, 耗时 {t1 - t0:.3f}s")

    # ---- 实验 6：参数量对比（手写 tokenizer 词表 vs 完整 GPT-2 词表）----
    print("\n" + "=" * 70)
    print("实验 6：词表大小对模型参数量的影响")
    print("=" * 70)
    for vs in [65, 300 + 65, 50257]:
        emb = vs * 384
        print(f"词表 {vs:6d} -> embedding 表 {emb:10d} 参数（384 维）")

    print("\n全部实验完成 ✅")


if __name__ == "__main__":
    main()
