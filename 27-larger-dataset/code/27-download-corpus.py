#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 27 课：下载更大的公开语料（Project Gutenberg 公共版权英文名著）。
用法: python 27-download-corpus.py
依赖: 无（只用标准库 urllib）。网络需要能访问 gutenberg.org。
输出: novels.txt（5 本名著正文拼接，约 8-10MB）
"""
import os
import re
import urllib.request

# 输出到脚本所在目录的上级（与 27-train.py 的 DATA_FILES["novels"] 一致）
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "novels.txt")

# (书名, Gutenberg 文本 ID)
BOOKS = [
    ("War and Peace",             "2600"),
    ("Moby Dick",                 "2701"),
    ("Pride and Prejudice",       "1342"),
    ("Great Expectations",        "1400"),
    ("Les Miserables",            "135"),
    ("Anna Karenina",             "1399"),
]

BASE = "https://www.gutenberg.org/cache/epub/{}/pg{}.txt"


def strip_gutenberg(text):
    """去掉 Gutenberg 页眉页脚，只保留正文。
    正文夹在 '*** START OF ... EBOOK ***' 和 '*** END OF ... EBOOK ***' 之间。"""
    m_start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.S)
    m_end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.S)
    if m_start and m_end:
        text = text[m_start.end():m_end.start()]
    return text


def main():
    parts = []
    total = 0
    for title, gid in BOOKS:
        url = BASE.format(gid, gid)
        print(f"下载 {title} <- {url}")
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        body = strip_gutenberg(raw).strip()
        total += len(body)
        parts.append(f"@@ {title} @@\n" + body)
        print(f"  {len(body):,} 字符")
    out = "\n\n".join(parts)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"完成: {OUT_PATH} 共 {total:,} 字符 ({total/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
