#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 27 课：加载训练好的 checkpoint，用不同温度生成对比样本。
用法:
  python 27-eval.py --ckpt ckpt-bpe-shakes-step1000.pt --prompt "KING HENRY VI:\n"
  python 27-eval.py --ckpt ckpt-bpe-novels-step1000.pt --prompt "CHAPTER I.\n\nIt was a bright cold day in April,\n"
依赖: torch / tiktoken / _27_gpt.py
"""
import argparse
import json

import torch
import tiktoken

from _27_gpt import MyGPT

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temps", default="0.5,0.8,1.2")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    print(f"checkpoint: step={ckpt.get('step')} val_loss={ckpt.get('val_loss'):.4f} data={ckpt.get('data')}")
    model = MyGPT(vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
                  n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
                  dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = enc.encode(args.prompt) if args.prompt else []
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    for t in [float(x) for x in args.temps.split(",")]:
        torch.manual_seed(42)
        out = model.generate(idx, args.tokens, temperature=t)
        text = enc.decode(out[0].tolist())
        print(f"\n===== T={t} =====")
        print(text)


if __name__ == "__main__":
    main()
