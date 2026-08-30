#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 30 课：把模型做成可玩的东西——终端聊天小工具（REPL）
=====================================================
系列 30 课的全部积木，今天拼成一个能对话的玩具。

支持两个"人格"：
  - pirate  海盗管家（28 课微调产物，会接话、会 arr!）
  - shake   莎士比亚剧作家（26 课训练产物，会写剧本）

交互命令（在提示符后输入）：
  /help                显示帮助
  /model pirate|shake  切换人格
  /temp 0.8            设置 temperature（默认 0.8）
  /topk 40             设置 top-k（默认 40）
  /topp 0.9            设置 top-p（默认 0.9）
  /len 150             设置生成长度（默认 150）
  /quit                退出

运行（Mac mini / Apple Silicon，torch 2.12.1，venv 已装 torch+numpy）：
    python 30-chat.py --model pirate
    python 30-chat.py --model shake
    python 30-chat.py --demo          # 非交互演示模式：跑一组固定问题，给文章/文档用

依赖：torch, numpy（venv 已装）
模型权重：默认在 nanoGPT 项目目录里找（见 find_ckpt），也可用 --ckpt 显式指定。
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

from _30_gpt import MyGPT

# ---------------------------------------------------------------------------
# 一、采样三件套（第 25 课）：temperature -> top-k -> top-p -> softmax -> 抽签
# ---------------------------------------------------------------------------

def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    """给定模型对下一个字符的分数 logits（形状 [vocab]），按采样策略抽一个字符 id。"""
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = float("-inf")
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[mask] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    """自回归生成：每步把已生成的 token 喂回模型，用 sample_next 抽下一个。"""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        idx_next = sample_next(logits[0, -1, :], temperature, top_k, top_p)
        idx = torch.cat((idx, idx_next.unsqueeze(0)), dim=1)
    return idx


# ---------------------------------------------------------------------------
# 二、模型加载：两套 checkpoint 格式都要认识
#   A. 本系列 MyGPT 直接保存的 ckpt（24/26/28 课产物，键名已是 blocks.0.xxx）
#   B. nanoGPT 官方 ckpt（out-shakespeare-char/ckpt.pt，键名 transformer.h.0.xxx）
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device):
    """加载 checkpoint，返回 (model, meta)：
    meta = {"tag": "pirate"/"shake"/"unknown", "iter": ...}，用于界面显示。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    is_mygpt = any(k.startswith("blocks.") or k.startswith("wte.") for k in sd)
    if is_mygpt:                                 # 本系列 MyGPT 保存格式（blocks.0.xxx）
        mcfg = dict(ck["model_args"])
        # model_args 里可能带 bias/dropout，MyGPT 这两个参数已显式传，先弹掉
        mcfg.pop("bias", None)
        mcfg.pop("dropout", None)
        model = MyGPT(bias=False, tie_weights=True, max_block=256, dropout=0.0, **mcfg)
        missing, unexpected = model.load_state_dict(sd, strict=False)
    else:                                        # nanoGPT 官方格式（transformer. 前缀）
        cfg = ck.get("config", {})
        mcfg = {
            "vocab_size": sd["transformer.wte.weight"].shape[0],
            "block_size": sd["transformer.wpe.weight"].shape[0],
            "n_layer": cfg.get("n_layer", 6),
            "n_head": cfg.get("n_head", 6),
            "n_embd": cfg.get("n_embd", 384),
        }
        model = MyGPT(bias=False, tie_weights=True, max_block=256, dropout=0.0, **mcfg)
        rename = {}
        for k, v in sd.items():
            if ".attn.masked_bias" in k:
                continue
            if k == "transformer.wte.weight":
                nk = "wte.weight"
            elif k == "transformer.wpe.weight":
                nk = "wpe.weight"
            elif k == "transformer.ln_f.weight":
                nk = "ln_f.weight"
            elif k == "transformer.ln_f.bias":
                nk = "ln_f.bias"
            elif k == "lm_head.weight":
                nk = "lm_head.weight"
            elif k.startswith("transformer.h."):
                nk = "blocks." + k[len("transformer.h."):]
            else:
                continue
            rename[nk] = v
        missing, unexpected = model.load_state_dict(rename, strict=False)
    # attn.bias 因果掩码 buffer 是确定性重建的，跳过不算漏参数
    real_missing = [m for m in missing if not m.endswith(".attn.bias")]
    assert not real_missing, f"还有参数没对上: {real_missing}"
    assert not unexpected, f"多出来的参数: {unexpected}"
    return model.to(device)


def find_ckpt(tag, explicit=None):
    """按人格找 checkpoint。--ckpt 显式指定优先；否则在常见位置搜索。"""
    if explicit:
        return explicit
    base = os.path.expanduser("~/projects/main-agent/learning-nanogpt-from-0-to-1")
    candidates = {
        "pirate": [
            os.path.join(base, "28-finetune-dialogue/code/ckpt-finetune-dialogue.pt"),
        ],
        "shake": [
            os.path.expanduser("~/projects/main-agent/nanoGPT/out-shakespeare-char/ckpt.pt"),
            os.path.join(base, "26-milestone-train/code/ckpts/ckpt-milestone-step4500.pt"),
        ],
    }
    for p in candidates.get(tag, []):
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# 三、字符词表（65 字符，和训练时完全一致）
# ---------------------------------------------------------------------------

def build_vocab():
    """从莎士比亚数据构建 65 字符词表。数据文件缺失时用硬编码兜底。"""
    data_path = os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char/input.txt")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = f.read()
        chars = sorted(list(set(data)))
    except FileNotFoundError:
        # 兜底：65 个字符的固定清单（shakespeare_char 的完整字符集）
        chars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';',
                 '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                 'v', 'w', 'x', 'y', 'z']
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


# ---------------------------------------------------------------------------
# 四、生成文本封装
# ---------------------------------------------------------------------------

def complete(stoi, itos, model, prompt, temperature=0.8, top_k=40, top_p=0.9,
             max_new_tokens=150, stop=None, device="mps"):
    """给定 prompt（字符串），生成续写。stop 是停止子串（出现即截断）。"""
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = generate(model, idx, max_new_tokens, temperature=temperature,
                   top_k=top_k, top_p=top_p)[0].tolist()
    text = prompt + "".join(itos[i] for i in out)
    if stop:
        cut = text.find(stop, len(prompt))
        if cut != -1:
            text = text[:cut]
    return text


def pirate_reply(stoi, itos, model, question, temperature=0.8, top_k=40, top_p=0.9,
                 max_new_tokens=120, device="mps"):
    """海盗管家模式：把问题拼成 USER:/ASSISTANT: 格式，生成回答（到下一个 USER: 或换行对）。"""
    prompt = f"USER: {question}\nASSISTANT:"
    text = complete(stoi, itos, model, prompt, temperature, top_k, top_p,
                    max_new_tokens, stop="\n\n", device=device)
    # 微调模型有个小毛病：答完会自己把问题抄一遍接下一轮（28 课彩蛋）。
    # 演示输出取最后一个 ASSISTANT: 之后的内容，界面干净。
    return text.split("ASSISTANT:")[-1].strip()


# ---------------------------------------------------------------------------
# 五、主程序：REPL 交互 / 演示模式
# ---------------------------------------------------------------------------

HELP = """
[命令]
  /help                显示本帮助
  /model pirate|shake  切换人格（重新加载模型，约 10 秒）
  /temp 0.8            设置 temperature（默认 0.8）
  /topk 40             设置 top-k（默认 40）
  /topp 0.9            设置 top-p（默认 0.9）
  /len 150             设置生成长度（默认 150）
  /quit                退出
[普通输入]
  pirate 人格：直接提问，模型以海盗管家口吻回答
  shake  人格：输入任意开头（如 KING ），模型续写莎士比亚腔
""".strip("\n")


def run_repl(stoi, itos, device):
    cur_tag = None
    model = None
    ckpt_path = None
    temperature, top_k, top_p, max_len = 0.8, 40, 0.9, 150

    def load(tag):
        nonlocal model, cur_tag, ckpt_path
        path = find_ckpt(tag, args.ckpt)
        if path is None:
            print(f"[30-chat] 找不到 {tag} 的 checkpoint，请用 --ckpt 指定路径")
            return
        model = load_model(path, device)
        cur_tag, ckpt_path = tag, path
        name = "海盗管家" if tag == "pirate" else "莎士比亚剧作家"
        print(f"[30-chat] 已加载 {name} <- {os.path.basename(path)}")

    load(args.model)
    if model is None:
        sys.exit(1)
    print(f"[30-chat] 系列第 30 课小工具。输入 /help 看命令，/quit 退出。\n")

    while True:
        try:
            line = input(f"[{cur_tag}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[30-chat] 再见！")
            break
        if not line:
            continue
        if line.startswith("/"):
            parts = line.split()
            cmd = parts[0].lower()
            if cmd == "/quit":
                print("[30-chat] 再见！")
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/model" and len(parts) >= 2:
                load(parts[1])
            elif cmd == "/temp" and len(parts) >= 2:
                temperature = float(parts[1])
                print(f"[30-chat] temperature = {temperature}")
            elif cmd == "/topk" and len(parts) >= 2:
                top_k = int(parts[1])
                print(f"[30-chat] top-k = {top_k}")
            elif cmd == "/topp" and len(parts) >= 2:
                top_p = float(parts[1])
                print(f"[30-chat] top-p = {top_p}")
            elif cmd == "/len" and len(parts) >= 2:
                max_len = int(parts[1])
                print(f"[30-chat] 生成长度 = {max_len}")
            else:
                print("[30-chat] 不认识这个命令，输入 /help 查看")
            continue
        # 普通输入：按人格生成
        if cur_tag == "pirate":
            reply = pirate_reply(stoi, itos, model, line, temperature, top_k, top_p,
                                 max_len, device)
            print(f"\nASSISTANT: {reply}\n")
        else:
            text = complete(stoi, itos, model, line, temperature, top_k, top_p,
                            max_len, device)
            print(f"\n{text}\n")


def run_demo(stoi, itos, device):
    """演示模式：固定问题跑几轮真实对话，输出给文章/文档用。"""
    torch.manual_seed(42)
    print("=" * 60)
    print("demo 1：海盗管家问答（28 课微调模型，temp=0.8 topk=40 topp=0.9）")
    print("=" * 60)
    p_ckpt = find_ckpt("pirate", args.ckpt)
    pirate = load_model(p_ckpt, device) if p_ckpt else None
    questions = [
        "what is your name?",
        "where is the treasure hidden?",
        "tell me a joke.",
        "what do you think of the king?",
    ]
    if pirate is not None:
        for q in questions:
            torch.manual_seed(42)
            r = pirate_reply(stoi, itos, pirate, q, 0.8, 40, 0.9, 120, device)
            print(f"USER: {q}")
            print(f"ASSISTANT: {r}\n")
    else:
        print("（找不到海盗管家 checkpoint，跳过）\n")

    print("=" * 60)
    print("demo 2：莎士比亚续写（26 课模型，temp=0.8 topk=40 topp=0.9）")
    print("=" * 60)
    s_ckpt = find_ckpt("shake", args.ckpt)
    shake = load_model(s_ckpt, device) if s_ckpt else None
    if shake is not None:
        for p in ["KING HENRY:", "ROMEO:"]:
            torch.manual_seed(42)
            t = complete(stoi, itos, shake, p, 0.8, 40, 0.9, 150, device)
            print(f"prompt: {p!r}")
            print(t)
            print()
    else:
        print("（找不到莎士比亚 checkpoint，跳过）\n")

    print("=" * 60)
    print("demo 3：温度旋钮——训练集内问题 vs 训练集外问题")
    print("=" * 60)
    if pirate is not None:
        for q in ("what do you eat for dinner?", "what is your favorite color?"):
            print(f"--- 问题: {q} ---")
            for temp in (0.2, 0.8, 1.5):
                torch.manual_seed(42)
                r = pirate_reply(stoi, itos, pirate, q, temp, 40, 0.9, 120, device)
                print(f"temp={temp}: {r}")
            print()
    else:
        print("（跳过）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手搓大模型 30 课：终端聊天小工具")
    parser.add_argument("--model", choices=["pirate", "shake"], default="pirate")
    parser.add_argument("--ckpt", type=str, default=None, help="显式指定 checkpoint 路径")
    parser.add_argument("--demo", action="store_true", help="非交互演示模式")
    args = parser.parse_args()

    torch.manual_seed(42)
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {DEVICE}")

    STOI, ITOS = build_vocab()
    if args.demo:
        run_demo(STOI, ITOS, DEVICE)
    else:
        run_repl(STOI, ITOS, DEVICE)
