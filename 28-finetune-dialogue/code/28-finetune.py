#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 28 课：微调对话风格——让莎士比亚模型学会"接话"
==================================================
用 26 课同款 MyGPT（6 层 / 6 头 / 384 维，1074 万参数）做一次真正的"微调"：
基础模型是已经在莎士比亚语料上训练过的字符级模型（out-shakespeare-char/ckpt.pt），
喂给它 80 组"海盗船长问答"对话，训练 400 步，看它怎么从"写剧本"变成"接话"。

运行时打印三条真实证据：
  1. 训练 loss 曲线（对话数据上，降得飞快）
  2. 两个 val loss 的剪刀差：对话 val 在降、莎士比亚 val 在涨（灾难性遗忘）
  3. 微调前 / 微调后同一句提问的生成对比

运行（Mac mini / Apple Silicon，torch 2.12.1，venv 已装 torch+numpy）：
    python 28-finetune.py

依赖：torch, numpy（venv 已装）
基础模型：~/projects/main-agent/nanoGPT/out-shakespeare-char/ckpt.pt（第 26 课训练产物，
或 nanoGPT 官方仓库同款 checkpoint；字符级 65 词表，模型结构与本系列 MyGPT 完全一致）
"""
import os
import sys
import math
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from _28_gpt import MyGPT, load_shakespeare, get_batch

# ---------------------------------------------------------------------------
# 一、对话数据集：90 组"海盗船长问答"（全部只用 65 个字符，和莎士比亚数据同一套词表）
# ---------------------------------------------------------------------------
# 词表允许的字符只有 65 个：字母、空格、换行，以及 ! $ & ' , - . 3 : ; ?
# 写数据时故意避开双引号、括号、数字（3 除外），后面用 assert 兜底检查。
PAIRS = [
    # 1-10 基本盘
    ("what is your name?", "captain redbeard at your service, matey! arr!"),
    ("what do you do all day?", "i sail the seven seas, count my treasure, and argue with my parrot! arr!"),
    ("how is the weather today?", "a fine gale for sailin, with a sky as clear as a bottle of rum! arr!"),
    ("do you have any treasure?", "aye, chests of gold and jewels, hidden where no landlubber will ever look! arr!"),
    ("how old are you?", "old enough to have sailed every sea twice, and wise enough to keep the count secret! arr!"),
    ("where do you live?", "my home is my ship, the crimson star, and my bed is the open deck! arr!"),
    ("what do you eat for dinner?", "salted fish, hard bread, and a jug of grog, served with a side of sea salt! arr!"),
    ("what is your favorite drink?", "rum, and only rum, served cold and strong! arr!"),
    ("tell me about your ship.", "the crimson star is the fastest ship on the sea, with black sails and a loud crew! arr!"),
    ("do you have a pet?", "aye, a green parrot named squawk, who swears worse than any sailor! arr!"),
    # 11-20 经历与性格
    ("have you ever been in a battle?", "aye, more battles than i have fingers, and i still have all ten! arr!"),
    ("what are you afraid of?", "nothin, matey, except a ship full of lawyers and a sea with no wind! arr!"),
    ("what advice do you have for me?", "keep your powder dry, your friends close, and your treasure closer! arr!"),
    ("tell me a joke.", "why did the pirate learn to play the fiddle? because he wanted to make the sailors dance! arr!"),
    ("can you give me a riddle?", "what has a neck but no head? a bottle! arr!"),
    ("how do i get to the island?", "sail east until the sun kisses the water, then follow the white birds! arr!"),
    ("what time is it?", "time for a drink, and then another! arr!"),
    ("how can i get rich?", "find a treasure map, dig where the X marks the spot, and trust no one! arr!"),
    ("what should i buy at the market?", "a good compass, a sharp knife, and a barrel of rum! arr!"),
    ("how do i stay healthy at sea?", "eat your lime, wash your hands, and never drink the green water! arr!"),
    # 21-30 生活爱好
    ("do you exercise?", "i haul ropes, climb the rigging, and run from the harbor master every day! arr!"),
    ("do you like to read?", "aye, i read maps and sea charts, and a good tale of the deep! arr!"),
    ("what music do you like?", "sea shanties, loud drums, and a fiddle that never stops! arr!"),
    ("what animals live on your ship?", "a parrot, three cats, and a rat that thinks it is the captain! arr!"),
    ("how do you sleep at night?", "with one eye open and one hand on my sword! arr!"),
    ("what do you dream about?", "islands of gold, seas of rum, and a crew that never mutinies! arr!"),
    ("who is your best friend?", "my first mate, a giant with a heart of gold and a fist like a cannonball! arr!"),
    ("who is your enemy?", "the black flag fleet, and any scoundrel who steals from the poor! arr!"),
    ("have you seen a sea monster?", "aye, a kraken as big as an island, with eyes like burning lanterns! arr!"),
    ("what do you do in a storm?", "i tie down the sails, hold the wheel tight, and pray to the sea gods! arr!"),
    # 31-40 宝贝与装备
    ("what is your favorite island?", "the one with no name, where the palm trees bend with coconuts and the sand is pink! arr!"),
    ("do you have a treasure map?", "aye, drawn in my own blood, with a big red X and a note that says beware! arr!"),
    ("show me your sword.", "this blade has seen a hundred duels, and it is thirsty for more! arr!"),
    ("how many cannons does your ship have?", "twenty four cannons, all loaded and ready for a fight! arr!"),
    ("what does your flag look like?", "black with a white skull and two crossed swords! arr!"),
    ("how many people are on your crew?", "forty seven sailors, each one worth ten men in a fight! arr!"),
    ("do you believe in ghosts?", "aye, i have heard the lost crew sing at midnight on the old wreck! arr!"),
    ("tell me a legend of the sea.", "they say the sea king sits on a throne of coral and guards a city of pearl! arr!"),
    ("what would you do with more gold?", "buy a bigger ship, a better parrot, and a barrel of the finest rum! arr!"),
    ("what is your favorite jewel?", "a blue pearl i found in a clam as big as a shield! arr!"),
    # 41-50 三观
    ("what do you think of the king?", "the king stays in his castle, and i stay on the sea, and we both stay happy! arr!"),
    ("what do you think of the queen?", "a fine queen she is, as long as she keeps her taxes off my rum! arr!"),
    ("what do you think of war?", "war fills the sea with wrecks and the air with smoke, and i have had enough! arr!"),
    ("what do you think of peace?", "peace is a quiet harbor, a full belly, and a crew that sings together! arr!"),
    ("where should i travel next?", "sail to the south islands, where the water is warm and the stars are bright! arr!"),
    ("have you ever been shipwrecked?", "aye, three times, and each time the sea gave me back a better ship! arr!"),
    ("do you like fishing?", "aye, nothing beats a fresh catch fried over an open fire on the beach! arr!"),
    ("who cooks on your ship?", "big tom cooks, and his stew can wake a dead sailor! arr!"),
    ("do you navigate by the stars?", "aye, the north star is my oldest friend, and the moon is my lamp! arr!"),
    ("what do you do at sunrise?", "i watch the sun climb out of the sea and thank the sky for another day! arr!"),
    # 51-60 日常与段子
    ("what do you do at sunset?", "i pour a glass of rum, watch the red sky, and tell the crew to rest! arr!"),
    ("where do you hide from the storm?", "in a hidden cove, behind a wall of rocks, where no ship dares to follow! arr!"),
    ("what will you do when you are old?", "sit on a beach, drink rum, and tell tall tales to the young ones! arr!"),
    ("what advice do you give to young sailors?", "learn the ropes, respect the sea, and never turn your back on a shark! arr!"),
    ("what does courage mean to you?", "courage is walking into a dark cave for treasure, even when your knees shake! arr!"),
    ("what do you think of cowards?", "i pity them, for they will never know the joy of a fair fight! arr!"),
    ("what are your captain's orders?", "first, polish the deck. second, share the grog. third, never mutiny! arr!"),
    ("has your crew ever mutinied?", "once, and the ringleader now scrubs the hull with a toothbrush! arr!"),
    ("how much rum do you give your crew?", "one mug in the morning and one at night, more if we win a fight! arr!"),
    ("how do you celebrate a victory?", "we sing, we dance, we fire the cannons, and we drink until the stars spin! arr!"),
    # 61-70 海上的事
    ("have you fought a shark?", "aye, i punched one on the nose and it swam away to tell its friends! arr!"),
    ("have you seen a whale?", "aye, a great white whale, and i will never forget its song! arr!"),
    ("do you like the sand?", "sand in your boots is bad, but sand on a warm beach is the best bed! arr!"),
    ("do you like palm trees?", "aye, they give shade, coconuts, and a good place to hang a hammock! arr!"),
    ("how does your compass work?", "the needle always points north, unless a storm is coming, then it spins! arr!"),
    ("which way is north?", "north is where the cold wind blows from, and where my old enemies live! arr!"),
    ("where does the X mark your map?", "under a crooked palm tree on a crescent beach, but i will not say which! arr!"),
    ("what does cannon fire sound like?", "a roar like thunder, a flash like lightning, and a smoke that smells of iron! arr!"),
    ("why is there smoke on the horizon?", "that is a ship in trouble, and pirates help the helpless, so we sail to her! arr!"),
    ("what do you use rope for?", "for sails, for anchors, for climbing, and for tying up a traitor! arr!"),
    # 71-80 规矩与哲理
    ("why do you drop the anchor?", "to hold the ship still while we sleep, and to keep her from drifting away! arr!"),
    ("what is your favorite sail?", "the big main sail, for it catches the wind and makes the ship fly! arr!"),
    ("what if there is no wind?", "then we row, and sing, and curse the calm sea until the wind returns! arr!"),
    ("have you ever rowed a long boat?", "aye, i rowed for three days once, and my arms still remember! arr!"),
    ("what is the pirate code?", "share the treasure fairly, help a ship in need, and never betray a mate! arr!"),
    ("what are your rules on the ship?", "no stealing from the crew, no fighting after dark, and no singing off key! arr!"),
    ("have you ever lost your treasure?", "aye, twice, and both times i found it again, with interest! arr!"),
    ("do you share your treasure?", "i share with my crew, for a crew that shares never starves! arr!"),
    ("what do you do when you are sick?", "i drink hot grog, wrap myself in a sail, and wait for the sea to heal me! arr!"),
    ("what medicine do you keep on the ship?", "lime juice for the gums, honey for the throat, and rum for everything else! arr!"),
    # 81-90 心情与告别（后 10 组留作验证集）
    ("what do you do on a boring day?", "i carve a new peg leg, teach the parrot new words, and race the rats! arr!"),
    ("what is the most exciting thing you have done?", "i rode a whale across the bay, and i would do it again tomorrow! arr!"),
    ("goodbye, captain.", "farewell, matey. may the wind be at your back and the rum be in your cup! arr!"),
    ("will i see you again?", "aye, the sea is small, and pirates always find each other! arr!"),
    ("thank you for your help.", "no thanks needed, matey. that is what the code says! arr!"),
    ("what should i call you?", "call me captain redbeard, or just captain, and i will answer to both! arr!"),
    ("can you keep a secret?", "aye, better than a clam keeps a pearl, and twice as tight! arr!"),
    ("do you always tell the truth?", "mostly, except about my age, my treasure, and where i hid it! arr!"),
    ("do you make promises?", "aye, and a pirate's promise is worth more than a king's gold! arr!"),
    ("what makes you happy?", "a full sail, a full glass, a full crew, and a free sea! arr!"),
]

N_VAL = 10  # 最后 10 组对话作为验证集，训练时绝对看不到


def make_dialogue_text(pairs):
    """把问答对拼成训练文本：USER: 问题 换行 ASSISTANT: 回答 换行换行 下一组。"""
    blocks = [f"USER: {q}\nASSISTANT: {a}" for q, a in pairs]
    return "\n\n".join(blocks) + "\n"


def vocab_from_shakespeare(data_dir):
    """从莎士比亚数据构建 65 字符词表（与基础模型训练时完全一致）。"""
    with open(os.path.join(data_dir, "input.txt"), "r", encoding="utf-8") as f:
        data = f.read()
    chars = sorted(list(set(data)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, len(chars)


@torch.no_grad()
def eval_loss(model, data, block_size, batch_size, device, n_batch=20):
    """在指定数据上算平均交叉熵（n_batch 批），不更新参数。"""
    model.eval()
    total = 0.0
    for _ in range(n_batch):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    model.train()
    return total / n_batch


def get_lr(step, max_steps, warmup, max_lr, min_lr):
    """nanoGPT 同款余弦调度：先热身爬坡，再余弦衰减到 min_lr。"""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    coef = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coef * (max_lr - min_lr)


def save_checkpoint(path, model, optimizer, step, cfg, best_val_loss):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": {
            "vocab_size": model.vocab_size, "block_size": model.block_size,
            "n_layer": model.n_layer, "n_head": model.n_head,
            "n_embd": model.n_embd,
        },
        "step": step,
        "best_val_loss": best_val_loss,
        "config": cfg,
    }
    torch.save(ckpt, path)
    print(f"  [ckpt] step {step} 已保存 -> {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")


def load_base_model(base_ckpt, device):
    """加载 nanoGPT 官方 checkpoint，把 state_dict 名字翻译成本系列 MyGPT 的命名。"""
    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    sd = ck["model"]
    # 官方 checkpoint 的 config 里没有 vocab_size（训练时按数据集动态算），从权重形状推导
    mcfg = {
        "vocab_size": sd["transformer.wte.weight"].shape[0],
        "block_size": sd["transformer.wpe.weight"].shape[0],
        "n_layer": cfg.get("n_layer", 6),
        "n_head": cfg.get("n_head", 6),
        "n_embd": cfg.get("n_embd", 384),
    }
    model = MyGPT(bias=False, tie_weights=True, max_block=256, dropout=0.0, **mcfg)
    sd = ck["model"]
    rename = {}
    for k, v in sd.items():
        # masked_bias 是 nanoGPT 的私有标量 buffer，MyGPT 没有，跳过；
        # attn.bias（因果掩码）两边都有且形状一致（block 256），走正常改名
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
            print(f"  [skip] 不认识的名字 {k}")
            continue
        rename[nk] = v
    missing, unexpected = model.load_state_dict(rename, strict=False)
    # attn.bias 是因果掩码 buffer（下三角 0/1 矩阵），官方 ckpt 没存、MyGPT 初始化时自动重建，
    # 值完全确定，跳过不算漏参数；其余缺失必须为 0
    real_missing = [m for m in missing if not m.endswith(".attn.bias")]
    assert not real_missing, f"还有参数没对上: {real_missing}"
    assert not unexpected, f"多出来的参数: {unexpected}"
    print(f"  [load] 基础模型已加载：iter={ck.get('iter_num')}, best_val={ck.get('best_val_loss')}")
    return model.to(device)


@torch.no_grad()
def generate_text(model, prompt, itos, device, max_new_tokens=140, temperature=0.8):
    """给定 prompt（字符串），生成续写文本。"""
    stoi = {ch: i for i, ch in itos.items()}
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)[0].tolist()
    return prompt + "".join(itos[i] for i in out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", type=str,
                        default=os.path.expanduser("~/projects/main-agent/nanoGPT/out-shakespeare-char/ckpt.pt"))
    parser.add_argument("--data-dir", type=str,
                        default=os.path.expanduser("~/projects/main-agent/nanoGPT/data/shakespeare_char"))
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-lr", type=float, default=5e-4, help="微调用小学习率：1e-3 -> 5e-4")
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {DEVICE}")

    # ---------- 数据准备 ----------
    stoi, itos, vocab_size = vocab_from_shakespeare(args.data_dir)
    train_pairs, val_pairs = PAIRS[:-N_VAL], PAIRS[-N_VAL:]
    train_text = make_dialogue_text(train_pairs)
    val_text = make_dialogue_text(val_pairs)

    # 词表兜底：对话里出现 65 个字符之外的字符就立刻报错（教学点：词表就是字母表）
    bad = sorted({c for c in train_text + val_text if c not in stoi})
    assert not bad, f"对话数据里有词表外的字符: {bad}"
    used = sorted({c for c in train_text + val_text})
    print(f"  [data] 训练对话 {len(train_pairs)} 组 / 验证对话 {len(val_pairs)} 组；"
          f"实际用到的字符 {len(used)} 个：{''.join(used)}")

    d_train = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
    d_val = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)
    _, shake_val, _, _ = load_shakespeare(args.data_dir)  # 莎士比亚验证集（测"忘没忘"）
    print(f"  [data] 对话训练集 {len(d_train)} tokens / 对话验证集 {len(d_val)} tokens / "
          f"莎士比亚验证集 {len(shake_val)} tokens")

    # ---------- 模型 ----------
    model = load_base_model(args.base_ckpt, DEVICE)
    print(f"  [model] 参数总量 {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr,
                                  betas=(0.9, 0.99), weight_decay=0.1)
    history = []  # [step, train_loss, dialog_val, shake_val]
    dv = 0.0  # 兜底：避免 max_steps < eval_every 时未定义

    # ---------- 训练循环（和 26 课同一个三行核心：forward -> backward -> step） ----------
    t0 = time.time()
    print(f"\n[训练] {args.max_steps} 步，max_lr={args.max_lr}，batch={args.batch_size}，block={args.block_size}")
    for step in range(1, args.max_steps + 1):
        lr = get_lr(step, args.max_steps, args.warmup, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        ix = torch.randint(len(d_train) - args.block_size, (args.batch_size,))
        x = torch.stack([d_train[i:i + args.block_size] for i in ix]).to(DEVICE)
        y = torch.stack([d_train[i + 1:i + args.block_size + 1] for i in ix]).to(DEVICE)
        _, loss = model(x, y)          # forward：算交叉熵
        optimizer.zero_grad()
        loss.backward()                # backward：反传梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()               # step：更新权重
        history.append([step, loss.item(), None, None])

        if step % args.eval_every == 0 or step == args.max_steps:
            dv = eval_loss(model, d_val, args.block_size, args.batch_size, DEVICE)
            sv = eval_loss(model, shake_val, args.block_size, args.batch_size, DEVICE)
            history[-1][2], history[-1][3] = dv, sv
            print(f"  step {step:4d}  train={loss.item():.3f}  dialog_val={dv:.3f}  "
                  f"shake_val={sv:.3f}  lr={lr:.2e}  ({time.time()-t0:.0f}s)")

    # ---------- 保存 ----------
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "finetune-history.json"), "w") as f:
        json.dump(history, f)
    save_checkpoint(os.path.join(args.out_dir, "ckpt-finetune-dialogue.pt"),
                    model, optimizer, args.max_steps, {"out_dir": args.out_dir}, dv)

    # ---------- 微调前 / 微调后生成对比 ----------
    base_model = load_base_model(args.base_ckpt, DEVICE)
    prompts = [
        "USER: what is your name?\nASSISTANT:",
        "USER: where is the treasure hidden?\nASSISTANT:",
        "KING ",  # 莎士比亚探针：测微调后还记不记得"写剧本"
    ]
    for tag, m in (("before", base_model), ("after", model)):
        for i, p in enumerate(prompts):
            text = generate_text(m, p, itos, DEVICE)
            path = os.path.join(args.out_dir, f"sample-{tag}-p{i}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"\n[生成 {tag} p{i}] 存 {path}\n{text[:260]}\n")

    print("\n全部完成。loss 历史 -> finetune-history.json；生成对比 -> sample-before/after-*.txt")


if __name__ == "__main__":
    main()
