# -*- coding: utf-8 -*-
"""
手搓大模型 第18课：GPT 全景——读 model.py 的真实实验
运行命令：
  ~/projects/main-agent/nanoGPT/.venv/bin/python 18-gpt-panorama.py
依赖：torch 2.12.1（venv 已装），nanoGPT 仓库（model.py / data/shakespeare_char）
"""
import os, sys, pickle, importlib.util
import torch

NANOGPT = os.path.expanduser("~/projects/main-agent/nanoGPT")

# ---------- 加载 nanoGPT 的 model.py（不改一行代码） ----------
spec = importlib.util.spec_from_file_location("nanogpt_model", os.path.join(NANOGPT, "model.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["nanogpt_model"] = mod
spec.loader.exec_module(mod)
GPT, GPTConfig = mod.GPT, mod.GPTConfig

torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device = {device}")

# ---------- 1. 实例化 baby GPT（系列标准配置：6层/6头/384维/256上下文/65字符词表） ----------
config = GPTConfig(vocab_size=65, block_size=256, n_layer=6, n_head=6,
                   n_embd=384, dropout=0.2, bias=False)
model = GPT(config)
model.to(device)

print("\n===== 1. 参数量：整个 GPT 的每一分钱花在哪 =====")
total = sum(p.numel() for p in model.parameters())
print(f"parameters() 实际统计到的参数总数: {total:,}")
print(f"get_num_params()（扣掉位置编码的计数口径）: {model.get_num_params():,}")

def count(prefix):
    n = sum(p.numel() for (pn, p) in model.named_parameters() if pn.startswith(prefix))
    return n

wte = count("transformer.wte")
wpe = count("transformer.wpe")
ln_f = count("transformer.ln_f")
lm_head = count("lm_head")
attn = sum(count(f"transformer.h.{i}.attn") for i in range(config.n_layer))
mlp = sum(count(f"transformer.h.{i}.mlp") for i in range(config.n_layer))
ln_blk = sum(count(f"transformer.h.{i}.ln_1") + count(f"transformer.h.{i}.ln_2") for i in range(config.n_layer))

print(f"\n{'组件':<38}{'参数量':>12}{'占比':>9}")
print("-" * 62)
for name, n in [("transformer.wte 词嵌入表 65x384", wte),
                ("transformer.wpe 位置编码表 256x384", wpe),
                ("6 个块 × attn（QKV 投影+输出投影）", attn),
                ("6 个块 × mlp（放大4倍+缩回）", mlp),
                ("6 个块 × ln_1/ln_2（LayerNorm）", ln_blk),
                ("transformer.ln_f 最终 LayerNorm", ln_f),
                ("lm_head 输出投影（与 wte 共享权重）", lm_head)]:
    print(f"{name:<38}{n:>12,}{n / total * 100:>8.2f}%")
print("-" * 62)
print(f"{'合计（共享权重只算一次）':<38}{total:>12,}{100:>8.2f}%")

# ---------- 2. 权重绑定（weight tying）：输出层用的就是词嵌入表 ----------
print("\n===== 2. weight tying：lm_head 和 wte 是不是同一份权重？ =====")
same = model.transformer.wte.weight is model.lm_head.weight
print(f"model.transformer.wte.weight is model.lm_head.weight → {same}")
print(f"wte.weight 形状: {tuple(model.transformer.wte.weight.shape)}，lm_head.weight 形状: {tuple(model.lm_head.weight.shape)}")
if same:
    print("结论：词嵌入矩阵被输出层复用——最后预测字符用的，就是开头的查表。")

# ---------- 3. forward 全流程：形状怎么变 ----------
print("\n===== 3. forward 数据流：每一步的形状（真实 batch） =====")
data_dir = os.path.join(NANOGPT, "data", "shakespeare_char")
train = np_memmap = None
import numpy as np
train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
B, T = 4, 64
raw = train[: B * (T + 1)].astype(np.int64)
batch = torch.from_numpy(raw).view(B, T + 1)[:, :-1].to(device)
targets = torch.from_numpy(raw).view(B, T + 1)[:, 1:].to(device)
print(f"输入 idx:        {tuple(batch.shape)}  (B={B} 个片段 × T={T} 个字符)")
print(f"目标 targets:    {tuple(targets.shape)}  (每个位置的下一个字符)")

model.eval()
with torch.no_grad():
    x = batch
    tok_emb = model.transformer.wte(x)
    print(f"wte 查表后:       {tuple(tok_emb.shape)}  (每个字符变成 384 维向量)")
    pos = torch.arange(0, T, dtype=torch.long, device=device)
    pos_emb = model.transformer.wpe(pos)
    print(f"wpe 查表后:       {tuple(pos_emb.shape)}  (位置向量，自动广播到每个样本)")
    x = model.transformer.drop(tok_emb + pos_emb)
    print(f"相加+dropout 后:  {tuple(x.shape)}  (语义=词义+位置)")
    for i, block in enumerate(model.transformer.h):
        x = block(x)
    print(f"穿过 6 个块后:    {tuple(x.shape)}  (形状一点没变——这是块能无限叠的原因)")
    x = model.transformer.ln_f(x)
    print(f"最终 LayerNorm:   {tuple(x.shape)}")
    logits = model.lm_head(x)
    print(f"lm_head 输出:     {tuple(logits.shape)}  (每个位置对 65 个字符各打一个分)")
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    print(f"交叉熵 loss:      {loss.item():.4f}  (随机初始化的真实值，约等于 ln(65)=4.17 附近)")
    print(f"logits 最后一维和: {logits.shape[-1]} == 词表大小 65 → 每个位置一个 65 维概率分布")

# ---------- 4. 加载训练 1000 步的 checkpoint，真的生成一段文本 ----------
print("\n===== 4. 用训练过的模型生成文本（1000 步 checkpoint，val loss 1.52） =====")
ckpt_path = os.path.join(NANOGPT, "out-shakespeare-char", "ckpt.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
print(f"加载完成，训练步数 iter_num = {ckpt['iter_num']}，best_val_loss = {ckpt['best_val_loss'].item():.4f}")
model.eval()

with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
itos = meta["itos"]  # {idx: char}

seed_text = "First Citizen:\nBefore we proceed any further, hear me speak.\n"
seed_ids = [meta["stoi"][ch] for ch in seed_text]
idx = torch.tensor([seed_ids], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=250, temperature=0.8, top_k=40)
generated = "".join(itos[int(i)] for i in out[0].tolist())
print("\n----- 生成结果（种子 + 模型续写 250 个字符） -----")
print(generated)
