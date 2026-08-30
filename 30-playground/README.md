# 第 30 课：结业——把模型做成可玩的东西

> 本节代码：✅ 见 `code/`（30-chat.py：终端聊天小工具 REPL，支持海盗管家/莎士比亚双人格、温度与采样旋钮、交互式对话；_30_gpt.py：MyGPT 模型复用）
>
> 本系列《手搓大模型：从零构建 NanoGPT》的第三十课，也是结业课。
> 目标读者：零基础。前 29 课把 GPT 拆成了零件、又亲手装了回去；今天把这些零件拼成一件能玩的东西——一个真的能聊天的终端程序。

## 一句话直觉

30 天攒的不是一个模型，是一整套"会说话的积木"——今天把它们装进一个终端程序，喂一句话，模型真的会接话。整个小工具的核心代码只有 100 行左右：加载 checkpoint → 编码 → 生成循环 → 采样 → 解码，每一步都是前 29 课写过的老朋友。

## 真实对话（Mac mini 实测，一字未改）

```
[pirate] > what is your name?
ASSISTANT: captain redbeard at your service, matey! arr!

[pirate] > tell me a joke.
ASSISTANT: why did the pirate learn to play the fiddle? because he wanted to make the sailors dance! arr!

[pirate] > what do you think of the king?
ASSISTANT: the king stays in his castle, and i stay on the sea, and we both stay happy! arr!
```

## 小工具 = 三块积木

| 积木 | 函数 | 来源 |
|------|------|------|
| 采样三件套 | `sample_next` | 第 25 课原样搬来 |
| 生成循环 | `generate` | 第 25 课原样搬来 |
| 模型加载 | `load_model` | 支持本系列 MyGPT 格式 + nanoGPT 官方格式 |

外加一个 REPL 主循环（交互界面）和一个 `pirate_reply`（对话格式拼接，第 28 课的技巧）。

## 运行

```bash
# 海盗管家模式（28 课微调模型，会接话）
python 30-chat.py --model pirate

# 莎士比亚模式（26 课训练模型，会写剧本）
python 30-chat.py --model shake

# 非交互演示模式（跑一组固定问题，给文档/测试用）
python 30-chat.py --demo
```

交互命令：`/help` 帮助、`/model pirate|shake` 切人格、`/temp 0.8` 温度、`/topk 40`、`/topp 0.9`、`/len 150`、`/quit` 退出。

## 真实实验数据

1. **海盗管家问答**：训练集内问题背得漂亮，训练集外问题"风格对、内容靠运气"；
2. **莎士比亚续写**：同一个 `generate`，喂给剧作家就是莎剧腔——模型没变，变的是训练数据教出来的说话方式；
3. **温度对比**：训练集内问题三个温度答得一字不差（微调只有 9424 个 token，模型背下来了）；训练集外问题高温开始"编"。

## 配图

- `images/30-chat-structure.png`：小工具结构图（加载 → 生成循环 → 采样）
- `images/30-roadmap.png`：30 课完整路线图（阶段一~五 + 进阶路径）
- `images/cover-30.png`：封面

## 环境

Mac mini（Apple Silicon / MPS）+ PyTorch 2.12.1。模型 checkpoint：28 课微调产物（`ckpt-finetune-dialogue.pt`）或 nanoGPT `out-shakespeare-char/ckpt.pt`；找不到时用 `--ckpt` 显式指定路径。

## 系列仓库

https://github.com/flyingcloud-code/learning-nanogpt-from-0-to-1
