# 纯 NumPy 中文 GPT 训练代码 — 设计文档

**日期**: 2026-06-30
**状态**: 已批准

## 概述

实现一个完整的、端到端的 GPT 模型训练代码，使用纯 NumPy（无深度学习框架依赖），基于中文语料进行训练。模型规模约 1M 参数，确保在个人机器（CPU）上可流畅运行。

## 技术选型

| 维度 | 选择 |
|------|------|
| 框架 | 纯 NumPy |
| 分词 | 字符级（Character-level） |
| 模型规模 | ~1M 参数 |
| 位置编码 | 可学习位置嵌入 |
| 训练器 | Adam + 梯度裁剪 + 余弦退火 |
| 代码组织 | 单文件脚本 `train_gpt.py` |
| 语料来源 | 自动下载公开中文数据集 |

## 模块设计

### 1. 数据模块 (`download_corpus`, `build_vocab`, `prepare_data`)

- 自动下载中文语料（优先使用 `chinese-poetry` GitHub 仓库，备选维基百科中文摘要）
- 字符级分词：统计频率最高的 3000 个字符，构建词表
- 文本转为 token id 序列，按 `seq_len=128` 切分训练样本
- 生成 (input, target) 对：target 是 input 向右偏移一位

### 2. 模型模块 (`GPTModel` 及其子模块)

**模型超参数（~1M 参数）**：

| 参数 | 值 |
|------|-----|
| vocab_size | 3000 |
| d_model | 64 |
| n_heads | 4 |
| head_dim | 16 |
| n_layers | 4 |
| d_ff | 256 |
| max_seq_len | 128 |
| dropout | 0.1 |

**子模块**：

- `MultiHeadAttention`: 缩放点积注意力 + 因果掩码
- `FeedForward`: Linear → ReLU → Linear
- `TransformerBlock`: Pre-LN → Self-Attention → Residual → Pre-LN → FFN → Residual
- `GPTModel`: TokenEmbedding + PositionEmbedding + N×TransformerBlock + LayerNorm + LM_Head

### 3. 训练模块 (`train`)

- Adam 优化器（纯 NumPy 实现，含一阶/二阶动量）
- 梯度裁剪（max_norm=1.0）
- 学习率余弦退火（warmup 500 步 + cosine decay）
- Batch size: 32
- 每 100 步输出 loss
- 支持模型保存/加载（numpy `.npz` 格式）

### 4. 生成模块 (`generate`)

- 输入上文，自回归生成后续 token
- 支持 temperature 采样
- 支持 top-k 采样
- 将生成的 token id 转为中文文本输出

## 数据流

```
[中文语料下载] → [字符级分词] → [Token ID 序列] → [按 seq_len 切分]
                                                          ↓
[文本生成] ← [Token → 字符] ← [自回归采样] ← [GPT Model] ←→ [训练循环]
                                               ↑↓
                                         [Adam 优化器]
```

## 文件结构

```
gpt-demo/
├── train_gpt.py          # 主文件：数据 + 模型 + 训练 + 生成
├── requirements.txt      # 依赖：numpy, requests, tqdm
├── README.md             # 使用说明
└── README
```

实际上用户选择单文件脚本，所以 `requirements.txt` 和 `README.md` 可选。

## 关键设计决策

1. **Pre-LN 而非 Post-LN**：Pre-LN（先 LayerNorm 再进入子层）训练更稳定，适合小模型
2. **可学习位置编码**：比正弦编码简单，且对小模型足够
3. **字符级分词**：词表仅 ~3000，避免了复杂分词器的实现负担
4. **因果掩码**：确保模型只能看到过去的信息，符合 GPT 的自回归特性
5. **交叉熵损失**：标准的语言模型训练目标

## 验证方式

- 运行 `python train_gpt.py` 应能完成训练并输出 loss 下降曲线
- 训练完成后输入"床前明月光"，模型应能生成连贯的中文文本
- Loss 应在几百步内明显下降（从 ~8 降至 ~4-5）
