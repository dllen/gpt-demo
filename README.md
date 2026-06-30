# 中文 GPT —— 纯 NumPy 从零实现

一个可运行的 GPT (Decoder-Only Transformer) 训练脚本，**不依赖 PyTorch / TensorFlow / JAX**，仅使用 NumPy 实现完整的前向传播、反向传播和训练流程。支持中文语料训练与文本生成。

## 快速开始

```bash
python train_gpt.py
```

或指定训练时长（需要 `train.sh`）：

```bash
./train.sh            # 默认训练 1 小时
./train.sh 30m        # 训练 30 分钟
./train.sh 2h         # 训练 2 小时
./train.sh 90m        # 自定义 90 分钟
```

无需额外安装（仅需 `numpy`，可选 `tqdm` 显示进度条）。

## 项目结构

```
gpt-demo/
├── train_gpt.py              # 核心脚本 (~1100 行，纯 NumPy)
├── train_gpt.ipynb           # Jupyter Notebook (含可视化与交互演示)
├── train.sh                  # 训练时长控制脚本
├── _duration.py              # 预估训练耗时辅助脚本
├── gpt_chinese.npz           # 训练好的模型参数
├── corpus.txt                # 本地语料缓存 (自动生成)
└── docs/                     # 文档
```

## 模型架构

| 组件 | 实现 |
|------|------|
| 架构 | Pre-LN Transformer Decoder |
| 层数 | 4 |
| 模型维度 | 64 |
| 注意力头数 | 4 (每头维度 16) |
| 前馈维度 | 256 |
| 最大序列长度 | 128 |
| 词表大小 | 5000 (字符级) |
| 参数量 | ~460K |
| 激活函数 | ReLU |
| 正则化 | Dropout (0.1) |
| 权重共享 | Embedding ↔ LM Head |

## 训练配置

| 超参数 | 值 |
|--------|-----|
| Batch Size | 64 |
| 学习率 | 3e-3 |
| 训练轮数 | 60 (默认，可通过环境变量覆盖) |
| Warmup Steps | 500 |
| 梯度裁剪 | 1.0 (全局范数) |
| 优化器 | Adam (β1=0.9, β2=0.999) |
| 学习率调度 | Linear Warmup + Cosine Decay |

## 代码模块

| 模块 | 功能 |
|------|------|
| 模块 0 | 全局超参数配置 |
| 模块 1 | 数据下载与处理（含内置古诗词 + AI/科普 fallback） |
| 模块 2 | 模型组件（LayerNorm、Multi-Head Attention、FFN、Transformer Block） |
| 模块 3 | 交叉熵损失函数 |
| 模块 4 | Adam 优化器 + 梯度裁剪 |
| 模块 5 | 训练循环 + 模型保存/加载 |
| 模块 6 | 文本生成（Temperature + Top-K 采样 + 简繁转换） |
| 模块 7 | 主函数入口 |

## 文本生成

支持 `temperature`、`top-k` 采样，以及简体中文到繁体的自动转换（匹配全唐诗训练语料）：

```python
generate("春眠不觉晓", params, char2idx, idx2char,
         max_new_tokens=50, temperature=0.5, top_k=20)
```

训练完成后自动演示多个 prompt 的生成效果。

演示 prompts 包括：`春眠不觉晓`、`人生若只如初见`、`人工智能`、`大江东去`、`红色`。

## 语料

- **在线语料**：自动从 GitHub (chinese-poetry/chinese-poetry) 下载全唐诗（约 5 万首）
- **本地缓存**：`corpus.txt`（首次下载后自动保存，后续直接从缓存加载）
- **内置 fallback**：经典古诗词（100+ 首）+ AI/ML 科普文本（网络不可用时使用）
- **滑动窗口**：序列长度 128，步长为 64（50% 重叠）

## 验证

- **梯度检查**：float64 数值梯度检验，所有参数相对误差 < 1e-8
- **端到端训练**：Loss 正常下降，生成文本连贯

## 依赖

```
numpy>=1.20
tqdm>=4.60   # 可选，用于进度条
requests     # 可选，用于下载在线语料
```

## License

MIT
