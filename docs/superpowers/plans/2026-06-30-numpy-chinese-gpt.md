# 纯 NumPy 中文 GPT 训练代码 — 实现计划

**目标**：编写一个单文件 Python 脚本 `train_gpt.py`，实现完整的 GPT 训练流程

**架构**：单文件包含数据下载、字符分词、GPT 模型（MultiHeadAttention + FeedForward + TransformerBlock）、Adam 优化器、训练循环和文本生成

**技术栈**：纯 NumPy，可选 requests（下载语料）、tqdm（进度条）

## 全局约束

- 纯 NumPy 实现，不依赖 PyTorch/TensorFlow/JAX
- 模型参数约 1M（vocab=3000, d_model=64, n_heads=4, n_layers=4, d_ff=256, max_seq_len=128）
- 关键代码必须有中文注释
- 单文件脚本，直接 `python train_gpt.py` 可运行
- CPU 上可流畅训练

---

## 任务：编写完整的 train_gpt.py

### 文件结构

- 创建：`train_gpt.py`

### 实现内容（按代码顺序）

#### 1. 依赖导入与超参数配置

```python
import numpy as np
import os
import json
from collections import Counter

# 尝试导入可选依赖
try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ==================== 超参数配置 ====================
VOCAB_SIZE = 3000        # 词表大小
D_MODEL = 64             # 嵌入维度
N_HEADS = 4              # 注意力头数
HEAD_DIM = D_MODEL // N_HEADS  # 每头维度 = 16
N_LAYERS = 4             # Transformer 层数
D_FF = 256               # FFN 隐藏层维度
MAX_SEQ_LEN = 128        # 最大序列长度
DROPOUT = 0.1            # Dropout 比率（训练时用，推理关闭）

# 训练超参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
CLIP_GRAD = 1.0          # 梯度裁剪阈值
WARMUP_STEPS = 500       # 预热步数
PRINT_EVERY = 100        # 每隔多少步打印一次 loss
SAVE_EVERY = 1000        # 每隔多少步保存模型
MODEL_SAVE_PATH = "gpt_chinese.npz"  # 模型保存路径
```

#### 2. 数据下载与语料处理

函数：`download_corpus()`
- 从公开数据源下载中文语料
- 首选：下载 `chinese-poetry` 全唐诗 JSON 文件
- 备选：如果下载失败，生成内置的示例中文文本
- 返回：合并后的中文文本字符串

函数：`build_vocab(text, vocab_size)`
- 统计数据中每个字符的出现频率
- 保留频率最高的 `vocab_size - 4` 个字符
- 添加特殊 token：`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
- 返回：`char2idx` 字典, `idx2char` 字典

函数：`prepare_data(text, char2idx, seq_len)`
- 将文本转为 token id 列表
- 按 `seq_len + 1` 长度切分（+1 因为 target 要偏移一位）
- 构建 (input, target) 对：input = tokens[:-1], target = tokens[1:]
- 返回：`np.array(inputs)`, `np.array(targets)`，shape 均为 (num_samples, seq_len)

函数：`get_batch(inputs, targets, batch_size)`
- 随机采样一个 batch
- 返回：batch_input, batch_target，shape 均为 (batch_size, seq_len)

#### 3. 模型组件

函数：`xavier_init(shape)`
- Xavier/Glorot 初始化
- 返回：`np.array`，使用 `np.sqrt(2.0 / (fan_in + fan_out))` 缩放

类/函数组：模型参数初始化 `init_model()`
- Token Embedding 矩阵：(vocab_size, d_model) — Xavier 初始化
- Position Embedding 矩阵：(max_seq_len, d_model) — Xavier 初始化
- 每层 TransformerBlock：
  - LayerNorm 参数：ln1_gamma (d_model,), ln1_beta (d_model,)
  - Attention: W_q (d_model, d_model), W_k (d_model, d_model), W_v (d_model, d_model), W_o (d_model, d_model)
  - LayerNorm 参数：ln2_gamma, ln2_beta
  - FFN: W1 (d_model, d_ff), b1 (d_ff,), W2 (d_ff, d_model), b2 (d_model,)
- 最终 LayerNorm：ln_f_gamma, ln_f_beta
- LM Head：与 Token Embedding 共享权重
- 返回：参数字典 `params`

函数：`layer_norm(x, gamma, beta, eps=1e-5)`
- 对最后一维做归一化
- mean = mean(x, axis=-1, keepdims=True)
- var = var(x, axis=-1, keepdims=True)
- return gamma * (x - mean) / sqrt(var + eps) + beta
- 返回：归一化结果（同时保留 mean 和 var 用于反向传播简化实现）

函数：`causal_mask(seq_len)`
- 创建下三角掩码矩阵
- 返回：shape (seq_len, seq_len) 的布尔/浮点矩阵，上三角为 -inf

函数：`multi_head_attention(x, W_q, W_k, W_v, W_o, mask, training=True)`
- Q = x @ W_q, K = x @ W_k, V = x @ W_v
- 拆分多头：reshape 为 (batch, n_heads, seq_len, head_dim)
- Attention scores = Q @ K^T / sqrt(head_dim)
- 应用因果掩码（上三角设为 -1e9）
- Softmax
- output = softmax @ V
- 合并多头：reshape 回 (batch, seq_len, d_model)
- output = merged @ W_o
- 返回：输出（训练时可选 dropout）

函数：`feedforward(x, W1, b1, W2, b2, training=True)`
- h = x @ W1 + b1
- h = ReLU(h)  # np.maximum(0, h)
- out = h @ W2 + b2
- 返回：输出（训练时可选 dropout）

函数：`transformer_block(x, block_params, mask, training=True)`
- Pre-LN: h = layer_norm(x, ln1_gamma, ln1_beta)
- h = multi_head_attention(h, ..., mask)
- x = x + h  # 残差连接
- Pre-LN: h2 = layer_norm(x, ln2_gamma, ln2_beta)
- h2 = feedforward(h2, ...)
- x = x + h2  # 残差连接
- 返回：x

函数：`gpt_forward(x, params, training=True)`
- token_emb = params['token_embedding'][x]  # (batch, seq_len, d_model)
- pos_emb = params['position_embedding'][:seq_len]  # (seq_len, d_model)
- h = token_emb + pos_emb
- mask = causal_mask(seq_len)
- for i in range(n_layers):
    h = transformer_block(h, params[f'block_{i}'], mask, training)
- h = layer_norm(h, params['ln_f_gamma'], params['ln_f_beta'])
- logits = h @ params['token_embedding'].T  # 权重共享
- 返回：logits (batch, seq_len, vocab_size)

#### 4. 损失函数与反向传播

函数：`cross_entropy_loss(logits, targets)`
- logits shape: (batch*seq_len, vocab_size)
- targets shape: (batch*seq_len,)
- 数值稳定的交叉熵：
  - logits_max = max(logits, axis=-1, keepdims=True)
  - logits_stable = logits - logits_max
  - log_sum_exp = log(sum(exp(logits_stable), axis=-1))
  - loss = -logits_stable[range(N), targets] + log_sum_exp
  - return mean(loss)
- 返回：标量 loss

函数：`backpropagate(logits, targets, cache)`
- 这里使用数值梯度检查来验证反向传播的正确性
- 对于纯 NumPy 实现，有两种选择：
  - (a) 反向传播的手写实现（复杂但快）
  - (b) 数值梯度（简单但慢，只用于验证）

考虑到项目目标是"能够运行并看到效果"，我们使用**手动实现的反向传播**，但由于手动为 GPT 写反向传播极其繁琐。我们采用**简化的自动微分方案**：

实际方案：**Numba-style 简单数值梯度用在小网络不可行（太慢），所以必须手动反向传播。**

最终方案：编写完整的手动反向传播。这是最复杂的部分，需要仔细处理每个操作的梯度。

#### 5. 优化器

类：`AdamOptimizer`
- 维护每个参数的一阶动量 m 和二阶动量 v
- 方法：`step(params, grads, lr, t)`
  - t 是当前步数（用于偏差修正）
  - m = beta1 * m + (1 - beta1) * grads
  - v = beta2 * v + (1 - beta2) * grads^2
  - m_hat = m / (1 - beta1^t)
  - v_hat = v / (1 - beta2^t)
  - params -= lr * m_hat / (sqrt(v_hat) + eps)

函数：`clip_gradients(grads, max_norm)`
- 计算所有梯度范数的平方和
- 如果总范数 > max_norm，则按比值缩放所有梯度

函数：`get_lr(step, warmup_steps, base_lr)`
- 前 warmup_steps 步：lr = base_lr * step / warmup_steps
- 之后：余弦退火 lr = base_lr * 0.5 * (1 + cos(pi * (step - warmup_steps) / total_steps))

#### 6. 训练循环

函数：`train()`
- 下载语料 → 构建词表 → 准备数据
- 初始化模型参数
- 初始化 Adam 优化器
- 循环训练：
  - 采样 batch
  - 前向传播得到 logits
  - 计算 loss
  - 反向传播计算梯度
  - 梯度裁剪
  - 计算当前学习率
  - 优化器更新参数
  - 定期打印 loss
  - 定期保存模型
- 返回：最终 params

#### 7. 文本生成

函数：`generate(prompt, params, char2idx, idx2char, max_new_tokens=100, temperature=0.8, top_k=50)`
- 将 prompt 转为 token id 序列
- 循环 max_new_tokens 次：
  - 取最后 MAX_SEQ_LEN 个 token 作为输入
  - 前向传播（training=False）得到最后一个 token 的 logits
  - temperature 缩放：logits /= temperature
  - top-k 过滤：只保留概率最高的 k 个 token，其余设为 -inf
  - softmax 转为概率
  - 从概率分布中采样下一个 token
  - 拼接到序列末尾
- 将 token id 序列转回中文文本
- 返回：完整字符串

#### 8. 模型保存与加载

函数：`save_model(params, path)`
- 使用 `np.savez_compressed(path, **params)` 保存所有参数

函数：`load_model(path)`
- 使用 `np.load(path)` 加载，返回参数字典

#### 9. 主函数

```python
def main():
    print("="*50)
    print("  纯 NumPy 中文 GPT 训练")
    print("="*50)
    
    # 1. 下载语料
    text = download_corpus()
    
    # 2. 构建词表
    char2idx, idx2char = build_vocab(text, VOCAB_SIZE)
    
    # 3. 准备数据
    inputs, targets = prepare_data(text, char2idx, MAX_SEQ_LEN)
    
    # 4. 检查是否有已保存的模型
    start_epoch = 0
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n发现已保存的模型 {MODEL_SAVE_PATH}，加载中...")
        saved_params = load_model(MODEL_SAVE_PATH)
        # 验证参数形状是否匹配
        model_params = init_model()  # 用保存的参数覆盖
        for key in model_params:
            if key in saved_params:
                model_params[key] = saved_params[key]
    else:
        model_params = init_model()
    
    # 5. 训练
    model_params = train(model_params, inputs, targets, char2idx, idx2char)
    
    # 6. 文本生成演示
    print("\n" + "="*50)
    print("  文本生成演示")
    print("="*50)
    
    prompts = ["床前明月光",", "春眠不觉晓", "你好世界"]
    for prompt in prompts:
        result = generate(prompt, model_params, char2idx, idx2char, max_new_tokens=50)
        print(f"\n输入: {prompt}")
        print(f"生成: {result}")
```

### 关键注意事项

1. **反向传播是核心难点**：纯 NumPy 实现 GPT 的反向传播需要非常仔细地为每个操作编写梯度计算。这里提供完整的实现。

2. **梯度计算中需要注意的维度变化**：特别是多头注意力的 split/merge 操作的梯度回传。

3. **数值稳定性**：softmax 和 cross_entropy 都要减去最大值。

4. **dropout 实现**：训练时随机生成 mask，推理时关闭。

5. **因果掩码**：确保模型不能看到未来信息，mask 矩阵为下三角。

### 验证方法

```bash
# 安装依赖（仅需 numpy）
pip install numpy

# 可选依赖
pip install requests tqdm

# 运行训练
python train_gpt.py
```

预期：
- 输出 loss 应该在前 100 步内从 ~8 降到 ~5 以下
- 最终保存模型文件 `gpt_chinese.npz`
- 文本生成能输出合理的中文字符

---

## 交付物

| 文件 | 说明 |
|------|------|
| `train_gpt.py` | 完整的端到端 GPT 训练脚本（约 800-1000 行） |
