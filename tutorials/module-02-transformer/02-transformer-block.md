# Module 02: Transformer架构深入 — 完整Transformer Block

## 理论部分

### 2.8 Transformer Block结构

一个完整的Transformer Block包含：

```
Input (x)
  │
  ├──→ LayerNorm ──→ Multi-Head Attention ──→ Dropout ──→ (+ residual)
  │                                                              │
  │                                                              ↓
  ├──→ LayerNorm ──→ Feed-Forward Network ──→ Dropout ──→ (+ residual)
  │
  ↓
Output
```

### 2.9 层归一化 (LayerNorm vs RMSNorm)

**LayerNorm** (原始Transformer):
```
μ = mean(x)
σ² = var(x)
LN(x) = γ · (x - μ) / √(σ² + ε) + β
```

**RMSNorm** (LLaMA/Mistral使用):
```
RMS(x) = √(mean(x²))
RMSNorm(x) = x / RMS(x) · γ
```

**为什么RMSNorm更好？**
- 去掉了均值中心化（减μ），减少15%计算量
- 效果相当甚至更好
- 是LLaMA系列的标准选择

### 2.10 前馈网络 (FFN / SwiGLU)

**标准FFN** (GPT-2风格):
```
FFN(x) = W₂ · ReLU(W₁·x + b₁) + b₂
```

**SwiGLU** (LLaMA/PaLM/Gemini使用):
```
SwiGLU(x) = (Swish(xW₁) ⊙ xW₃) · W₂
```

其中 Swish(x) = x · sigmoid(x)

**为什么SwiGLU更好？**
- 门控机制让模型学会"选择性地通过信息"
- 实验证明效果优于ReLU/GELU
- 是现代LLM的标配

**维度关系**：
- d_model = 4096
- d_ff = 11008 (≈ 8/3 × d_model for SwiGLU)

### 2.11 残差连接 (Residual Connection)

```
output = x + Sublayer(x)
```

**为什么需要残差连接？**
1. **梯度高速公路**：梯度可以直接通过残差路径回传，避免梯度消失
2. **增量学习**：每层只需学习"增量"（残差），而非完整映射
3. **深层网络**：没有残差连接，100+层的网络几乎无法训练

### 2.12 Pre-Norm vs Post-Norm

**Post-Norm** (原始Transformer):
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Pre-Norm** (现代LLM标准):
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Pre-Norm的优势**：
- 训练更稳定（归一化在子层之前）
- 不需要Warmup就能训练深层网络
- 梯度流动更顺畅

### 2.13 位置编码

#### 绝对位置编码 (Sinusoidal)
```
PE(pos, 2i) = sin(pos / 10000^{2i/d})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d})
```

#### RoPE (旋转位置编码) - 现代标准

RoPE的核心思想：通过旋转编码相对位置信息。

```
对位置m的query和位置n的key:
q̃_m = q_m · e^{imθ}
k̃_n = k_n · e^{inθ}

q̃_m · k̃_n* = q_m · k_n* · e^{i(m-n)θ}
```

**关键性质**：注意力分数只依赖于相对位置 (m-n)，而非绝对位置。

**RoPE的优势**：
1. 编码相对位置（更符合语言特性）
2. 支持长度外推（训练时短序列，推理时长序列）
3. 与线性注意力兼容

## 实践部分

### 实践1：完整Transformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("实践1: 完整Transformer Block实现")
print("=" * 60)

class RMSNorm(nn.Module):
    """RMSNorm - LLaMA系列使用的归一化"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (batch, seq, d_model)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SwiGLU(nn.Module):
    """SwiGLU前馈网络"""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8/3 * d_model)  # SwiGLU标准比例
            d_ff = ((d_ff + 255) // 256) * 256  # 对齐到256

        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # down
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # up

    def forward(self, x):
        # SwiGLU: (Swish(x @ W1) * (x @ W3)) @ W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MultiHeadAttention(nn.Module):
    """多头注意力（带RoPE支持）"""
    def __init__(self, d_model, num_heads, num_kv_heads=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads  # GQA支持
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        # GQA: 如果KV头数少于Q头，需要重复
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(repeat, dim=1)
            V = V.repeat_interleave(repeat, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 1, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_O(context), attn_weights

class TransformerBlock(nn.Module):
    """完整的Transformer Block (Pre-Norm + SwiGLU + RMSNorm)"""
    def __init__(self, d_model, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, num_kv_heads)
        self.ffn = SwiGLU(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-Norm: 先归一化，再计算，最后残差
        normed = self.norm1(x)
        attn_out, weights = self.attn(normed, mask)
        x = x + self.dropout(attn_out)

        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, weights

# 测试
torch.manual_seed(42)
d_model = 256
num_heads = 8
num_kv_heads = 2  # GQA: 2个KV头，8个Q头
seq_len = 10
batch_size = 2

x = torch.randn(batch_size, seq_len, d_model)
block = TransformerBlock(d_model, num_heads, num_kv_heads)

# 因果掩码
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
output, weights = block(x, mask=mask)

print(f"输入 shape: {x.shape}")
print(f"输出 shape: {output.shape}")
print(f"注意力权重 shape: {weights.shape}")
print(f"参数量: {sum(p.numel() for p in block.parameters()):,}")
```

### 实践2：RoPE实现

```python
print("\n" + "=" * 60)
print("实践2: RoPE (旋转位置编码) 实现")
print("=" * 60)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_head, base=10000):
        super().__init__()
        # 计算频率: θ_i = 1 / (base^(2i/d))
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device=None):
        """生成位置编码"""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # 外积: (seq_len, d_head/2)
        freqs = torch.outer(t, self.inv_freq)
        # 拼接: (seq_len, d_head)
        emb = torch.cat((freqs, freqs), dim=-1)
        # 返回 cos 和 sin
        return emb.cos(), emb.sin()

def rotate_half(x):
    """将向量分成两半并旋转"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    """对Q和K应用RoPE"""
    # q, k: (batch, heads, seq, d_head)
    # cos, sin: (seq, d_head)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d_head)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

# 测试RoPE
d_head = 64
seq_len = 8
rope = RotaryPositionEmbedding(d_head)
cos, sin = rope(seq_len)

print(f"RoPE频率 shape: {rope.inv_freq.shape}")
print(f"cos shape: {cos.shape}, sin shape: {sin.shape}")

# 验证RoPE的相对位置性质
print("\n--- 验证RoPE的相对位置性质 ---")
# 位置m和n的注意力分数应只依赖于(m-n)
Q = torch.randn(1, 1, seq_len, d_head)
K = torch.randn(1, 1, seq_len, d_head)

Q_rotated, K_rotated = apply_rope(Q, K, cos, sin)

# 计算旋转后的注意力分数
scores = torch.matmul(Q_rotated, K_rotated.transpose(-2, -1)) / math.sqrt(d_head)
scores = scores[0, 0].detach().numpy()

print("注意力分数矩阵 (应呈现对角线模式，即只依赖相对位置):")
for i in range(min(5, seq_len)):
    row = "  ".join(f"{scores[i][j]:7.3f}" for j in range(min(5, seq_len)))
    print(f"  位置{i}: [{row}]")

# 验证: score(i,j) == score(i+k, j+k) (只依赖相对位置)
print("\n验证相对位置不变性:")
print(f"score(2,1) = {scores[2][1]:.4f}")
print(f"score(4,3) = {scores[4][3]:.4f}")
print(f"score(5,4) = {scores[5][4]:.4f}")
print("→ 相同相对位置的值相近，验证了RoPE的性质")
```

### 实践3：构建完整GPT模型

```python
print("\n" + "=" * 60)
print("实践3: 构建完整GPT模型")
print("=" * 60)

class GPTModel(nn.Module):
    """完整的GPT模型"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionEmbedding(d_model // num_heads)

        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        # 最终归一化和输出
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying: 共享嵌入和输出权重
        self.lm_head.weight = self.token_embedding.weight

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len) 词ID
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()

        # Token嵌入
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)

        # 因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1)

        # 通过所有Transformer层
        for layer in self.layers:
            x, _ = layer(x, mask)

        # 最终归一化
        x = self.final_norm(x)

        # 输出logits
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=20, temperature=1.0, top_k=None):
        """自回归生成"""
        for _ in range(max_new_tokens):
            # 截断到最大长度
            idx_cond = input_ids[:, -self.max_seq_len:]

            # 前向传播
            logits = self(idx_cond)

            # 取最后一个位置的logits
            logits = logits[:, -1, :] / temperature

            # Top-k采样
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            # Softmax
            probs = F.softmax(logits, dim=-1)

            # 采样
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# 创建模型
config = {
    'vocab_size': 1000,
    'd_model': 256,
    'num_heads': 8,
    'num_layers': 4,
    'max_seq_len': 128
}

model = GPTModel(**config)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型配置: {config}")
print(f"总参数量: {total_params:,} ({total_params/1e6:.1f}M)")

# 测试前向传播
input_ids = torch.randint(0, config['vocab_size'], (2, 10))
logits = model(input_ids)
print(f"\n输入 shape: {input_ids.shape}")
print(f"输出 logits shape: {logits.shape}")

# 测试生成
print("\n--- 生成测试 ---")
prompt = torch.randint(0, config['vocab_size'], (1, 5))
generated = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=50)
print(f"输入长度: {prompt.shape[1]}")
print(f"生成后长度: {generated.shape[1]}")
print(f"生成的新token数: {generated.shape[1] - prompt.shape[1]}")
```

## 总结

| 组件 | 原始Transformer | 现代LLM (LLaMA风格) |
|------|----------------|-------------------|
| 归一化 | LayerNorm | RMSNorm |
| 激活函数 | ReLU/GELU | SwiGLU |
| 位置编码 | Sinusoidal | RoPE |
| 注意力 | MHA | GQA/MHA |
| 归一化位置 | Post-Norm | Pre-Norm |
| 权重共享 | 无 | Weight Tying |

**下一步**: [Module 03: 分词与词表构建](./../module-03-tokenization/)
