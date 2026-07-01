# Module 02: Transformer架构深入 — Self-Attention机制

## 理论部分

### 2.1 为什么需要注意力？

在Transformer之前，RNN/LSTM处理序列是顺序的：

```
RNN: h₁ → h₂ → h₃ → ... → hₙ  (串行，难以并行)
```

**核心问题**：
1. 长距离依赖难以捕捉（梯度消失）
2. 无法并行计算（训练慢）
3. 信息瓶颈（所有信息压缩到固定大小的隐藏状态）

**注意力机制的解决方案**：每个位置直接"关注"所有其他位置，一步到位。

### 2.2 Self-Attention的直觉

想象你在读这句话："**猫**坐在垫子上，因为它饿了"

要理解"它"指代什么，你需要将"它"与前面的词关联起来。Self-Attention就是让模型自动学习这种关联。

**三个核心概念**：
- **Query (查询)**：当前词在问"我需要什么信息？"
- **Key (键)**：每个词在说"我提供什么信息？"
- **Value (值)**：每个词的实际内容

### 2.3 Self-Attention的数学

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

**逐步分解**：

**Step 1: 线性投影**
```
Q = X · W_Q    (查询矩阵)
K = X · W_K    (键矩阵)
V = X · W_V    (值矩阵)
```

**Step 2: 计算注意力分数**
```
Scores = Q · Kᵀ / √d_k
```
- Q·Kᵀ：计算每对(token_i, token_j)的相似度
- /√d_k：缩放，防止点积过大导致softmax梯度消失

**Step 3: Softmax归一化**
```
Weights = softmax(Scores)
```
- 每行和为1，表示当前token对其他token的注意力分配

**Step 4: 加权求和**
```
Output = Weights · V
```
- 每个位置的输出是所有位置Value的加权平均

### 2.4 为什么除以 √d_k？

**数学证明**：

假设 q, k ∈ ℝ^{d_k} 的每个分量独立，均值为0，方差为1。

```
q · k = Σᵢ qᵢkᵢ
E[q·k] = Σᵢ E[qᵢ]E[kᵢ] = 0
Var(q·k) = Σᵢ Var(qᵢkᵢ) = Σᵢ Var(qᵢ)Var(kᵢ) = d_k
```

点积的方差是 d_k。当 d_k = 64 时，点积的标准差 = 8。

**问题**：点积值很大 → softmax进入梯度极小的饱和区 → 梯度消失

**解决**：除以 √d_k 使方差归一化为1，保持梯度稳定。

### 2.5 多头注意力 (Multi-Head Attention)

单头注意力只能学习一种关联模式。多头注意力允许模型同时关注不同类型的关系：

```
MultiHead(Q,K,V) = Concat(head₁, head₂, ..., headₕ) · W_O

headᵢ = Attention(Q·W_Qᵢ, K·W_Kᵢ, V·W_Vᵢ)
```

**直觉**：
- Head 1 可能学习语法关系（主谓一致）
- Head 2 可能学习语义关系（同义词）
- Head 3 可能学习位置关系（相邻词）
- ...

**维度关系**：
- d_model = 768 (总维度)
- h = 12 (头数)
- d_k = d_v = d_model / h = 64 (每头维度)

### 2.6 因果掩码 (Causal Mask)

在自回归语言模型中，生成第t个词时只能看到前t-1个词，不能"偷看"未来：

```
掩码矩阵 (seq_len=4):
┌ 0  -∞  -∞  -∞ ┐
│ 0   0  -∞  -∞ │
│ 0   0   0  -∞ │
└ 0   0   0   0 ┘
```

- 0：允许关注
- -∞：禁止关注（softmax后变为0）

### 2.7 计算复杂度

Self-Attention的时间复杂度：O(n²·d)
- n: 序列长度
- d: 模型维度

**瓶颈**：序列长度n的平方复杂度限制了长上下文处理能力。

## 实践部分

### 实践1：从零实现Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=" * 60)
print("实践1: 从零实现Self-Attention")
print("=" * 60)

class SelfAttention(nn.Module):
    """单头Self-Attention"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (seq_len, seq_len) 或 None
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        Q = self.W_Q(x)  # (batch, seq, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        # 注意力分数: Q @ K^T / sqrt(d_k)
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))

        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)

        # 加权求和
        output = torch.bmm(attention_weights, V)

        return output, attention_weights

# 测试
torch.manual_seed(42)
batch_size, seq_len, d_model = 2, 5, 16

x = torch.randn(batch_size, seq_len, d_model)
attn = SelfAttention(d_model)
output, weights = attn(x)

print(f"输入 shape: {x.shape}")
print(f"输出 shape: {output.shape}")
print(f"注意力权重 shape: {weights.shape}")
print(f"注意力权重 (第一个batch):\n{weights[0].detach().numpy().round(3)}")
print(f"每行之和: {weights[0].sum(dim=-1).detach().numpy()}")
```

### 实践2：多头注意力实现

```python
print("\n" + "=" * 60)
print("实践2: 多头注意力实现")
print("=" * 60)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性投影
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        """(batch, seq, d_model) → (batch, num_heads, seq, d_k)"""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # 线性投影并分头
        Q = self.split_heads(self.W_Q(x))  # (batch, heads, seq, d_k)
        K = self.split_heads(self.W_K(x))
        V = self.split_heads(self.W_V(x))

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 1, float('-inf'))

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)

        # 加权求和
        context = torch.matmul(attention_weights, V)  # (batch, heads, seq, d_k)

        # 合并多头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        # 最终线性变换
        output = self.W_O(context)

        return output, attention_weights

# 测试多头注意力
d_model = 64
num_heads = 8
seq_len = 6

x = torch.randn(1, seq_len, d_model)
mha = MultiHeadAttention(d_model, num_heads)

# 创建因果掩码
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
output, weights = mha(x, mask=causal_mask)

print(f"d_model={d_model}, num_heads={num_heads}, d_k={d_model//num_heads}")
print(f"输入 shape: {x.shape}")
print(f"输出 shape: {output.shape}")
print(f"注意力权重 shape: {weights.shape} (batch, heads, seq, seq)")

# 验证因果掩码
print(f"\n第一个头的注意力权重:\n{weights[0, 0].detach().numpy().round(3)}")
print("→ 上三角为0，符合因果掩码要求")
```

### 实践3：注意力权重可视化

```python
print("\n" + "=" * 60)
print("实践3: 注意力权重可视化")
print("=" * 60)

def visualize_attention(tokens, attention_weights, title="Attention Weights"):
    """文本形式可视化注意力权重"""
    n = len(tokens)
    weights = attention_weights.detach().numpy()

    print(f"\n{title}")
    print("       ", "  ".join(f"{t:>6s}" for t in tokens))
    for i, token in enumerate(tokens):
        row = "  ".join(f"{weights[i][j]:6.3f}" for j in range(n))
        bar = "".join("█" * int(weights[i][j] * 20) for j in range(n))
        print(f"{token:>6s} [{row}]")

# 模拟一个句子的注意力
tokens = ["The", "cat", "sat", "on", "mat"]
seq_len = len(tokens)
d_model = 32
num_heads = 4

x = torch.randn(1, seq_len, d_model)
mha = MultiHeadAttention(d_model, num_heads)
_, weights = mha(x)

# 可视化每个头
for h in range(num_heads):
    visualize_attention(tokens, weights[0, h], f"Head {h+1}")

# 实践4: 缩放因子的重要性
print("\n" + "=" * 60)
print("实践4: 缩放因子 √d_k 的重要性")
print("=" * 60)

def attention_without_scaling(Q, K, V):
    """不使用缩放的注意力"""
    scores = torch.bmm(Q, K.transpose(1, 2))  # 不除以 sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, V), weights

def attention_with_scaling(Q, K, V):
    """使用缩放的注意力"""
    d_k = Q.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, V), weights

# 测试不同维度下的影响
for d_k in [16, 64, 256, 1024]:
    Q = torch.randn(1, 4, d_k)
    K = torch.randn(1, 4, d_k)
    V = torch.randn(1, 4, d_k)

    _, w_no_scale = attention_without_scaling(Q, K, V)
    _, w_scale = attention_with_scaling(Q, K, V)

    # 计算注意力权重的熵（越低表示越尖锐）
    entropy_no_scale = -torch.sum(w_no_scale * torch.log(w_no_scale + 1e-10), dim=-1).mean()
    entropy_scale = -torch.sum(w_scale * torch.log(w_scale + 1e-10), dim=-1).mean()

    print(f"\nd_k={d_k:4d}: 无缩放熵={entropy_no_scale:.4f}, 有缩放熵={entropy_scale:.4f}")
    print(f"  无缩放最大注意力: {w_no_scale[0].max().item():.4f}")
    print(f"  有缩放最大注意力: {w_scale[0].max().item():.4f}")

print("\n→ 维度越大，不缩放的注意力越尖锐（熵越低），梯度越容易消失")
print("→ 缩放使注意力分布更平滑，保持有效梯度")
```

## 总结

| 组件 | 公式/作用 |
|------|----------|
| Q, K, V | 通过线性投影从输入得到 |
| 注意力分数 | QKᵀ/√d_k，衡量token间相似度 |
| Softmax | 归一化为概率分布 |
| 输出 | 对V的加权平均 |
| 多头 | 并行学习多种关联模式 |
| 因果掩码 | 防止关注未来位置 |
| 缩放因子 | 防止点积过大导致梯度消失 |

**下一步**: [Module 03: 分词与词表构建](./../module-03-tokenization/)
