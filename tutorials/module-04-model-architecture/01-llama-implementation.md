---
layout: page
title: "Module 04: 模型架构实现 — 构建LLaMA-style模型"
---
# Module 04: 模型架构实现 — 构建LLaMA-style模型

## 理论部分

### 4.1 现代LLM架构演进

```
GPT (2018) → GPT-2 (2019) → GPT-3 (2020) → LLaMA (2023) → LLaMA 3 (2024)
   │              │              │              │              │
 LayerNorm    LayerNorm     LayerNorm     RMSNorm      RMSNorm
 GELU         GELU          GELU          SwiGLU       SwiGLU
 Learned PE   Learned PE    Learned PE    RoPE         RoPE
 MHA          MHA           MHA           MHA/GQA      GQA
 Post-Norm    Post-Norm     Post-Norm     Pre-Norm     Pre-Norm
```

### 4.2 关键组件对比

| 组件 | GPT-2 | LLaMA 3 | 影响 |
|------|-------|---------|------|
| 归一化 | LayerNorm | RMSNorm | +15%速度 |
| 激活 | GELU | SwiGLU | 更好效果 |
| 位置编码 | Learned | RoPE | 长度外推 |
| 注意力 | MHA | GQA | 推理更快 |
| 归一化位置 | Post-Norm | Pre-Norm | 训练稳定 |
| 权重共享 | 无 | 有 | -30%参数 |

### 4.3 Grouped Query Attention (GQA)

GQA是MHA和MQA之间的折中：

```
MHA:  num_kv_heads = num_heads (每个头有自己的KV)
GQA:  num_kv_heads < num_heads (多个Q头共享KV)
MQA:  num_kv_heads = 1 (所有Q头共享同一对KV)
```

**为什么GQA？**
- MHA: 效果好但KV Cache大
- MQA: KV Cache小但质量下降
- GQA: 在质量和效率间取得平衡

### 4.4 模型参数量计算

```
Embedding:  V × d_model
Attention:  4 × d_model² (W_Q, W_K, W_V, W_O)
FFN:        3 × d_model × d_ff (w1, w2, w3 for SwiGLU)
LayerNorm:  2 × d_model (每层两个RMSNorm)
Total/Layer: 4d² + 3d·d_ff + 2d
Total:       V·d + N·(4d² + 3d·d_ff + 2d)
```

**示例 (LLaMA-7B)**：
- d=4096, N=32, d_ff=11008, V=32000
- ≈ 6.7B 参数

## 实践部分

### 实践1：完整LLaMA模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, d_head, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, seq_len, device=None):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.d_k = d_model // num_heads
        self.repeat = num_heads // self.num_kv_heads

        self.W_Q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin, mask=None):
        B, S, _ = x.size()
        Q = self.W_Q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)

        Q, K = apply_rope(Q, K, cos, sin)

        if self.repeat > 1:
            K = K.repeat_interleave(self.repeat, dim=1)
            V = V.repeat_interleave(self.repeat, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 1, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_O(out)

class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = int(8/3 * d_model)
        d_ff = ((d_ff + 255) // 256) * 256
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None):
        super().__init__()
        self.attn = Attention(d_model, num_heads, num_kv_heads)
        self.ffn = FFN(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x

class LLaMAModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_kv_heads=None, max_seq=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(d_model // num_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, num_kv_heads) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight Tying

    def forward(self, input_ids):
        B, S = input_ids.size()
        x = self.embed(input_ids) * math.sqrt(self.embed.embedding_dim)
        cos, sin = self.rope(S, input_ids.device)
        mask = torch.triu(torch.ones(S, S, device=input_ids.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, ids, max_new=20, temp=0.8, top_k=50):
        for _ in range(max_new):
            logits = self(ids[:, -2048:])[:, -1, :] / temp
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
        return ids

# 创建模型并统计参数
config = {'vocab_size': 32000, 'd_model': 512, 'num_heads': 8,
          'num_kv_heads': 2, 'num_layers': 6}
model = LLaMAModel(**config)
params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {params/1e6:.1f}M")
print(f"模型大小: {params * 2 / 1024 / 1024:.1f} MB (fp16)")

# 测试
ids = torch.randint(0, 32000, (1, 20))
logits = model(ids)
print(f"输入: {ids.shape} → 输出: {logits.shape}")
```

## 总结

| 组件 | 实现要点 |
|------|---------|
| RMSNorm | 无均值中心化，更快 |
| RoPE | 旋转编码相对位置 |
| GQA | KV头少于Q头，节省KV Cache |
| SwiGLU | 门控激活，效果更好 |
| Pre-Norm | 先归一化再计算 |
| Weight Tying | 共享嵌入和输出权重 |

**下一步**: [Module 05: 预训练](./../module-05-pretraining/)
