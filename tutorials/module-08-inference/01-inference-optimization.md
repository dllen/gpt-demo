# Module 08: 推理优化 — KV Cache、量化与推测解码

## 理论部分

### 8.1 推理瓶颈

自回归生成是逐token进行的，每生成一个token都需要完整的前向传播：

```
生成第1个token: 处理prompt (n个token)
生成第2个token: 处理n+1个token  ← 重复计算了前n个token!
生成第3个token: 处理n+2个token  ← 又重复计算了!
...
```

**核心问题**: 大量重复计算导致推理速度慢。

### 8.2 KV Cache

**原理**: 缓存已计算的K和V，避免重复计算。

```
不使用KV Cache:
  每步计算: Q·K^T → 需要所有K
  复杂度: O(n²) 每步

使用KV Cache:
  只计算新token的Q, K, V
  将新K, V追加到缓存
  复杂度: O(n) 每步
```

**显存占用**: `2 × num_layers × num_heads × d_head × seq_len × sizeof(dtype)`

### 8.3 量化

将模型权重从高精度转为低精度：

| 精度 | 位宽 | 相对大小 | 质量损失 |
|------|------|---------|---------|
| FP32 | 32 | 100% | 无 |
| FP16 | 16 | 50% | 极小 |
| BF16 | 16 | 50% | 极小 |
| INT8 | 8 | 25% | 小 |
| INT4 | 4 | 12.5% | 中 |
| NF4 | 4 | 12.5% | 中 (非均匀) |

**量化方法**:
- PTQ (Post-Training Quantization): 训练后量化
- QAT (Quantization-Aware Training): 量化感知训练
- GPTQ/AWQ: 基于校准数据的权重量化

### 8.4 推测解码 (Speculative Decoding)

**核心思想**: 用小模型快速生成草稿，大模型验证。

```
1. 小模型(Draft)快速生成 k 个token (草稿)
2. 大模型(Target)并行验证这 k 个token
3. 接受的token保留，拒绝的从正确分布重新采样

加速比: 通常 2-3x
条件: 小模型质量不能太差
```

### 8.5 PagedAttention (vLLM)

借鉴操作系统虚拟内存管理：

```
传统KV Cache: 连续分配 → 碎片多，浪费大
PagedAttention: 分页管理 → 按需分配，接近零浪费

效果: 显存利用率从20-40%提升到接近100%
```

## 实践部分

### 实践1：KV Cache实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time

print("=" * 60)
print("实践1: KV Cache实现与性能对比")
print("=" * 60)

class AttentionWithCache(nn.Module):
    """带KV Cache的注意力"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None, mask=None):
        B, S, _ = x.size()
        Q = self.W_Q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        # 更新KV Cache
        if kv_cache is not None:
            K = torch.cat([kv_cache['K'], K], dim=2)
            V = torch.cat([kv_cache['V'], V], dim=2)

        new_cache = {'K': K, 'V': V}

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_O(out), new_cache

# 性能对比
d_model = 512
num_heads = 8
seq_len = 512
new_tokens = 20

attn = AttentionWithCache(d_model, num_heads)
prompt = torch.randn(1, seq_len, d_model)

# 不使用KV Cache: 每步处理所有token
print("--- 不使用KV Cache ---")
start = time.time()
x = prompt
for i in range(new_tokens):
    new_token = torch.randn(1, 1, d_model)
    x = torch.cat([x, new_token], dim=1)
    out, _ = attn(x[:, -new_tokens-1:], kv_cache=None)
time_without_cache = time.time() - start
print(f"  时间: {time_without_cache*1000:.1f}ms")

# 使用KV Cache: 每步只处理新token
print("--- 使用KV Cache ---")
start = time.time()
# 先处理prompt
out, cache = attn(prompt, kv_cache=None)
x = prompt
for i in range(new_tokens):
    new_token = torch.randn(1, 1, d_model)
    x = torch.cat([x, new_token], dim=1)
    out, cache = attn(new_token, kv_cache=cache)
time_with_cache = time.time() - start
print(f"  时间: {time_with_cache*1000:.1f}ms")
print(f"  加速比: {time_without_cache/time_with_cache:.1f}x")
```

### 实践2：量化实现

```python
print("\n" + "=" * 60)
print("实践2: 权重量化")
print("=" * 60)

def quantize_weight(weight, bits=4):
    """简单的权重量化 (对称量化)"""
    # 计算缩放因子
    max_val = weight.abs().max()
    scale = max_val / (2 ** (bits - 1) - 1)

    # 量化: float → int
    weight_quant = torch.round(weight / scale).clamp(
        -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    )

    return weight_quant, scale

def dequantize_weight(weight_quant, scale):
    """反量化: int → float"""
    return weight_quant * scale

# 测试量化
weight = torch.randn(256, 256)
print(f"原始权重: shape={weight.shape}, 范围=[{weight.min():.3f}, {weight.max():.3f}]")

for bits in [8, 4, 2]:
    w_q, scale = quantize_weight(weight, bits)
    w_reconstructed = dequantize_weight(w_q, scale)

    # 计算误差
    mse = ((weight - w_reconstructed) ** 2).mean()
    original_size = weight.numel() * 4  # FP32
    quantized_size = weight.numel() * bits // 8

    print(f"\n{bits}-bit量化:")
    print(f"  MSE: {mse:.6f}")
    print(f"  压缩比: {original_size/quantized_size:.1f}x")
    print(f"  量化范围: [{w_q.min():.0f}, {w_q.max():.0f}]")
    print(f"  缩放因子: {scale:.6f}")

# 实践3: 推测解码
print("\n" + "=" * 60)
print("实践3: 推测解码 (Speculative Decoding)")
print("=" * 60)

def speculative_decode(target_model, draft_model, prompt, k=4, max_tokens=20):
    """
    推测解码

    Args:
        target_model: 大模型 (准确但慢)
        draft_model: 小模型 (快速但可能不准)
        prompt: 输入prompt
        k: 每次推测的token数
        max_tokens: 最大生成token数
    """
    generated = prompt.clone()
    total_draft = 0
    total_accepted = 0

    while generated.size(1) < prompt.size(1) + max_tokens:
        # Step 1: 小模型快速生成k个token
        draft_tokens = []
        draft_probs = []
        for _ in range(k):
            logits = draft_model(generated[:, -1:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            token = torch.multinomial(probs, 1)
            draft_tokens.append(token)
            draft_probs.append(probs)
            generated = torch.cat([generated, token], dim=1)

        total_draft += k

        # Step 2: 大模型并行验证
        target_logits = target_model(generated)
        target_probs = F.softmax(target_logits[:, -k-1:-1, :], dim=-1)

        # Step 3: 接受/拒绝
        accepted = 0
        for i in range(k):
            token = draft_tokens[i]
            p_target = target_probs[:, i, token.squeeze(-1)]
            p_draft = draft_probs[i][:, token.squeeze(-1)]

            # 接受概率
            accept_prob = torch.min(torch.ones_like(p_target), p_target / (p_draft + 1e-10))

            if torch.rand(1).item() < accept_prob.mean().item():
                accepted += 1
                total_accepted += 1
            else:
                # 拒绝: 从修正分布中重新采样
                corrected = torch.clamp(target_probs[:, i, :] - draft_probs[i], min=0)
                corrected = corrected / (corrected.sum(dim=-1, keepdim=True) + 1e-10)
                new_token = torch.multinomial(corrected, 1)
                generated = torch.cat([generated[:, :-(k-i)], new_token], dim=1)
                break

        if accepted < k:
            generated = generated[:, :-(k - accepted)]

    return generated, total_accepted, total_draft

print("推测解码流程:")
print("1. 小模型快速生成k个token草稿")
print("2. 大模型并行验证所有草稿token")
print("3. 按概率接受/拒绝")
print("4. 加速比通常2-3x")
```

## 总结

| 技术 | 加速 | 显存节省 | 实现难度 |
|------|------|---------|---------|
| KV Cache | 2-5x | - | 低 |
| FP16/BF16 | 1.5-2x | 50% | 极低 |
| INT8量化 | 1.5-2x | 75% | 中 |
| INT4量化 | 2-3x | 87.5% | 中 |
| 推测解码 | 2-3x | - | 中 |
| PagedAttention | - | 60-80% | 中 |

**下一步**: [Module 09: RAG系统](./../module-09-rag/)
