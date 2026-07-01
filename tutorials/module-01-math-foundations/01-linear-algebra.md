# Module 01: 数学基础与预备知识 — 线性代数

## 理论部分

### 1.1 向量与矩阵

大模型的一切运算都建立在向量与矩阵之上。理解线性代数是深入LLM的第一步。

#### 向量 (Vector)

向量是有序的数字列表，可以表示一个点在多维空间中的位置：

```
v = [v₁, v₂, v₃, ..., vₙ] ∈ ℝⁿ
```

在LLM中，一个词可以被表示为一个高维向量（嵌入向量），维度通常是 256、768、4096 等。

#### 矩阵 (Matrix)

矩阵是二维数字数组，可以看作多个向量的组合：

```
M = ┌ m₁₁  m₁₂  ...  m₁ₙ ┐
    │ m₂₁  m₂₂  ...  m₂ₙ │
    │  ⋮    ⋮   ⋱    ⋮   │
    └ mₘ₁  mₘ₂  ...  mₘₙ ┘  ∈ ℝᵐˣⁿ
```

在LLM中，权重矩阵、注意力分数、批处理数据都以矩阵形式存储。

### 1.2 矩阵乘法

矩阵乘法是Transformer中最核心的操作。给定 `A ∈ ℝᵐˣᵏ` 和 `B ∈ ℝᵏˣⁿ`：

```
C = AB, 其中 cᵢⱼ = Σₖ aᵢₖ · bₖⱼ
```

**直觉理解**：矩阵乘法是"行×列"的点积，结果矩阵的每个元素是A的行向量与B的列向量的相似度。

在Transformer中，注意力计算 `QKᵀ` 本质上就是计算查询向量与键向量的相似度矩阵。

### 1.3 点积与相似度

两个向量的点积（内积）：

```
a · b = Σᵢ aᵢbᵢ = ||a|| ||b|| cos(θ)
```

**几何意义**：点积衡量两个向量的"方向一致性"。
- 点积 > 0：方向相近（锐角）
- 点积 = 0：正交（90°）
- 点积 < 0：方向相反（钝角）

在注意力机制中，Q和K的点积直接决定了注意力权重的分配。

### 1.4 特征值与特征向量

对于方阵A，如果存在非零向量v和标量λ，使得：

```
Av = λv
```

则λ称为特征值，v称为特征向量。

**直觉**：特征向量是在A变换下方向不变的向量，特征值表示在该方向上的"拉伸倍数"。

在LLM中，特征值分析用于理解优化动态、梯度爆炸/消失问题。

### 1.5 范数

范数衡量向量的"大小"：

```
L1范数: ||v||₁ = Σ|vᵢ|
L2范数: ||v||₂ = √(Σvᵢ²)
∞范数: ||v||∞ = max|vᵢ|
```

在LLM中：
- L2归一化用于嵌入向量比较
- 梯度裁剪使用L2范数防止梯度爆炸
- LayerNorm内部使用L2范数

### 1.6 Softmax函数

Softmax将任意实数向量转换为概率分布：

```
softmax(zᵢ) = e^{zᵢ} / Σⱼ e^{zⱼ}
```

**性质**：
- 输出值域 (0, 1)
- 所有输出之和为 1
- 保持输入的相对顺序

在LLM中，Softmax用于：
1. 注意力权重的归一化（attention scores → attention weights）
2. 输出层生成词表上的概率分布

### 1.7 矩阵的迹与行列式

**迹 (Trace)**：方阵对角线元素之和
```
tr(A) = Σᵢ aᵢᵢ
```

**行列式 (Determinant)**：衡量矩阵所表示的线性变换的"体积缩放因子"
```
det(A) = 0  →  矩阵奇异（不可逆）
det(A) > 0  →  保持方向
det(A) < 0  →  翻转方向
```

## 实践部分

### 实践1：NumPy实现基本运算

```python
import numpy as np

# 1.8 向量创建与基本运算
print("=" * 50)
print("1.8 向量创建与基本运算")
print("=" * 50)

# 创建向量（模拟词嵌入）
word_embedding = np.array([0.2, -0.5, 0.8, 0.1, -0.3])
print(f"词嵌入向量: {word_embedding}")
print(f"维度: {word_embedding.shape}")

# 向量加减
v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([4.0, 5.0, 6.0])
print(f"\nv1 + v2 = {v1 + v2}")
print(f"v1 - v2 = {v1 - v2}")

# 标量乘法
print(f"2 * v1 = {2 * v1}")

# 1.9 点积与相似度
print("\n" + "=" * 50)
print("1.9 点积与相似度计算")
print("=" * 50)

def dot_product(a, b):
    """计算两个向量的点积"""
    return np.sum(a * b)

def cosine_similarity(a, b):
    """计算余弦相似度"""
    return dot_product(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# 模拟三个词的嵌入
cat = np.array([0.8, 0.6, 0.1, 0.2])
dog = np.array([0.7, 0.5, 0.2, 0.3])
car = np.array([0.1, 0.2, 0.9, 0.8])

print(f"cat · dog (余弦相似度): {cosine_similarity(cat, dog):.4f}")
print(f"cat · car (余弦相似度): {cosine_similarity(cat, car):.4f}")
print(f"dog · car (余弦相似度): {cosine_similarity(dog, car):.4f}")
print("→ cat和dog更相似，符合直觉")

# 1.10 矩阵乘法
print("\n" + "=" * 50)
print("1.10 矩阵乘法")
print("=" * 50)

# 模拟注意力计算中的 Q * K^T
Q = np.random.randn(3, 4)  # 3个查询，每个4维
K = np.random.randn(3, 4)  # 3个键，每个4维

# 注意力分数 = Q * K^T / sqrt(d_k)
d_k = K.shape[-1]
attention_scores = Q @ K.T / np.sqrt(d_k)
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"Attention scores shape: {attention_scores.shape}")
print(f"Attention scores:\n{attention_scores}")

# 1.11 Softmax实现
print("\n" + "=" * 50)
print("1.11 Softmax实现")
print("=" * 50)

def softmax(x, axis=-1):
    """数值稳定的Softmax实现"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# 对注意力分数应用softmax
attention_weights = softmax(attention_scores, axis=-1)
print(f"Attention weights (每行和为1):\n{attention_weights}")
print(f"每行之和: {attention_weights.sum(axis=-1)}")

# 1.12 范数计算
print("\n" + "=" * 50)
print("1.12 范数计算")
print("=" * 50)

v = np.array([3.0, -4.0, 0.0, 1.0])
print(f"向量 v: {v}")
print(f"L1范数: {np.linalg.norm(v, ord=1):.4f}")
print(f"L2范数: {np.linalg.norm(v, ord=2):.4f}")
print(f"∞范数: {np.linalg.norm(v, ord=np.inf):.4f}")

# 1.13 特征值分解
print("\n" + "=" * 50)
print("1.13 特征值分解")
print("=" * 50)

# 创建一个对称矩阵（如协方差矩阵）
A = np.array([[4, 2], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eigh(A)
print(f"矩阵 A:\n{A}")
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 验证: A @ v = λ * v
for i in range(len(eigenvalues)):
    lhs = A @ eigenvectors[:, i]
    rhs = eigenvalues[i] * eigenvectors[:, i]
    print(f"\n特征值 {eigenvalues[i]:.4f}:")
    print(f"  A @ v = {lhs}")
    print(f"  λ * v = {rhs}")
    print(f"  验证: {np.allclose(lhs, rhs)}")
```

### 实践2：PyTorch张量操作

```python
import torch
import torch.nn.functional as F

print("=" * 50)
print("PyTorch 张量操作基础")
print("=" * 50)

# 1.14 张量创建
print("\n--- 张量创建 ---")
# 从列表创建
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"从列表创建:\n{x}")

# 随机张量（模拟嵌入层输出）
torch.manual_seed(42)
embeddings = torch.randn(2, 3, 4)  # (batch_size, seq_len, d_model)
print(f"\n随机嵌入 (batch=2, seq=3, dim=4):\n{embeddings}")

# 1.15 矩阵乘法
print("\n--- 矩阵乘法 ---")
Q = torch.randn(2, 3, 4)  # (batch, seq_len, d_k)
K = torch.randn(2, 3, 4)  # (batch, seq_len, d_k)

# 批量矩阵乘法
scores = torch.bmm(Q, K.transpose(1, 2)) / (4 ** 0.5)
print(f"Q shape: {Q.shape}")
print(f"K^T shape: {K.transpose(1,2).shape}")
print(f"Scores shape: {scores.shape}")

# 使用einsum（更灵活）
scores_einsum = torch.einsum('bqd,bkd->bqk', Q, K) / (4 ** 0.5)
print(f"Einsum结果一致: {torch.allclose(scores, scores_einsum)}")

# 1.16 Softmax与LogSoftmax
print("\n--- Softmax与交叉熵 ---")
logits = torch.randn(2, 5)  # (batch_size, vocab_size)
probs = F.softmax(logits, dim=-1)
log_probs = F.log_softmax(logits, dim=-1)

print(f"Logits:\n{logits}")
print(f"\nProbabilities (和为1):\n{probs}")
print(f"每行之和: {probs.sum(dim=-1)}")

# 交叉熵损失
targets = torch.tensor([1, 3])  # 真实标签
loss = F.cross_entropy(logits, targets)
print(f"\n交叉熵损失: {loss.item():.4f}")

# 1.17 范数与归一化
print("\n--- 范数与归一化 ---")
x = torch.randn(2, 4)
print(f"原始张量:\n{x}")
print(f"L2范数 (每行): {x.norm(dim=-1)}")

# L2归一化
x_normalized = F.normalize(x, p=2, dim=-1)
print(f"\nL2归一化后:\n{x_normalized}")
print(f"归一化后L2范数: {x_normalized.norm(dim=-1)}")

# 1.18 张量变形操作
print("\n--- 张量变形 ---")
x = torch.arange(24)
print(f"原始: {x.shape} → {x}")

# reshape
x_2d = x.reshape(4, 6)
print(f"\nreshape(4,6): {x_2d.shape}")
print(x_2d)

# view (要求内存连续)
x_3d = x.reshape(2, 3, 4)
print(f"\nreshape(2,3,4): {x_3d.shape}")

# transpose
x_t = x_3d.transpose(1, 2)
print(f"\ntranspose(1,2): {x_t.shape}")

# permute
x_p = x_3d.permute(0, 2, 1)
print(f"permute(0,2,1): {x_p.shape}")

# 1.19 广播机制
print("\n--- 广播机制 ---")
a = torch.randn(2, 3, 4)
b = torch.randn(4)       # 会自动广播到 (2, 3, 4)
c = torch.randn(1, 3, 1) # 会自动广播到 (2, 3, 4)

print(f"a shape: {a.shape}")
print(f"b shape: {b.shape} → 广播后: {(a + b).shape}")
print(f"c shape: {c.shape} → 广播后: {(a + c).shape}")
```

### 实践3：理解注意力中的线性代数

```python
import torch
import torch.nn.functional as F
import numpy as np

print("=" * 60)
print("实践3: 注意力机制中的线性代数")
print("=" * 60)

# 模拟一个简单的注意力计算
torch.manual_seed(42)

batch_size = 1
seq_len = 4
d_model = 8

# 输入: "我 爱 深度 学习" 四个token
X = torch.randn(batch_size, seq_len, d_model)
print(f"输入 X shape: {X.shape} (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")

# 权重矩阵
W_Q = torch.randn(d_model, d_model)
W_K = torch.randn(d_model, d_model)
W_V = torch.randn(d_model, d_model)
W_O = torch.randn(d_model, d_model)

# 计算 Q, K, V
Q = X @ W_Q  # (batch, seq, d_model)
K = X @ W_K
V = X @ W_V

print(f"\nQ shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# 注意力分数: Q @ K^T / sqrt(d_k)
d_k = d_model
scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
print(f"\n注意力分数 shape: {scores.shape}")
print(f"注意力分数矩阵:\n{scores[0].detach().numpy().round(3)}")

# 上三角掩码（因果掩码）
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores_masked = scores.masked_fill(mask, float('-inf'))
print(f"\n掩码后的注意力分数:\n{scores_masked[0].detach().numpy().round(3)}")

# Softmax归一化
attn_weights = F.softmax(scores_masked, dim=-1)
print(f"\n注意力权重 (每行和为1):\n{attn_weights[0].detach().numpy().round(3)}")
print(f"每行之和: {attn_weights[0].sum(dim=-1).detach().numpy()}")

# 加权求和: Attention = softmax(QK^T/√d_k) @ V
output = attn_weights @ V
print(f"\n输出 shape: {output.shape}")

# 最终线性变换
final_output = output @ W_O
print(f"最终输出 shape: {final_output.shape}")

# 可视化注意力权重
print("\n--- 注意力权重可视化 ---")
tokens = ["我", "爱", "深度", "学习"]
weights = attn_weights[0].detach().numpy()

print("      ", "  ".join(f"{t:>6s}" for t in tokens))
for i, token in enumerate(tokens):
    row = "  ".join(f"{weights[i][j]:6.3f}" for j in range(seq_len))
    print(f"{token:>6s} [{row}]")
```

## 总结

本模块涵盖了大模型所需的线性代数核心知识：

| 概念 | 在LLM中的应用 |
|------|-------------|
| 向量/矩阵 | 词嵌入、权重矩阵、批处理数据 |
| 矩阵乘法 | Q@K^T、线性层、注意力计算 |
| 点积 | 注意力分数、相似度计算 |
| Softmax | 注意力权重、输出概率分布 |
| 范数 | 归一化、梯度裁剪 |
| 特征值 | 优化分析、梯度问题诊断 |

**下一步**: [Module 02: Transformer架构深入](./../module-02-transformer/)
