# Module 01: 数学基础与预备知识 — 概率论与优化

## 理论部分

### 2.1 概率论基础

#### 概率分布

概率分布描述了一个随机变量取各个值的可能性。在LLM中，模型输出的是词表上的概率分布。

```
离散分布: P(X = xᵢ) = pᵢ,  Σpᵢ = 1
连续分布: ∫f(x)dx = 1
```

#### 条件概率与链式法则

条件概率 P(Y|X) 表示在X发生的条件下Y发生的概率。

```
P(X, Y) = P(Y|X) · P(X) = P(X|Y) · P(Y)
```

语言模型的核心就是条件概率：给定前文，预测下一个词。

```
P(w₁, w₂, ..., wₙ) = ∏ᵢ P(wᵢ | w₁, ..., wᵢ₋₁)
```

这就是自回归生成的数学基础。

#### 最大似然估计 (MLE)

给定观测数据，找到最可能产生这些数据的参数：

```
θ* = argmax_θ P(D|θ) = argmax_θ Σ log P(xᵢ|θ)
```

在语言模型训练中，我们最大化训练数据的对数似然，等价于最小化交叉熵损失。

### 2.2 信息论基础

#### 熵 (Entropy)

熵衡量随机变量的不确定性：

```
H(X) = -Σ P(x) log P(x)
```

- 确定性事件: H = 0
- 均匀分布: H = log(n)（最大熵）

#### 交叉熵 (Cross-Entropy)

衡量两个概率分布之间的差异：

```
H(P, Q) = -Σ P(x) log Q(x)
```

在LLM训练中：
- P 是真实分布（one-hot向量，真实词位置为1）
- Q 是模型预测的分布（softmax输出）
- 交叉熵损失 = -log P_model(真实词|上下文)

#### KL散度 (Kullback-Leibler Divergence)

衡量两个分布的"距离"：

```
D_KL(P||Q) = Σ P(x) log(P(x)/Q(x)) = H(P,Q) - H(P)
```

在RLHF中，KL散度用于限制策略模型不要偏离参考模型太远。

#### 困惑度 (Perplexity)

困惑度是语言模型的核心评价指标：

```
PPL = exp(H(P, Q)) = exp(-1/N Σ log P(wᵢ|context))
```

**直觉**：困惑度表示模型在预测下一个词时的"平均选择数"。
- PPL = 1：完美预测
- PPL = V（词表大小）：完全随机猜测
- GPT-4在常见基准上PPL约为 10-20

### 2.3 优化基础

#### 梯度下降

梯度下降是训练神经网络的基本方法：

```
θ_{t+1} = θ_t - η · ∇L(θ_t)
```

其中 η 是学习率，∇L 是损失函数的梯度。

#### 反向传播 (Backpropagation)

反向传播利用链式法则计算损失对每个参数的梯度：

```
∂L/∂W = ∂L/∂y · ∂y/∂W
```

在Transformer中，梯度从输出层一路传播回嵌入层，经过注意力、FFN、归一化等所有子层。

#### 学习率调度

```
常数:     η(t) = η₀
线性衰减:  η(t) = η₀(1 - t/T)
余弦退火:  η(t) = η_min + (η₀ - η_min)(1 + cos(πt/T))/2
Warmup:   η(t) = η₀ · min(t/T_warmup, 1)
```

现代LLM通常使用 **Warmup + Cosine Decay** 策略。

#### Adam优化器

Adam结合了动量和自适应学习率：

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t        (一阶矩/动量)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²       (二阶矩/自适应)
m̂_t = m_t / (1-β₁^t)                  (偏差校正)
v̂_t = v_t / (1-β₂^t)
θ_{t+1} = θ_t - η·m̂_t / (√v̂_t + ε)
```

AdamW在Adam基础上加入了权重衰减（L2正则化），是LLM训练的标准选择。

## 实践部分

### 实践1：概率分布与采样

```python
import numpy as np
import torch
import torch.nn.functional as F

print("=" * 60)
print("实践1: 概率分布与采样")
print("=" * 60)

# 2.4 模拟模型输出的logits
logits = np.array([2.0, 1.0, 0.5, -0.5, -1.0])
print(f"模型输出 logits: {logits}")

# Softmax转换为概率
probs = np.exp(logits) / np.sum(np.exp(logits))
print(f"Softmax概率: {np.round(probs, 4)}")
print(f"概率之和: {probs.sum():.6f}")

# 2.5 温度缩放 (Temperature Scaling)
def softmax_with_temperature(logits, temperature=1.0):
    """温度越高分布越平坦，越低越尖锐"""
    scaled_logits = logits / temperature
    exp_x = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_x / np.sum(exp_x)

for temp in [0.5, 1.0, 2.0]:
    probs_t = softmax_with_temperature(logits, temp)
    print(f"\nTemperature={temp}: {np.round(probs_t, 4)}")
    print(f"  最大概率: {probs_t.max():.4f}, 熵: {-np.sum(probs_t * np.log(probs_t + 1e-10)):.4f}")

# 2.6 Top-k 采样
def top_k_sampling(logits, k=3):
    """只从概率最高的k个词中采样"""
    top_k_indices = np.argsort(logits)[-k:]
    top_k_probs = softmax_with_temperature(logits[top_k_indices], 1.0)
    chosen = np.random.choice(top_k_indices, p=top_k_probs)
    return chosen

print(f"\nTop-3采样 (10次):")
for _ in range(10):
    idx = top_k_sampling(logits, k=3)
    print(f"  选中索引: {idx}, logit: {logits[idx]:.2f}")

# 2.7 Top-p (Nucleus) 采样
def top_p_sampling(logits, p=0.9):
    """从累积概率超过p的最小词集合中采样"""
    sorted_indices = np.argsort(logits)[::-1]
    sorted_probs = softmax_with_temperature(logits[sorted_indices], 1.0)
    cumsum = np.cumsum(sorted_probs)
    # 找到累积概率超过p的位置
    cutoff = np.searchsorted(cumsum, p) + 1
    valid_indices = sorted_indices[:cutoff]
    valid_probs = softmax_with_temperature(logits[valid_indices], 1.0)
    chosen = np.random.choice(valid_indices, p=valid_probs)
    return chosen

print(f"\nTop-p (p=0.9) 采样 (10次):")
for _ in range(10):
    idx = top_p_sampling(logits, p=0.9)
    print(f"  选中索引: {idx}, logit: {logits[idx]:.2f}")
```

### 实践2：信息论计算

```python
print("\n" + "=" * 60)
print("实践2: 信息论计算")
print("=" * 60)

# 2.8 计算熵
def entropy(probs):
    """计算分布的熵"""
    return -np.sum(probs * np.log2(probs + 1e-10))

# 不同分布的熵
uniform = np.ones(10) / 10
peaky = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02, 0, 0, 0, 0])
deterministic = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

print(f"均匀分布熵: {entropy(uniform):.4f} bits (最大)")
print(f"尖锐分布熵: {entropy(peaky):.4f} bits")
print(f"确定性分布熵: {entropy(deterministic):.4f} bits (最小)")

# 2.9 交叉熵与KL散度
def cross_entropy(p, q):
    """计算交叉熵 H(p, q)"""
    return -np.sum(p * np.log2(q + 1e-10))

def kl_divergence(p, q):
    """计算KL散度 D_KL(p || q)"""
    return np.sum(p * np.log2((p + 1e-10) / (q + 1e-10)))

# 真实分布 (one-hot)
p = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
# 模型预测分布
q_good = np.array([0.01, 0.01, 0.9, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
q_bad = np.array([0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])

print(f"\n好模型的交叉熵: {cross_entropy(p, q_good):.4f} bits")
print(f"差模型的交叉熵: {cross_entropy(p, q_bad):.4f} bits")
print(f"\n好模型的KL散度: {kl_divergence(p, q_good):.4f}")
print(f"差模型的KL散度: {kl_divergence(p, q_bad):.4f}")

# 2.10 困惑度计算
def perplexity(log_probs):
    """从对数概率计算困惑度"""
    return np.exp(-np.mean(log_probs))

# 模拟一个句子的对数概率
good_log_probs = np.log([0.8, 0.7, 0.9, 0.6, 0.85])
bad_log_probs = np.log([0.2, 0.3, 0.15, 0.25, 0.1])

print(f"\n好模型困惑度: {perplexity(good_log_probs):.2f}")
print(f"差模型困惑度: {perplexity(bad_log_probs):.2f}")
print("→ 困惑度越低，模型越好")
```

### 实践3：优化器实现

```python
print("\n" + "=" * 60)
print("实践3: 从零实现优化器")
print("=" * 60)

# 2.11 手动实现SGD + Momentum
class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in self.params]

    def step(self, gradients):
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            param += self.velocities[i]

    def zero_grad(self):
        self.velocities = [np.zeros_like(p) for p in self.params]

# 2.12 手动实现AdamW
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p) for p in self.params]  # 一阶矩
        self.v = [np.zeros_like(p) for p in self.params]  # 二阶矩

    def step(self, gradients):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            # 权重衰减
            param *= (1 - self.lr * self.weight_decay)

            # 更新矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            # 偏差校正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 更新参数
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# 2.13 测试优化器
print("\n--- 优化 f(x,y) = x² + 2y² ---")
print("最小值在 (0, 0)")

# 初始参数
params = [np.array([3.0]), np.array([2.0])]  # 起点 (3, 2)
optimizer = AdamW(params, lr=0.1)

print(f"初始: x={params[0][0]:.4f}, y={params[1][0]:.4f}")

for step in range(50):
    # 梯度: df/dx = 2x, df/dy = 4y
    grads = [2 * params[0], 4 * params[1]]
    optimizer.step(grads)
    if step % 10 == 9:
        print(f"Step {step+1}: x={params[0][0]:.6f}, y={params[1][0]:.6f}")

# 2.14 余弦退火学习率
print("\n--- 余弦退火学习率调度 ---")
import matplotlib
matplotlib.use('Agg')

epochs = 100
lr_max = 3e-4
lr_min = 3e-5

lrs = []
for epoch in range(epochs):
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / epochs))
    lrs.append(lr)

print(f"初始学习率: {lrs[0]:.6f}")
print(f"中间学习率: {lrs[50]:.6f}")
print(f"最终学习率: {lrs[-1]:.6f}")

# 2.15 Warmup + Cosine Decay
def get_lr(step, warmup_steps, total_steps, lr_max, lr_min=1e-6):
    """Warmup + Cosine Decay 学习率调度"""
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))

warmup_steps = 200
total_steps = 10000
lr_max = 3e-4

print(f"\nWarmup阶段 (step 0): lr = {get_lr(0, warmup_steps, total_steps, lr_max):.6e}")
print(f"Warmup阶段 (step 100): lr = {get_lr(100, warmup_steps, total_steps, lr_max):.6e}")
print(f"Warmup结束 (step 200): lr = {get_lr(200, warmup_steps, total_steps, lr_max):.6e}")
print(f"训练中期 (step 5000): lr = {get_lr(5000, warmup_steps, total_steps, lr_max):.6e}")
print(f"训练结束 (step 9999): lr = {get_lr(9999, warmup_steps, total_steps, lr_max):.6e}")
```

## 总结

| 概念 | 在LLM中的应用 |
|------|-------------|
| 条件概率 | 自回归语言建模 P(wₜ\|w<t) |
| 最大似然估计 | 训练目标 = 最大化对数似然 |
| 交叉熵 | 训练损失函数 |
| 困惑度 | 模型质量评估指标 |
| KL散度 | RLHF中的策略约束 |
| 温度/Top-k/Top-p | 文本生成采样策略 |
| AdamW | 标准优化器 |
| Warmup+Cosine | 标准学习率调度 |

**下一步**: [Module 02: Transformer架构深入](./../module-02-transformer/)
