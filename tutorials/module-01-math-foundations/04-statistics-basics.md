# Module 01: 数学基础 — 统计与概率论基础

> 用后端监控的思维理解统计学

## 写给后端程序员的导读

后端工程师每天看的监控指标（P99延迟、QPS、错误率）本质上就是统计量。机器学习中的统计概念是你已经熟悉的东西：

```
后端监控                    ML统计
─────────────────────────────────────────
平均值 (avg response time)  → 期望 E[X]
P99延迟                     → 分位数
错误率                      → 概率
QPS波动                     → 方差/标准差
服务间延迟相关性             → 协方差/相关系数
流量分布                    → 概率分布
异常检测                    → 假设检验
A/B测试结果对比              → 置信区间
```

## 理论部分

### 4.1 描述统计

#### 集中趋势

```
均值 (Mean):     μ = (x₁ + x₂ + ... + xₙ) / n
中位数 (Median): 排序后中间的值
众数 (Mode):    出现最频繁的值
```

**后端类比**：
- 均值 = 平均响应时间（容易被极端值拉高）
- 中位数 = 中位数响应时间（更能代表"典型"用户体验）

#### 离散程度

```
方差:  σ² = Σ(xᵢ - μ)² / n        (每个值偏离均值的平均平方距离)
标准差: σ = √σ²                     (与原始数据同单位)
```

**后端类比**：
- 低标准差 = 服务响应稳定（用户体验一致）
- 高标准差 = 服务响应不稳定（忽快忽慢）

#### 分位数

```
P50 = 中位数: 50%的请求低于此值
P95: 95%的请求低于此值
P99: 99%的请求低于此值
```

**后端类比**：P99延迟就是"最慢的1%请求有多慢"。

### 4.2 概率分布

#### 正态分布 (Gaussian Distribution)

```
f(x) = (1/√(2πσ²)) · e^{-(x-μ)²/(2σ²)}
```

**特征**：
- 钟形曲线，关于均值对称
- 68-95-99.7规则：数据落在 μ±σ/μ±2σ/μ±3σ 的概率分别为68%/95%/99.7%

**后端类比**：大多数服务的响应时间近似正态分布——大部分请求集中在平均值附近，极端快/慢的都是少数。

#### 均匀分布

```
f(x) = 1/(b-a)  对于 x ∈ [a, b]
```

**后端类比**：Round-Robin负载均衡下，请求被分配到每台服务器的概率是均匀的。

#### 伯努利分布与二项分布

```
伯努利: P(X=1)=p, P(X=0)=1-p    (单次试验)
二项:   P(X=k) = C(n,k) p^k (1-p)^{n-k}  (n次独立试验中成功k次)
```

**后端类比**：
- 伯努利 = 单次请求成功/失败
- 二项 = n次请求中有k次成功

### 4.3 协方差与相关系数

#### 协方差

```
Cov(X, Y) = E[(X-μₓ)(Y-μᵧ)]
```

- Cov > 0: X和Y同向变化
- Cov < 0: X和Y反向变化
- Cov = 0: X和Y无线性关系

#### 相关系数

```
ρ(X,Y) = Cov(X,Y) / (σₓ · σᵧ) ∈ [-1, 1]
```

**后端类比**：
- CPU利用率与响应时间的相关系数 ≈ 0.8（强正相关）
- 内存使用率与404错误率的相关系数 ≈ 0（无关）

### 4.4 大数定律与中心极限定理

**大数定律**：样本量越大，样本均值越接近期望值。

```
当 n → ∞ 时, (X₁+...+Xₙ)/n → E[X]
```

**后端类比**：监控10次请求的平均响应时间波动大，监控10000次就稳定了。

**中心极限定理**：独立随机变量之和近似正态分布（无论原始分布是什么）。

**后端类比**：不管单个请求的延迟分布长什么样，100个请求的平均延迟总是近似正态分布。

### 4.5 最大似然估计 (MLE)

给定观测数据，找到最可能产生这些数据的参数：

```
θ* = argmax_θ P(数据|θ)
```

**后端类比**：根据错误日志反推最可能的故障原因——"哪种参数设置最可能产生我们观察到的现象？"

## 实践部分

### 实践1：描述统计计算

```python
import numpy as np
import torch

print("=" * 60)
print("实践1: 描述统计 — 像看监控一样看数据")
print("=" * 60)

# 模拟API响应时间数据 (ms)
np.random.seed(42)
response_times = np.random.lognormal(mean=3.5, sigma=0.5, size=10000)
response_times = np.clip(response_times, 10, 2000)  # 限制范围

print(f"API响应时间统计 (n={len(response_times)}):")
print(f"  均值 (avg):   {np.mean(response_times):.1f} ms")
print(f"  中位数 (P50): {np.median(response_times):.1f} ms")
print(f"  P95:          {np.percentile(response_times, 95):.1f} ms")
print(f"  P99:          {np.percentile(response_times, 99):.1f} ms")
print(f"  标准差:       {np.std(response_times):.1f} ms")
print(f"  最小值:       {np.min(response_times):.1f} ms")
print(f"  最大值:       {np.max(response_times):.1f} ms")

# 均值 vs 中位数
print(f"\n  均值({np.mean(response_times):.1f}) > 中位数({np.median(response_times):.1f})")
print(f"  → 右偏分布: 少量慢请求拉高了平均值")
print(f"  → 中位数更能代表'典型'用户体验")

# 异常检测 (3σ原则)
mean = np.mean(response_times)
std = np.std(response_times)
threshold = mean + 3 * std
anomalies = response_times[response_times > threshold]
print(f"\n  异常检测 (μ+3σ = {threshold:.1f}ms):")
print(f"  异常请求数: {len(anomalies)} ({len(anomalies)/len(response_times)*100:.2f}%)")
```

### 实践2：概率分布

```python
print("\n" + "=" * 60)
print("实践2: 概率分布")
print("=" * 60)

# 4.6 正态分布
print("\n--- 正态分布 ---")
mu, sigma = 100, 15  # 均值100，标准差15
samples = np.random.normal(mu, sigma, 10000)

print(f"正态分布 N(μ={mu}, σ={sigma}):")
print(f"  理论: 68% 在 [{mu-sigma}, {mu+sigma}]")
print(f"  实际: {np.mean((samples >= mu-sigma) & (samples <= mu+sigma))*100:.1f}%")
print(f"  理论: 95% 在 [{mu-2*sigma}, {mu+2*sigma}]")
print(f"  实际: {np.mean((samples >= mu-2*sigma) & (samples <= mu+2*sigma))*100:.1f}%")

# 4.7 二项分布 — 模拟请求成功率
print("\n--- 二项分布: 请求成功率 ---")
n_requests = 100
p_success = 0.95
n_trials = 1000

successes = np.random.binomial(n_requests, p_success, n_trials)
print(f"每次{n_requests}个请求, 成功率{p_success*100}%:")
print(f"  平均成功数: {successes.mean():.1f}")
print(f"  标准差: {successes.std():.1f}")
print(f"  理论标准差: {np.sqrt(n_requests * p_success * (1-p_success)):.1f}")
print(f"  全部成功的概率: {p_success}^{n_requests} = {p_success**n_requests:.6f}")
print(f"  → 即使95%成功率, 100个请求全部成功的概率也很低!")

# 4.8 泊松分布 — 模拟请求到达
print("\n--- 泊松分布: 请求到达率 ---")
lambda_rate = 5  # 平均每秒5个请求
arrivals = np.random.poisson(lambda_rate, 10000)

print(f"请求到达 (λ={lambda_rate}/s):")
print(f"  平均到达率: {arrivals.mean():.2f}/s")
print(f"  方差: {arrivals.var():.2f} (泊松分布中均值=方差)")
print(f"  P(0请求): {np.mean(arrivals == 0)*100:.2f}%")
print(f"  P(>10请求): {np.mean(arrivals > 10)*100:.2f}%")
```

### 实践3：协方差与相关性

```python
print("\n" + "=" * 60)
print("实践3: 协方差与相关性分析")
print("=" * 60)

# 4.9 模拟服务监控指标
np.random.seed(42)
n = 1000

# 生成相关的监控指标
cpu_usage = np.random.uniform(20, 90, n)  # CPU利用率 20-90%
memory_usage = cpu_usage * 0.6 + np.random.normal(0, 5, n)  # 内存与CPU相关
response_time = cpu_usage * 2 + np.random.normal(0, 10, n)  # 响应时间与CPU相关
error_rate = np.random.uniform(0, 5, n)  # 错误率与CPU无关

# 计算相关系数
def correlation(x, y):
    """计算皮尔逊相关系数"""
    x_centered = x - x.mean()
    y_centered = y.mean()
    return np.sum(x_centered * (y - y.mean())) / (np.sqrt(np.sum(x_centered**2)) * np.sqrt(np.sum((y - y.mean())**2)) + 1e-10)

pairs = [
    ("CPU利用率", cpu_usage, "响应时间", response_time),
    ("CPU利用率", cpu_usage, "内存使用率", memory_usage),
    ("CPU利用率", cpu_usage, "错误率", error_rate),
]

for name1, x, name2, y in pairs:
    r = correlation(x, y)
    strength = "强正相关" if r > 0.7 else ("中等正相关" if r > 0.3 else ("弱相关" if r > 0.1 else "几乎无关"))
    print(f"  {name1} vs {name2}: r={r:.4f} ({strength})")

print("\n→ CPU与响应时间强相关: 高CPU导致慢响应")
print("→ CPU与内存中等相关: 有一定关联")
print("→ CPU与错误率无关: 错误率是独立因素")
```

### 实践4：大数定律验证

```python
print("\n" + "=" * 60)
print("实践4: 大数定律验证")
print("=" * 60)

# 4.10 模拟掷骰子
np.random.seed(42)
true_mean = 3.5  # 公平骰子的期望

sample_sizes = [10, 100, 1000, 10000, 100000]
print("掷骰子 — 样本均值收敛到期望值 (3.5):")
for n in sample_sizes:
    samples = np.random.randint(1, 7, size=n)
    sample_mean = samples.mean()
    error = abs(sample_mean - true_mean)
    print(f"  n={n:6d}: 均值={sample_mean:.4f}, 误差={error:.4f}")

print("\n→ 样本量越大，样本均值越接近理论期望值")

# 4.11 中心极限定理验证
print("\n--- 中心极限定理 ---")
print("原始分布: 指数分布 (严重右偏)")
print("样本均值的分布: 近似正态分布")

lambda_exp = 0.5
sample_means = []
for _ in range(10000):
    samples = np.random.exponential(1/lambda_exp, size=100)
    sample_means.append(samples.mean())

sample_means = np.array(sample_means)
print(f"  原始分布均值: {1/lambda_exp:.2f}")
print(f"  样本均值均值: {sample_means.mean():.2f}")
print(f"  样本均值标准差: {sample_means.std():.4f}")
print(f"  理论标准差: {(1/lambda_exp)/np.sqrt(100):.4f}")
print(f"  偏度: {((sample_means - sample_means.mean())**3).mean() / sample_means.std()**3:.4f} (接近0=对称)")
```

### 实践5：PyTorch中的统计操作

```python
print("\n" + "=" * 60)
print("实践5: PyTorch中的统计操作")
print("=" * 60)

# 4.12 张量统计
x = torch.randn(100, 10)  # 100个样本, 10个特征
print(f"数据 shape: {x.shape}")
print(f"每个特征的均值: {x.mean(dim=0)[:5].numpy().round(3)} ...")
print(f"每个特征的标准差: {x.std(dim=0)[:5].numpy().round(3)} ...")

# 4.13 批归一化 (BatchNorm) — 后端视角
# 就像API网关对请求做标准化处理
batch_norm = torch.nn.BatchNorm10(num_features=10)
normalized = batch_norm(x)
print(f"\n批归一化后:")
print(f"  均值 ≈ 0: {normalized.mean(dim=0)[:5].numpy().round(6)}")
print(f"  标准差 ≈ 1: {normalized.std(dim=0)[:5].numpy().round(3)}")

# 4.14 层归一化 (LayerNorm) — 每个样本独立归一化
layer_norm = torch.nn.LayerNorm(normalized_shape=10)
output = layer_norm(x)
print(f"\n层归一化后 (每行独立):")
print(f"  每行均值 ≈ 0: {output.mean(dim=1)[:5].numpy().round(6)}")
print(f"  每行标准差 ≈ 1: {output.std(dim=1)[:5].numpy().round(3)}")

# 4.15 采样与分布
print("\n--- 从分布中采样 ---")
# 正态分布
normal_samples = torch.distributions.Normal(0, 1).sample((1000,))
print(f"正态分布 N(0,1): 均值={normal_samples.mean():.4f}, 标准差={normal_samples.std():.4f}")

# 分类分布 (Categorical) — 模型输出
probs = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.25])
cat_dist = torch.distributions.Categorical(probs)
samples = cat_dist.sample((1000,))
print(f"\n分类分布 (概率={probs.numpy()}):")
for i in range(5):
    print(f"  类别{i}: 理论={probs[i]:.2f}, 实际={torch.sum(samples == i).item()/1000:.3f}")
```

## 总结

| 统计概念 | 后端类比 | ML应用 |
|---------|---------|--------|
| 均值/中位数 | 平均/P50响应时间 | 数据标准化 |
| 方差/标准差 | 响应时间波动 | 权重初始化 |
| 正态分布 | 响应时间分布 | 假设检验 |
| 二项分布 | 请求成功/失败 | 分类问题 |
| 泊松分布 | 请求到达率 | 事件建模 |
| 相关系数 | 指标间关联 | 特征选择 |
| 大数定律 | 监控数据收敛 | 训练稳定性 |
| 中心极限定理 | 聚合指标正态性 | 置信区间 |

**下一步**: [数学速查手册](./05-math-cheat-sheet.md)
