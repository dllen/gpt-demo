---
layout: page
title: "Module 01: 数学基础 — 微积分：导数、梯度与链式法则"
---
# Module 01: 数学基础 — 微积分：导数、梯度与链式法则

> 理解反向传播的数学基础

## 写给后端程序员的导读

后端开发中的"依赖追踪"和"链路传播"与微积分中的链式法则本质相同：

```
后端: 服务A → 调用服务B → 调用服务C
      A的错误率 = f(B的错误率) = g(C的错误率)

ML:   损失L → 依赖输出y → 依赖参数w
     dL/dw = dL/dy · dy/dw  (链式法则)
```

## 理论部分

### 3.1 导数 (Derivative)

#### 直觉

导数 = 函数在某点的"变化率" = 曲线的切线斜率

```
f(x) = x²
f'(x) = 2x

f'(3) = 6  →  在x=3处，x增加1，f(x)增加约6
f'(0) = 0  →  在x=0处，函数几乎不变（极值点）
```

**后端类比**：导数就像"弹性系数"——输入变化1%，输出变化多少？

#### 常见导数公式

```
f(x) = xⁿ       →  f'(x) = nxⁿ⁻¹
f(x) = eˣ       →  f'(x) = eˣ
f(x) = ln(x)    →  f'(x) = 1/x
f(x) = sin(x)   →  f'(x) = cos(x)
f(x) = σ(x)     →  f'(x) = σ(x)(1-σ(x))  (Sigmoid)
```

### 3.2 偏导数 (Partial Derivative)

当函数有多个输入时，对每个输入分别求导：

```
f(x, y) = x² + 3xy + y²

∂f/∂x = 2x + 3y    (把y当常数，对x求导)
∂f/∂y = 3x + 2y    (把x当常数，对y求导)
```

**后端类比**：微服务中，接口响应时间对CPU利用率的偏导数 = "其他条件不变时，CPU增加1%，响应时间变化多少？"

### 3.3 梯度 (Gradient)

梯度是所有偏导数组成的向量，指向函数增长最快的方向：

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**关键性质**：
- 梯度方向 = 函数增长最快的方向
- 负梯度方向 = 函数下降最快的方向
- 梯度大小 = 该方向的陡峭程度

**后端类比**：梯度就像"性能瓶颈分析"——告诉你优化哪个参数对减少延迟最有效。

### 3.4 链式法则 (Chain Rule)

链式法则是反向传播的数学基础：

```
若: h(x) = f(g(x))
则: h'(x) = f'(g(x)) · g'(x)

一般形式:
dh/dx = dh/dg · dg/dx
```

**多层链式法则**：

```
L = f(g(h(x)))
dL/dx = df/dg · dg/dh · dh/dx
```

**后端类比**：调用链 A→B→C 的错误传播：
```
A的错误 = A对B的敏感度 × B的错误
B的错误 = B对C的敏感度 × C的错误
A的错误 = A对B的敏感度 × B对C的敏感度 × C的错误
```

### 3.5 计算图 (Computational Graph)

神经网络的每一层都可以看作计算图中的一个节点：

```
输入 x → [线性: xW+b] → [激活: ReLU] → [线性: xW+b] → 输出 ŷ
         ↓                    ↓                    ↓
       中间值a1             中间值a2             中间值a3

反向传播:
∂L/∂W₁ = ∂L/∂a₃ · ∂a₃/∂a₂ · ∂a₂/∂a₁ · ∂a₁/∂W₁
```

## 实践部分

### 实践1：数值导数与解析导数

```python
import numpy as np
import torch

print("=" * 60)
print("实践1: 导数的数值计算与PyTorch自动求导")
print("=" * 60)

# 3.6 数值导数 (有限差分法)
def numerical_derivative(f, x, h=1e-5):
    """用有限差分近似导数: f'(x) ≈ (f(x+h) - f(x-h)) / 2h"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 测试: f(x) = x², f'(x) = 2x
f = lambda x: x ** 2
for x in [0.0, 1.0, 3.0, -2.0]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:5.1f}: 数值导数={numerical:.6f}, 解析导数={analytical:.6f}, 误差={abs(numerical-analytical):.2e}")

# 3.7 PyTorch自动求导 (Autograd)
print("\n--- PyTorch Autograd ---")

# 创建一个需要梯度的张量
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x.item()}, requires_grad = {x.requires_grad}")

# 前向计算: y = x²
y = x ** 2
print(f"y = x² = {y.item()}")

# 反向传播: 计算 dy/dx
y.backward()
print(f"dy/dx = {x.grad.item()}  (解析解: 2*3 = 6)")

# 3.8 多变量偏导数
print("\n--- 多变量偏导数 ---")

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# f(x,y) = x² + 3xy + y²
f = x**2 + 3*x*y + y**2
f.backward()

print(f"f(x,y) = x² + 3xy + y²")
print(f"在点 (x=2, y=3):")
print(f"  ∂f/∂x = {x.grad.item()}  (解析: 2*2 + 3*3 = 13)")
print(f"  ∂f/∂y = {y.grad.item()}  (解析: 3*2 + 2*3 = 12)")
```

### 实践2：链式法则验证

```python
print("\n" + "=" * 60)
print("实践2: 链式法则验证")
print("=" * 60)

# 3.9 简单链式法则: h(x) = f(g(x)) = (3x+1)²
# h'(x) = 2(3x+1)·3 = 6(3x+1)
x = torch.tensor(2.0, requires_grad=True)

# 前向: g(x) = 3x+1, f(g) = g²
g = 3 * x + 1
h = g ** 2

# 反向传播
h.backward()

# 数值验证
x_val = 2.0
analytical = 6 * (3 * x_val + 1)  # h'(x) = 6(3x+1)
print(f"h(x) = (3x+1)²")
print(f"h'(x) = 6(3x+1)")
print(f"h'(2) = {analytical}")
print(f"PyTorch: dh/dx = {x.grad.item()}")
print(f"匹配: {np.isclose(x.grad.item(), analytical)}")

# 3.10 多层链式法则 (模拟神经网络的一层)
print("\n--- 模拟神经网络单层的前向与反向 ---")

# 输入
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# 权重
W = torch.tensor([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]], requires_grad=True)
b = torch.tensor([0.1, 0.2], requires_grad=True)

# 前向: z = x @ W + b, a = ReLU(z)
z = x @ W + b
a = torch.relu(z)
loss = a.sum()  # 简化的损失

print(f"输入 x: {x.detach().numpy()}")
print(f"权重 W:\n{W.detach().numpy()}")
print(f"线性输出 z = xW+b: {z.detach().numpy()}")
print(f"激活输出 a = ReLU(z): {a.detach().numpy()}")
print(f"损失: {loss.item():.4f}")

# 反向传播
loss.backward()

print(f"\n梯度 dL/dW:\n{W.grad.numpy()}")
print(f"梯度 dL/db: {b.grad.numpy()}")
print(f"梯度 dL/dx: {x.grad.numpy()}")
print("→ 梯度告诉我们: 调整哪个权重对减少损失最有效")
```

### 实践3：梯度下降可视化

```python
print("\n" + "=" * 60)
print("实践3: 梯度下降 — 最直观的优化")
print("=" * 60)

# 3.11 一维梯度下降
def gradient_descent_1d():
    """用梯度下降找 f(x) = x² 的最小值"""
    x = 5.0  # 初始值
    lr = 0.1  # 学习率
    history = [x]

    for step in range(20):
        grad = 2 * x  # f'(x) = 2x
        x = x - lr * grad
        history.append(x)

    return history

history = gradient_descent_1d()
print("梯度下降找 f(x)=x² 的最小值:")
print(f"初始: x = 5.0")
for i, x in enumerate(history):
    if i % 5 == 0:
        print(f"  Step {i:2d}: x = {x:.6f}, f(x) = {x**2:.6f}")
print(f"最终: x = {history[-1]:.6f} → 接近真实最小值 x=0")

# 3.12 不同学习率的影响
print("\n--- 不同学习率的影响 ---")
for lr in [0.01, 0.1, 0.5, 1.0, 1.1]:
    x = 5.0
    for _ in range(20):
        x = x - lr * 2 * x
    status = "收敛" if abs(x) < 0.01 else ("震荡" if abs(x) > 5 else "缓慢收敛")
    print(f"  lr={lr:.2f}: x={x:.6f} ({status})")

# 3.13 二维梯度下降 (更贴近实际)
print("\n--- 二维梯度下降 ---")
print("优化 f(x,y) = x² + 2y² (椭圆抛物面)")

x, y = 3.0, 2.0
lr = 0.1
for step in range(30):
    # 梯度: ∇f = [2x, 4y]
    grad_x = 2 * x
    grad_y = 4 * y
    x -= lr * grad_x
    y -= lr * grad_y
    if step % 5 == 4:
        f_val = x**2 + 2*y**2
        print(f"  Step {step+1}: x={x:.6f}, y={y:.6f}, f={f_val:.6f}")

print(f"\n→ 收敛到 (0,0), 即全局最小值")
```

### 实践4：理解反向传播

```python
print("\n" + "=" * 60)
print("实践4: 反向传播完整示例")
print("=" * 60)

# 3.14 手动实现一个两层网络的反向传播
torch.manual_seed(42)

# 数据: 2个样本, 3维输入, 2类输出
X = torch.randn(2, 3)
y_true = torch.tensor([0, 1])

# 第一层: 3 → 4
W1 = torch.randn(3, 4, requires_grad=True)
b1 = torch.randn(4, requires_grad=True)

# 第二层: 4 → 2
W2 = torch.randn(4, 2, requires_grad=True)
b2 = torch.randn(2, requires_grad=True)

# 前向传播
z1 = X @ W1 + b1       # 线性变换
a1 = torch.relu(z1)     # 激活
z2 = a1 @ W2 + b2       # 输出层
loss = torch.nn.functional.cross_entropy(z2, y_true)

print(f"输入 X shape: {X.shape}")
print(f"第一层: {X.shape[1]} → {W1.shape[1]}")
print(f"第二层: {W2.shape[0]} → {W2.shape[1]}")
print(f"损失: {loss.item():.4f}")

# 反向传播
loss.backward()

# 查看梯度
print(f"\n梯度信息:")
print(f"  dL/dW1: shape={W1.grad.shape}, 范数={W1.grad.norm():.4f}")
print(f"  dL/db1: shape={b1.grad.shape}, 范数={b1.grad.norm():.4f}")
print(f"  dL/dW2: shape={W2.grad.shape}, 范数={W2.grad.norm():.4f}")
print(f"  dL/db2: shape={b2.grad.shape}, 范数={b2.grad.norm():.4f}")

# 梯度范数分析
print(f"\n梯度范数分析:")
print(f"  第一层权重梯度范数: {W1.grad.norm():.4f}")
print(f"  第二层权重梯度范数: {W2.grad.norm():.4f}")
ratio = W1.grad.norm() / (W2.grad.norm() + 1e-8)
if ratio > 10:
    print(f"  → 第一层梯度更大，更新更快")
elif ratio < 0.1:
    print(f"  → 第二层梯度更大（可能梯度消失）")
else:
    print(f"  → 梯度分布较均匀")

# 3.15 参数更新
lr = 0.01
with torch.no_grad():
    W1 -= lr * W1.grad
    b1 -= lr * b1.grad
    W2 -= lr * W2.grad
    b2 -= lr * b2.grad

# 验证损失下降
z1_new = X @ W1 + b1
a1_new = torch.relu(z1_new)
z2_new = a1_new @ W2 + b2
loss_new = torch.nn.functional.cross_entropy(z2_new, y_true)

print(f"\n更新后损失: {loss_new.item():.4f} (之前: {loss.item():.4f})")
print(f"损失下降: {(loss.item() - loss_new.item()):.4f}")
print("→ 梯度下降使损失减小，模型在'学习'")
```

## 总结

| 概念 | 公式 | 后端类比 | 在ML中的作用 |
|------|------|---------|-------------|
| 导数 | f'(x) | 弹性系数 | 衡量参数变化对输出的影响 |
| 偏导数 | ∂f/∂x | 控制变量法 | 多参数时单独分析每个参数 |
| 梯度 | ∇f | 瓶颈分析 | 决定参数更新方向 |
| 链式法则 | dh/dx = Σ ∂h/∂gᵢ · ∂gᵢ/∂x | 调用链追踪 | 反向传播的核心 |
| 计算图 | 节点=操作, 边=数据流 | 服务依赖图 | 自动求导的基础 |

**下一步**: [概率论与统计基础](./02-probability-optimization.md)
