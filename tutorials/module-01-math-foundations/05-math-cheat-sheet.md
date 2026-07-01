# Module 01: 数学基础 — 速查手册

> 一页纸的ML数学参考

## 核心公式速查

### 线性代数

```
向量点积:     a·b = Σ aᵢbᵢ = ||a|| ||b|| cos(θ)
矩阵乘法:     C = AB, cᵢⱼ = Σₖ aᵢₖbₖⱼ
转置:         (AB)ᵀ = BᵀAᵀ
逆矩阵:       AA⁻¹ = I
行列式:       det(AB) = det(A)det(B)
特征分解:     A = QΛQ⁻¹
SVD:          A = UΣVᵀ
```

### 微积分

```
基本导数:
  d/dx xⁿ = nxⁿ⁻¹
  d/dx eˣ = eˣ
  d/dx ln(x) = 1/x
  d/dx σ(x) = σ(x)(1-σ(x))
  d/dx tanh(x) = 1 - tanh²(x)
  d/dx ReLU(x) = 1 if x>0 else 0

链式法则:
  dh/dx = dh/dg · dg/dx

常见梯度:
  ∇_W (xW) = xᵀ
  ∇_x (xW) = Wᵀ
  ∇_W ||XW-Y||² = 2Xᵀ(XW-Y)
```

### 概率论

```
条件概率:    P(A|B) = P(A∩B) / P(B)
全概率:      P(A) = Σ P(A|Bᵢ)P(Bᵢ)
贝叶斯:      P(A|B) = P(B|A)P(A) / P(B)
期望:        E[X] = Σ xᵢP(xᵢ)
方差:        Var(X) = E[(X-μ)²] = E[X²] - (E[X])²
协方差:      Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]
```

### 信息论

```
熵:          H(X) = -Σ p(x)log p(x)
交叉熵:      H(p,q) = -Σ p(x)log q(x)
KL散度:      D_KL(p||q) = Σ p(x)log(p(x)/q(x))
困惑度:      PPL = exp(H(p,q))
```

### 优化

```
梯度下降:    θ = θ - η∇L(θ)
Adam:        m = β₁m + (1-β₁)g
             v = β₂v + (1-β₂)g²
             θ = θ - η·m̂/(√v̂+ε)
Softmax:     σ(zᵢ) = e^{zᵢ}/Σe^{zⱼ}
```

## 常用分布

| 分布 | PMF/PDF | 均值 | 方差 | 场景 |
|------|---------|------|------|------|
| 伯努利 | pˣ(1-p)¹⁻ˣ | p | p(1-p) | 二分类 |
| 二项 | C(n,k)pᵏ(1-p)ⁿ⁻ᵏ | np | np(1-p) | n次独立试验 |
| 泊松 | λᵏe⁻λ/k! | λ | λ | 到达率 |
| 均匀 | 1/(b-a) | (a+b)/2 | (b-a)²/12 | 随机初始化 |
| 正态 | (1/√2πσ²)e^{-(x-μ)²/2σ²} | μ | σ² | 最常见 |
| 指数 | λe^{-λx} | 1/λ | 1/λ² | 等待时间 |

## 激活函数

```
Sigmoid:  σ(x) = 1/(1+e⁻ˣ)         → (0, 1)
Tanh:     tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) → (-1, 1)
ReLU:     max(0, x)                 → [0, +∞)
GELU:     x·Φ(x)                    → 平滑ReLU
Swish:    x·σ(x)                    → 自门控
SwiGLU:   (Swish(xW₁) ⊙ xW₃)·W₂   → LLaMA标配
```

## 损失函数

```
回归:
  MSE:  (1/n)Σ(yᵢ - ŷᵢ)²
  MAE:  (1/n)Σ|yᵢ - ŷᵢ|

分类:
  Binary CE: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
  Cross Entropy: -Σ yᵢ log(ŷᵢ)

语言模型:
  Next Token CE: -Σ log P(wₜ|w<t)
  Perplexity: exp(CE)
```

## 矩阵维度速查

```
嵌入:        X = embed(input)        → (B, S, d_model)
线性层:      Y = X @ W + b           → (B, S, d_out) = (B,S,d_in) @ (d_in,d_out)
注意力Q:     Q = X @ W_Q             → (B, S, d_model)
注意力分数:   scores = Q @ K.T        → (B, S, S)
注意力输出:   out = softmax(scores) @ V → (B, S, d_v)
LayerNorm:   (B, S, d) → (B, S, d)   (最后一维归一化)
```

## 参数量计算

```
Embedding:     V × d_model
Attention:     4 × d_model²          (W_Q, W_K, W_V, W_O)
FFN (标准):    2 × d_model × d_ff    (W_up, W_down)
FFN (SwiGLU):  3 × d_model × d_ff    (w1, w2, w3)
RMSNorm:       d_model               (一个可学习参数)
Total/Layer:   4d² + 3d·d_ff + 2d
Total Model:   V·d + N_layers × (4d² + 3d·d_ff + 2d)

示例 (d=4096, d_ff=11008, N=32, V=32000):
  ≈ 32000×4096 + 32×(4×4096² + 3×4096×11008 + 2×4096)
  ≈ 131M + 32×(67M + 135M + 8K)
  ≈ 131M + 32×202M
  ≈ 6.6B (LLaMA-7B)
```

## 显存估算

```
参数:         N_params × bytes_per_param
梯度:         = 参数
优化器状态(Adam): = 2 × 参数 (m + v)
激活:         B × S × d × N_layers × 2 (forward+backward)

7B模型 FP16:
  参数:    7B × 2 = 14 GB
  梯度:    7B × 2 = 14 GB
  优化器:  7B × 4 × 2 = 56 GB (FP32的m和v)
  总计:    ~84 GB (全量微调)

7B模型 QLoRA (4-bit):
  参数:    7B × 0.5 = 3.5 GB (4-bit)
  LoRA:    ~10M × 2 = 20 MB
  梯度:    ~10M × 2 = 20 MB
  优化器:  ~10M × 4 × 2 = 80 MB
  总计:    ~4 GB (可单卡训练)
```

## 常用Python/PyTorch操作

```python
# 创建
x = torch.randn(B, S, d)           # 随机张量
x = torch.zeros(B, S, d)           # 零张量
x = torch.eye(n)                   # 单位矩阵
x = torch.arange(10)               # [0,1,...,9]

# 变形
x.reshape(B, S, d)                 # 改变形状
x.view(B, S, d)                    # 改变形状(内存连续)
x.transpose(1, 2)                  # 转置两维
x.permute(0, 2, 1)                 # 任意维度重排
x.unsqueeze(0)                     # 增加维度
x.squeeze()                        # 移除大小为1的维度

# 数学
torch.matmul(a, b) 或 a @ b        # 矩阵乘法
torch.bmm(a, b)                    # 批量矩阵乘法
torch.einsum('bqd,bkd->bqk', q, k) # 爱因斯坦求和
x.sum(dim=-1)                      # 求和
x.mean(dim=-1)                     # 均值
x.norm(dim=-1)                     # L2范数

# 索引
x[mask]                            # 布尔索引
x[:, :, :d//2]                     # 切片
torch.gather(x, dim, index)        # 按索引收集
torch.masked_fill(x, mask, val)    # 掩码填充

# 自动求导
x = torch.tensor(..., requires_grad=True)
loss.backward()                    # 计算梯度
x.grad                             # 访问梯度
with torch.no_grad(): ...          # 禁用梯度
optimizer.step()                   # 更新参数
optimizer.zero_grad()              # 清零梯度

# 常用函数
F.softmax(x, dim=-1)               # Softmax
F.cross_entropy(logits, targets)   # 交叉熵
F.normalize(x, p=2, dim=-1)        # L2归一化
F.relu(x)                          # ReLU
F.silu(x)                          # SiLU/Swish
F.gelu(x)                          # GELU
torch.clamp(x, min, max)           # 裁剪
```

## 调试检查清单

```
训练不收敛:
  □ 学习率是否太大/太小？
  □ 梯度是否爆炸/消失？(check grad norm)
  □ 数据是否有问题？(check input range)
  □ 损失函数是否正确？
  □ 标签是否正确对齐？

显存不足:
  □ 减小batch size
  □ 使用梯度累积
  □ 使用混合精度 (AMP)
  □ 使用梯度检查点
  □ 使用QLoRA代替全量微调

过拟合:
  □ 增加数据量
  □ 增加正则化 (dropout, weight decay)
  □ 减小模型大小
  □ 早停 (early stopping)
```

---

**Module 01 完成!** 继续学习: [Module 02: Transformer架构深入](../module-02-transformer/)
