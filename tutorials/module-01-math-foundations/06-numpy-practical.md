# Module 01: 数学基础 — NumPy实战手册

> 用NumPy亲手实现所有ML核心运算

## 写给后端程序员的导读

NumPy就像SQL之于数据库——是Python数据操作的"标准查询语言"。作为后端工程师，你已经熟悉：

```
SQL                          NumPy
─────────────────────────────────────────
SELECT * FROM table    →    array[rows, columns]
WHERE age > 25         →    array[array[:, 1] > 25]
ORDER BY score DESC    →    array[np.argsort(-array[:, 2])]
GROUP BY category      →    np.bincount(labels)
JOIN ON id             →    np.concatenate([a, b], axis=1)
```

## 实践1：NumPy基础操作

```python
import numpy as np

print("=" * 60)
print("NumPy基础操作 — 像操作数据库一样操作数组")
print("=" * 60)

# 1. 创建数组 (类似 INSERT)
print("\n--- 创建数组 ---")
# 从列表创建 (类似 INSERT INTO)
a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
print(f"从列表: {a}")

# 零矩阵 (类似 CREATE TABLE with default 0)
zeros = np.zeros((3, 4))
print(f"零矩阵 (3×4):\n{zeros}")

# 全1矩阵
ones = np.ones((2, 3))
print(f"全1矩阵 (2×3):\n{ones}")

# 单位矩阵 (类似 PRIMARY KEY 的唯一性)
eye = np.eye(4)
print(f"单位矩阵 (4×4):\n{eye}")

# 随机数组 (类似 RAND() 函数)
np.random.seed(42)
rand = np.random.randn(3, 4)  # 标准正态分布
print(f"随机矩阵 (3×4):\n{rand.round(3)}")

# 均匀分布
uniform = np.random.uniform(-1, 1, (2, 3))
print(f"均匀分布 [-1,1]:\n{uniform.round(3)}")

# 范围数组 (类似 generate_series)
arange = np.arange(0, 10, 2)
print(f"arange(0,10,2): {arange}")

linspace = np.linspace(0, 1, 5)
print(f"linspace(0,1,5): {linspace}")

# 2. 数组属性 (类似 DESCRIBE TABLE)
print("\n--- 数组属性 ---")
arr = np.random.randn(3, 4, 5)
print(f"shape (维度/表结构): {arr.shape}")
print(f"ndim (维度数): {arr.ndim}")
print(f"size (总元素数): {arr.size}")
print(f"dtype (数据类型): {arr.dtype}")
print(f"itemsize (每元素字节): {arr.itemsize}")
print(f"nbytes (总字节): {arr.nbytes}")

# 3. 索引与切片 (类似 SELECT ... WHERE)
print("\n--- 索引与切片 ---")
arr = np.arange(24).reshape(4, 6)
print(f"原始数组 (4×6):\n{arr}")

# 取第2行
print(f"第2行: {arr[1]}")

# 取第3列
print(f"第3列: {arr[:, 2]}")

# 取子矩阵 (类似 LIMIT + OFFSET)
print(f"子矩阵 [1:3, 2:5]:\n{arr[1:3, 2:5]}")

# 条件筛选 (类似 WHERE)
mask = arr > 15
print(f">15 的元素: {arr[mask]}")

# 布尔索引
print(f"偶数元素: {arr[arr % 2 == 0]}")

# Fancy indexing (类似 WHERE id IN (1,3))
print(f"第1,3行: {arr[[1, 3]]}")

# 4. 变形操作 (类似 ALTER TABLE)
print("\n--- 变形操作 ---")
a = np.arange(12)
print(f"原始: {a.shape} → {a}")

# reshape (类似改变表结构)
b = a.reshape(3, 4)
print(f"reshape(3,4): {b.shape}")
print(b)

# 转置 (类似 PIVOT)
c = b.T
print(f"转置: {c.shape}")
print(c)

# 展平 (类似 UNNEST)
d = b.flatten()
print(f"展平: {d.shape} → {d}")

# 增加/删除维度
e = a[np.newaxis, :]  # (12,) → (1, 12)
print(f"增加维度: {e.shape}")

f = e.squeeze()  # (1, 12) → (12,)
print(f"压缩维度: {f.shape}")
```

## 实践2：线性代数运算

```python
print("\n" + "=" * 60)
print("线性代数运算 — ML的基石")
print("=" * 60)

# 5. 矩阵乘法 (类似 JOIN)
print("\n--- 矩阵乘法 ---")
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# 矩阵乘法
C = A @ B  # 或 np.dot(A, B)
print(f"A (3×4) @ B (4×5) = C ({C.shape})")
print(f"C:\n{C.round(3)}")

# 批量矩阵乘法
batch_A = np.random.randn(2, 3, 4)
batch_B = np.random.randn(2, 4, 5)
batch_C = batch_A @ batch_B
print(f"\n批量乘法: {batch_A.shape} @ {batch_B.shape} = {batch_C.shape}")

# einsum (最灵活的运算)
# 矩阵乘法
C_einsum = np.einsum('ij,jk->ik', A, B)
print(f"\neinsum结果一致: {np.allclose(C, C_einsum)}")

# 批量矩阵乘法
batch_C_einsum = np.einsum('bij,bjk->bik', batch_A, batch_B)
print(f"批量einsum一致: {np.allclose(batch_C, batch_C_einsum)}")

# 迹
trace = np.einsum('ii->', np.eye(3) * 5)
print(f"trace(5I) = {trace}")

# 对角线提取
D = np.arange(9).reshape(3, 3)
print(f"\n对角线: {np.einsum('ii->i', D)}")

# 外积 (类似 CROSS JOIN)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5])
outer = np.einsum('i,j->ij', v1, v2)
print(f"外积:\n{outer}")

# 6. 点积与相似度
print("\n--- 点积与相似度 ---")
def dot_product(a, b):
    """向量点积"""
    return np.sum(a * b)

def cosine_similarity(a, b):
    """余弦相似度"""
    return dot_product(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# 模拟词向量
word_vectors = {
    'king':   np.array([0.8, 0.6, 0.1, 0.2]),
    'queen':  np.array([0.7, 0.7, 0.2, 0.1]),
    'man':    np.array([0.7, 0.5, 0.1, 0.3]),
    'woman':  np.array([0.6, 0.6, 0.2, 0.2]),
    'apple':  np.array([0.1, 0.2, 0.8, 0.7]),
}

# 计算相似度矩阵
words = list(word_vectors.keys())
n = len(words)
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = cosine_similarity(word_vectors[words[i]], word_vectors[words[j]])

print("余弦相似度矩阵:")
print(f"{'':>8s}", "".join(f"{w:>8s}" for w in words))
for i, w in enumerate(words):
    row = "".join(f"{sim_matrix[i,j]:8.3f}" for j in range(n))
    print(f"{w:>8s}{row}")

# 经典类比: king - man + woman ≈ queen
result = word_vectors['king'] - word_vectors['man'] + word_vectors['woman']
sim_with_queen = cosine_similarity(result, word_vectors['queen'])
print(f"\nking - man + woman ≈ queen: 相似度={sim_with_queen:.4f}")

# 7. 范数与归一化
print("\n--- 范数与归一化 ---")
v = np.array([3.0, -4.0, 0.0, 1.0])
print(f"向量: {v}")
print(f"L1范数: {np.linalg.norm(v, ord=1):.4f}")
print(f"L2范数: {np.linalg.norm(v, ord=2):.4f}")
print(f"∞范数: {np.linalg.norm(v, ord=np.inf):.4f}")

# L2归一化 (单位向量)
v_normalized = v / (np.linalg.norm(v) + 1e-8)
print(f"L2归一化后: {v_normalized}")
print(f"归一化后L2范数: {np.linalg.norm(v_normalized):.6f}")

# 批量归一化
def batch_norm(X, eps=1e-8):
    """对每列做归一化"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + eps)

X = np.random.randn(100, 5) * np.array([10, 1, 100, 0.1, 50]) + np.array([5, 0, -20, 1, 10])
X_norm = batch_norm(X)
print(f"\n归一化前 - 均值: {X.mean(axis=0).round(2)}, 标准差: {X.std(axis=0).round(2)}")
print(f"归一化后 - 均值: {np.abs(X_norm.mean(axis=0)).round(6)}, 标准差: {X_norm.std(axis=0).round(3)}")

# 8. 特征值分解
print("\n--- 特征值分解 ---")
# 协方差矩阵 (对称)
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eigh(A)  # eigh用于对称矩阵
print(f"特征值: {eigenvalues.round(4)}")
print(f"特征向量:\n{eigenvectors.round(4)}")

# 验证: A @ v = λ * v
for i in range(len(eigenvalues)):
    lhs = A @ eigenvectors[:, i]
    rhs = eigenvalues[i] * eigenvectors[:, i]
    print(f"  特征值{eigenvalues[i]:.4f}: 验证 {np.allclose(lhs, rhs)}")

# SVD分解 (更通用)
U, S, Vt = np.linalg.svd(A)
print(f"\nSVD奇异值: {S.round(4)}")
print(f"重构误差: {np.linalg.norm(U @ np.diag(S) @ Vt - A):.2e}")

# 9. Softmax与LogSoftmax
print("\n--- Softmax ---")
def softmax(x, axis=-1):
    """数值稳定的Softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def log_softmax(x, axis=-1):
    """数值稳定的LogSoftmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))

# 模拟模型输出
logits = np.random.randn(3, 5)  # 3个样本, 5个类别
probs = softmax(logits)
log_probs = log_softmax(logits)

print(f"Logits:\n{logits.round(3)}")
print(f"\nSoftmax概率 (每行和=1):\n{probs.round(4)}")
print(f"每行之和: {probs.sum(axis=1).round(6)}")
print(f"\nLogSoftmax:\n{log_probs.round(4)}")
print(f"验证: exp(log_softmax) ≈ softmax: {np.allclose(np.exp(log_probs), probs)}")
```

## 实践3：统计与概率运算

```python
print("\n" + "=" * 60)
print("统计与概率运算")
print("=" * 60)

# 10. 描述统计
print("\n--- 描述统计 ---")
np.random.seed(42)
data = np.random.lognormal(mean=3.5, sigma=0.5, size=10000)

print(f"数据量: {len(data)}")
print(f"均值: {np.mean(data):.2f}")
print(f"中位数: {np.median(data):.2f}")
print(f"标准差: {np.std(data):.2f}")
print(f"方差: {np.var(data):.2f}")
print(f"最小值: {np.min(data):.2f}")
print(f"最大值: {np.max(data):.2f}")
print(f"P25: {np.percentile(data, 25):.2f}")
print(f"P50: {np.percentile(data, 50):.2f}")
print(f"P75: {np.percentile(data, 75):.2f}")
print(f"P95: {np.percentile(data, 95):.2f}")
print(f"P99: {np.percentile(data, 99):.2f}")

# 11. 协方差与相关系数
print("\n--- 协方差与相关系数 ---")
n = 500
x = np.random.randn(n)
y = 0.8 * x + 0.6 * np.random.randn(n)  # y与x相关
z = np.random.randn(n)  # z与x无关

# 协方差矩阵
data_matrix = np.stack([x, y, z], axis=1)
cov_matrix = np.cov(data_matrix.T)
print("协方差矩阵:")
print(cov_matrix.round(4))

# 相关系数矩阵
corr_matrix = np.corrcoef(data_matrix.T)
print("\n相关系数矩阵:")
print(corr_matrix.round(4))
print(f"x与y相关系数: {corr_matrix[0,1]:.4f} (强相关)")
print(f"x与z相关系数: {corr_matrix[0,2]:.4f} (几乎无关)")

# 12. 直方图与分布
print("\n--- 直方图与分布 ---")
samples = np.random.randn(10000)
hist, bin_edges = np.histogram(samples, bins=20)
print("标准正态分布直方图:")
for i, count in enumerate(hist):
    bar = "█" * (count // 20)
    print(f"  [{bin_edges[i]:6.2f}, {bin_edges[i+1]:6.2f}]: {count:4d} {bar}")

# 13. 随机采样
print("\n--- 随机采样 ---")
# 从分布中采样
probs = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
samples = np.random.choice(len(probs), size=10000, p=probs)
counts = np.bincount(samples, minlength=len(probs))
print(f"理论概率: {probs}")
print(f"实际频率: {counts / counts.sum()}")
print(f"最大偏差: {np.max(np.abs(probs - counts / counts.sum())):.4f}")

# 加权采样
weights = np.array([1.0, 2.0, 3.0, 4.0])
weights = weights / weights.sum()
samples = np.random.choice(4, size=1000, p=weights)
print(f"\n加权采样: 权重={weights.round(3)}")
print(f"采样结果: {np.bincount(samples)}")
```

## 实践4：NumPy实现ML核心算法

```python
print("\n" + "=" * 60)
print("NumPy实现ML核心算法")
print("=" * 60)

# 14. 从零实现线性回归
print("\n--- 线性回归 (最小二乘法) ---")
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 3)
true_w = np.array([2.0, -1.5, 0.5])
true_b = 1.0
y = X @ true_w + true_b + np.random.randn(n_samples) * 0.1

# 解析解: w = (X^T X)^{-1} X^T y
X_with_bias = np.column_stack([X, np.ones(n_samples)])
w_analytical = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T y
print(f"真实权重: {np.append(true_w, true_b).round(4)}")
print(f"估计权重: {w_analytical.round(4)}")
print(f"预测MSE: {np.mean((X_with_bias @ w_analytical - y)**2):.6f}")

# 15. 从零实现逻辑回归
print("\n--- 逻辑回归 (梯度下降) ---")
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# 生成二分类数据
n = 200
X1 = np.random.randn(n // 2, 2) + np.array([2, 2])
X2 = np.random.randn(n // 2, 2) + np.array([-2, -2])
X_cls = np.vstack([X1, X2])
y_cls = np.array([1] * (n // 2) + [0] * (n // 2))

# 添加偏置
X_cls_b = np.column_stack([X_cls, np.ones(n)])

# 梯度下降
w = np.zeros(3)
lr = 0.1
for epoch in range(100):
    z = X_cls_b @ w
    pred = sigmoid(z)
    grad = X_cls_b.T @ (pred - y_cls) / n
    w -= lr * grad

    if epoch % 20 == 0:
        loss = -np.mean(y_cls * np.log(pred + 1e-8) + (1 - y_cls) * np.log(1 - pred + 1e-8))
        acc = np.mean((pred > 0.5) == y_cls)
        print(f"  Epoch {epoch:3d}: loss={loss:.4f}, acc={acc:.4f}")

print(f"\n最终权重: {w.round(4)}")
print(f"最终准确率: {np.mean((sigmoid(X_cls_b @ w) > 0.5) == y_cls):.4f}")

# 16. 从零实现K-Means聚类
print("\n--- K-Means聚类 ---")
def kmeans(X, k=3, max_iter=100, tol=1e-4):
    """K-Means聚类"""
    n = X.shape[0]
    # 随机初始化中心点
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices].copy()

    for iteration in range(max_iter):
        # 分配: 每个点分配到最近的中心
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)

        # 更新: 重新计算中心点
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i] for i in range(k)])

        # 检查收敛
        shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
        centroids = new_centroids

        if shift < tol:
            print(f"  收敛于第{iteration+1}轮")
            break

    return labels, centroids

# 生成聚类数据
np.random.seed(42)
X_kmeans = np.vstack([
    np.random.randn(50, 2) + np.array([0, 0]),
    np.random.randn(50, 2) + np.array([5, 5]),
    np.random.randn(50, 2) + np.array([0, 5]),
])

labels, centroids = kmeans(X_kmeans, k=3)
print(f"聚类中心:\n{centroids.round(3)}")
print(f"每类数量: {np.bincount(labels)}")

# 17. 从零实现PCA
print("\n--- PCA降维 ---")
def pca(X, n_components=2):
    """PCA降维"""
    # 1. 中心化
    X_centered = X - X.mean(axis=0)

    # 2. 协方差矩阵
    cov = np.cov(X_centered.T)

    # 3. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 4. 取前n_components个最大特征值对应的特征向量
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    components = eigenvectors[:, idx]

    # 5. 投影
    X_pca = X_centered @ components

    # 解释方差比
    explained_variance_ratio = eigenvalues[idx] / eigenvalues.sum()

    return X_pca, components, explained_variance_ratio

# 高维数据降维
np.random.seed(42)
X_high = np.random.randn(200, 10) @ np.random.randn(10, 10) + np.random.randn(200, 10) * 0.1
X_pca, components, var_ratio = pca(X_high, n_components=2)

print(f"原始维度: {X_high.shape[1]}")
print(f"降维后: {X_pca.shape[1]}")
print(f"解释方差比: {var_ratio.round(4)}")
print(f"保留信息: {var_ratio.sum()*100:.1f}%")
```

## 实践5：NumPy性能优化

```python
print("\n" + "=" * 60)
print("NumPy性能优化 — 向量化 vs 循环")
print("=" * 60)

import time

# 18. 向量化 vs Python循环
print("\n--- 向量化 vs 循环 ---")
n = 1000000
a = np.random.randn(n)
b = np.random.randn(n)

# Python循环
start = time.time()
result_loop = np.zeros(n)
for i in range(n):
    result_loop[i] = a[i] * b[i] + a[i]
time_loop = time.time() - start

# NumPy向量化
start = time.time()
result_vec = a * b + a
time_vec = time.time() - start

print(f"数据量: {n:,}")
print(f"Python循环: {time_loop*1000:.1f}ms")
print(f"NumPy向量化: {time_vec*1000:.1f}ms")
print(f"加速比: {time_loop/time_vec:.0f}x")
print(f"结果一致: {np.allclose(result_loop, result_vec)}")

# 19. 广播机制
print("\n--- 广播机制 ---")
# 场景: 计算一组点到一组中心的距离
points = np.random.randn(1000, 3)    # 1000个3D点
centers = np.random.randn(10, 3)     # 10个中心

# 方法1: 循环 (慢)
start = time.time()
dist_loop = np.zeros((1000, 10))
for i in range(1000):
    for j in range(10):
        dist_loop[i, j] = np.sqrt(np.sum((points[i] - centers[j])**2))
time_loop = time.time() - start

# 方法2: 广播 (快)
start = time.time()
# points: (1000, 1, 3), centers: (1, 10, 3) → (1000, 10, 3)
dist_broadcast = np.sqrt(np.sum((points[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
time_broadcast = time.time() - start

print(f"循环: {time_loop*1000:.1f}ms")
print(f"广播: {time_broadcast*1000:.1f}ms")
print(f"加速比: {time_loop/time_broadcast:.0f}x")
print(f"结果一致: {np.allclose(dist_loop, dist_broadcast)}")

# 20. 内存布局优化
print("\n--- 内存布局 ---")
n = 10000
# C-order (行优先, 默认)
a_c = np.random.randn(n, n)
# F-order (列优先)
a_f = np.asfortranarray(a_c)

# 按行求和 (C-order更快)
start = time.time()
for _ in range(100):
    a_c.sum(axis=1)
time_c_row = time.time() - start

start = time.time()
for _ in range(100):
    a_f.sum(axis=1)
time_f_row = time.time() - start

print(f"C-order 行求和: {time_c_row*1000:.1f}ms")
print(f"F-order 行求和: {time_f_row*1000:.1f}ms")
print(f"C-order行优先更快: {time_c_row < time_f_row}")

# 21. 原地操作 (节省内存)
print("\n--- 原地操作 ---")
a = np.random.randn(1000, 1000)
b = np.random.randn(1000, 1000)

# 非原地: 创建新数组
start = time.time()
for _ in range(1000):
    c = a + b  # 分配新内存
time_new = time.time() - start

# 原地: 复用内存
start = time.time()
for _ in range(1000):
    a += b  # 原地修改
time_inplace = time.time() - start

print(f"非原地操作: {time_new*1000:.1f}ms")
print(f"原地操作: {time_inplace*1000:.1f}ms")
print(f"原地操作更快: {time_inplace < time_new}")

# 22. 使用np.where替代循环
print("\n--- np.where替代循环 ---")
data = np.random.randn(1000000)

# 循环方式
start = time.time()
result_loop = np.zeros_like(data)
for i in range(len(data)):
    if data[i] > 0:
        result_loop[i] = data[i] ** 2
    else:
        result_loop[i] = 0
time_loop = time.time() - start

# np.where方式
start = time.time()
result_where = np.where(data > 0, data ** 2, 0)
time_where = time.time() - start

print(f"循环: {time_loop*1000:.1f}ms")
print(f"np.where: {time_where*1000:.1f}ms")
print(f"加速比: {time_loop/time_where:.0f}x")
```

## 实践6：NumPy实现注意力机制

```python
print("\n" + "=" * 60)
print("NumPy实现Self-Attention")
print("=" * 60)

# 23. 纯NumPy实现Self-Attention
def numpy_attention(X, W_Q, W_K, W_V, W_O):
    """
    纯NumPy实现Self-Attention

    Args:
        X: (batch, seq_len, d_model)
        W_Q, W_K, W_V, W_O: (d_model, d_model)
    Returns:
        output: (batch, seq_len, d_model)
        attention_weights: (batch, seq_len, seq_len)
    """
    # 线性投影
    Q = X @ W_Q  # (batch, seq, d_model)
    K = X @ W_K
    V = X @ W_V

    # 注意力分数
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)  # (batch, seq, seq)

    # 因果掩码
    seq_len = X.shape[1]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    scores[:, mask] = -np.inf

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 加权求和
    context = attention_weights @ V  # (batch, seq, d_model)

    # 最终线性变换
    output = context @ W_O

    return output, attention_weights

# 测试
np.random.seed(42)
batch_size, seq_len, d_model = 2, 6, 32

X = np.random.randn(batch_size, seq_len, d_model)
W_Q = np.random.randn(d_model, d_model) * 0.02
W_K = np.random.randn(d_model, d_model) * 0.02
W_V = np.random.randn(d_model, d_model) * 0.02
W_O = np.random.randn(d_model, d_model) * 0.02

output, attn_weights = numpy_attention(X, W_Q, W_K, W_V, W_O)

print(f"输入 shape: {X.shape}")
print(f"输出 shape: {output.shape}")
print(f"注意力权重 shape: {attn_weights.shape}")
print(f"注意力权重 (第一个batch):\n{attn_weights[0].round(3)}")
print(f"每行之和: {attn_weights[0].sum(axis=1).round(6)}")
print(f"因果掩码验证 (上三角为0): {np.allclose(attn_weights[0][np.triu_indices(seq_len, k=1)], 0)}")

# 24. 纯NumPy实现LayerNorm
print("\n--- NumPy实现LayerNorm ---")
def layer_norm(X, gamma=None, beta=None, eps=1e-6):
    """LayerNorm"""
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + eps)
    if gamma is not None:
        X_norm = X_norm * gamma
    if beta is not None:
        X_norm = X_norm + beta
    return X_norm

def rms_norm(X, weight=None, eps=1e-6):
    """RMSNorm (LLaMA使用)"""
    rms = np.sqrt(np.mean(X ** 2, axis=-1, keepdims=True) + eps)
    X_norm = X / rms
    if weight is not None:
        X_norm = X_norm * weight
    return X_norm

X_ln = np.random.randn(2, 5, 8)
gamma = np.ones(8)
beta = np.zeros(8)

X_ln_result = layer_norm(X_ln, gamma, beta)
X_rms_result = rms_norm(X_ln, gamma)

print(f"LayerNorm后 (每行均值≈0, 标准差≈1):")
print(f"  均值: {X_ln_result[0].mean(axis=-1).round(6)}")
print(f"  标准差: {X_ln_result[0].std(axis=-1).round(3)}")

print(f"\nRMSNorm后:")
print(f"  RMS: {np.sqrt(np.mean(X_rms_result[0]**2, axis=-1)).round(6)}")

# 25. 纯NumPy实现简单的训练循环
print("\n--- NumPy实现简单训练循环 ---")
np.random.seed(42)

# 数据: y = 2x + 1 + 噪声
n_samples = 100
X_train = np.random.randn(n_samples, 1)
y_train = 2 * X_train + 1 + np.random.randn(n_samples, 1) * 0.1

# 参数初始化
W = np.random.randn(1, 1) * 0.01
b = np.zeros((1,))

# 训练
lr = 0.1
for epoch in range(100):
    # 前向
    y_pred = X_train @ W + b
    loss = np.mean((y_pred - y_train) ** 2)

    # 反向
    grad_pred = 2 * (y_pred - y_train) / n_samples
    grad_W = X_train.T @ grad_pred
    grad_b = np.sum(grad_pred, axis=0)

    # 更新
    W -= lr * grad_W
    b -= lr * grad_b

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: loss={loss:.6f}, W={W[0,0]:.4f}, b={b[0]:.4f}")

print(f"\n真实: W=2.0, b=1.0")
print(f"估计: W={W[0,0]:.4f}, b={b[0]:.4f}")
```

## 总结

| 操作 | NumPy函数 | 后端类比 |
|------|----------|---------|
| 创建数组 | `np.array()`, `np.zeros()` | INSERT |
| 索引筛选 | `array[mask]` | WHERE |
| 矩阵乘法 | `a @ b` | JOIN |
| 聚合 | `sum()`, `mean()`, `std()` | GROUP BY |
| 变形 | `reshape()`, `transpose()` | ALTER TABLE |
| 广播 | 自动扩展 | 自动类型转换 |
| 条件 | `np.where()` | CASE WHEN |
| 随机 | `np.random.*` | RAND() |

**下一步**: [Module 02: Transformer架构深入](../module-02-transformer/)
