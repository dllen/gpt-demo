---
layout: page
title: "Module 01: 数学基础 — 后端开发者速成指南"
---
# Module 01: 数学基础 — 后端开发者速成指南

> 用你已有的后端知识，快速理解机器学习所需的数学

## 写给后端程序员的导读

作为后端工程师，你已经掌握了理解ML数学所需的大部分直觉。本节将建立你已知概念与ML数学之间的映射。

### 概念对照表

| 后端概念 | ML/数学概念 | 联系 |
|---------|-----------|------|
| API 请求/响应 | 张量 (Tensor) | 都是结构化数据的流动 |
| 数据库记录 | 向量 (Vector) | 一条记录 = 一个多维特征向量 |
| 数据库表 | 矩阵 (Matrix) | 多行记录 = 二维矩阵 |
| SQL WHERE 条件 | 掩码 (Mask) | 筛选/过滤某些元素 |
| JOIN 操作 | 矩阵乘法 | 按某种规则组合两个表 |
| 索引 (Index) | 嵌入 (Embedding) | 将数据映射到便于查找的表示 |
| 缓存 (Cache) | KV Cache | 存储已计算结果避免重复计算 |
| 负载均衡 | Softmax | 将请求分配到多个服务器 |
| 限流/权重 | 注意力权重 | 决定资源分配比例 |
| 日志聚合 | 梯度聚合 | 多个节点的结果汇总 |
| 配置参数 | 超参数 (Hyperparameters) | 控制行为的外部配置 |
| 数据库迁移 | 反向传播 | 从目标出发逐层更新 |
| 单元测试 | 梯度检验 | 验证计算正确性 |
| 性能监控 | 损失曲线 | 跟踪指标变化趋势 |

### 后端思维 → ML思维

```
后端: 输入 → 处理函数 → 输出
ML:   输入 → 模型(参数化函数) → 预测

后端: 优化SQL查询 → 减少响应时间
ML:   优化损失函数 → 减少预测误差

后端: 数据库索引加速查询
ML:   嵌入空间加速相似度搜索

后端: 微服务拆分降低耦合
ML:   神经网络分层提取特征

后端: 灰度发布验证新功能
ML:   A/B测试验证新模型
```

## 实践：用后端直觉理解张量

```python
import torch

print("=" * 60)
print("用后端视角理解ML数据结构")
print("=" * 60)

# 一条数据库记录 → 一个向量
user_record = {
    'user_id': 42,
    'age': 28,
    'login_count': 156,
    'purchase_amount': 299.5,
    'is_vip': 1
}

# 转为向量 (就像把数据库记录转为特征向量)
user_vector = torch.tensor([42.0, 28.0, 156.0, 299.5, 1.0])
print(f"用户记录 → 向量: {user_vector}")
print(f"形状 (字段数): {user_vector.shape}")

# 一个数据库表 → 一个矩阵
users_table = torch.tensor([
    [42.0, 28.0, 156.0, 299.5, 1.0],   # 用户1
    [43.0, 35.0, 89.0, 150.0, 0.0],     # 用户2
    [44.0, 22.0, 230.0, 520.0, 1.0],    # 用户3
    [45.0, 45.0, 12.0, 0.0, 0.0],       # 用户4
])
print(f"\n用户表 → 矩阵:\n{users_table}")
print(f"形状 (行数, 列数): {users_table.shape}")
print(f"  → {users_table.shape[0]} 条记录, {users_table.shape[1]} 个字段")

# SQL WHERE → 掩码操作
# SQL: SELECT * FROM users WHERE is_vip = 1
is_vip_column = users_table[:, 4]  # 第5列是is_vip
vip_mask = is_vip_column == 1.0
vip_users = users_table[vip_mask]
print(f"\nVIP用户 (WHERE is_vip=1):\n{vip_users}")

# SQL: SELECT * FROM users WHERE age > 25 AND purchase_amount > 100
age_mask = users_table[:, 1] > 25
purchase_mask = users_table[:, 3] > 100
combined_mask = age_mask & purchase_mask
filtered = users_table[combined_mask]
print(f"\n年龄>25 且 消费>100 的用户:\n{filtered}")

# JOIN → 矩阵乘法 (简化示例)
# 用户-商品评分矩阵 (类似订单表)
ratings = torch.tensor([
    [5.0, 3.0, 0.0, 1.0],   # 用户对4个商品的评分
    [4.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 5.0],
    [1.0, 0.0, 0.0, 4.0],
    [0.0, 1.0, 5.0, 4.0],
])
print(f"\n评分矩阵 (用户×商品):\n{ratings}")

# 商品相似度 = 评分矩阵的转置乘以自身
# (类似: 买了A的人还买了B → 协同过滤)
item_similarity = ratings.T @ ratings
print(f"\n商品相似度矩阵 (商品×商品):\n{item_similarity}")
print("→ 值越大表示两个商品越常被同一用户喜欢")

# 索引 → 嵌入 (Embedding)
# 类似数据库索引: 将ID映射到可比较的空间
num_users = 1000
embedding_dim = 16
user_embedding = torch.nn.Embedding(num_users, embedding_dim)

# 获取用户ID=42的嵌入向量
user_id = torch.tensor([42])
user_vec = user_embedding(user_id)
print(f"\n用户ID=42 的嵌入向量 (16维):\n{user_vec.detach().numpy().round(3)}")
print("→ 相似用户的嵌入向量在空间中也相近")

# 缓存 → KV Cache
# 类似Redis缓存: 存储已计算的结果
class SimpleKVCache:
    """模拟Redis缓存的KV Cache"""
    def __init__(self):
        self.cache = {}  # 就像 Redis 的 key-value 存储

    def get(self, key):
        return self.cache.get(key, None)

    def set(self, key, value):
        self.cache[key] = value

    def has(self, key):
        return key in self.cache

cache = SimpleKVCache()
cache.set("user:42:profile", user_vector)
print(f"\nKV Cache 命中: {cache.has('user:42:profile')}")
print(f"KV Cache 未命中: {cache.has('user:99:profile')}")

# 负载均衡 → Softmax
# 类似Nginx权重分配: 将请求按权重分配到多台服务器
server_logits = torch.tensor([2.0, 1.0, 0.5])  # 服务器权重/得分
allocation = torch.softmax(server_logits, dim=0)
print(f"\n服务器分配比例 (Softmax):")
for i, prob in enumerate(allocation):
    print(f"  服务器{i+1}: {prob:.1%}")
print("→ 所有比例之和 = 100%")

# 限流 → 注意力权重
# 类似令牌桶算法: 决定每个请求分配多少资源
attention_scores = torch.tensor([3.0, 1.0, 0.5, 0.2])
attention_weights = torch.softmax(attention_scores, dim=0)
print(f"\n注意力权重 (资源分配):")
for i, w in enumerate(attention_weights):
    print(f"  Token {i}: {w:.1%}")
print("→ 重要的token获得更多'计算资源'")
```

## 后端开发者的ML学习路线图

```
阶段1: 能跑通 (Week 1-2)
  ├── 理解张量 = 多维数组
  ├── 理解模型 = 参数化的函数
  ├── 理解训练 = 调参使函数输出正确
  └── 跑通第一个训练脚本

阶段2: 能调试 (Week 3-4)
  ├── 理解梯度 = 参数该往哪调
  ├── 理解损失 = 当前调得有多差
  ├── 理解学习率 = 每次调多少
  └── 能诊断训练不收敛的问题

阶段3: 能优化 (Week 5-8)
  ├── 理解反向传播 = 高效计算梯度
  ├── 理解正则化 = 防止过拟合
  ├── 理解优化器 = 更聪明的调参策略
  └── 能独立调优模型性能

阶段4: 能创新 (Week 9+)
  ├── 理解注意力 = 动态权重分配
  ├── 理解表示学习 = 自动特征工程
  └── 能设计新的模型结构
```

## 关键心态转变

```
后端思维                    ML思维
─────────────────────────────────────────
"代码逻辑是确定的"     →   "模型行为是概率的"
"bug是错误，要修复"    →   "噪声是数据的一部分"
"性能优化是减少计算"   →   "训练是增加计算换泛化"
"接口契约是严格的"     →   "输入输出是连续的"
"状态是精确管理的"     →   "表示是自动学习的"
"测试是验证正确性"     →   "评估是度量概率分布"
```

**下一步**: [线性代数基础](./01-linear-algebra.md)
