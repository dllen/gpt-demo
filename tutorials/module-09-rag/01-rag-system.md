# Module 09: RAG系统 — 检索增强生成

## 理论部分

### 9.1 为什么需要RAG？

LLM的局限性：
- **幻觉**: 编造不存在的事实
- **知识过时**: 训练数据有截止日期
- **缺乏专业领域知识**: 对垂直领域了解不足
- **无法引用来源**: 不知道答案从哪来

**RAG的解决方案**: 在生成前先从外部知识库检索相关信息。

### 9.2 RAG架构

```
┌─────────────────────────────────────────────────┐
│                   RAG Pipeline                    │
│                                                   │
│  用户查询 → [检索器] → 相关文档 → [生成器] → 回答  │
│                ↑                    ↑             │
│           [知识库]              [LLM模型]         │
│                                                   │
└─────────────────────────────────────────────────┘
```

### 9.3 检索器 (Retriever)

**流程**:
```
文档 → 分块 → 嵌入(向量化) → 向量数据库
查询 → 嵌入(向量化) → 相似度搜索 → Top-K文档
```

**嵌入模型**:
- OpenAI text-embedding-3
- BGE (BAAI General Embedding)
- E5
- Sentence Transformers

**相似度计算**:
```
余弦相似度: cos(q, d) = q·d / (||q||·||d||)
内积:       score = q·d
欧氏距离:   dist = ||q - d||
```

### 9.4 分块策略

| 策略 | 块大小 | 优点 | 缺点 |
|------|--------|------|------|
| 固定大小 | 256-1024 tokens | 简单 | 可能切断语义 |
| 递归分块 | 可变 | 保留语义 | 实现复杂 |
| 语义分块 | 可变 | 语义完整 | 计算量大 |
| 滑动窗口 | 固定+重叠 | 保留上下文 | 冗余存储 |

### 9.5 RAG变体

```
Naive RAG:     查询 → 检索 → 拼接 → 生成
Advanced RAG:  查询改写 → 检索 → 重排序 → 过滤 → 生成
Modular RAG:   多检索器 + 多策略 + 融合
GraphRAG:      知识图谱 + 向量检索
Agentic RAG:   Agent自主决定检索策略
```

## 实践部分

### 实践1：向量检索实现

```python
import numpy as np
from collections import Counter
import re

print("=" * 60)
print("实践1: 从零实现向量检索")
print("=" * 60)

class SimpleVectorStore:
    """简单的向量存储与检索"""
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.vectors = []

    def add_documents(self, documents):
        """添加文档到向量库"""
        for doc in documents:
            self.documents.append(doc)
            # 简单的词袋嵌入 (实际使用预训练嵌入模型)
            vec = self._simple_embed(doc)
            self.vectors.append(vec)

        self.vectors = np.array(self.vectors)
        # L2归一化
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norms + 1e-8)

    def _simple_embed(self, text):
        """简单的词袋嵌入 (仅用于演示)"""
        words = re.findall(r'\w+', text.lower())
        vec = np.zeros(self.embedding_dim)
        for word in words:
            # 用hash将词映射到维度
            idx = hash(word) % self.embedding_dim
            vec[idx] += 1
        return vec

    def search(self, query, top_k=3):
        """检索最相关的文档"""
        query_vec = self._simple_embed(query)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # 余弦相似度
        scores = self.vectors @ query_vec

        # Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(scores[idx]),
                'index': int(idx)
            })

        return results

# 创建知识库
knowledge_base = [
    "Python是一种高级编程语言，由Guido van Rossum于1991年创建。",
    "机器学习是人工智能的一个分支，让计算机从数据中学习。",
    "深度学习是机器学习的子集，使用多层神经网络。",
    "Transformer架构由Vaswani等人在2017年提出，是现代LLM的基础。",
    "RAG（检索增强生成）结合检索系统和语言模型来生成更准确的回答。",
    "LoRA是一种参数高效微调方法，通过低秩矩阵适配大模型。",
    "KV Cache是推理优化技术，缓存已计算的键值对加速生成。",
    "BPE（字节对编码）是GPT系列模型使用的分词算法。",
]

store = SimpleVectorStore(embedding_dim=256)
store.add_documents(knowledge_base)

# 测试检索
queries = [
    "什么是Transformer？",
    "如何微调大模型？",
    "推理加速技术有哪些？",
]

for query in queries:
    print(f"\n查询: {query}")
    results = store.search(query, top_k=3)
    for i, r in enumerate(results):
        print(f"  [{i+1}] 相似度: {r['score']:.4f} | {r['document'][:50]}...")
```

### 实践2：完整RAG Pipeline

```python
print("\n" + "=" * 60)
print("实践2: 完整RAG Pipeline")
print("=" * 60)

class SimpleRAG:
    """简化的RAG系统"""
    def __init__(self, vector_store, llm_model=None):
        self.store = vector_store
        self.llm = llm_model

    def retrieve(self, query, top_k=3):
        """检索相关文档"""
        return self.store.search(query, top_k)

    def format_context(self, results):
        """格式化检索结果为上下文"""
        context = "\n".join([
            f"[文档{i+1}] (相关度: {r['score']:.3f})\n{r['document']}"
            for i, r in enumerate(results)
        ])
        return context

    def generate_prompt(self, query, context):
        """构建RAG prompt"""
        prompt = f"""基于以下参考信息回答问题。如果参考信息不足，请说明。

参考信息:
{context}

问题: {query}

回答:"""
        return prompt

    def answer(self, query, top_k=3):
        """完整的RAG回答流程"""
        # 1. 检索
        results = self.retrieve(query, top_k)

        # 2. 格式化上下文
        context = self.format_context(results)

        # 3. 构建prompt
        prompt = self.generate_prompt(query, context)

        # 4. 生成回答 (这里简化，实际调用LLM)
        answer = f"[基于{len(results)}条检索结果生成的回答]"

        return {
            'query': query,
            'retrieved': results,
            'prompt': prompt,
            'answer': answer,
        }

# 使用RAG
rag = SimpleRAG(store)

query = "Transformer在LLM中有什么作用？"
result = rag.answer(query, top_k=3)

print(f"查询: {result['query']}")
print(f"\n检索到 {len(result['retrieved'])} 条相关文档:")
for i, doc in enumerate(result['retrieved']):
    print(f"  [{i+1}] {doc['document'][:60]}... (score: {doc['score']:.3f})")
print(f"\n构建的Prompt:\n{result['prompt']}")
```

### 实践3：嵌入模型使用

```python
print("\n" + "=" * 60)
print("实践3: 使用预训练嵌入模型")
print("=" * 60)

print("""
使用Sentence Transformers进行文档嵌入:

from sentence_transformers import SentenceTransformer
import numpy as np

# 加载嵌入模型
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')

# 编码文档
documents = ["文档1...", "文档2...", "文档3..."]
doc_embeddings = model.encode(documents, normalize_embeddings=True)

# 编码查询
query = "什么是RAG？"
query_embedding = model.encode([query], normalize_embeddings=True)

# 计算相似度
scores = query_embedding @ doc_embeddings.T
top_k = np.argsort(scores[0])[::-1][:3]

for idx in top_k:
    print(f"文档{idx}: 相似度={scores[0][idx]:.4f}")
""")

# 模拟嵌入和检索
np.random.seed(42)
embedding_dim = 768

# 模拟嵌入
doc_embeddings = np.random.randn(8, embedding_dim).astype(np.float32)
doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

query_embedding = np.random.randn(1, embedding_dim).astype(np.float32)
query_embedding = query_embedding / np.linalg.norm(query_embedding)

# 检索
scores = (query_embedding @ doc_embeddings.T)[0]
top_k = np.argsort(scores)[::-1][:3]

print("模拟嵌入检索结果:")
for i, idx in enumerate(top_k):
    print(f"  Top-{i+1}: 文档{idx}, 相似度={scores[idx]:.4f}")
```

## 总结

| 组件 | 选择 | 推荐 |
|------|------|------|
| 嵌入模型 | BGE/E5/OpenAI | BGE-base (中文) |
| 向量数据库 | FAISS/Chroma/Milvus | FAISS (简单) / Milvus (生产) |
| 分块大小 | 256-1024 tokens | 512 + 50重叠 |
| 重排序 | Cross-Encoder | bge-reranker |
| Top-K | 3-10 | 5 |

**下一步**: [Module 10: Agent系统](./../module-10-agents/)
