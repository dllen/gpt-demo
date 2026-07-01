---
layout: page
title: "Module 03: 分词与词表构建"
---
# Module 03: 分词与词表构建

## 理论部分

### 3.1 为什么需要分词？

计算机不能直接处理文本，必须将文本转换为数字。分词是文本到数字的第一步：

```
原始文本: "我爱你中国"
分词结果: ["我", "爱", "你", "中国"]
Token IDs: [1234, 5678, 9012, 3456]
```

### 3.2 分词粒度对比

| 粒度 | 示例 | 优点 | 缺点 |
|------|------|------|------|
| 字符级 | 我/爱/你/中/国 | 词表小 | 序列太长，语义弱 |
| 词级 | 我/爱/你/中国 | 语义明确 | 词表爆炸，OOV问题 |
| 子词级 | 我/爱/你/中国 | 平衡词表与语义 | 需要学习分词器 |

现代LLM几乎都使用**子词分词**（Subword Tokenization）。

### 3.3 BPE (Byte Pair Encoding)

BPE是最广泛使用的分词算法（GPT-2/3/4, LLaMA使用）。

**算法流程**：

```
1. 初始化: 将文本拆分为字符 + 结束符
   "low" → ["l", "o", "w", "</w>"]

2. 统计所有相邻对频率:
   ("l","o"): 5次, ("o","w"): 5次, ("w","</w>"): 5次

3. 合并最频繁的对:
   ("l","o") → "lo"
   "low" → ["lo", "w", "</w>"]

4. 重复2-3，直到达到目标词表大小
```

**BPE的特点**：
- 高频词保持完整（"the" → 1个token）
- 低频词被拆分成子词（"unbelievably" → "un" + "believ" + "ably"）
- 词表大小通常 32K-100K

### 3.4 Byte-Level BPE (GPT-2)

标准BPE在非ASCII文本上可能产生未知token。Byte-Level BPE的改进：

```
1. 先将所有文本转为UTF-8字节
2. 在字节级别运行BPE
3. 词表大小固定为256（所有可能的字节值）+ 合并操作

优点: 不存在任何语言的OOV问题
```

### 3.5 SentencePiece (LLaMA使用)

SentencePiece是Google开发的分词框架，LLaMA使用它。

**与BPE的区别**：
1. 直接在原始文本上训练（不需要预分词）
2. 支持Unigram语言模型（不只是贪心合并）
3. 将空格作为特殊token `_` 处理

### 3.6 Token长度与模型性能

```
英文: 1 token ≈ 4 characters ≈ 0.75 words
中文: 1 token ≈ 1-2 characters
代码: 1 token ≈ 3-4 characters
```

**实际影响**：
- GPT-4的8K上下文 ≈ 6000英文单词 ≈ 4000中文字
- 更细的分词 → 更多token → 更长序列 → 更高计算成本
- 更粗的分词 → 更少token → 信息损失

## 实践部分

### 实践1：从零实现BPE

```python
import re
from collections import Counter, defaultdict

print("=" * 60)
print("实践1: 从零实现BPE分词器")
print("=" * 60)

class BPETokenizer:
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.vocab = {}        # token → id
        self.merges = []       # 合并规则列表
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def train(self, text):
        """训练BPE分词器"""
        # Step 1: 预分词
        words = self.pattern.findall(text)
        print(f"预分词得到 {len(words)} 个词")

        # Step 2: 初始化词频统计（字符级）
        word_freq = Counter(words)
        splits = {word: list(word) + ['</w>'] for word in word_freq}

        # Step 3: 初始词表（所有字符）
        chars = set()
        for word_splits in splits.values():
            chars.update(word_splits)
        self.vocab = {char: i for i, char in enumerate(sorted(chars))}
        print(f"初始词表大小: {len(self.vocab)}")

        # Step 4: 迭代合并
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            # 统计所有相邻对频率
            pair_freq = Counter()
            for word, freq in word_freq.items():
                w_splits = splits[word]
                for j in range(len(w_splits) - 1):
                    pair_freq[(w_splits[j], w_splits[j+1])] += freq

            if not pair_freq:
                break

            # 找到最频繁的对
            best_pair = max(pair_freq, key=pair_freq.get)
            best_freq = pair_freq[best_pair]

            if best_freq < 2:
                print(f"提前停止: 最高频对只出现{best_freq}次")
                break

            # 合并
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)

            # 更新所有词的拆分
            for word in word_freq:
                w_splits = splits[word]
                new_splits = []
                j = 0
                while j < len(w_splits):
                    if j < len(w_splits) - 1 and (w_splits[j], w_splits[j+1]) == best_pair:
                        new_splits.append(new_token)
                        j += 2
                    else:
                        new_splits.append(w_splits[j])
                        j += 1
                splits[word] = new_splits

            if (i + 1) % 50 == 0:
                print(f"  合并 {i+1}/{num_merges}, 词表大小: {len(self.vocab)}, 最新合并: {best_pair} → '{new_token}' (频率: {best_freq})")

        print(f"\n最终词表大小: {len(self.vocab)}")
        print(f"合并规则数: {len(self.merges)}")

    def encode(self, text):
        """将文本编码为token IDs"""
        words = self.pattern.findall(text)
        result = []

        for word in words:
            # 初始拆分为字符
            word_tokens = list(word) + ['</w>']

            # 应用所有合并规则
            for merge_pair in self.merges:
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i+1]) == merge_pair:
                        new_tokens.append(word_tokens[i] + word_tokens[i+1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens

            # 转换为IDs
            for token in word_tokens:
                if token in self.vocab:
                    result.append(self.vocab[token])

        return result

    def decode(self, token_ids):
        """将token IDs解码为文本"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(id, '<UNK>') for id in token_ids]
        text = ''.join(tokens)
        return text.replace('</w>', ' ').strip()

# 训练BPE
training_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language, in particular how to program computers to process and analyze
large amounts of natural language data. The goal is a computer capable of
understanding the contents of documents, including the contextual nuances of
the language within them. Challenges in natural language processing frequently
involve speech recognition, natural language understanding, and natural language
generation. Natural language processing has many applications including machine
translation, text summarization, sentiment analysis, and chatbot development.
Deep learning models have achieved state-of-the-art results in many NLP tasks.
The transformer architecture has become the foundation for modern language models.
"""

tokenizer = BPETokenizer(vocab_size=200)
tokenizer.train(training_text)

# 测试编码解码
test_text = "Natural language processing is amazing"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print(f"\n--- 编码解码测试 ---")
print(f"原文: {test_text}")
print(f"Token IDs: {encoded}")
print(f"Token数: {len(encoded)}")
print(f"解码: {decoded}")
print(f"还原正确: {test_text.lower().replace(' ', '') == decoded.replace(' ', '')}")
```

### 实践2：使用HuggingFace Tokenizers

```python
print("\n" + "=" * 60)
print("实践2: 使用HuggingFace Tokenizers训练BPE")
print("=" * 60)

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    # 创建BPE分词器
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    # 训练
    trainer = trainers.BpeTrainer(
        vocab_size=1000,
        min_frequency=2,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
        show_progress=True
    )

    # 训练数据
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Transformers have revolutionized natural language processing.",
        "Large language models can generate human-like text.",
        "Deep learning models require large amounts of data.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Pre-training and fine-tuning are key techniques in modern NLP.",
        "Tokenization is the first step in processing text for language models.",
    ]

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # 测试
    test_sentences = [
        "The transformer model is powerful.",
        "Machine learning transforms data into insights.",
    ]

    for sent in test_sentences:
        output = tokenizer.encode(sent)
        print(f"\n原文: {sent}")
        print(f"Tokens: {output.tokens}")
        print(f"IDs: {output.ids}")
        print(f"Token数: {len(output.ids)}")

except ImportError:
    print("请安装: pip install tokenizers")

# 实践3: 分析不同分词器的效果
print("\n" + "=" * 60)
print("实践3: 分析分词效果")
print("=" * 60)

def analyze_tokenization(text, tokenizer_fn, name):
    """分析分词效果"""
    tokens = tokenizer_fn(text)
    print(f"\n{name}:")
    print(f"  文本: {text}")
    print(f"  Tokens: {tokens}")
    print(f"  Token数: {len(tokens)}")
    print(f"  字符数: {len(text)}")
    print(f"  压缩比: {len(text)/len(tokens):.2f} chars/token")
    return len(tokens)

# 测试文本
texts = {
    "英文": "The transformer architecture has revolutionized NLP.",
    "中文": "Transformer架构彻底改变了自然语言处理领域。",
    "代码": "def forward(x): return self.attn(x) + x",
    "混合": "使用PyTorch实现Transformer模型(model = Transformer())",
}

# 简单字符级分词
char_tokenize = lambda text: list(text)

# 简单空格分词
space_tokenize = lambda text: text.split()

for lang, text in texts.items():
    print(f"\n{'='*40}")
    print(f"语言: {lang}")
    n_chars = analyze_tokenization(text, char_tokenize, "字符级")
    n_spaces = analyze_tokenization(text, space_tokenize, "空格级")
    print(f"  字符级/空格级: {n_chars}/{n_spaces}")
```

### 实践3：词表大小与OOV分析

```python
print("\n" + "=" * 60)
print("实践3: 词表大小与OOV分析")
print("=" * 60)

# 模拟不同词表大小的效果
sample_vocabularies = {
    "tiny (100)": [chr(i) for i in range(32, 132)],  # 100个字符
    "small (1K)": [f"token_{i}" for i in range(1000)],
    "medium (10K)": [f"token_{i}" for i in range(10000)],
    "large (50K)": [f"token_{i}" for i in range(50000)],
}

test_words = ["hello", "world", "transformer", "你好", "🤖", "unbelievably", "GPT-4"]

for vocab_name, vocab in sample_vocabularies.items():
    vocab_set = set(vocab)
    oov = [w for w in test_words if w not in vocab_set]
    coverage = (len(test_words) - len(oov)) / len(test_words) * 100
    print(f"\n词表 {vocab_name}:")
    print(f"  覆盖率: {coverage:.0f}%")
    print(f"  OOV词: {oov if oov else '无'}")

print("\n→ 词表越大覆盖率越高，但嵌入层参数也越多")
print("→ 需要在覆盖率和计算成本之间权衡")
```

## 总结

| 算法 | 使用模型 | 特点 |
|------|---------|------|
| BPE | GPT-2/3/4 | 贪心合并，简单高效 |
| Byte-Level BPE | GPT-2/3/4 | 无OOV，字节级 |
| SentencePiece (BPE) | LLaMA, T5 | 语言无关，支持空格 |
| SentencePiece (Unigram) | T5, ALBERT | 概率模型，更灵活 |
| WordPiece | BERT | 最大似然合并 |

**下一步**: [Module 04: 模型架构实现](./../module-04-model-architecture/)
