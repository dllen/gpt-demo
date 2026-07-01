---
layout: page
title: "Module 11: 模型评估"
---
# Module 11: 模型评估

## 理论部分

### 11.1 评估维度

```
┌─────────────────────────────────────────┐
│            模型评估维度                    │
├─────────────┬─────────────┬─────────────┤
│   能力评估   │   效率评估   │   安全评估   │
├─────────────┼─────────────┼─────────────┤
│ 知识问答     │ 推理速度     │ 有害内容     │
│ 数学推理     │ 显存占用     │ 偏见检测     │
│ 代码生成     │ 吞吐量       │ 隐私保护     │
│ 多语言       │ 首token延迟  │ 真实性       │
│ 长文本       │ 成本         │ 指令遵循     │
└─────────────┴─────────────┴─────────────┘
```

### 11.2 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 准确率 | correct/total | 分类/选择题 |
| 困惑度 | exp(-ΣlogP/N) | 语言建模 |
| F1 | 2·P·R/(P+R) | 信息抽取 |
| BLEU | n-gram精确度 | 翻译/生成 |
| ROUGE | n-gram召回率 | 摘要 |
| Pass@k | 至少1次正确 | 代码生成 |

### 11.3 主流基准测试

| 基准 | 类型 | 语言 | 任务 |
|------|------|------|------|
| MMLU | 知识 | 英文 | 57学科选择题 |
| C-Eval | 知识 | 中文 | 52学科选择题 |
| GSM8K | 数学 | 英文 | 小学数学 |
| HumanEval | 代码 | 英文 | Python代码生成 |
| MT-Bench | 对话 | 英文 | 多轮对话质量 |
| IFEval | 指令 | 英文 | 指令遵循 |
| LongBench | 长文本 | 双语 | 长上下文理解 |

### 11.4 评估方法

```
自动评估:
  - 精确匹配 (EM)
  - n-gram重叠 (BLEU/ROUGE)
  - 模型评分 (LLM-as-Judge)

人工评估:
  - 偏好比较 (A/B Test)
  - Likert量表评分
  - 盲评

混合评估:
  - 自动筛选 + 人工复核
  - 模型初评 + 人工校验
```

## 实践部分

### 实践1：基础指标计算

```python
import numpy as np
from collections import Counter
import re

print("=" * 60)
print("实践1: 基础评估指标计算")
print("=" * 60)

def accuracy(predictions, targets):
    """准确率"""
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(predictions)

def precision_recall_f1(predictions, targets, num_classes=10):
    """精确率、召回率、F1"""
    results = {}
    for c in range(num_classes):
        tp = sum((p == c and t == c) for p, t in zip(predictions, targets))
        fp = sum((p == c and t != c) for p, t in zip(predictions, targets))
        fn = sum((p != c and t == c) for p, t in zip(predictions, targets))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        results[c] = {'precision': precision, 'recall': recall, 'f1': f1}
    return results

def bleu_score(reference, hypothesis, max_n=4):
    """BLEU分数计算"""
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(get_ngrams(ref_tokens, n))
        hyp_ngrams = Counter(get_ngrams(hyp_tokens, n))

        clipped = sum(min(hyp_ngrams[g], ref_ngrams[g]) for g in hyp_ngrams)
        total = max(sum(hyp_ngrams.values()), 1)

        if total > 0:
            scores.append(clipped / total)
        else:
            scores.append(0)

    # 几何平均
    if all(s > 0 for s in scores):
        geo_mean = np.exp(np.mean(np.log(scores)))
    else:
        geo_mean = 0

    # 简短惩罚
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    return bp * geo_mean

def perplexity(log_probs):
    """困惑度"""
    return np.exp(-np.mean(log_probs))

# 测试
predictions = [1, 2, 3, 4, 5, 1, 2, 3]
targets =     [1, 2, 3, 4, 6, 1, 2, 4]

print(f"准确率: {accuracy(predictions, targets):.4f}")

reference = "The cat sat on the mat"
hypothesis = "The cat is on the mat"
print(f"BLEU分数: {bleu_score(reference, hypothesis):.4f}")

log_probs = np.log([0.8, 0.7, 0.9, 0.6, 0.85])
print(f"困惑度: {perplexity(log_probs):.2f}")
```

### 实践2：使用评估框架

```python
print("\n" + "=" * 60)
print("实践2: 使用评估框架")
print("=" * 60)

print("""
# 使用lm-evaluation-harness
pip install lm-eval

# 评估模型
lm_eval --model hf \
    --model_args pretrained=model_name \
    --tasks hellaswag,mmlu,gsm8k \
    --device cuda:0 \
    --batch_size 8

# 使用OpenCompass
pip install opencompass
opencompass --models hf_llama_7b --datasets mmlu ceval gsm8k

# 使用EvalScope
pip install evalscope
evalscope eval \
    --model model_name \
    --datasets mmlu ceval \
    --limit 100
""")

# 实践3: LLM-as-Judge
print("\n--- LLM-as-Judge评估 ---")

def llm_judge(question, answer_a, answer_b, criterion="helpfulness"):
    """使用LLM作为评判者"""
    prompt = f"""请比较以下两个回答，判断哪个更好。

评判标准: {criterion}

问题: {question}

回答A: {answer_a}

回答B: {answer_b}

请只输出 "A" 或 "B"，表示哪个回答更好。"""
    # 实际中调用LLM API
    return prompt

question = "什么是机器学习？"
answer_a = "机器学习是AI的一个分支。"
answer_b = "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进，而无需显式编程。"

prompt = llm_judge(question, answer_a, answer_b)
print("LLM-as-Judge Prompt:")
print(prompt)
```

## 总结

| 评估类型 | 工具 | 适用场景 |
|---------|------|---------|
| 知识 | MMLU, C-Eval | 通用知识 |
| 数学 | GSM8K, MATH | 推理能力 |
| 代码 | HumanEval, MBPP | 编程能力 |
| 对话 | MT-Bench, AlpacaEval | 对话质量 |
| 长文本 | LongBench, LVEval | 长上下文 |
| 安全 | ToxicChat, SafetyBench | 安全性 |

**下一步**: [Module 12: 部署与LLMOps](./../module-12-deployment/)
