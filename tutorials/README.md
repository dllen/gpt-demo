# 大模型全栈教程系列

> 从理论到实践，构建你自己的大语言模型系统

## 系列简介

本教程系列基于以下优秀开源项目，系统性地讲解大语言模型全栈技术：

| 参考项目 | 说明 |
|---------|------|
| [tiny-universe](https://github.com/datawhalechina/tiny-universe) | 大模型白盒子构建指南 - 全手搓Tiny-Universe |
| [code-your-own-llm](https://github.com/datawhalechina/code-your-own-llm) | 全栈式大语言模型参考指南 |
| [llm-action](https://github.com/liguodongiot/llm-action) | 大模型技术原理与实战经验 |
| [llm-resource](https://github.com/liguodongiot/llm-resource) | LLM全栈优质资源汇总 |
| [how-to-train-your-gpt](https://github.com/raiyanyahya/how-to-train-your-gpt) | 从零构建现代LLM |
| [llm_wiki](https://github.com/nashsu/llm_wiki) | LLM驱动的个人知识库 |

## 学习路径

```
┌─────────────────────────────────────────────────────────────┐
│                   大模型全栈技术栈                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Module 01 ─── 数学基础与预备知识                              │
│       │                                                     │
│  Module 02 ─── Transformer架构深入                            │
│       │                                                     │
│  Module 03 ─── 分词与词表构建                                  │
│       │                                                     │
│  Module 04 ─── 模型架构实现 (GPT/LLaMA)                       │
│       │                                                     │
│  Module 05 ─── 预训练 (Pre-training)                         │
│       │                                                     │
│  Module 06 ─── 微调技术 (SFT/LoRA/QLoRA)                     │
│       │                                                     │
│  Module 07 ─── 对齐技术 (RLHF/DPO)                           │
│       │                                                     │
│  Module 08 ─── 推理优化 (KV Cache/量化/vLLM)                  │
│       │                                                     │
│  Module 09 ─── RAG系统 (检索增强生成)                          │
│       │                                                     │
│  Module 10 ─── Agent系统 (智能体)                             │
│       │                                                     │
│  Module 11 ─── 模型评估                                      │
│       │                                                     │
│  Module 12 ─── 部署与LLMOps                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 模块列表

| 模块 | 主题 | 理论 | 实践 | 难度 |
|------|------|------|------|------|
| [Module 01](./module-01-math-foundations/) | 数学基础与预备知识 | 线性代数、微积分、概率统计、优化 | NumPy/PyTorch基础 | ⭐ |
| [Module 02](./module-02-transformer/) | Transformer架构深入 | Self-Attention、多头注意力 | 手写Transformer | ⭐⭐ |
| [Module 03](./module-03-tokenization/) | 分词与词表构建 | BPE、SentencePiece | 训练自己的Tokenizer | ⭐⭐ |
| [Module 04](./module-04-model-architecture/) | 模型架构实现 | RoPE、RMSNorm、SwiGLU、GQA | 实现LLaMA-style模型 | ⭐⭐⭐ |
| [Module 05](./module-05-pretraining/) | 预训练 | 数据工程、分布式训练、优化器 | 预训练Tiny模型 | ⭐⭐⭐ |
| [Module 06](./module-06-finetuning/) | 微调技术 | SFT、LoRA、QLoRA原理 | 微调实践 | ⭐⭐⭐ |
| [Module 07](./module-07-alignment/) | 对齐技术 | RLHF、DPO、PPO | 偏好对齐实践 | ⭐⭐⭐⭐ |
| [Module 08](./module-08-inference/) | 推理优化 | KV Cache、量化、推测解码 | 推理引擎实现 | ⭐⭐⭐ |
| [Module 09](./module-09-rag/) | RAG系统 | 向量检索、嵌入模型 | 搭建RAG系统 | ⭐⭐⭐ |
| [Module 10](./module-10-agents/) | Agent系统 | ReAct、工具调用、规划 | 构建Agent | ⭐⭐⭐⭐ |
| [Module 11](./module-11-evaluation/) | 模型评估 | 评测指标、基准测试 | 评估框架使用 | ⭐⭐ |
| [Module 12](./module-12-deployment/) | 部署与LLMOps | 服务化、监控、CI/CD | 模型部署实践 | ⭐⭐⭐ |

## Module 01 详细结构

Module 01 专为**后端程序员**设计，用你已有的工程直觉快速理解ML数学：

| 文件 | 内容 | 适合人群 |
|------|------|---------|
| [00-backend-dev-crash-course.md](./module-01-math-foundations/00-backend-dev-crash-course.md) | 后端概念→ML概念映射，张量的后端视角 | 所有后端开发者 |
| [01-linear-algebra.md](./module-01-math-foundations/01-linear-algebra.md) | 向量、矩阵、点积、Softmax | 需要线性代数基础 |
| [02-probability-optimization.md](./module-01-math-foundations/02-probability-optimization.md) | 概率论、信息论、优化基础 | 需要概率论基础 |
| [03-calculus-derivatives.md](./module-01-math-foundations/03-calculus-derivatives.md) | 导数、梯度、链式法则、反向传播 | 理解训练过程必读 |
| [04-statistics-basics.md](./module-01-math-foundations/04-statistics-basics.md) | 描述统计、分布、协方差、大数定律 | 数据分析和评估 |
| [05-math-cheat-sheet.md](./module-01-math-foundations/05-math-cheat-sheet.md) | 一页纸公式速查手册 | 随时查阅 |

**推荐学习顺序**：`00 → 03 → 01 → 02 → 04 → 05(速查)`

## 学习建议

1. **顺序学习**：建议按模块顺序学习，每个模块都以前置模块为基础
2. **理论+实践**：每个模块包含理论讲解和动手实践，建议先理解理论再写代码
3. **代码复现**：鼓励从零实现每个组件，而非直接调用高级API
4. **实验记录**：建议记录每次实验的参数、结果和分析

## 环境要求

```bash
# 基础环境
Python >= 3.10
PyTorch >= 2.0
CUDA >= 11.8 (GPU训练)

# 推荐工具
pip install torch transformers datasets tokenizers
pip install numpy matplotlib tqdm
pip install accelerate peft bitsandbytes
```

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - LLaMA论文
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) - GPT-4技术报告
- [InstructGPT](https://arxiv.org/abs/2203.02155) - RLHF经典论文
