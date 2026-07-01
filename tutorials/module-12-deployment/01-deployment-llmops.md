---
layout: page
title: "Module 12: 部署与LLMOps"
---
# Module 12: 部署与LLMOps

## 理论部分

### 12.1 部署架构

```
┌─────────────────────────────────────────────────┐
│                 生产部署架构                       │
│                                                  │
│  用户 → [负载均衡] → [推理服务集群] → [模型]      │
│              ↓              ↓          ↓         │
│         [监控]          [缓存]    [GPU集群]      │
│              ↓              ↓                   │
│         [日志]          [限流]                   │
└─────────────────────────────────────────────────┘
```

### 12.2 推理框架对比

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| vLLM | PagedAttention, 高吞吐 | 生产部署 |
| SGLang | RadixAttention, 批量推理 | 生产部署 |
| TGI | 简单易用 | 快速部署 |
| TensorRT-LLM | NVIDIA优化 | NVIDIA GPU |
| LMDeploy | 全链路优化 | 端侧/云端 |
| Ollama | 极简部署 | 本地开发 |

### 12.3 服务化关键指标

| 指标 | 说明 | 目标 |
|------|------|------|
| TTFT | Time to First Token | < 500ms |
| TPOT | Time Per Output Token | < 50ms |
| Throughput | tokens/second | 越高越好 |
| Latency | 端到端延迟 | < 2s |
| Availability | 可用性 | > 99.9% |

### 12.4 部署策略

```
单机部署:
  单GPU → 单实例 → 适合开发/小流量

多机部署:
  多GPU → 多实例 + 负载均衡 → 适合生产

Serverless:
  按需启动 → 冷启动问题 → 适合低频调用

边缘部署:
  量化模型 → 端侧推理 → 适合隐私场景
```

### 12.5 LLMOps流水线

```
开发 → 测试 → 训练 → 评估 → 部署 → 监控
  ↑                                    │
  └──────────── 反馈循环 ──────────────┘
```

## 实践部分

### 实践1：模型服务化

```python
print("=" * 60)
print("实践1: 模型服务化")
print("=" * 60)

print("""
# 使用vLLM部署
pip install vllm

# 方式1: 命令行启动
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000 \
    --gpu-memory-utilization 0.9

# 方式2: Python代码
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["Hello, how are you?"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)

# 调用API (兼容OpenAI格式)
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
""")

# 实践2: FastAPI服务
print("\n--- FastAPI模型服务 ---")

fastapi_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI(title="LLM Inference API")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = False

class ChatResponse(BaseModel):
    content: str
    usage: dict
    latency_ms: float

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    start = time.time()
    try:
        # 这里调用模型推理
        response_text = "模型生成的回答..."
        latency = (time.time() - start) * 1000

        return ChatResponse(
            content=response_text,
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}
'''
print(fastapi_code)
```

### 实践2：监控与日志

```python
print("\n" + "=" * 60)
print("实践2: 监控与日志")
print("=" * 60)

print("""
# Prometheus + Grafana 监控

# 关键指标:
# - inference_latency_seconds (推理延迟)
# - tokens_generated_total (生成token数)
# - requests_total (请求总数)
# - gpu_utilization (GPU利用率)
# - gpu_memory_used (显存使用)

# 使用Prometheus客户端
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('llm_requests_total', 'Total requests')
LATENCY = Histogram('llm_latency_seconds', 'Request latency')
TOKENS = Counter('llm_tokens_total', 'Total tokens generated')
GPU_MEMORY = Gauge('gpu_memory_used_bytes', 'GPU memory usage')

# 记录指标
REQUEST_COUNT.inc()
with LATENCY.time():
    # 推理...
    TOKENS.inc(num_tokens)
GPU_MEMORY.set(current_memory)
""")

# 实践3: Docker部署
print("\n--- Docker部署 ---")

dockerfile = """
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install torch vllm fastapi uvicorn

COPY model/ /app/model/
COPY serve.py /app/

WORKDIR /app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
"""

print("Dockerfile:")
print(dockerfile)

# 实践4: 模型版本管理
print("\n--- 模型版本管理 ---")

print("""
# MLflow / Weights & Biases 管理模型版本

import mlflow

with mlflow.start_run():
    # 记录参数
    mlflow.log_params({
        "model": "llama-2-7b",
        "lr": 2e-4,
        "lora_rank": 16,
        "epochs": 3
    })

    # 记录指标
    mlflow.log_metrics({
        "eval_loss": 0.45,
        "perplexity": 12.3,
        "accuracy": 0.87
    })

    # 保存模型
    mlflow.pytorch.log_model(model, "model")

    # 注册模型
    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "llama-2-7b-finetuned"
    )
""")
```

### 实践3：CI/CD流水线

```python
print("\n" + "=" * 60)
print("实践3: CI/CD流水线")
print("=" * 60)

print("""
# GitHub Actions 示例
# .github/workflows/llm-pipeline.yml

name: LLM Training Pipeline

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # 每周运行

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  train:
    needs: test
    runs-on: gpu-runner
    steps:
      - name: Train model
        run: python train.py --config configs/default.yaml
      - name: Evaluate
        run: python eval.py --model outputs/model
      - name: Check quality gate
        run: |
          python check_metrics.py --threshold accuracy=0.8

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Build Docker image
        run: docker build -t llm-service:${{ github.sha }} .
      - name: Push to registry
        run: docker push registry/llm-service:${{ github.sha }}
      - name: Deploy to staging
        run: kubectl set image deployment/llm-service *=registry/llm-service:${{ github.sha }}
      - name: Integration tests
        run: pytest tests/integration/
      - name: Deploy to production
        run: kubectl rollout status deployment/llm-service
""")
```

## 总结

| 方面 | 工具/方法 | 推荐 |
|------|---------|------|
| 推理框架 | vLLM / SGLang | vLLM |
| 服务化 | FastAPI + Docker | 标准方案 |
| 监控 | Prometheus + Grafana | 标准方案 |
| 模型管理 | MLflow / W&B | MLflow |
| CI/CD | GitHub Actions | 标准方案 |
| 编排 | Kubernetes | 生产必需 |

---

## 系列总结

恭喜你完成了大模型全栈教程系列！回顾我们学到的：

```
Module 01: 数学基础 → 线性代数、概率论、优化
Module 02: Transformer → Self-Attention、多头注意力
Module 03: 分词 → BPE、SentencePiece
Module 04: 模型架构 → LLaMA-style完整实现
Module 05: 预训练 → 训练循环、分布式训练
Module 06: 微调 → LoRA、QLoRA
Module 07: 对齐 → RLHF、DPO
Module 08: 推理优化 → KV Cache、量化、推测解码
Module 09: RAG → 向量检索、检索增强生成
Module 10: Agent → ReAct、工具调用
Module 11: 评估 → 基准测试、指标
Module 12: 部署 → 服务化、监控、CI/CD
```

**继续学习**:
- 阅读论文: Attention Is All You Need, LLaMA, InstructGPT
- 参与开源: nanoGPT, tiny-universe
- 实践项目: 从零训练一个Tiny LLM
