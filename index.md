---
layout: default
---

<div class="home">
  <header class="hero">
    <h1>{{ site.title }}</h1>
    <p class="tagline">{{ site.description }}</p>
    <div class="cta-buttons">
      <a href="{{ '/tutorials/README' | relative_url }}" class="btn btn-primary">开始学习</a>
      <a href="https://github.com/dllen/gpt-demo" class="btn btn-secondary" target="_blank">GitHub</a>
    </div>
  </header>

  <section class="features">
    <h2>教程特色</h2>
    <div class="feature-grid">
      <div class="feature-card">
        <h3>🎯 面向后端开发者</h3>
        <p>用你熟悉的工程概念类比ML知识，快速建立直觉</p>
      </div>
      <div class="feature-card">
        <h3>📐 理论+实践</h3>
        <p>每个模块包含数学原理和可运行的Python/NumPy代码</p>
      </div>
      <div class="feature-card">
        <h3>🔧 从零实现</h3>
        <p>不依赖高级API，手写Transformer、LoRA、RAG等核心组件</p>
      </div>
      <div class="feature-card">
        <h3>📊 全栈覆盖</h3>
        <p>从数学基础到生产部署，12个模块覆盖LLM完整技术栈</p>
      </div>
    </div>
  </section>

  <section class="modules">
    <h2>学习模块</h2>
    <div class="module-grid">
      <a href="{{ '/tutorials/module-01-math-foundations/' | relative_url }}" class="module-card">
        <span class="module-num">01</span>
        <h4>数学基础</h4>
        <p>线代、微积分、概率统计、NumPy实战</p>
        <span class="difficulty">⭐</span>
      </a>
      <a href="{{ '/tutorials/module-02-transformer/' | relative_url }}" class="module-card">
        <span class="module-num">02</span>
        <h4>Transformer</h4>
        <p>Self-Attention、多头注意力、RoPE</p>
        <span class="difficulty">⭐⭐</span>
      </a>
      <a href="{{ '/tutorials/module-03-tokenization/' | relative_url }}" class="module-card">
        <span class="module-num">03</span>
        <h4>分词</h4>
        <p>BPE算法、从零实现Tokenizer</p>
        <span class="difficulty">⭐⭐</span>
      </a>
      <a href="{{ '/tutorials/module-04-model-architecture/' | relative_url }}" class="module-card">
        <span class="module-num">04</span>
        <h4>模型架构</h4>
        <p>LLaMA-style完整实现</p>
        <span class="difficulty">⭐⭐⭐</span>
      </a>
      <a href="{{ '/tutorials/module-05-pretraining/' | relative_url }}" class="module-card">
        <span class="module-num">05</span>
        <h4>预训练</h4>
        <p>训练循环、分布式、优化技术</p>
        <span class="difficulty">⭐⭐⭐</span>
      </a>
      <a href="{{ '/tutorials/module-06-finetuning/' | relative_url }}" class="module-card">
        <span class="module-num">06</span>
        <h4>微调</h4>
        <p>LoRA、QLoRA、PEFT</p>
        <span class="difficulty">⭐⭐⭐</span>
      </a>
      <a href="{{ '/tutorials/module-07-alignment/' | relative_url }}" class="module-card">
        <span class="module-num">07</span>
        <h4>对齐</h4>
        <p>RLHF、DPO、PPO</p>
        <span class="difficulty">⭐⭐⭐⭐</span>
      </a>
      <a href="{{ '/tutorials/module-08-inference/' | relative_url }}" class="module-card">
        <span class="module-num">08</span>
        <h4>推理优化</h4>
        <p>KV Cache、量化、推测解码</p>
        <span class="difficulty">⭐⭐⭐</span>
      </a>
      <a href="{{ '/tutorials/module-09-rag/' | relative_url }}" class="module-card">
        <span class="module-num">09</span>
        <h4>RAG</h4>
        <p>向量检索、检索增强生成</p>
        <span class="difficulty">⭐⭐⭐</span>
      </a>
      <a href="{{ '/tutorials/module-10-agents/' | relative_url }}" class="module-card">
        <span class="module-num">10</span>
        <h4>Agent</h4>
        <p>ReAct、工具调用、多Agent</p>
        <span class="difficulty">⭐⭐⭐⭐</span>
      </a>
      <a href="{{ '/tutorials/module-11-evaluation/' | relative_url }}" class="module-card">
        <span class="module-num">11</span>
        <h4>评估</h4>
        <p>基准测试、指标、框架</p>
        <span class="difficulty">⭐⭐</span>
      </a>
      <a href="{{ '/tutorials/module-12-deployment/' | relative_url }}" class="module-card">
        <span class="module-num">12</span>
        <h4>部署</h4>
        <p>服务化、监控、CI/CD</p>
        <span class="difficulty">⭐⭐⭐</span>
      </a>
    </div>
  </section>

  <section class="references">
    <h2>参考项目</h2>
    <ul class="ref-list">
      <li><a href="https://github.com/datawhalechina/tiny-universe" target="_blank">tiny-universe</a> — 大模型白盒子构建指南</li>
      <li><a href="https://github.com/datawhalechina/code-your-own-llm" target="_blank">code-your-own-llm</a> — 全栈式大语言模型参考指南</li>
      <li><a href="https://github.com/liguodongiot/llm-action" target="_blank">llm-action</a> — 大模型技术原理与实战经验</li>
      <li><a href="https://github.com/liguodongiot/llm-resource" target="_blank">llm-resource</a> — LLM全栈优质资源汇总</li>
      <li><a href="https://github.com/raiyanyahya/how-to-train-your-gpt" target="_blank">how-to-train-your-gpt</a> — 从零构建现代LLM</li>
      <li><a href="https://github.com/nashsu/llm_wiki" target="_blank">llm_wiki</a> — LLM驱动的个人知识库</li>
    </ul>
  </section>
</div>
