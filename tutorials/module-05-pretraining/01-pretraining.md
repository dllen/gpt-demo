# Module 05: 预训练 (Pre-training)

## 理论部分

### 5.1 预训练的目标

预训练是让模型从海量文本中学习语言的一般性知识：

```
目标函数: L = -Σ log P(wₜ | w₁, w₂, ..., wₜ₋₁)

即: 给定前面的词，最大化预测下一个词的概率
```

### 5.2 预训练数据

| 数据集 | 规模 | 语言 | 特点 |
|--------|------|------|------|
| Common Crawl | 数TB | 多语言 | 网页爬取，需大量清洗 |
| C4 | 800GB | 英文 | Common Crawl清洗版 |
| The Pile | 800GB | 英文 | 多领域混合 |
| RedPajama | 数TB | 英文 | 开源复现LLaMA |
| 中文维基 | 数GB | 中文 | 高质量但规模小 |
| 悟道 | 数TB | 中文 | 中文大规模 |

### 5.3 数据处理流水线

```
原始数据 → 去重 → 质量过滤 → 语言识别 → PII去除 → 分词 → 训练
```

**关键步骤**：
1. **去重**: 去除重复文档（MinHash/SimHash）
2. **质量过滤**: 去除低质量内容（长度、符号比例、困惑度）
3. **PII去除**: 去除个人身份信息

### 5.4 训练配置

```
模型: d_model=4096, layers=32, heads=32 (7B级别)
数据: 1-2万亿 tokens
批次: batch_size=4M tokens (约1024个4K长度的序列)
学习率: 3e-4 with cosine decay, warmup 2000步
优化器: AdamW (β1=0.9, β2=0.95, wd=0.1)
硬件: 数百到数千张A100/H100
时间: 数周到数月
```

### 5.5 显存优化技术

| 技术 | 节省 | 代价 |
|------|------|------|
| 混合精度 (BF16) | 50% | 精度损失可忽略 |
| 梯度检查点 | 60-70% | 30%训练时间 |
| 梯度累积 | 线性 | 训练时间 |
| Flash Attention | 20-40% | 无 |
| DeepSpeed ZeRO | 线性(参数分片) | 通信开销 |

### 5.6 分布式训练策略

```
数据并行 (DP): 每卡完整模型，数据分片
    → 简单，但每卡需要容纳完整模型

张量并行 (TP): 每层切分到多卡
    → 需要高速互联 (NVLink)

流水线并行 (PP): 不同层放不同卡
    → 有气泡(bubble)开销

ZeRO (Zero Redundancy Optimizer):
    Stage 1: 优化器状态分片
    Stage 2: + 梯度分片
    Stage 3: + 参数分片
```

## 实践部分

### 实践1：训练循环实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

print("=" * 60)
print("实践1: 完整的预训练训练循环")
print("=" * 60)

class TextDataset(Dataset):
    """简单的文本数据集"""
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class WarmupCosineScheduler:
    """Warmup + Cosine Decay 学习率调度"""
    def __init__(self, optimizer, warmup_steps, total_steps, lr_max, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.lr_max * self.current_step / self.warmup_steps
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))

def train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum_steps=1, max_grad_norm=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # 前向传播
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        loss = loss / grad_accum_steps

        # 反向传播
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * y.numel()
        total_tokens += y.numel()

        if step % 100 == 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(min(avg_loss, 100))
            lr = scheduler.get_lr()
            print(f"  Step {step}: loss={avg_loss:.4f}, ppl={perplexity:.2f}, lr={lr:.2e}")

    return total_loss / total_tokens

# 创建模拟数据
vocab_size = 1000
d_model = 128
num_heads = 4
num_layers = 4
seq_len = 64

# 模拟token数据
num_tokens = 10000
tokens = torch.randint(0, vocab_size, (num_tokens,)).tolist()
dataset = TextDataset(tokens, seq_len)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 创建模型 (使用Module 04的简化版)
class SimpleGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model*4, batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, x):
        h = self.embed(x) * math.sqrt(d_model)
        h = self.transformer(h, is_causal=True)
        return self.lm_head(self.norm(h))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleGPT().to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"设备: {device}")

# 训练配置
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
total_steps = len(dataloader) * 2  # 2个epoch
scheduler = WarmupCosineScheduler(optimizer, warmup_steps=50, total_steps=total_steps, lr_max=3e-4)

# 训练
print("\n--- 开始训练 ---")
for epoch in range(2):
    print(f"\nEpoch {epoch + 1}:")
    avg_loss = train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum_steps=2)
    print(f"Epoch {epoch+1} 完成, 平均loss: {avg_loss:.4f}, 困惑度: {math.exp(min(avg_loss, 100)):.2f}")
```

### 实践2：混合精度训练

```python
print("\n" + "=" * 60)
print("实践2: 混合精度训练 (AMP)")
print("=" * 60)

try:
    from torch.cuda.amp import autocast, GradScaler

    def train_step_amp(model, x, y, optimizer, scaler):
        """混合精度训练步骤"""
        optimizer.zero_grad()

        with autocast():  # 自动混合精度
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        scaler.scale(loss).backward()      # 缩放损失
        scaler.unscale_(optimizer)          # 反缩放梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)              # 更新参数
        scaler.update()                     # 调整缩放因子

        return loss.item()

    print("混合精度训练可用 (需要CUDA)")
    print("优势: 2x速度, 50%显存, 精度损失可忽略")

except ImportError:
    print("AMP需要CUDA环境")

# 实践3: 梯度累积
print("\n--- 梯度累积 ---")
print("当GPU放不下大batch时，使用梯度累积模拟大batch")
print("effective_batch = micro_batch * grad_accum_steps * num_gpus")
print("例如: 8 * 4 * 8 = 256 (等效batch size)")
```

### 实践3：数据加载优化

```python
print("\n" + "=" * 60)
print("实践3: 高效数据加载")
print("=" * 60)

class MemoryMappedDataset(Dataset):
    """内存映射数据集 - 处理大规模数据"""
    def __init__(self, data_path, seq_len):
        self.seq_len = seq_len
        # 实际项目中使用 numpy.memmap 或类似方法
        # 这里用模拟数据
        self.data = torch.randint(0, 10000, (1000000,))
        self.length = len(self.data) - seq_len - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y

# DataLoader最佳实践
def create_dataloader(dataset, batch_size, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,      # 多进程加载
        pin_memory=True,              # 加速CPU→GPU传输
        prefetch_factor=2,            # 预取数据
        persistent_workers=True,      # 保持worker进程
        drop_last=True,               # 丢弃不完整的最后一批
    )

print("DataLoader最佳实践:")
print("  num_workers: 4-8 (根据CPU核心数)")
print("  pin_memory: True (加速CPU→GPU)")
print("  prefetch_factor: 2-4")
print("  persistent_workers: True")
```

## 总结

| 技术 | 作用 | 推荐设置 |
|------|------|---------|
| 混合精度 | 加速+省显存 | BF16 |
| 梯度累积 | 模拟大batch | 根据显存调整 |
| 梯度裁剪 | 训练稳定 | max_norm=1.0 |
| Flash Attention | 加速+省显存 | 默认开启 |
| Warmup+Cosine | 学习率调度 | warmup=总步数1% |
| AdamW | 优化器 | β=(0.9,0.95), wd=0.1 |

**下一步**: [Module 06: 微调技术](./../module-06-finetuning/)
