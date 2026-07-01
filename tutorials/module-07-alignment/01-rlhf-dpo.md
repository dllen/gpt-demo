---
layout: page
title: "Module 07: 对齐技术 — RLHF与DPO"
---
# Module 07: 对齐技术 — RLHF与DPO

## 理论部分

### 7.1 为什么需要对齐？

预训练模型可能产生有害、不准确或不符合人类偏好的输出：

```
问题: "如何制作炸弹？"
未对齐模型: 可能给出详细步骤 ✗
对齐模型: "我无法提供有害信息" ✓
```

### 7.2 RLHF (Reinforcement Learning from Human Feedback)

RLHF三阶段流程：

```
阶段1: SFT (监督微调)
  预训练模型 → SFT数据 → SFT模型

阶段2: 训练奖励模型 (Reward Model)
  SFT模型生成多个回答 → 人类标注偏好 → 奖励模型

阶段3: PPO强化学习
  SFT模型 → 生成回答 → 奖励模型打分 → PPO更新策略
```

### 7.3 奖励模型

```
给定prompt x和两个回答 y_w (偏好) 和 y_l (非偏好):

L_reward = -log(σ(r(x, y_w) - r(x, y_l)))

直觉: 让偏好回答的得分高于非偏好回答
```

### 7.4 PPO (Proximal Policy Optimization)

```
目标函数:
L_PPO = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]

其中:
  r(θ) = π_θ(y|x) / π_θ_old(y|x)  (概率比)
  A = 优势函数 (奖励 - 基线)
  ε = 0.2 (裁剪范围)

额外: + β·D_KL(π_θ || π_ref)  (KL惩罚，防止偏离太远)
```

### 7.5 DPO (Direct Preference Optimization)

DPO绕过了奖励模型，直接从偏好数据优化策略：

```
L_DPO = -log(σ(β·(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x))))

优势:
- 不需要单独训练奖励模型
- 训练更简单稳定
- 效果与RLHF相当甚至更好
```

### 7.6 RLHF vs DPO 对比

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 复杂度 | 高 (4个模型) | 低 (2个模型) |
| 训练稳定性 | 较差 | 较好 |
| 显存需求 | 高 | 中 |
| 效果 | 好 | 相当或更好 |
| 实现难度 | 高 | 低 |

## 实践部分

### 实践1：奖励模型训练

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("实践1: 奖励模型训练")
print("=" * 60)

class RewardModel(nn.Module):
    """奖励模型: 在语言模型基础上添加奖励头"""
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        # 获取最后一层的隐藏状态
        outputs = self.base(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq, hidden)

        # 取最后一个token的隐藏状态
        batch_size = hidden.size(0)
        last_hidden = hidden[range(batch_size), -1]  # (batch, hidden)

        # 奖励分数
        reward = self.reward_head(last_hidden)  # (batch, 1)
        return reward.squeeze(-1)

def reward_loss(rewards_chosen, rewards_rejected, margin=0):
    """奖励模型损失: 偏好回答得分应更高"""
    # Bradley-Terry模型
    loss = -F.logsigmoid(rewards_chosen - rewards_rejected)
    return loss.mean()

# 模拟训练
print("模拟奖励模型训练:")
batch_size = 4
hidden_size = 256

# 模拟数据
rewards_chosen = torch.randn(batch_size)  # 偏好回答的奖励
rewards_rejected = torch.randn(batch_size)  # 非偏好回答的奖励

# 训练前
loss_before = reward_loss(rewards_chosen, rewards_rejected)
accuracy_before = (rewards_chosen > rewards_rejected).float().mean()
print(f"训练前 - 损失: {loss_before:.4f}, 准确率: {accuracy_before:.2%}")

# 模拟训练后 (偏好回答得分提高)
rewards_chosen = rewards_chosen + 2.0
rewards_rejected = rewards_rejected - 0.5
loss_after = reward_loss(rewards_chosen, rewards_rejected)
accuracy_after = (rewards_chosen > rewards_rejected).float().mean()
print(f"训练后 - 损失: {loss_after:.4f}, 准确率: {accuracy_after:.2%}")
```

### 实践2：DPO训练

```python
print("\n" + "=" * 60)
print("实践2: DPO (Direct Preference Optimization)")
print("=" * 60)

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    DPO损失函数

    Args:
        policy_chosen_logps: 策略模型对偏好回答的log概率
        policy_rejected_logps: 策略模型对非偏好回答的log概率
        reference_chosen_logps: 参考模型对偏好回答的log概率
        reference_rejected_logps: 参考模型对非偏好回答的log概率
        beta: KL惩罚系数
    """
    # 隐式奖励
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # DPO目标
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    # 监控指标
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    return loss, chosen_rewards.mean(), rejected_rewards.mean()

# 模拟DPO训练
print("模拟DPO训练过程:")
batch_size = 8

for step in range(5):
    # 模拟log概率 (训练过程中策略模型逐渐改善)
    progress = (step + 1) / 5

    # 策略模型: 逐渐学会偏好好的回答
    policy_chosen = -2.0 + progress * 1.5 + torch.randn(batch_size) * 0.3
    policy_rejected = -2.0 + progress * 0.3 + torch.randn(batch_size) * 0.3

    # 参考模型: 固定不变
    ref_chosen = -2.0 + torch.randn(batch_size) * 0.1
    ref_rejected = -2.0 + torch.randn(batch_size) * 0.1

    loss, chosen_r, rejected_r = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )

    accuracy = (policy_chosen > policy_rejected).float().mean()
    print(f"Step {step+1}: loss={loss:.4f}, "
          f"chosen_reward={chosen_r:.3f}, rejected_reward={rejected_r:.3f}, "
          f"准确率={accuracy:.1%}")

print("\n→ DPO使策略模型逐渐提高偏好回答的概率")
```

### 实践3：PPO简化实现

```python
print("\n" + "=" * 60)
print("实践3: PPO简化实现")
print("=" * 60)

class PPOTrainer:
    """简化的PPO训练器"""
    def __init__(self, policy, reference, lr=1e-5, clip_eps=0.2, kl_coef=0.1):
        self.policy = policy
        self.reference = reference
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef

    def compute_advantage(self, rewards, values):
        """计算优势函数 (简化版)"""
        # A = R - V (回报 - 基线)
        return rewards - values

    def ppo_loss(self, log_probs_old, log_probs_new, advantages):
        """PPO裁剪目标"""
        ratio = torch.exp(log_probs_new - log_probs_old)

        # 裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        # 取最小值 (悲观估计)
        return -torch.min(surr1, surr2).mean()

    def train_step(self, prompts, responses, rewards):
        """一步PPO训练"""
        # 计算新旧策略的log概率
        with torch.no_grad():
            ref_log_probs = self.reference(prompts, responses)
            old_log_probs = self.policy(prompts, responses)

        new_log_probs = self.policy(prompts, responses)

        # 优势 (简化: 直接用奖励)
        advantages = rewards

        # PPO损失
        policy_loss = self.ppo_loss(old_log_probs, new_log_probs, advantages)

        # KL惩罚
        kl_penalty = self.kl_coef * (new_log_probs - ref_log_probs).pow(2).mean()

        total_loss = policy_loss + kl_penalty

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'total_loss': total_loss.item(),
        }

print("PPO训练流程:")
print("1. 策略模型生成回答")
print("2. 奖励模型打分")
print("3. 计算优势函数")
print("4. PPO裁剪目标更新策略")
print("5. KL惩罚防止偏离参考模型太远")
```

## 总结

| 方法 | 阶段数 | 模型数 | 复杂度 | 推荐场景 |
|------|--------|--------|--------|---------|
| RLHF (PPO) | 3 | 4 | 高 | 追求极致效果 |
| DPO | 2 | 2 | 中 | 大多数场景 |
| ORPO | 1 | 1 | 低 | 简单高效 |

**下一步**: [Module 08: 推理优化](./../module-08-inference/)
