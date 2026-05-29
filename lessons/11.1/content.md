# 11.1 RLHF完整Pipeline实现

## 课程概述

本节课介绍RLHF（基于人类反馈的强化学习）的完整实现Pipeline。RLHF是ChatGPT、Claude等大模型对齐人类偏好的核心技术，整个流程包含三个阶段：监督微调（SFT）、奖励模型训练（RM）、以及基于PPO的强化学习微调。我们将深入讲解每个阶段的技术细节、实现方法、以及三个模型如何协同工作。通过本节课，你将理解从原始预训练模型到对齐模型的全流程，掌握RLHF的系统架构和工程实现要点。

## 学习目标

- 理解RLHF三阶段Pipeline的整体架构和数据流
- 掌握奖励模型的架构设计、训练方法和Bradley-Terry损失函数
- 理解PPO在RLHF中的特殊实现：优势估计、KL惩罚、经验回放
- 掌握多模型（Actor、Critic、Reference、Reward）协调管理的方法
- 能够诊断RLHF中的常见失败模式（如reward hacking）
- 理解RLHF效果的评估指标和方法

## 前置知识

- 理解语言模型预训练的基本概念
- 熟悉LoRA微调的基本原理（参考第7章内容）
- 了解强化学习基本概念：策略、奖励、价值函数
- 熟悉PPO算法的核心机制（参考第4章内容）
- 了解人类反馈数据的收集和标注方法

---

## 1. RLHF Pipeline概述

### 1.1 三阶段架构

RLHF的完整Pipeline包含三个顺序执行的阶段，每个阶段输出作为下一个阶段的输入：

```
RLHF完整Pipeline数据流：

阶段1: SFT (监督微调)
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  预训练模型      │ ──▶ │  有监督微调      │ ──▶ │   SFT模型       │
│  (Base LM)      │     │  (SFT Data)      │     │ (Supervised)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘

阶段2: Reward Model (奖励模型)
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  偏好数据        │     │  奖励模型训练    │     │   奖励模型       │
│ (Preference)    │ ──▶ │ (Bradley-Terry) │ ──▶ │   (RM)          │
└─────────────────┘     └─────────────────┘     └─────────────────┘

阶段3: PPO微调 (强化学习)
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  SFT模型         │ ──▶ │  PPO强化学习     │ ──▶ │   对齐模型       │
│ (Actor)          │ +RM │  (PPO+KL)       │     │ (Aligned LM)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**阶段一 SFT**：使用人类标注的问答对，对预训练模型进行监督微调，得到基础的对齐模型。这一阶段让模型学会按照指令生成回答。

**阶段二 Reward Model**：训练一个奖励模型，能够预测人类对回答的偏好。输入是(提示, 回答)对，输出是标量分数。

**阶段三 PPO微调**：使用PPO算法，利用奖励模型提供的奖励信号微调SFT模型，同时约束模型不能偏离初始SFT模型太远。

### 1.2 四模型架构

RLHF中实际涉及四个模型的协同工作：

| 模型 | 作用 | 参数更新 | 冻结状态 |
|-----|------|---------|---------|
| Actor (策略模型) | 生成回答的策略网络 | 是 | 否 |
| Critic (价值网络) | 估计长期价值 | 是 | 否 |
| Reference (参考模型) | 计算KL散度约束 | 否 | 是 |
| Reward (奖励模型) | 提供奖励信号 | 否 | 是 |

```
四模型交互流程：

          ┌──────────────────────────────────┐
          │        PPO强化学习阶段            │
          │                                  │
   Prompt ──▶ Actor ──▶ Response             │
                 │                           │
                 ▼                           │
          ┌─────────────┐                    │
          │  Reward     │◀── 提供奖励分数    │
          │  Model      │                    │
          └─────────────┘                    │
                 │                           │
                 ▼                           │
          ┌─────────────┐                    │
          │  Reference  │◀── KL约束计算      │
          │  Model      │                    │
          └─────────────┘                    │
                 │                           │
                 ▼                           │
          ┌─────────────┐                    │
          │   Critic    │◀── 价值估计         │
          │   Model     │                    │
          └─────────────┘                    │
                 │                           │
                 ▼                           │
          ┌─────────────┐                    │
          │  PPO Update │                    │
          └─────────────┘                    │
                 │                           │
                 └───────────────────────────┘
```

---

## 2. 阶段一：监督微调（SFT）

### 2.1 SFT的作用

SFT是RLHF的起点，其目标是将预训练模型（已完成大规模无监督训练）微调到具备基本指令遵循能力的状态。

```
SFT训练数据格式：
{
  "prompt": "解释什么是机器学习",
  "response": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测，而无需明确编程。"
}
```

SFT阶段使用交叉熵损失函数，让模型学习在给定prompt下生成高质量response的能力。

### 2.2 SFT与RLHF的关系

SFT模型作为PPO阶段中Actor和Reference的初始化基础：

- **Actor初始化**：SFT模型的权重作为Actor的起点
- **Reference初始化**：同一份权重复制给Reference（冻结）
- **Critic初始化**：从SFT模型初始化，但输出变为价值标量

这种设计确保了PPO训练开始时，Actor与Reference行为相似，KL散度约束是有意义的。

---

## 3. 阶段二：奖励模型训练

### 3.1 奖励模型架构

奖励模型基于语言模型改造，输出一个标量分数表示回答质量：

```python
# 奖励模型架构
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # 预训练语言模型
        self.value_head = nn.Linear(
            base_model.config.hidden_size, 
            1, 
            bias=False
        )  # 标量输出头
        
    def forward(self, input_ids, attention_mask):
        # 获取最后一个token的隐藏状态
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state[:, -1, :]  # [batch, hidden]
        # 输出奖励分数
        reward = self.value_head(last_hidden)  # [batch, 1]
        return reward.squeeze(-1)  # [batch]
```

关键设计点：
- **输入**：完整的(提示+回答)序列
- **输出**：单个标量分数（一个回答对应一个分数）
- **提取位置**：使用最后一个token的隐藏状态预测奖励

### 3.2 偏好数据与Bradley-Terry模型

奖励模型训练使用偏好数据（Preference Data），每条数据包含：

```json
{
  "prompt": "什么是量子计算？",
  "chosen": "量子计算是一种利用量子力学原理进行信息处理的计算方式...",
  "rejected": "量子计算就是量子力学在计算上的应用。"
}
```

偏好对生成过程：
1. 对同一个prompt，让模型生成多个候选回答
2. 人类标注者对这些回答排序
3. 将排序转换为pairwise偏好数据

**Bradley-Terry模型**假设人类对两个回答的偏好概率可以建模为：

$$
P(preferred | A, B) = \frac{exp(r(A))}{exp(r(A)) + exp(r(B))}
$$

其中 $r(A)$ 和 $r(B)$ 是两个回答的奖励分数。

### 3.3 奖励模型损失函数

奖励模型使用对比损失函数（Contrastive Loss）训练：

```python
def reward_model_loss(reward_chosen, reward_rejected):
    """
    Bradley-Terry风格的对比损失
    reward_chosen: 被选中的回答的奖励分数
    reward_rejected: 被拒绝的回答的奖励分数
    """
    # 偏好概率（sigmoid形式）
    prob = torch.sigmoid(reward_chosen - reward_rejected)
    # 负对数似然
    loss = -torch.log(prob + 1e-8)
    return loss.mean()
```

梯度推导：
- 如果 $r_{chosen} > r_{rejected}$，损失减小，模型学习到正确排序
- 如果 $r_{chosen} < r_{rejected}$，损失增大，梯度反向传播调整参数

### 3.4 奖励归一化

奖励分数的绝对值没有物理意义，重要的是相对差异。但在PPO中，奖励值会影响优势估计，因此需要对奖励进行归一化：

```python
def normalize_rewards(rewards):
    """在PPO训练中对奖励进行标准化"""
    mean = rewards.mean()
    std = rewards.std()
    # 防止除零
    std = torch.where(std > 0, std, torch.ones_like(std))
    return (rewards - mean) / std
```

训练时的归一化策略：
- 每个batch内归一化
- 记录滑动平均用于推理时的参考
- 避免极端reward值主导训练

---

## 4. 阶段三：PPO强化学习微调

### 4.1 PPO在RLHF中的特殊实现

标准PPO应用于游戏或机器人控制，而RLHF中的PPO有以下关键差异：

| 维度 | 标准PPO | RLHF-PPO |
|-----|--------|----------|
| 状态空间 | 低维向量（位置、速度） | 高维文本语义空间 |
| 动作空间 | 离散有限（上下左右） | 超大规模词表（50000+） |
| 奖励频率 | 每步有即时奖励 | 仅序列结束时才有奖励 |
| 序列长度 | 短（几十到几百步） | 长（几百到几千token） |

### 4.2 经验收集阶段

PPO训练采用**批量收集（Batch Collection）**策略，而非单步更新：

```python
def collect_experience(policy, prompts, reward_model, reference_model, 
                       mini_batch_size=8, generation_length=128):
    """
    收集PPO训练经验
    """
    all_data = []
    
    for prompt_batch in batched(prompts, mini_batch_size):
        # 1. 使用当前策略生成回答
        responses = policy.generate(
            input_ids=prompt_batch,
            max_new_tokens=generation_length
        )
        
        # 2. 计算奖励分数
        rewards = reward_model(prompt_batch, responses)
        
        # 3. 计算与参考模型的KL散度
        kl_penalty = compute_kl_divergence(
            policy, reference_model, prompt_batch, responses
        )
        
        # 4. 组装经验数据
        for i in range(len(prompt_batch)):
            all_data.append({
                'prompt': prompt_batch[i],
                'response': responses[i],
                'reward': rewards[i],
                'kl_penalty': kl_penalty[i]
            })
    
    return all_data
```

### 4.3 优势估计与奖励分配

RLHF中只有序列结束时的单一奖励，需要通过优势估计将奖励分配到每个token位置：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    GAE (Generalized Advantage Estimation) 计算优势
    rewards: [seq_len] 奖励序列
    values: [seq_len] 价值估计序列
    """
    advantages = []
    gae = 0
    
    # 从后向前计算
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # 最后一个位置，bootstrap用0
            next_value = 0
        else:
            next_value = values[t + 1]
        
        # TD误差
        delta = rewards[t] + gamma * next_value - values[t]
        # GAE累加
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages)
```

### 4.4 PPO裁剪目标函数

PPO的核心是裁剪后的策略目标函数：

```python
def ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    """
    PPO裁剪损失函数
    old_log_probs: 旧策略的动作log概率
    new_log_probs: 新策略的动作log概率
    advantages: 优势估计
    """
    # 策略比率
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 未裁剪目标
    surr1 = ratio * advantages
    
    # 裁剪目标
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surr2 = clipped_ratio * advantages
    
    # 取较小值（悲观下限）
    loss = -torch.min(surr1, surr2)
    
    return loss.mean()
```

### 4.5 参考模型KL惩罚

KL散度约束防止Actor过度偏离Reference，保持模型输出的多样性和安全性：

```python
def compute_kl_divergence(policy_model, reference_model, input_ids, response_ids):
    """
    计算响应序列上Actor和Reference的KL散度
    """
    # Reference模型（不计算梯度）
    with torch.no_grad():
        ref_logits = reference_model(input_ids, attention_mask)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    
    # Actor模型
    policy_logits = policy_model(input_ids, attention_mask)
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    
    # KL散度 = sum(p * log(p/q))
    kl = policy_log_probs - ref_log_probs  # [seq_len, vocab_size]
    
    # 只计算response部分的KL
    response_len = response_ids.size(1)
    response_kl = kl[:, -response_len:].sum(dim=-1)
    
    return response_kl
```

完整的PPO损失函数组合：

```python
def total_rlhf_loss(ppo_loss, kl_penalty, kl_coef=0.1):
    """
    RLHF完整损失 = PPO目标 + KL惩罚
    """
    return ppo_loss + kl_coef * kl_penalty
```

KL系数选择：
- **过大（>0.3）**：模型不敢偏离参考，训练效果差
- **过小（<0.01）**：模型可能过度拟合奖励，失去原有能力

---

## 5. Pipeline管理与优化

### 5.1 三阶段模型管理

RLHF三个阶段需要管理好模型检查点：

| 阶段 | 输入模型 | 输出模型 | 保存内容 |
|-----|---------|---------|---------|
| SFT | 预训练模型 | SFT模型 | 全量权重 |
| RM训练 | SFT模型 | 奖励模型 | 全量权重 |
| PPO | SFT + RM | 对齐模型 | 全量权重（Actor） |

检查点存储建议：
```
rlhf_checkpoints/
├── stage1_sft/
│   ├── checkpoint_1000/
│   └── checkpoint_2000/
├── stage2_rm/
│   ├── checkpoint_best/
│   └── checkpoint_final/
└── stage3_ppo/
    ├── checkpoint_1000/
    └── checkpoint_final/
```

### 5.2 显存管理

RLHF需要同时在显存中保存多个模型副本，必须进行优化：

| 优化技术 | 显存节省 | 实现方式 |
|---------|---------|---------|
| 梯度累积 | 4-8倍 | 虚拟增大batch size |
| 模型并行 | 线性扩展 | 张量并行、流水线并行 |
| 量化存储 | 50-75% | FP16/INT8压缩 |
| 梯度检查点 | ~30% | 用计算换显存 |

典型的7B模型RLHF显存分配：
- Actor（训练）：~14GB
- Reference（推理）：~7GB
- Reward（推理）：~7GB
- Critic（训练）：~14GB
- 激活值：~8GB

总计：约50GB，需要多卡或量化支持。

### 5.3 混合精度训练

使用BF16进行RLHF训练，防止数值溢出：

```python
# 混合精度配置
training_args = TrainingArguments(
    bf16=True,  # 使用BF16精度
    fp16=False,
    # 梯度缩放避免溢出
    gradient_checkpointing=True,
    # 梯度裁剪
    max_grad_norm=1.0,
)
```

---

## 6. 实践考量与故障排除

### 6.1 常见失败模式

**Reward Hacking（奖励黑客）**

表现：模型学会"欺骗"奖励模型，而非真正提升回答质量。

症状：
- 回答变长但质量下降
- 奖励模型分数持续上升，但人类评估下降
- 模型生成语法正确但语义空洞的内容

诊断方法：
1. 监控奖励模型与人类评估的一致性
2. 检查生成文本的困惑度（perplexity）
3. 观察KL散度是否突然下降（过度利用reward）

**模式崩溃（Mode Collapse）**

表现：模型输出变得单调，丧失多样性。

症状：
- 重复相同句式
- 词汇多样性下降
- KL散度接近0（与参考几乎相同）

### 6.2 KL散度监控

KL散度是RLHF健康度的关键指标：

```python
def monitor_training(kl_history, reward_history):
    """
    监控训练健康度
    """
    avg_kl = np.mean(kl_history[-100:])
    kl_std = np.std(kl_history[-100:])
    
    # 健康范围
    if avg_kl < 2.0:
        print("⚠️ KL过低，模型可能未充分学习")
    elif avg_kl > 10.0:
        print("⚠️ KL过高，模型偏离参考太远")
    else:
        print("✅ KL处于健康范围")
    
    return avg_kl, kl_std
```

KL异常的处理策略：
- **KL过高**：增大KL系数，或减少PPO更新幅度
- **KL过低**：减小KL系数，或增加探索

### 6.3 训练稳定性

RLHF训练不稳定的主要原因：
1. 奖励信号稀疏
2. 策略更新幅度过大
3. 价值函数估计不准确

稳定化技术：
- **奖励缩放**：将奖励标准化到合理范围
- **价值裁剪**：限制Critic的预测范围
- **早停机制**：KL超过阈值时停止更新

---

## 7. 评估方法

### 7.1 自动评估指标

| 指标 | 计算方式 | 说明 |
|-----|---------|------|
| Win Rate | 对比基线模型的胜率 | 需要参考模型或人工标注 |
| Reward Mean | 平均奖励分数 | 监控训练进度 |
| KL Divergence | 与参考模型的KL | 监控策略偏移 |
| Response Length | 生成长度分布 | 检测模式崩溃 |

### 7.2 人类评估

最可靠的RLHF评估方法是人类评估：

1. **成对比较**：给定同一prompt的两个回答，标注者选择更好的
2. **评分量表**：对回答的多个维度（有用性、安全性、准确性）打分
3. **红队测试**：专门测试模型的安全性漏洞

### 7.3 段阶式评估Pipeline

```python
def evaluate_rlhf_model(model, test_prompts, human_eval_samples=100):
    """
    完整RLHF评估流程
    """
    # 1. 自动指标
    auto_metrics = compute_auto_metrics(model, test_prompts)
    
    # 2. 奖励模型评估
    rm_scores = reward_model.evaluate(test_prompts)
    
    # 3. 采样用于人类评估
    samples = model.sample(test_prompts[:human_eval_samples])
    
    # 4. 计算综合得分
    final_score = (
        0.3 * auto_metrics['win_rate'] + 
        0.3 * rm_scores['mean'] +
        0.4 * human_eval_score  # 需要人工标注
    )
    
    return final_score, {
        'auto': auto_metrics,
        'reward_model': rm_scores,
        'samples': samples
    }
```

---

## 8. 总结

### 8.1 核心要点

- **RLHF三阶段**：SFT → Reward Model → PPO微调，每阶段承担不同角色
- **奖励模型**：基于语言模型改造，使用Bradley-Terry损失学习偏好关系
- **PPO特殊性**：序列级奖励分配、KL约束、与参考模型对比
- **多模型协调**：Actor/Critic训练，Reference/Reward冻结

### 8.2 工程要点

- 检查点管理：每个阶段完成后保存模型，用于下一阶段初始化
- 显存优化：梯度累积、量化、模型并行是必需技术
- 监控指标：KL散度、奖励均值、响应长度联合监控

### 8.3 进阶方向

- DPO（Direct Preference Optimization）：绕过PPO的直接偏好优化
- GRPO（Group Relative Policy Optimization）：无参考模型的群体相对优化
- 过程奖励模型：每步而非仅序列结束提供奖励信号

---

## 延伸阅读

1. **InstructGPT论文**：Training language models to follow instructions with human feedback（OpenAI，2022）
2. **RLHF综述**：A Comprehensive Overview of Reinforcement Learning from Human Feedback (RLHF)
3. **PPO论文**：Proximal Policy Optimization Algorithms（Schulman et al., 2017）
4. **DeepSpeed Chat**：https://github.com/microsoft/DeepSpeed
5. **TRLX框架**：HuggingFace的RLHF实现库

---

## 复习题

1. **问题一**：解释RLHF中四个模型（Actor、Critic、Reference、Reward）的各自作用，以及它们在PPO更新阶段如何交互。

2. **问题二**：奖励模型的Bradley-Terry损失函数如何工作？如果被选中的回答得分低于被拒绝的回答，梯度会如何调整？

3. **问题三**：为什么RLHF中需要对奖励进行归一化？假设没有归一化，可能出现什么问题？

4. **问题四**：你观察到KL散度在训练过程中持续下降到接近0，但奖励分数还在上升。这表明发生了什么？你会如何调整训练参数？