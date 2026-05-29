# 3.4 PPO代码实现：CartPole/Atari游戏

## 课程概述

本课时是PPO算法的实践篇，我们将从零开始实现完整的PPO算法，并在CartPole环境中训练验证。课程首先梳理PPO与上节课Actor-Critic架构的联系，然后逐步实现策略网络、价值网络、GAE优势估计和PPO裁剪损失函数。通过在CartPole上的训练实验，观察PPO如何实现稳定的策略更新，并分析关键超参数的影响。最后，我们讨论如何将这一框架从游戏环境迁移到大语言模型的RLHF场景。

**学习目标**
- 掌握PPO算法的完整实现流程，理解各个模块的数学原理
- 从零实现Actor-Critic双网络架构，包括前向传播和梯度更新
- 理解GAE（广义优势估计）的实现及其在偏差-方差权衡中的作用
- 掌握PPO裁剪损失函数的实现，理解min操作和clip操作的协同机制
- 通过CartPole训练实验，观察PPO的收敛特性和关键超参数的影响
- 理解从游戏环境到LLM RLHF的场景迁移，包括状态表示、动作空间和奖励设计的差异

**前置知识**：Actor-Critic架构、GAE优势估计、PyTorch基础、NumPy基础

---

## 1. PPO与Actor-Critic的回顾

### 1.1 算法框架联系

回顾上一章的Actor-Critic架构，PPO是其改进版本，引入了两个关键机制来解决原始Actor-Critic的稳定性问题：

**原始Actor-Critic的问题**：策略更新幅度难以控制，可能导致训练崩溃。

**PPO的改进**：
1. **重要性采样**：允许使用旧策略收集的数据多次更新，提高数据效率
2. **裁剪机制**：限制策略比率的变化幅度，确保每次更新不会偏离旧策略太远

```
┌─────────────────────────────────────────────────────────┐
│                    PPO 算法框架                          │
├─────────────────────────────────────────────────────────┤
│  1. 数据收集阶段                                          │
│     使用当前策略 π_θ_old 与环境交互                       │
│     收集 (s_t, a_t, r_t, s_{t+1}, done_t) 序列           │
│                                                         │
│  2. 优势估计阶段                                          │
│     使用价值网络 V_ω 估计状态价值                         │
│     计算 GAE(γ, λ) 优势估计 Â_t                           │
│                                                         │
│  3. 策略更新阶段（多轮更新）                               │
│     计算策略比率 r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)│
│     应用裁剪损失：L^{CLIP}(θ) = E[min(r_t(θ)Â_t, clip(...))]│
│     更新 Actor 网络参数 θ                                 │
│     更新 Critic 网络参数 ω                                │
└─────────────────────────────────────────────────────────┘
```

### 1.2 PPO损失函数详解

PPO的核心损失函数结合了重要性采样和裁剪机制：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

**min操作的物理含义**：取未裁剪目标和裁剪目标的较小值，确保损失不会超过裁剪目标。这意味着只有当策略比率在允许范围内时，才允许优势函数充分发挥作用；一旦超出范围，损失被裁剪，不再增加。

**clip操作的物理含义**：将策略比率限制在 [1-ε, 1+ε] 范围内，防止策略发生剧烈变化。通常 ε = 0.2，意味着策略概率最多变化20%。

---

## 2. 网络架构实现

### 2.1 策略网络（Actor）

策略网络接收状态，输出动作的概率分布。在CartPole中，状态是4维向量，动作是2个离散动作（向左/向右推）。

```python
class Actor(nn.Module):
    """策略网络：输入状态，输出动作概率分布"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        """
        Args:
            state: 状态张量，形状 (batch_size, state_dim) 或 (state_dim,)
        Returns:
            probs: 动作概率分布，形状 (batch_size, action_dim) 或 (action_dim,)
        """
        return self.network(state)
```

**网络设计要点**：
- 使用Tanh激活函数，输出范围 (-1, 1)，适合连续状态空间的特征提取
- 输出层使用Softmax，确保动作概率和为1
- 隐藏层维度64是经验值，CartPole任务不需要更深更宽的网络

### 2.2 价值网络（Critic）

价值网络评估状态的价值，即从当前状态开始能获得的期望累积折扣奖励。

```python
class Critic(nn.Module):
    """价值网络：输入状态，输出状态价值估计"""
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Args:
            state: 状态张量
        Returns:
            value: 状态价值标量
        """
        return self.network(state)
```

**网络设计要点**：
- 输出层无激活函数，输出任意实数值（价值可以是负的）
- 与Actor网络结构对称，共享相同的特征提取层设计

### 2.3 并行双网络结构

在完整PPO实现中，Actor和Critic通常共享部分特征提取层，但为清晰起见，我们使用独立网络：

```python
class ActorCritic(nn.Module):
    """Actor-Critic双网络结构"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)
    
    def forward(self, state):
        """同时返回策略概率和状态价值"""
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value
```

---

## 3. GAE优势估计实现

### 3.1 从TD误差到GAE

TD误差定义：
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GAE优势估计：
$$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

实际实现时使用有限窗口截断：

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算GAE优势估计
    
    Args:
        rewards: 奖励张量，形状 (T,)
        values: 价值张量，形状 (T+1,)，最后一项是结束时的价值（通常为0）
        dones: 是否结束标志，形状 (T,)
        gamma: 折扣因子
        lam: GAE参数，控制偏差-方差权衡
    
    Returns:
        advantages: 优势张量，形状 (T,)
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T)
    
    # 从后向前递推计算GAE
    # A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    gae = 0
    for t in reversed(range(T)):
        # TD误差：r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        # 递推：A_t = δ_t + γλ * A_{t+1}
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    return advantages
```

### 3.2 GAE的超参数影响

| 参数 | 值 | 效果 |
|-----|-----|------|
| λ → 0 | 接近TD(0) | 低方差，高偏差，快速收敛但可能不稳定 |
| λ → 1 | 接近Monte Carlo | 无偏，高方差，收敛慢但稳定 |
| 典型值 | λ = 0.95~0.98 | 平衡偏差和方差 |

### 3.3 优势估计的标准化

在计算PPO损失前，通常需要对优势进行标准化：

```python
def normalize_advantages(advantages):
    """标准化优势函数"""
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

标准化不改变优势的方向（正负号），但使得不同batch的优势具有可比尺度，有利于训练稳定性。

---

## 4. PPO更新实现

### 4.1 PPO损失函数计算

```python
def compute_ppo_loss(probs, old_probs, actions, advantages, epsilon=0.2):
    """
    计算PPO裁剪损失
    
    Args:
        probs: 新策略的动作概率，形状 (T, action_dim)
        old_probs: 旧策略的动作概率（对数概率），形状 (T,)
        actions: 实际采取的动作，形状 (T,)
        advantages: 优势估计，形状 (T,)
        epsilon: 裁剪参数，通常为0.2
    
    Returns:
        actor_loss: 策略损失
        ratio: 策略比率（用于监控）
    """
    # 创建概率分布
    dist = torch.distributions.Categorical(probs)
    
    # 计算新策略下动作的对数概率
    log_probs = dist.log_prob(actions)
    
    # 计算策略比率：π_θ(a|s) / π_{θ_old}(a|s)
    # 由于old_probs存储的是对数概率，所以比率 = exp(log_π_θ - log_π_{θ_old})
    ratio = torch.exp(log_probs - old_probs)
    
    # 计算未裁剪的目标：r(θ) * A
    surr1 = ratio * advantages
    
    # 计算裁剪后的目标：clip(r(θ), 1-ε, 1+ε) * A
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    
    # 取两者较小值（当A>0时限制增加，A<0时限制减少）
    # 再取负号是因为我们最小化损失来最大化目标
    actor_loss = -torch.min(surr1, surr2).mean()
    
    return actor_loss, ratio
```

### 4.2 价值网络损失

```python
def compute_critic_loss(values, returns):
    """
    计算价值网络损失
    
    Args:
        values: 价值网络预测，形状 (T,)
        returns: 回报目标，形状 (T,)
    
    Returns:
        critic_loss: 价值损失（均方误差）
    """
    return nn.MSELoss()(values.squeeze(), returns)
```

### 4.3 完整PPO更新步骤

```python
def ppo_update(ppo_agent, states, actions, old_log_probs, rewards, dones, 
               epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
    """
    PPO完整更新步骤
    
    Args:
        ppo_agent: PPO智能体，包含actor和critic网络
        states: 状态序列
        actions: 动作序列
        old_log_probs: 旧策略的对数概率
        rewards: 奖励序列
        dones: 结束标志序列
        epsilon: 裁剪参数
        value_coef: 价值损失系数
        entropy_coef: 熵正则化系数
    """
    # 1. 计算价值估计（用于GAE）
    with torch.no_grad():
        values = ppo_agent.critic(states).squeeze()
    
    # 2. 计算GAE优势估计
    # 在末尾添加一个0值作为最后状态的价值
    values_extended = torch.cat([values, torch.zeros(1)])
    advantages = compute_gae(rewards, values_extended, dones, 
                            gamma=ppo_agent.gamma, lam=ppo_agent.lam)
    advantages = normalize_advantages(advantages)
    
    # 计算回报目标：A + V = GAE + baseline ≈ Q值
    returns = advantages + values
    
    # 3. 多轮更新
    for _ in range(ppo_agent.update_epochs):
        # 前向传播获取当前策略概率和价值估计
        probs, values_pred = ppo_agent(states)
        
        # 计算PPO策略损失
        actor_loss, ratio = compute_ppo_loss(
            probs, old_log_probs, actions, advantages, epsilon
        )
        
        # 计算价值损失
        critic_loss = compute_critic_loss(values_pred, returns)
        
        # 总损失 = 策略损失 + value_coef * 价值损失 + 熵正则化
        # 熵正则化鼓励探索
        entropy = dist.entropy().mean()
        total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
        
        # 反向传播更新
        ppo_agent.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(ppo_agent.parameters(), 0.5)
        ppo_agent.optimizer.step()
```

---

## 5. CartPole训练实验

### 5.1 训练脚本结构

```python
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PPOTrainingConfig:
    """PPO训练配置"""
    gamma: float = 0.99           # 折扣因子
    lam: float = 0.95              # GAE参数
    epsilon: float = 0.2           # PPO裁剪参数
    lr: float = 3e-4               # 学习率
    update_epochs: int = 10        # 每次收集数据后的更新轮数
    batch_size: int = 64           # mini-batch大小
    max_steps: int = 500           # 每个episode的最大步数
    num_episodes: int = 2000      # 训练episode数量
    early_stop_threshold: float = 475.0  # 提前停止阈值

def train_cartpole(config=PPOTrainingConfig()):
    """CartPole训练主函数"""
    # 创建环境
    env = gym.make('CartPole-v1')
    
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n             # 2
    
    # 创建PPO智能体
    agent = PPOAgent(state_dim, action_dim, config.lr, 
                   config.gamma, config.lam, config.epsilon)
    
    # 训练记录
    episode_rewards = []
    policy_losses = []
    value_losses = []
    
    # 训练循环
    for episode in range(config.num_episodes):
        # 收集一个episode的数据
        states, actions, rewards, dones, log_probs = collect_episode(
            env, agent, config.max_steps
        )
        
        # 转换为张量
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        log_probs = torch.stack(log_probs)
        
        # PPO更新
        actor_loss, critic_loss = ppo_update(
            agent, states, actions, log_probs, rewards, dones,
            epsilon=config.epsilon
        )
        
        # 记录
        episode_rewards.append(sum(rewards))
        policy_losses.append(actor_loss)
        value_losses.append(critic_loss)
        
        # 打印训练进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1}, Avg Reward (last 50): {avg_reward:.2f}")
            
            # 提前停止
            if avg_reward >= config.early_stop_threshold:
                print(f"Task solved at episode {episode+1}!")
                break
    
    env.close()
    return episode_rewards, policy_losses, value_losses
```

### 5.2 关键超参数的影响

**学习率 (lr)**：
- 学习率过高：训练不稳定，损失剧烈波动
- 学习率过低：收敛慢，需要更多episodes
- 推荐起始值：3e-4，可根据任务调整

**折扣因子 (gamma)**：
- γ = 0.99：重视长期奖励，适合需要前瞻性决策的任务
- γ = 0.95：更重视即时奖励
- CartPole通常用0.99

**GAE参数 (lam)**：
- λ = 0.95~0.98：平衡偏差和方差
- λ = 0.99：更稳定但收敛较慢

**裁剪参数 (epsilon)**：
- ε = 0.2：允许策略概率最多变化20%
- ε = 0.1：更保守的更新
- ε = 0.3：更大的更新幅度

**更新轮数 (update_epochs)**：
- 通常4~10轮
- 更多的更新轮数提高数据利用效率
- 过多可能导致过拟合旧数据

### 5.3 训练结果可视化

训练过程中应监控以下指标：
1. **Episode Reward**：验证是否收敛
2. **Policy Loss**：确保损失在合理范围
3. **Value Loss**：价值估计的准确性
4. **Policy Ratio**：监控策略变化幅度

---

## 6. 从游戏到LLM：场景迁移

### 6.1 环境差异对比

| 维度 | CartPole | Atari | LLM RLHF |
|------|---------|-------|----------|
| 状态空间 | 4维连续（位置、速度、角度） | 游戏画面（210×160像素） | 词序列（token IDs） |
| 动作空间 | 2个离散动作 | 有限按键组合 | 词表大小（通常32k+） |
| 奖励 | 每步+1 | 游戏得分变化 | 人类偏好/任务完成度 |
| Episode长度 | 最多500步 | 有限 | 可变长度 |
| 状态表示 | 数值向量 | 图像 | embedding向量 |

### 6.2 Atari游戏的特殊处理

Atari游戏相比CartPole有几个关键区别：

**视觉输入处理**：
```python
class AtariNet(nn.Module):
    """Atari游戏的卷积网络"""
    def __init__(self, action_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 输入4帧灰度图
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def forward(self, x):
        x = x.float() / 255.0  # 归一化像素值
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

**经验回放与多环境并行**：Atari训练通常使用多个并行环境收集数据，提高样本效率。

### 6.3 LLM RLHF的适配

将PPO框架应用于LLM对齐时，关键变化：

**状态表示**：从数值向量/图像变为token序列的embedding

**动作空间**：从2-18个离散动作变为词表大小（32000+）

**奖励获取**：从实时奖励变为延迟奖励（整个序列生成完成后）

```python
class LLMActor(nn.Module):
    """LLM策略网络"""
    def __init__(self, backbone, action_dim):
        super().__init__()
        self.backbone = backbone  # 预训练语言模型
        self.lm_head = nn.Linear(backbone.config.vocab_size, action_dim)
    
    def forward(self, input_ids):
        """前向传播计算动作概率（下一个token）"""
        outputs = self.backbone(input_ids)
        logits = self.lm_head(outputs.last_hidden_state)
        probs = torch.softmax(logits, dim=-1)
        return probs
```

**reward模型的使用**：LLM场景通常使用reward model来估计序列的价值，这在RLHF中扮演Critic的角色。

---

## 7. 完整代码结构

### 7.1 ppo.py 模块结构

```python
"""
PPO算法完整实现
包含：网络定义、GAE计算、PPO损失、训练逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from typing import Tuple

# === 网络定义 ===

class Actor(nn.Module):
    """策略网络"""
    pass

class Critic(nn.Module):
    """价值网络"""
    pass

class ActorCritic(nn.Module):
    """Actor-Critic双网络"""
    pass

# === 优势估计 ===

def compute_gae(rewards, values, dones, gamma, lam):
    """计算GAE优势估计"""
    pass

def normalize_advantages(advantages):
    """标准化优势函数"""
    pass

# === PPO损失 ===

def compute_ppo_loss(probs, old_log_probs, actions, advantages, epsilon):
    """计算PPO裁剪损失"""
    pass

def compute_critic_loss(values, returns):
    """计算价值损失"""
    pass

# === PPO智能体 ===

class PPOAgent:
    """PPO智能体，包含网络和优化器"""
    pass

# === 数据收集 ===

def collect_episode(env, agent, max_steps):
    """收集一个episode的数据"""
    pass

# === 训练 ===

def ppo_update(agent, states, actions, old_log_probs, rewards, dones, epsilon):
    """PPO完整更新步骤"""
    pass

def train(env, agent, config):
    """训练主循环"""
    pass
```

### 7.2 核心代码清单

| 文件 | 功能 | 关键函数/类 |
|-----|-----|------------|
| ppo.py | PPO算法实现 | Actor, Critic, compute_gae, compute_ppo_loss |
| train_cartpole.py | CartPole训练 | train_cartpole, PPOTrainingConfig |
| requirements.txt | 依赖包 | gymnasium, torch, numpy |

---

## 本章小结

1. **网络架构**：Actor-Critic双网络结构，Actor输出动作概率，Critic估计状态价值

2. **GAE优势估计**：通过指数加权多步TD误差实现偏差-方差权衡，λ参数控制权衡

3. **PPO损失函数**：结合重要性采样和裁剪机制，确保稳定的策略更新

4. **超参数影响**：学习率、γ、λ、ε等参数需要根据任务调优

5. **CartPole实验**：验证PPO算法收敛性，观察训练曲线变化

6. **LLM迁移**：从游戏到语言模型，关键变化是状态表示、动作空间规模和奖励机制

---

## 延伸阅读

- Schulman et al. 2017: "Proximal Policy Optimization Algorithms (PPO)"
- Schulman et al. 2015: "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
- gymnasium文档: https://gymnasium.farama.org/
- PPO-PyTorch实现参考: https://github.com/openai/spinningup

---

## 思考题

1. 在PPO的min操作中，当优势A > 0和A < 0时，min操作分别起到什么作用？

2. GAE的参数λ如何影响优势估计的偏差和方差？请解释为什么λ=0.95比λ=0表现更稳定。

3. PPO使用重要性采样允许用旧数据更新，但可能导致分布偏移问题。裁剪机制如何缓解这一问题？

4. 在将PPO从CartPole迁移到LLM RLHF时，最大的挑战是什么？如何在代码层面实现这种迁移？