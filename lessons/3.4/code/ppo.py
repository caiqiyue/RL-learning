"""
PPO算法完整实现
基于PyTorch的纯PPO实现，不依赖外部RL库

包含模块：
1. 网络定义：Actor（策略网络）、Critic（价值网络）
2. GAE优势估计：广义优势估计实现
3. PPO损失：裁剪损失函数计算
4. PPO智能体：封装网络和优化器
5. 工具函数：数据收集、训练辅助
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
from typing import Tuple, List


# =============================================================================
# 1. 网络定义
# =============================================================================


class Actor(nn.Module):
    """
    策略网络（Actor）
    输入状态，输出动作的概率分布

    Architecture:
        - 输入层: state_dim -> 64
        - 隐藏层: 64 -> 64 (Tanh激活)
        - 输出层: 64 -> action_dim (Softmax输出动作概率)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态张量，形状 (batch_size, state_dim) 或 (state_dim,)

        Returns:
            probs: 动作概率分布，形状 (batch_size, action_dim)
        """
        return self.network(state)


class Critic(nn.Module):
    """
    价值网络（Critic）
    输入状态，输出该状态的价值估计

    Architecture:
        - 输入层: state_dim -> 64
        - 隐藏层: 64 -> 64 (Tanh激活)
        - 输出层: 64 -> 1 (状态价值)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态张量

        Returns:
            value: 状态价值标量
        """
        return self.network(state)


class ActorCritic(nn.Module):
    """
    Actor-Critic双网络结构
    同时包含策略网络和价值网络，共享特征提取（可选）
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同时返回策略概率和状态价值

        Args:
            state: 状态张量

        Returns:
            probs: 动作概率分布
            value: 状态价值估计
        """
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value


# =============================================================================
# 2. GAE优势估计
# =============================================================================


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """
    计算GAE（广义优势估计）

    GAE通过指数加权平均组合多步TD误差，实现偏差-方差权衡：
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

    其中 δ_t = r_t + γV(s_{t+1}) - V(s_t) 是TD误差

    Args:
        rewards: 奖励张量，形状 (T,)
        values: 价值张量，形状 (T+1,)，最后一项是结束时的价值（通常为0）
        dones: 是否结束标志，形状 (T,)
        gamma: 折扣因子，控制远期奖励的重要性
        lam: GAE参数（lambda），控制偏差-方差权衡
            - lam=0：只考虑当前步TD误差（低方差高偏差）
            - lam=1：考虑完整轨迹回报（无偏高方差）
            - 典型值：0.95~0.98

    Returns:
        advantages: GAE优势估计，形状 (T,)

    Example:
        >>> rewards = torch.tensor([1.0, 1.0, 1.0])
        >>> values = torch.tensor([2.5, 1.8, 1.0, 0.0])  # 最后一个是V(s_T+1)=0
        >>> dones = torch.tensor([False, False, False])
        >>> advantages = compute_gae(rewards, values, dones)
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T)

    # 从后向前递推计算GAE
    gae = 0
    for t in reversed(range(T)):
        # TD误差：r_t + γV(s_{t+1}) - V(s_t)
        # 当done=True时，不需要考虑未来价值（V(s_{t+1})被置0）
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]

        # 递推公式：A_t = δ_t + γλ * (1-done_t) * A_{t+1}
        # 当episode结束时(done=True)，下一状态没有价值，不继续递推
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    return advantages


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """
    标准化优势函数

    标准化不改变优势的方向（正负号），但：
    1. 使得不同batch的优势具有可比尺度
    2. 有利于训练稳定性
    3. 在PPO中帮助控制梯度大小

    Args:
        advantages: 原始优势估计，形状 (T,)

    Returns:
        标准化后的优势，形状 (T,)
    """
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


# =============================================================================
# 3. PPO损失函数
# =============================================================================


def compute_ppo_loss(
    probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算PPO裁剪损失函数

    L^{CLIP}(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

    其中：
    - r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t) 是策略比率
    - A_t 是优势函数
    - clip(r, 1-ε, 1+ε) 将策略比率限制在 [1-ε, 1+ε] 范围内

    Args:
        probs: 当前策略的动作概率，形状 (T, action_dim)
        old_log_probs: 旧策略的动作对数概率，形状 (T,)
        actions: 实际采取的动作，形状 (T,)
        advantages: 优势估计，形状 (T,)
        epsilon: 裁剪参数，通常为0.2
            - 允许策略概率最多变化20%
            - ε=0.1更保守，ε=0.3更新幅度更大

    Returns:
        actor_loss: 策略损失（负值，因为最小化损失）
        ratio: 策略比率（用于监控）

    Note:
        min操作的物理含义：
        - 当A>0（动作优于平均）：限制策略比率不要太大
        - 当A<0（动作差于平均）：限制策略比率不要太小
        这样确保损失不会超过裁剪目标
    """
    # 创建分类分布
    dist = Categorical(probs)

    # 计算新策略下动作的对数概率 log π_θ(a|s)
    log_probs = dist.log_prob(actions)

    # 计算策略比率：r_t(θ) = exp(log π_θ - log π_{θ_old})
    ratio = torch.exp(log_probs - old_log_probs)

    # 计算未裁剪的目标：r_t(θ) * A_t
    # 如果这个动作好（A>0），比率增大会增加损失（鼓励）
    # 如果这个动作差（A<0），比率增大会减少损失（鼓励）
    surr1 = ratio * advantages

    # 计算裁剪后的目标：clip(r_t, 1-ε, 1+ε) * A_t
    # 将策略比率限制在 [1-ε, 1+ε] 范围内
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    # 取两者较小值，然后取负号（因为要最小化损失来最大化目标）
    # 当A>0时：如果ratio太大，surr1会很大，但surr2被限制
    # 当A<0时：如果ratio太小（<1），surr1会变负（更负），但surr2被限制
    actor_loss = -torch.min(surr1, surr2).mean()

    return actor_loss, ratio


def compute_critic_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    计算价值网络（Critic）的损失

    使用均方误差损失，让价值网络准确估计状态价值
    L = E[(V(s) - G)^2]

    其中G是回报目标（通常是GAE优势 + 价值基线）

    Args:
        values: 价值网络预测，形状 (T,) 或 (T, 1)
        returns: 回报目标，形状 (T,)

    Returns:
        critic_loss: 均方误差损失
    """
    return nn.MSELoss()(values.squeeze(), returns)


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    计算策略分布的熵

    熵衡量策略的随机性：
    - 高熵：策略接近均匀分布，探索性强
    - 低熵：策略接近确定性， exploit性强

    熵正则化鼓励探索，防止策略过早收敛到确定性策略

    Args:
        probs: 动作概率分布

    Returns:
        entropy: 熵值
    """
    dist = Categorical(probs)
    return dist.entropy().mean()


# =============================================================================
# 4. PPO智能体
# =============================================================================


class PPOAgent:
    """
    PPO智能体

    封装Actor-Critic网络、优化器和超参数
    提供训练和推理接口
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        hidden_dim: int = 64,
    ):
        """
        初始化PPO智能体

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            lam: GAE参数
            epsilon: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            hidden_dim: 隐藏层维度
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # 创建Actor-Critic网络
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)

        # 创建优化器（可以分别设置学习率）
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # 训练历史记录
        self.rewards_history = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.ratios = []

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，同时获取策略概率和价值估计
        """
        return self.actor_critic(state)

    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        根据当前状态选择动作

        Args:
            state: 状态数组
            deterministic: 是否使用确定性策略（选择概率最大的动作）
                - False：随机采样，适合训练
                - True：选择概率最大的动作，适合评估

        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        state_tensor = torch.FloatTensor(state)

        with torch.no_grad():
            probs, value = self.forward(state_tensor)

        dist = Categorical(probs)

        if deterministic:
            action = probs.argmax().item()
        else:
            action = dist.sample()

        log_prob = dist.log_prob(torch.tensor(action))
        value = value.item()

        return action, log_prob.item(), value

    def evaluate_action(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定状态-动作对的价值

        用于计算损失时的前向传播

        Args:
            states: 状态批量
            actions: 动作批量

        Returns:
            log_probs: 动作的对数概率
            values: 价值估计
            entropy: 熵
        """
        probs, values = self.forward(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, values.squeeze(), entropy

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        update_epochs: int = 10,
    ) -> dict:
        """
        PPO更新步骤

        完整的PPO更新流程：
        1. 计算价值估计和GAE优势
        2. 多轮更新网络参数

        Args:
            states: 状态序列，形状 (T, state_dim)
            actions: 动作序列，形状 (T,)
            old_log_probs: 旧策略的对数概率，形状 (T,)
            rewards: 奖励序列，形状 (T,)
            dones: 结束标志，形状 (T,)
            update_epochs: 每次数据收集后的更新轮数

        Returns:
            metrics: 训练指标字典
        """
        # 1. 获取价值估计用于GAE
        with torch.no_grad():
            values = self.actor_critic.critic(states).squeeze()

        # 2. 扩展价值序列，末尾添加0（最后状态的价值）
        values_extended = torch.cat([values, torch.zeros(1)])

        # 3. 计算GAE优势估计
        advantages = compute_gae(rewards, values_extended, dones, self.gamma, self.lam)
        advantages = normalize_advantages(advantages)

        # 计算回报目标（用于价值网络训练）
        # 回报 = 优势 + 价值基线 ≈ Q值
        returns = advantages + values

        # 记录更新过程中的指标
        policy_losses = []
        value_losses = []
        entropies = []
        ratios = []

        # 4. 多轮更新
        for _ in range(update_epochs):
            # 前向传播获取当前策略和价值
            probs, values_pred = self.forward(states)

            # 计算策略损失
            actor_loss, ratio = compute_ppo_loss(
                probs, old_log_probs, actions, advantages, self.epsilon
            )

            # 计算价值损失
            critic_loss = compute_critic_loss(values_pred, returns)

            # 计算熵
            entropy = compute_entropy(probs)

            # 总损失 = 策略损失 + value_coef * 价值损失 - entropy_coef * 熵
            # 注意：熵项前是负号，因为要最大化熵（最小化负熵）
            total_loss = (
                actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            )

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)

            # 更新参数
            self.optimizer.step()

            # 记录指标
            policy_losses.append(actor_loss.item())
            value_losses.append(critic_loss.item())
            entropies.append(entropy.item())
            ratios.append(ratio.mean().item())

        # 记录历史
        self.policy_losses.append(np.mean(policy_losses))
        self.value_losses.append(np.mean(value_losses))
        self.entropies.append(np.mean(entropies))
        self.ratios.append(np.mean(ratios))

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
            "ratio": np.mean(ratios),
        }

    def save(self, path: str):
        """保存模型"""
        torch.save(self.actor_critic.state_dict(), path)

    def load(self, path: str):
        """加载模型"""
        self.actor_critic.load_state_dict(torch.load(path))


# =============================================================================
# 5. 数据收集
# =============================================================================


def collect_episode(
    env, agent: PPOAgent, max_steps: int = 500
) -> Tuple[List[torch.Tensor], List[int], List[float], List[float], List[float]]:
    """
    收集一个episode的数据

    使用当前策略与环境交互，收集状态、动作、奖励等数据

    Args:
        env: Gymnasium环境
        agent: PPO智能体
        max_steps: 单个episode的最大步数（CartPole默认500）

    Returns:
        states: 状态列表
        actions: 动作列表
        rewards: 奖励列表
        dones: 结束标志列表
        log_probs: 对数概率列表
    """
    states, actions, rewards, dones, log_probs = [], [], [], [], []

    state, _ = env.reset()
    done = False
    truncated = False

    step = 0
    while not (done or truncated) and step < max_steps:
        # 获取动作和价值
        action, log_prob, _ = agent.get_action(state, deterministic=False)

        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated

        # 存储数据
        states.append(torch.FloatTensor(state))
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))
        log_probs.append(log_prob)

        # 更新状态
        state = next_state
        step += 1

    return states, actions, rewards, dones, log_probs


def collect_batch(
    env, agent: PPOAgent, num_episodes: int, max_steps: int = 500
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    收集多个episode的数据作为一个batch

    Args:
        env: Gymnasium环境
        agent: PPO智能体
        num_episodes: 收集的episode数量
        max_steps: 单个episode最大步数

    Returns:
        批量数据（tensor格式）
    """
    all_states, all_actions, all_rewards, all_dones, all_log_probs = [], [], [], [], []

    for _ in range(num_episodes):
        states, actions, rewards, dones, log_probs = collect_episode(
            env, agent, max_steps
        )

        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_dones.extend(dones)
        all_log_probs.extend(log_probs)

    return (
        torch.stack(all_states),
        torch.tensor(all_actions),
        torch.FloatTensor(all_rewards),
        torch.FloatTensor(all_dones),
        torch.FloatTensor(all_log_probs),
    )


# =============================================================================
# 6. 训练辅助
# =============================================================================


def train(
    env,
    agent: PPOAgent,
    num_episodes: int,
    update_epochs: int = 10,
    print_interval: int = 10,
    max_steps: int = 500,
) -> dict:
    """
    PPO训练主循环

    Args:
        env: Gymnasium环境
        agent: PPO智能体
        num_episodes: 训练episode数量
        update_epochs: 每次数据收集后的更新轮数
        print_interval: 打印信息的间隔
        max_steps: 单个episode最大步数

    Returns:
        training_history: 训练历史记录
    """
    episode_rewards = []

    for episode in range(num_episodes):
        # 1. 收集一个episode的数据
        states, actions, rewards, dones, log_probs = collect_episode(
            env, agent, max_steps
        )

        # 转换为张量
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        log_probs = torch.FloatTensor(log_probs)

        # 2. PPO更新
        metrics = agent.update(
            states, actions, log_probs, rewards, dones, update_epochs
        )

        # 3. 记录
        episode_reward = rewards.sum().item()
        episode_rewards.append(episode_reward)
        agent.rewards_history.append(episode_reward)

        # 4. 打印进度
        if (episode + 1) % print_interval == 0:
            recent_rewards = episode_rewards[-print_interval:]
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Ratio: {metrics['ratio']:.4f}")
            print()

    return {
        "episode_rewards": episode_rewards,
        "policy_losses": agent.policy_losses,
        "value_losses": agent.value_losses,
        "entropies": agent.entropies,
        "ratios": agent.ratios,
    }


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    import gymnasium as gym

    # 创建CartPole环境
    env = gym.make("CartPole-v1")

    # 获取环境维度
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # 创建PPO智能体
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        epsilon=0.2,
    )

    # 训练
    history = train(env, agent, num_episodes=500, update_epochs=10)

    # 测试训练后的策略
    print("\nTesting trained agent...")
    test_episodes = 10
    test_rewards = []

    for _ in range(test_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _, _ = agent.get_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        test_rewards.append(total_reward)

    print(f"Average test reward: {np.mean(test_rewards):.2f}")

    env.close()
