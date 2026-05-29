"""
CartPole训练脚本
使用PPO算法在CartPole-v1环境上训练策略

CartPole环境说明：
- 状态空间（4维连续）：
  1. 小车位置 (-4.8 ~ 4.8 米)
  2. 小车速度 (-Inf ~ Inf)
  3. 杆子角度 (-41.8° ~ 41.8°)
  4. 杆子角速度 (-Inf ~ Inf)
- 动作空间（2个离散）：
  0: 向左推
  1: 向右推
- 奖励：每保持平衡一步得1分
- 终止条件：杆子角度超过±41.8°或小车位置超出±4.8
"""

import os
import sys
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from datetime import datetime

# 导入PPO实现
from ppo import PPOAgent, collect_episode, train


@dataclass
class PPOTrainingConfig:
    """PPO训练配置"""

    gamma: float = 0.99
    lam: float = 0.95
    epsilon: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 10
    batch_size: int = 64
    max_steps: int = 500
    num_episodes: int = 2000
    early_stop_threshold: float = 475.0
    print_interval: int = 50


def setup_environment():
    """
    创建并配置训练环境

    Returns:
        env: Gymnasium环境
    """
    env = gym.make("CartPole-v1")
    return env


def create_agent(
    state_dim: int, action_dim: int, config: PPOTrainingConfig
) -> PPOAgent:
    """
    创建PPO智能体

    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        config: 训练配置

    Returns:
        agent: PPO智能体实例
    """
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config.lr,
        gamma=config.gamma,
        lam=config.lam,
        epsilon=config.epsilon,
        value_coef=0.5,
        entropy_coef=0.01,
        hidden_dim=64,
    )
    return agent


def plot_training_curves(history: dict, save_path: str = None):
    """
    绘制训练曲线

    Args:
        history: 训练历史记录
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode奖励曲线
    ax1 = axes[0, 0]
    ax1.plot(history["episode_rewards"])
    ax1.axhline(y=475.0, color="r", linestyle="--", label="Target (475)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 策略损失曲线
    ax2 = axes[0, 1]
    ax2.plot(history["policy_losses"], color="orange")
    ax2.set_xlabel("Update Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Policy Loss")
    ax2.grid(True, alpha=0.3)

    # 价值损失曲线
    ax3 = axes[1, 0]
    ax3.plot(history["value_losses"], color="green")
    ax3.set_xlabel("Update Step")
    ax3.set_ylabel("Loss")
    ax3.set_title("Value Loss")
    ax3.grid(True, alpha=0.3)

    # 策略比率曲线（监控策略变化幅度）
    ax4 = axes[1, 1]
    ax4.plot(history["ratios"], color="purple")
    ax4.axhline(y=1.2, color="r", linestyle="--", alpha=0.5, label="Upper bound (1+ε)")
    ax4.axhline(y=0.8, color="r", linestyle="--", alpha=0.5, label="Lower bound (1-ε)")
    ax4.set_xlabel("Update Step")
    ax4.set_ylabel("Ratio")
    ax4.set_title("Policy Ratio (π_new / π_old)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")

    plt.show()


def plot_smoothedRewards(rewards: list, window: int = 50):
    """
    绘制滑动平均奖励曲线

    Args:
        rewards: 奖励列表
        window: 滑动窗口大小
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 计算滑动平均
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(rewards[start : i + 1]))

    # 绘制原始数据和滑动平均
    ax.plot(rewards, alpha=0.3, label="Raw Rewards")
    ax.plot(smoothed, linewidth=2, label=f"Smoothed (window={window})")
    ax.axhline(y=475.0, color="r", linestyle="--", label="Target (475)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Progress: Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate_agent(env, agent: PPOAgent, num_episodes: int = 10, render: bool = False):
    """
    评估训练后的智能体

    Args:
        env: Gymnasium环境
        agent: 训练好的PPO智能体
        num_episodes: 评估episode数量
        render: 是否渲染环境

    Returns:
        avg_reward: 平均奖励
        rewards: 所有评估episode的奖励列表
    """
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action, _, _ = agent.get_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if render:
                env.render()

        rewards.append(total_reward)

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    return avg_reward, std_reward, rewards


def train_cartpole(config: PPOTrainingConfig = PPOTrainingConfig()):
    """
    CartPole训练主函数

    Args:
        config: 训练配置

    Returns:
        agent: 训练好的智能体
        history: 训练历史记录
    """
    print("=" * 60)
    print("PPO Training on CartPole-v1")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Gamma: {config.gamma}")
    print(f"  - Lambda (GAE): {config.lam}")
    print(f"  - Epsilon (clip): {config.epsilon}")
    print(f"  - Learning rate: {config.lr}")
    print(f"  - Update epochs: {config.update_epochs}")
    print(f"  - Max steps per episode: {config.max_steps}")
    print(f"  - Total episodes: {config.num_episodes}")
    print("=" * 60)
    print()

    # 创建环境
    env = setup_environment()

    # 获取环境维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment info:")
    print(f"  - State dimension: {state_dim}")
    print(f"  - Action dimension: {action_dim}")
    print()

    # 创建智能体
    agent = create_agent(state_dim, action_dim, config)

    # 训练
    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    history = train(
        env=env,
        agent=agent,
        num_episodes=config.num_episodes,
        update_epochs=config.update_epochs,
        print_interval=config.print_interval,
        max_steps=config.max_steps,
    )

    end_time = datetime.now()
    duration = end_time - start_time
    print("-" * 60)
    print(f"Training completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print()

    # 评估
    print("Evaluating trained agent...")
    avg_reward, std_reward, test_rewards = evaluate_agent(env, agent, num_episodes=20)
    print(f"Evaluation results (20 episodes):")
    print(f"  - Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  - Min reward: {min(test_rewards)}")
    print(f"  - Max reward: {max(test_rewards)}")
    print()

    # 绘制训练曲线
    print("Plotting training curves...")
    plot_training_curves(history)

    # 关闭环境
    env.close()

    return agent, history


def main():
    """主程序入口"""
    # 默认配置
    config = PPOTrainingConfig()

    # 运行训练
    agent, history = train_cartpole(config)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
