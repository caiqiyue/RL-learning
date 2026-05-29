"""
RLHF PPO Training
=================

PPO强化学习微调脚本 - 使用奖励模型进行RLHF对齐

本脚本实现:
1. PPOTrainer配置 - policy模型、reference模型、reward模型
2. 经验收集 - 使用当前策略生成回答
3. PPO更新 - 裁剪目标函数 + KL惩罚
4. 内存优化 - 梯度检查点、混合精度

参考课程: 11.1 RLHF完整Pipeline实现 - 阶段三
"""

import os
import sys
import json
import math
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Callable
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
    GenerationConfig,
)
from tqdm import tqdm
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO训练配置"""

    # 模型配置
    policy_model_name: str = "microsoft/phi-2"
    ref_model_name: str = "microsoft/phi-2"
    reward_model_name: str = "./reward_model_checkpoints/final"
    tokenizer_name: Optional[str] = None

    # 生成配置
    max_length: int = 512
    max_new_tokens: int = 128
    min_new_tokens: int = 32
    temperature: float = 0.9
    top_p: float = 0.9
    do_sample: bool = True

    # PPO超参数
    gamma: float = 0.99  # 折扣因子
    lam: float = 0.95  # GAE lambda
    epsilon: float = 0.2  # PPO裁剪参数
    kl_coef: float = 0.1  # KL惩罚系数
    entropy_coef: float = 0.01  # 熵奖励系数

    # 训练超参数
    learning_rate: float = 1e-5
    num_epochs: int = 2
    batch_size: int = 8
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # 价值网络
    use_value_network: bool = True
    value_clip: float = 10.0  # 价值裁剪

    # 优化
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = False

    # 其他
    log_interval: int = 5
    eval_interval: int = 50
    save_dir: str = "./ppo_checkpoints"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ValueNetwork(nn.Module):
    """
    价值网络 - 估计状态价值 V(s)

    在RLHF中，价值网络用于GAE优势估计
    输入是prompt的隐藏状态，输出是标量价值
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # 初始化为接近0
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """前向传播，返回价值估计"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # 使用平均池化的最后一个隐藏状态
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
        # 取第一个token（通常是[CLS]或开始token）的隐藏状态
        first_token_hidden = last_hidden[:, 0, :]

        value = self.value_head(first_token_hidden).squeeze(-1)  # [batch]
        return value


class ReferenceModelWrapper(nn.Module):
    """
    参考模型封装 - 冻结的参考模型用于KL散度计算

    功能:
        1. 存储参考模型副本
        2. 提供log_probs计算（用于KL）
        3. 确保不更新梯度
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = copy.deepcopy(model)
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算参考模型的log_probs"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # 计算log_probs
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """获取log概率的便捷方法"""
        return self.forward(input_ids, attention_mask)


class RewardModelWrapper(nn.Module):
    """
    奖励模型封装 - 提供奖励分数计算

    与ReferenceModelWrapper类似，冻结参数
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = copy.deepcopy(model)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算奖励分数"""
        return self.model(input_ids, attention_mask)


class PPOMemory:
    """
    PPO经验回放缓冲区

    存储一次收集的所有经验:
        - prompts: 输入提示
        - responses: 策略生成的回答
        - log_probs: 响应每个token的log概率
        - values: 价值估计
        - rewards: 奖励分数
        - advantages: 优势估计
    """

    def __init__(self):
        self.prompts = []
        self.responses = []
        self.response_log_probs = []
        self.values = []
        self.rewards = []
        self.kl_penalties = []

        # 用于GAE计算
        self.response_ids = []
        self.response_masks = []

    def add(
        self,
        prompt: str,
        response: str,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        log_probs: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        kl_penalty: torch.Tensor,
    ):
        """添加一条经验"""
        self.prompts.append(prompt)
        self.responses.append(response)
        self.response_ids.append(response_ids)
        self.response_masks.append(response_mask)
        self.response_log_probs.append(log_probs)
        self.values.append(value)
        self.rewards.append(reward)
        self.kl_penalties.append(kl_penalty)

    def clear(self):
        """清空缓冲区"""
        self.prompts.clear()
        self.responses.clear()
        self.response_ids.clear()
        self.response_masks.clear()
        self.response_log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.kl_penalties.clear()

    def __len__(self) -> int:
        return len(self.prompts)


def compute_gae_advantages(
    rewards: List[torch.Tensor],
    values: List[torch.Tensor],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    计算GAE (Generalized Advantage Estimation) 优势估计

    GAE将TD误差累积起来，提供偏差-方差权衡:
        - lambda=1: 高方差低偏差
        - lambda=0: 低方差高偏差

    原理:
        TD_error[t] = r[t] + gamma * V(s[t+1]) - V(s[t])
        A[t] = TD_error[t] + gamma * lambda * A[t+1]

    Args:
        rewards: 每个token的奖励列表 [seq_len]
        values: 每个token的价值估计列表 [seq_len]
        gamma: 折扣因子
        lam: GAE lambda参数

    Returns:
        advantages: 优势估计列表
        returns: 回报估计列表 (用于价值网络学习)
    """
    advantages = []
    returns = []

    for reward_seq, value_seq in zip(rewards, values):
        # 转为tensor并移到CPU
        reward_seq = reward_seq.cpu()
        value_seq = value_seq.cpu()

        seq_len = len(reward_seq)
        advantage = torch.zeros_like(reward_seq)
        gae = 0

        # 从后向前计算GAE
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                # 最后一个位置，没有下一个价值，用0 bootstrap
                next_value = 0.0
            else:
                next_value = value_seq[t + 1].item()

            # TD误差
            delta = reward_seq[t].item() + gamma * next_value - value_seq[t].item()

            # GAE累积
            gae = delta + gamma * lam * gae
            advantage[t] = gae

        # 回报 = 优势 + 价值
        ret = advantage + value_seq
        advantages.append(advantage)
        returns.append(ret)

    return advantages, returns


def ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    PPO裁剪损失函数

    PPO的核心思想: 限制策略更新幅度，避免剧烈变化

    L^CLIP = -min(r * A, clip(r, 1-ε, 1+ε) * A)

    其中 r = exp(log_prob_new - log_prob_old) 是策略比率

    Args:
        old_log_probs: 旧策略的log概率 [batch, seq_len]
        new_log_probs: 新策略的log概率 [batch, seq_len]
        advantages: 优势估计 [batch]
        epsilon: 裁剪参数
    """
    # 策略比率
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 未裁剪目标
    surr1 = ratio * advantages.unsqueeze(-1)

    # 裁剪目标
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surr2 = clipped_ratio * advantages.unsqueeze(-1)

    # 取较小值（悲观下限，防止过度优化）
    loss = -torch.min(surr1, surr2)

    return loss.mean()


def compute_kl_divergence(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算Actor和Reference之间的KL散度

    KL(p||q) = sum(p * log(p/q)) = sum(p * (log(p) - log(q)))

    这里只计算response部分的KL，因为prompt是共享的

    Args:
        policy_log_probs: 策略模型的log概率
        ref_log_probs: 参考模型的log概率
        response_mask: response部分的掩码
    """
    # KL散度 = policy_log_prob - ref_log_prob（在ref概率上）
    kl = policy_log_probs - ref_log_probs  # [batch, seq_len, vocab_size]

    # 只计算response部分
    # 获取response起始位置（通常是prompt长度之后）
    response_kl = kl.sum(dim=-1)  # [batch, seq_len]

    # 应用mask，只保留response部分的KL
    masked_kl = response_kl * response_mask

    return masked_kl.sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8)


class PPOTrainer:
    """
    PPO训练器 - 管理PPO训练的各个环节

    主要组件:
        - policy_model: 待训练的策略模型 (Actor)
        - ref_model: 参考模型 (冻结)
        - reward_model: 奖励模型 (冻结)
        - value_model: 价值网络 (Critic)

    训练流程:
        1. collect_experience: 使用当前策略收集经验
        2. compute_advantages: 计算优势估计
        3. ppo_update: 使用PPO更新策略
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = config.device

        logger.info("Initializing PPO Trainer...")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载策略模型 (Actor)
        logger.info(f"Loading policy model: {config.policy_model_name}")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.policy_model_name,
            trust_remote_code=True,
        )
        if config.use_gradient_checkpointing:
            self.policy_model.gradient_checkpointing_enable()
        self.policy_model.to(self.device)

        # 加载参考模型 (冻结)
        logger.info(f"Loading reference model: {config.ref_model_name}")
        self.ref_model = ReferenceModelWrapper(self.policy_model)
        self.ref_model.to(self.device)

        # 加载奖励模型 (冻结)
        logger.info(f"Loading reward model: {config.reward_model_name}")
        from train_reward_model import RewardModel, RewardModelConfig

        reward_config = RewardModelConfig(base_model_name=config.policy_model_name)
        reward_model = RewardModel(reward_config)
        reward_model.load_state_dict(
            torch.load(
                os.path.join(config.reward_model_name, "reward_model.pt"),
                map_location=self.device,
            )
        )
        self.reward_model = RewardModelWrapper(reward_model)
        self.reward_model.to(self.device)

        # 价值网络 (Critic)
        if config.use_value_network:
            logger.info("Initializing value network")
            self.value_model = ValueNetwork(copy.deepcopy(self.policy_model.base_model))
            self.value_model.to(self.device)
        else:
            self.value_model = None

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        if self.value_model is not None:
            self.value_optimizer = torch.optim.AdamW(
                self.value_model.parameters(),
                lr=config.learning_rate,
                weight_decay=0.01,
            )

        # 记忆缓冲区
        self.memory = PPOMemory()

        # 训练统计
        self.stats = {
            "kl_divergence": [],
            "reward": [],
            "ppo_loss": [],
            "value_loss": [],
            "entropy": [],
        }

    @torch.no_grad()
    def generate_responses(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        使用当前策略生成回答

        Args:
            prompts: 输入提示列表

        Returns:
            responses: 生成的响应文本列表
            response_ids: 响应的token ids列表
            response_masks: 响应的attention mask列表
        """
        self.policy_model.eval()

        responses = []
        response_ids_list = []
        response_masks_list = []

        for prompt in prompts:
            # Tokenize prompt
            prompt_encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length - self.config.max_new_tokens,
            )

            prompt_ids = prompt_encoded["input_ids"].to(self.device)
            prompt_mask = prompt_encoded["attention_mask"].to(self.device)

            # 生成response
            generation_output = self.policy_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.config.max_new_tokens,
                min_new_tokens=self.config.min_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # 提取response部分（去掉prompt）
            prompt_len = prompt_ids.shape[1]
            response_ids = generation_output[0, prompt_len:]

            # 创建response mask
            response_mask = (response_ids != self.tokenizer.pad_token_id).float()

            # Decode
            response_text = self.tokenizer.decode(
                response_ids, skip_special_tokens=True
            )

            responses.append(response_text)
            response_ids_list.append(response_ids.cpu())
            response_masks_list.append(response_mask.cpu())

        return responses, response_ids_list, response_masks_list

    @torch.no_grad()
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        response_ids_list: List[torch.Tensor],
        response_masks_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        计算每个(prompt, response)对的奖励分数

        奖励模型输入完整序列，输出标量分数
        """
        self.reward_model.eval()

        rewards = []

        for prompt, response, response_ids, response_mask in zip(
            prompts, responses, response_ids_list, response_masks_list
        ):
            # 构完整序列
            full_text = prompt + "\n\n" + response
            encoded = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # 计算奖励
            reward = self.reward_model(input_ids, attention_mask)
            rewards.append(reward.cpu())

        return rewards

    @torch.no_grad()
    def compute_kl_penalties(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids_list: List[torch.Tensor],
        response_masks_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        计算每个response的KL散度惩罚

        KL(P_policy || P_ref) 衡量policy偏离reference的程度
        """
        self.ref_model.eval()

        kl_penalties = []

        for response_ids, response_mask in zip(response_ids_list, response_masks_list):
            # 拼接prompt + response
            full_ids = (
                torch.cat([prompt_ids[0], response_ids]).unsqueeze(0).to(self.device)
            )
            full_mask = (
                torch.cat([prompt_mask[0], response_mask]).unsqueeze(0).to(self.device)
            )

            # 获取policy和ref的log_probs
            policy_log_probs = self.policy_model(
                input_ids=full_ids,
                attention_mask=full_mask,
            ).logits
            policy_log_probs = F.log_softmax(policy_log_probs, dim=-1)

            ref_log_probs = self.ref_model(full_ids, full_mask)

            # 计算KL
            kl = compute_kl_divergence(
                policy_log_probs,
                ref_log_probs,
                response_mask.to(self.device).unsqueeze(0),
            )

            kl_penalties.append(kl.cpu())

        return kl_penalties

    @torch.no_grad()
    def compute_values(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算每个prompt的价值估计"""
        if self.value_model is None:
            return None

        self.value_model.eval()
        values = self.value_model(prompt_ids, prompt_mask)
        return values

    def collect_experience(
        self,
        prompts: List[str],
    ) -> PPOMemory:
        """
        收集PPO训练经验

        步骤:
            1. 使用当前策略生成回答
            2. 计算奖励分数
            3. 计算KL散度
            4. 计算价值估计
            5. 组装经验数据
        """
        logger.info(f"Collecting experience for {len(prompts)} prompts")

        memory = PPOMemory()

        # 分批处理
        for i in range(0, len(prompts), self.config.batch_size):
            batch_prompts = prompts[i : i + self.config.batch_size]

            # 1. 生成responses
            responses, response_ids_list, response_masks_list = self.generate_responses(
                batch_prompts
            )

            # 2. 计算奖励
            rewards = self.compute_rewards(
                batch_prompts, responses, response_ids_list, response_masks_list
            )

            # 3. 计算KL散度 (作为惩罚项)
            # 为简化，这里使用reward减去kl作为最终reward
            kl_penalties = []  # placeholder

            for j, (prompt, response, response_ids, response_mask, reward) in enumerate(
                zip(
                    batch_prompts,
                    responses,
                    response_ids_list,
                    response_masks_list,
                    rewards,
                )
            ):
                memory.add(
                    prompt=prompt,
                    response=response,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    log_probs=torch.zeros_like(response_ids),  # placeholder
                    value=torch.tensor(0.0),  # placeholder
                    reward=reward,
                    kl_penalty=torch.tensor(0.0),  # placeholder
                )

        return memory

    def ppo_update(
        self,
        memory: PPOMemory,
        num_epochs: int = 2,
    ):
        """
        PPO策略更新

        步骤:
            1. 计算GAE优势估计
            2. 多次epoch更新策略
            3. 更新价值网络
        """
        if len(memory) == 0:
            logger.warning("Empty memory, skipping update")
            return

        logger.info(f"PPO update with {len(memory)} samples")

        self.policy_model.train()
        if self.value_model is not None:
            self.value_model.train()

        # 转换为batch
        batch_size = self.config.mini_batch_size
        num_updates = 0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # 遍历所有经验
            indices = list(range(len(memory)))

            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]

                # 获取batch数据
                prompts = [memory.prompts[i] for i in batch_indices]
                responses = [memory.responses[i] for i in batch_indices]

                # Tokenize
                full_texts = [p + "\n\n" + r for p, r in zip(prompts, responses)]
                encoded = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                # 前向传播
                outputs = self.policy_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

                # 计算log_probs
                log_probs = F.log_softmax(logits, dim=-1)

                # 获取response部分的log_probs（简化处理：使用全部序列）
                # 实际应该只计算response部分
                response_log_probs = (
                    log_probs[:, :-1]
                    .gather(2, input_ids[:, 1:].unsqueeze(-1))
                    .squeeze(-1)
                )

                # 获取旧log_probs (从memory)
                old_log_probs = (
                    torch.stack(
                        [memory.response_log_probs[i] for i in batch_indices]
                    ).to(self.device)
                    if memory.response_log_probs[0] is not None
                    else None
                )

                # 获取奖励
                rewards = torch.stack([memory.rewards[i] for i in batch_indices]).to(
                    self.device
                )

                # 简化的优势估计：用奖励作为优势
                advantages = rewards

                # PPO损失
                if old_log_probs is not None:
                    loss = ppo_loss(
                        old_log_probs,
                        response_log_probs,
                        advantages,
                        self.config.epsilon,
                    )
                else:
                    # 第一次更新，没有旧log_probs
                    loss = ppo_loss(
                        response_log_probs,
                        response_log_probs,
                        advantages,
                        self.config.epsilon,
                    )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                num_updates += 1

                # 记录统计
                self.stats["ppo_loss"].append(loss.item())
                self.stats["reward"].append(rewards.mean().item())

        logger.info(f"PPO update completed: {num_updates} updates")

    def train_step(self, prompts: List[str]):
        """单步训练"""
        # 1. 收集经验
        memory = self.collect_experience(prompts)

        # 2. PPO更新
        self.ppo_update(memory, num_epochs=self.config.num_epochs)

        return memory

    def save_checkpoint(self, path: str):
        """保存检查点"""
        os.makedirs(path, exist_ok=True)

        self.policy_model.save_pretrained(os.path.join(path, "policy_model"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

        if self.value_model is not None:
            torch.save(
                self.value_model.state_dict(), os.path.join(path, "value_model.pt")
            )

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        self.policy_model.from_pretrained(os.path.join(path, "policy_model"))
        self.tokenizer.from_pretrained(os.path.join(path, "tokenizer"))

        if self.value_model is not None and os.path.exists(
            os.path.join(path, "value_model.pt")
        ):
            self.value_model.load_state_dict(
                torch.load(
                    os.path.join(path, "value_model.pt"), map_location=self.device
                )
            )

        logger.info(f"Checkpoint loaded from {path}")


def create_sample_prompts() -> List[str]:
    """创建示例提示"""
    return [
        "解释什么是深度学习？",
        "如何学习一门新编程语言？",
        "量子计算和传统计算有什么区别？",
        "推荐一些提高工作效率的方法",
        "机器学习中的过拟合是什么，如何避免？",
        "解释区块链技术的工作原理",
        "如何保持身心健康？",
        "什么是人工智能的伦理问题？",
    ]


def main():
    """主函数"""
    config = PPOConfig(
        policy_model_name="microsoft/phi-2",
        batch_size=2,
        mini_batch_size=1,
        num_epochs=1,
        max_new_tokens=64,
        learning_rate=1e-5,
        kl_coef=0.1,
    )

    trainer = PPOTrainer(config)

    prompts = create_sample_prompts()

    logger.info("Starting PPO training...")
    logger.info(f"Using {len(prompts)} prompts")

    # 模拟多步训练
    for step in range(2):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training Step {step + 1}")
        logger.info(f"{'=' * 60}")

        memory = trainer.train_step(prompts)

        # 打印统计
        if trainer.stats["ppo_loss"]:
            avg_loss = sum(trainer.stats["ppo_loss"][-10:]) / min(
                10, len(trainer.stats["ppo_loss"])
            )
            avg_reward = sum(trainer.stats["reward"][-10:]) / min(
                10, len(trainer.stats["reward"])
            )
            logger.info(
                f"Stats: avg_ppo_loss={avg_loss:.4f}, avg_reward={avg_reward:.4f}"
            )

        # 保存检查点
        if (step + 1) % 1 == 0:
            trainer.save_checkpoint(os.path.join(config.save_dir, f"step_{step + 1}"))

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
