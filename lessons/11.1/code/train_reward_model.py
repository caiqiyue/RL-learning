"""
RLHF Reward Model Training
==========================

奖励模型训练脚本 - 使用Bradley-Terry损失函数学习人类偏好

本脚本实现:
1. RewardModel类 - 基于预训练语言模型的奖励输出头
2. 偏好数据处理 - 处理(chosen, rejected)对
3. Bradley-Terry损失函数训练
4. 奖励归一化

参考课程: 11.1 RLHF完整Pipeline实现 - 阶段二
"""

import os
import sys
import json
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """奖励模型配置"""

    base_model_name: str = "microsoft/phi-2"  # 使用较小的模型用于演示
    tokenizer_name: Optional[str] = None
    max_length: int = 512
    hidden_size: Optional[int] = None  # 如果为None，从模型配置中获取
    use_flash_attention: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RewardModel(nn.Module):
    """
    奖励模型 - 基于语言模型改造，输出标量分数

    架构:
        输入: (prompt + response) 文本序列
        输出: 标量分数 (表示回答质量)

    关键设计:
        - 使用最后一个token的隐藏状态预测奖励
        - 线性输出头，无偏置
        - 支持梯度_checkpointing节省显存
    """

    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config

        # 加载预训练模型作为base
        logger.info(f"Loading base model: {config.base_model_name}")
        self.base_model = AutoModel.from_pretrained(
            config.base_model_name,
            trust_remote_code=True,
        )

        # 获取隐藏层大小
        self.hidden_size = config.hidden_size or self.base_model.config.hidden_size

        # 奖励输出头 - 将隐藏状态映射到标量分数
        self.reward_head = nn.Linear(self.hidden_size, 1, bias=False)

        # 初始化奖励头，使其输出接近0 (初始时chosen和rejected奖励相近)
        nn.init.zeros_(self.reward_head.weight)

        self.device = config.device
        self.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len] 输入token ids
            attention_mask: [batch_size, seq_len] 注意力掩码

        Returns:
            rewards: [batch_size] 每条样本的奖励分数
        """
        # 获取模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # 取最后一个token的隐藏状态
        # 这是奖励模型的标准做法，只用最后一个token预测奖励
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        last_token_hidden = last_hidden_state[:, -1, :]  # [batch, hidden]

        # 计算奖励分数
        reward = self.reward_head(last_token_hidden).squeeze(-1)  # [batch]

        return reward

    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """获取奖励分数的便捷方法"""
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)


class PreferenceDataset(Dataset):
    """
    偏好数据集 - 存储(提示, 被选回答, 被拒回答)三元组

    每条数据包含:
        - prompt: 输入提示
        - chosen: 人类偏好的回答
        - rejected: 人类拒绝的回答
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # 构建设签序列: prompt + response
        # 使用特殊分隔符区分prompt和response
        chosen_text = prompt + "\n\n" + chosen
        rejected_text = prompt + "\n\n" + rejected

        # Tokenize
        chosen_encoded = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        rejected_encoded = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_input_ids": chosen_encoded["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoded["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoded["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoded["attention_mask"].squeeze(0),
        }


def reward_model_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> torch.Tensor:
    """
    Bradley-Terry风格的对比损失函数

    原理:
        P(preferred) = sigmoid(r_chosen - r_rejected)
        损失 = -log(P(preferred))

    如果chosen的奖励高于rejected，损失减小
    如果chosen的奖励低于rejected，损失增大，梯度反向调整

    Args:
        reward_chosen: 被选回答的奖励分数 [batch_size]
        reward_rejected: 被拒回答的奖励分数 [batch_size]

    Returns:
        loss: 标量损失值
    """
    # 计算偏好概率 (sigmoid形式)
    # P = sigmoid(r_chosen - r_rejected)
    # 当r_chosen > r_rejected时，P > 0.5，损失为负对数似然
    difference = reward_chosen - reward_rejected
    prob = torch.sigmoid(difference)

    # 防止log(0)
    prob = torch.clamp(prob, min=1e-8, max=1 - 1e-8)

    # 负对数似然
    loss = -torch.log(prob)

    return loss.mean()


def compute_rewards(
    model: RewardModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """批量计算奖励分数"""
    return model(input_ids, attention_mask)


def normalize_rewards(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    奖励归一化 - 在PPO训练前对奖励进行标准化

    奖励的绝对值没有物理意义，重要的是相对差异
    归一化确保:
        1. 奖励均值为0，方差为1
        2. 防止极端reward值主导训练
        3. 稳定优势估计

    Args:
        rewards: 原始奖励分数
        eps: 防止除零的小常数

    Returns:
        normalized_rewards: 归一化后的奖励
        mean: 原始奖励均值
        std: 原始奖励标准差
    """
    mean = rewards.mean()
    std = rewards.std()

    # 防止std为0 (当所有奖励相同时)
    std = torch.where(std > 0, std, torch.ones_like(std))

    normalized = (rewards - mean) / (std + eps)

    return normalized, mean, std


@dataclass
class TrainingConfig:
    """训练配置"""

    # 模型
    base_model_name: str = "microsoft/phi-2"
    max_length: int = 512

    # 训练超参数
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # 优化器
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # 其他
    log_interval: int = 10
    eval_interval: int = 100
    save_dir: str = "./reward_model_checkpoints"
    seed: int = 42


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_sample_preference_data() -> List[Dict[str, str]]:
    """
    创建示例偏好数据

    实际应用中，这些数据来自人类标注
    每条数据包含同一prompt的两个回答及其偏好关系
    """
    return [
        {
            "prompt": "解释什么是机器学习？",
            "chosen": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测，而无需明确编程。机器学习算法会识别数据中的模式和规律，并用这些模式来预测新数据的标签或数值。",
            "rejected": "机器学习就是让电脑学习。",
        },
        {
            "prompt": "什么是量子计算？",
            "chosen": "量子计算是一种利用量子力学原理进行信息处理的计算方式。它使用量子比特作为基本信息单位，与传统比特的0或1不同，量子比特可以同时处于0和1的叠加状态。这使得量子计算机在处理某些特定问题时，如大数分解、药物研发等，能够比经典计算机快得多。",
            "rejected": "量子计算是用量子力学原理做计算。",
        },
        {
            "prompt": "如何学习编程？",
            "chosen": "学习编程建议：1) 选择一门入门语言如Python；2) 每天坚持编码练习；3) 完成小项目巩固知识；4) 阅读他人代码学习最佳实践；5) 加入编程社区交流经验。编程是实践性很强的技能，关键是多写多练。",
            "rejected": "想学编程就多写代码。",
        },
        {
            "prompt": "为什么天空是蓝色的？",
            "chosen": "天空呈现蓝色是因为瑞利散射效应。阳光进入大气层时，大气分子对不同波长光线散射程度不同。蓝光波长较短，散射更强烈，因此向四面八方散射的蓝光使我们看到的天空呈现蓝色。日出日落时由于光线穿过更厚的大气层，蓝光被进一步散射，只剩下红橙光，所以天空呈现红色。",
            "rejected": "因为蓝色光被散射了。",
        },
        {
            "prompt": "推荐一本好书",
            "chosen": "我推荐《人类简史》作者尤瓦尔·赫拉利。这本书概述了人类从远古到现代的发展历程，探讨了认知革命、农业革命和科学革命如何塑造了人类文明。内容涵盖历史学、生物学、物理学等多个学科，观点独特见解深刻，能帮助你重新思考人类自身和社会的本质。",
            "rejected": "《人类简史》是一本好书。",
        },
        {
            "prompt": "解释相对论",
            "chosen": "爱因斯坦的相对论包括狭义相对论和广义相对论。狭义相对论指出：1) 光速是宇宙中最快的速度；2) 时间和空间是相对的，取决于观察者的运动状态。广义相对论进一步指出：重力是时空弯曲的表现，质量越大的物体弯曲时空越厉害。这就是为什么地球围绕太阳转——太阳的巨大质量弯曲了周围的时空。",
            "rejected": "相对论是爱因斯坦提出的理论。",
        },
        {
            "prompt": "如何保持健康？",
            "chosen": "保持健康的建议：1) 规律运动，每周至少150分钟中等强度运动；2) 均衡饮食，多吃蔬菜水果全谷物；3) 保证7-8小时睡眠；4) 管理压力，通过冥想或兴趣爱好放松；5) 定期体检，及时发现健康问题；6) 保持社交，与家人朋友保持联系。健康是长期投资，需要养成好习惯。",
            "rejected": "想健康就要多运动。",
        },
        {
            "prompt": "什么是区块链？",
            "chosen": "区块链是一种分布式账本技术。简单说，它是一个由多方共同维护、不可篡改的数据库。数据被打包成区块，通过加密技术连成链条。每个区块都包含前一个区块的哈希值，形成链式结构。这使得区块链具有去中心化、可追溯、不可篡改的特点。比特币就是区块链的第一个应用。",
            "rejected": "区块链就是比特币。",
        },
    ]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """DataLoader的collate函数"""
    return {
        "prompt": [item["prompt"] for item in batch],
        "chosen_input_ids": torch.stack([item["chosen_input_ids"] for item in batch]),
        "chosen_attention_mask": torch.stack(
            [item["chosen_attention_mask"] for item in batch]
        ),
        "rejected_input_ids": torch.stack(
            [item["rejected_input_ids"] for item in batch]
        ),
        "rejected_attention_mask": torch.stack(
            [item["rejected_attention_mask"] for item in batch]
        ),
    }


def train_reward_model(
    config: TrainingConfig,
    output_dir: str = "./reward_model_checkpoints",
):
    """
    完整的奖励模型训练流程

    步骤:
        1. 加载预训练模型和tokenizer
        2. 准备偏好数据
        3. 训练循环: 前向传播 -> 计算Bradley-Terry损失 -> 反向传播
        4. 保存检查点
    """
    logger.info("=" * 60)
    logger.info("Starting Reward Model Training")
    logger.info("=" * 60)

    # 设置随机种子
    set_seed(config.seed)

    # 加载tokenizer
    logger.info(f"Loading tokenizer from {config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 创建奖励模型
    reward_config = RewardModelConfig(
        base_model_name=config.base_model_name,
        max_length=config.max_length,
    )
    model = RewardModel(reward_config)
    logger.info(
        f"Reward model created with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters"
    )

    # 准备数据
    logger.info("Preparing preference data")
    preference_data = create_sample_preference_data()
    dataset = PreferenceDataset(
        preference_data,
        tokenizer,
        max_length=config.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 学习率调度器
    total_steps = (
        len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 训练循环
    model.train()
    global_step = 0
    epoch_iterator = tqdm(range(config.num_epochs), desc="Epochs")

    for epoch in epoch_iterator:
        epoch_loss = 0.0
        num_batches = 0

        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        optimizer.zero_grad()

        for step, batch in enumerate(batch_iterator):
            # 获取chosen和rejected的奖励
            chosen_rewards = model(
                batch["chosen_input_ids"].to(model.device),
                batch["chosen_attention_mask"].to(model.device),
            )

            rejected_rewards = model(
                batch["rejected_input_ids"].to(model.device),
                batch["rejected_attention_mask"].to(model.device),
            )

            # 计算Bradley-Terry损失
            loss = reward_model_loss(chosen_rewards, rejected_rewards)

            # 梯度累积
            scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()

            # 记录
            epoch_loss += loss.item()
            num_batches += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # 日志
                if global_step % config.log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(
                        f"Step {global_step}: loss={avg_loss:.4f}, "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                    logger.info(
                        f"  Chosen rewards: mean={chosen_rewards.mean().item():.3f}, "
                        f"std={chosen_rewards.std().item():.3f}"
                    )
                    logger.info(
                        f"  Rejected rewards: mean={rejected_rewards.mean().item():.3f}, "
                        f"std={rejected_rewards.std().item():.3f}"
                    )

        # Epoch结束
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch + 1} completed: avg_loss={avg_epoch_loss:.4f}")

        # 保存检查点
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}")
        os.makedirs(checkpoint_path, exist_ok=True)

        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    # 保存最终模型
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Final model saved to {final_path}")

    # 评估
    logger.info("=" * 60)
    logger.info("Evaluating Reward Model")
    logger.info("=" * 60)
    evaluate_reward_model(model, tokenizer, device=model.device)

    return model, tokenizer


def evaluate_reward_model(
    model: RewardModel,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
):
    """
    评估奖励模型

    检验:
        1. Chosen回答的奖励是否高于rejected
        2. 奖励分数分布是否合理
    """
    model.eval()

    test_data = [
        {
            "prompt": "解释人工智能",
            "chosen": "人工智能（AI）是指计算机系统执行通常需要人类智能的任务的能力，包括视觉感知、语音识别、决策制定和语言翻译等。它通过机器学习和深度学习技术从大量数据中学习模式和规律，从而能够在新情况下做出预测或决策。",
            "rejected": "AI就是让机器像人一样思考。",
        },
        {
            "prompt": "什么是深度学习？",
            "chosen": "深度学习是机器学习的一个分支，使用多层神经网络（称为深度神经网络）来学习数据的层次化表示。与传统机器学习需要手工特征工程不同，深度学习能够自动从原始数据中学习有用的特征。它在图像识别、自然语言处理和语音识别等领域取得了突破性进展。",
            "rejected": "深度学习就是很深的机器学习。",
        },
    ]

    correct = 0
    total = len(test_data)

    for item in test_data:
        chosen_text = item["prompt"] + "\n\n" + item["chosen"]
        rejected_text = item["prompt"] + "\n\n" + item["rejected"]

        chosen_encoded = tokenizer(
            chosen_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        rejected_encoded = tokenizer(
            rejected_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            chosen_reward = model(
                chosen_encoded["input_ids"].to(device),
                chosen_encoded["attention_mask"].to(device),
            )
            rejected_reward = model(
                rejected_encoded["input_ids"].to(device),
                rejected_encoded["attention_mask"].to(device),
            )

        is_correct = chosen_reward > rejected_reward
        correct += int(is_correct)

        logger.info(f"Prompt: {item['prompt']}")
        logger.info(f"  Chosen reward: {chosen_reward.item():.4f}")
        logger.info(f"  Rejected reward: {rejected_reward.item():.4f}")
        logger.info(f"  Correct: {is_correct}")
        logger.info("")

    accuracy = correct / total
    logger.info(f"Evaluation Accuracy: {accuracy:.2%}")


def main():
    """主函数"""
    config = TrainingConfig(
        base_model_name="microsoft/phi-2",
        batch_size=2,
        num_epochs=3,
        learning_rate=1e-5,
        max_length=256,
    )

    model, tokenizer = train_reward_model(
        config,
        output_dir="./reward_model_checkpoints",
    )

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
