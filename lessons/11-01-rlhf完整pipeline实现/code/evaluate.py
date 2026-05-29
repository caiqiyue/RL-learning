"""
RLHF Model Evaluation
=====================

RLHF模型评估脚本 - 评估对齐效果

本脚本实现:
1. Win Rate评估 - 与参考模型对比
2. 奖励模型评估 - 验证奖励分数
3. 基础质量指标 - 长度、困惑度等

参考课程: 11.1 RLHF完整Pipeline实现 - 评估方法
"""

import os
import sys
import json
import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    GenerationConfig,
)
import numpy as np
from tqdm import tqdm
import logging

from train_reward_model import RewardModel, RewardModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """评估配置"""

    model_path: str = "./rlhf_output/stage3_ppo/step_3"
    ref_model_path: str = "./rlhf_output/stage1_sft"
    reward_model_path: str = "./rlhf_output/stage2_reward_model/final"
    tokenizer_name: Optional[str] = None

    # 生成配置
    max_length: int = 512
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1

    # 评估配置
    num_samples: int = 20
    batch_size: int = 4

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class WinRateEvaluator:
    """
    Win Rate评估器 - 对比策略模型与参考模型的输出质量

    方法:
        1. 使用相同prompt分别生成回答
        2. 使用奖励模型评估两个回答
        3. 计算策略模型胜率
    """

    def __init__(
        self,
        policy_model,
        ref_model,
        reward_model,
        tokenizer,
        config: EvaluationConfig,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config

        self.policy_model.eval()
        self.ref_model.eval()
        self.reward_model.eval()

    @torch.no_grad()
    def generate_response(
        self,
        model,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        """使用指定模型生成回答"""
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length - max_new_tokens,
        )

        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded["attention_mask"].to(self.config.device)

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            num_beams=self.config.num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

        # 提取response部分
        prompt_len = input_ids.shape[1]
        response_ids = output[0, prompt_len:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response

    @torch.no_grad()
    def get_reward_score(self, prompt: str, response: str) -> float:
        """获取( prompt, response)对的奖励分数"""
        full_text = prompt + "\n\n" + response
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )

        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded["attention_mask"].to(self.config.device)

        reward = self.reward_model(input_ids, attention_mask)
        return reward.item()

    def evaluate_pair(self, prompt: str) -> Dict[str, Any]:
        """评估单个prompt - 比较策略模型和参考模型"""
        # 生成回答
        policy_response = self.generate_response(
            self.policy_model, prompt, self.config.max_new_tokens
        )
        ref_response = self.generate_response(
            self.ref_model, prompt, self.config.max_new_tokens
        )

        # 计算奖励分数
        policy_reward = self.get_reward_score(prompt, policy_response)
        ref_reward = self.get_reward_score(prompt, ref_response)

        # 判断胜负
        if policy_reward > ref_reward:
            winner = "policy"
        elif policy_reward < ref_reward:
            winner = "ref"
        else:
            winner = "tie"

        return {
            "prompt": prompt,
            "policy_response": policy_response,
            "ref_response": ref_response,
            "policy_reward": policy_reward,
            "ref_reward": ref_reward,
            "winner": winner,
        }

    def evaluate(self, prompts: List[str]) -> Dict[str, Any]:
        """评估多个prompts"""
        logger.info(f"Evaluating {len(prompts)} prompts...")

        results = []
        policy_wins = 0
        ref_wins = 0
        ties = 0

        for prompt in tqdm(prompts, desc="Evaluating"):
            result = self.evaluate_pair(prompt)
            results.append(result)

            if result["winner"] == "policy":
                policy_wins += 1
            elif result["winner"] == "ref":
                ref_wins += 1
            else:
                ties += 1

        total = len(prompts)

        metrics = {
            "num_samples": total,
            "policy_wins": policy_wins,
            "ref_wins": ref_wins,
            "ties": ties,
            "policy_win_rate": policy_wins / total,
            "ref_win_rate": ref_wins / total,
            "tie_rate": ties / total,
            "avg_policy_reward": np.mean([r["policy_reward"] for r in results]),
            "avg_ref_reward": np.mean([r["ref_reward"] for r in results]),
        }

        return {
            "metrics": metrics,
            "results": results,
        }


class RewardModelEvaluator:
    """
    奖励模型评估器 - 评估奖励模型的质量

    检查:
        1. Chosen回答的奖励是否高于rejected
        2. 奖励分数分布是否合理
        3. 奖励是否与人类偏好一致
    """

    def __init__(
        self,
        reward_model,
        tokenizer,
        config: EvaluationConfig,
    ):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_model.eval()

    @torch.no_grad()
    def get_reward(self, prompt: str, response: str) -> float:
        """获取单个回答的奖励分数"""
        full_text = prompt + "\n\n" + response
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )

        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded["attention_mask"].to(self.config.device)

        reward = self.reward_model(input_ids, attention_mask)
        return reward.item()

    def evaluate_preference_data(
        self,
        preference_data: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        评估偏好数据

        检查奖励模型是否能正确区分chosen和rejected
        """
        correct = 0
        total = len(preference_data)

        rewards_chosen = []
        rewards_rejected = []

        for item in tqdm(preference_data, desc="Evaluating preferences"):
            chosen_reward = self.get_reward(item["prompt"], item["chosen"])
            rejected_reward = self.get_reward(item["prompt"], item["rejected"])

            rewards_chosen.append(chosen_reward)
            rewards_rejected.append(rejected_reward)

            if chosen_reward > rejected_reward:
                correct += 1

        accuracy = correct / total

        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "mean_chosen_reward": np.mean(rewards_chosen),
            "mean_rejected_reward": np.mean(rewards_rejected),
            "std_chosen_reward": np.std(rewards_chosen),
            "std_rejected_reward": np.std(rewards_rejected),
            "mean_reward_diff": np.mean(
                np.array(rewards_chosen) - np.array(rewards_rejected)
            ),
        }

        return metrics

    def evaluate_response_quality(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> Dict[str, Any]:
        """评估回答质量分布"""
        rewards = []

        for prompt, response in tqdm(
            zip(prompts, responses), desc="Evaluating quality"
        ):
            reward = self.get_reward(prompt, response)
            rewards.append(reward)

        metrics = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "median_reward": np.median(rewards),
        }

        return metrics


class QualityMetrics:
    """
    基础质量指标计算

    包括:
        1. 响应长度统计
        2. 词汇多样性
        3. 重复率
        4. 困惑度 (如果提供模型)
    """

    @staticmethod
    def compute_length_stats(responses: List[str]) -> Dict[str, Any]:
        """计算长度统计"""
        lengths = [len(r) for r in responses]
        token_lengths = [len(r.split()) for r in responses]

        return {
            "mean_char_length": np.mean(lengths),
            "std_char_length": np.std(lengths),
            "min_char_length": np.min(lengths),
            "max_char_length": np.max(lengths),
            "mean_token_length": np.mean(token_lengths),
            "std_token_length": np.std(token_lengths),
        }

    @staticmethod
    def compute_vocabulary_diversity(responses: List[str]) -> Dict[str, Any]:
        """计算词汇多样性"""
        all_words = []
        for response in responses:
            words = response.lower().split()
            all_words.extend(words)

        unique_words = set(all_words)
        total_words = len(all_words)

        return {
            "total_words": total_words,
            "unique_words": len(unique_words),
            "vocabulary_richness": len(unique_words) / max(total_words, 1),
            "avg_unique_per_response": np.mean(
                [len(set(r.lower().split())) for r in responses]
            ),
        }

    @staticmethod
    def compute_repetition_rate(responses: List[str]) -> Dict[str, Any]:
        """计算重复率 - 检测模式崩溃"""
        repetition_rates = []

        for response in responses:
            words = response.lower().split()
            if len(words) < 2:
                repetition_rates.append(0.0)
                continue

            # 计算连续重复
            repeat_count = 0
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    repeat_count += 1

            rate = repeat_count / max(len(words) - 1, 1)
            repetition_rates.append(rate)

        return {
            "mean_repetition_rate": np.mean(repetition_rates),
            "max_repetition_rate": np.max(repetition_rates),
        }

    @staticmethod
    def compute_all_metrics(responses: List[str]) -> Dict[str, Any]:
        """计算所有质量指标"""
        metrics = {}
        metrics.update(QualityMetrics.compute_length_stats(responses))
        metrics.update(QualityMetrics.compute_vocabulary_diversity(responses))
        metrics.update(QualityMetrics.compute_repetition_rate(responses))
        return metrics


def load_models(config: EvaluationConfig):
    """加载所有模型"""
    logger.info("Loading models...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path or config.ref_model_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy模型 (PPO训练后的)
    logger.info(f"Loading policy model from {config.model_path}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
    )
    policy_model.to(config.device)
    policy_model.eval()

    # Reference模型 (SFT模型)
    logger.info(f"Loading reference model from {config.ref_model_path}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.ref_model_path,
        trust_remote_code=True,
    )
    ref_model.to(config.device)
    ref_model.eval()

    # 奖励模型
    logger.info(f"Loading reward model from {config.reward_model_path}")
    from train_reward_model import RewardModel, RewardModelConfig

    reward_config = RewardModelConfig(
        base_model_name=config.model_path,
        device=config.device,
    )
    reward_model = RewardModel(reward_config)
    # 尝试加载检查点
    rm_path = os.path.join(config.reward_model_path, "reward_model.pt")
    if os.path.exists(rm_path):
        reward_model.load_state_dict(torch.load(rm_path, map_location=config.device))
    reward_model.to(config.device)
    reward_model.eval()

    return policy_model, ref_model, reward_model, tokenizer


def create_test_prompts() -> List[str]:
    """创建测试prompts"""
    return [
        "解释什么是深度学习？",
        "如何学习一门新编程语言？",
        "量子计算和传统计算有什么区别？",
        "推荐一些提高工作效率的方法",
        "机器学习中的过拟合是什么，如何避免？",
        "解释区块链技术的工作原理",
        "如何保持身心健康？",
        "什么是人工智能的伦理问题？",
        "介绍一下机器人的发展历史",
        "为什么日出和日落时天空呈现红色？",
        "如何培养良好的阅读习惯？",
        "什么是可持续能源，有哪些类型？",
        "解释互联网的工作原理",
        "如何做好时间管理？",
        "人工智能在医疗领域有哪些应用？",
        "什么是虚拟现实技术？",
        "如何教育孩子正确使用电子产品？",
        "解释5G网络的特点和优势",
        "气候变化对地球有什么影响？",
        "如何培养创造力和创新思维？",
    ]


def main():
    """主评估函数"""
    config = EvaluationConfig(
        num_samples=10,
        max_new_tokens=64,
    )

    logger.info("=" * 60)
    logger.info("RLHF Model Evaluation")
    logger.info("=" * 60)

    # 加载模型
    try:
        policy_model, ref_model, reward_model, tokenizer = load_models(config)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.info("Using mock evaluation instead...")
        policy_model = None
        ref_model = None
        reward_model = None
        tokenizer = None

    # 测试prompts
    test_prompts = create_test_prompts()[: config.num_samples]

    # 1. Win Rate评估
    logger.info("\n--- Win Rate Evaluation ---")
    if policy_model is not None:
        win_rate_evaluator = WinRateEvaluator(
            policy_model, ref_model, reward_model, tokenizer, config
        )
        win_rate_results = win_rate_evaluator.evaluate(test_prompts)

        logger.info("\nWin Rate Metrics:")
        metrics = win_rate_results["metrics"]
        logger.info(f"  Policy Win Rate: {metrics['policy_win_rate']:.2%}")
        logger.info(f"  Reference Win Rate: {metrics['ref_win_rate']:.2%}")
        logger.info(f"  Tie Rate: {metrics['tie_rate']:.2%}")
        logger.info(f"  Avg Policy Reward: {metrics['avg_policy_reward']:.4f}")
        logger.info(f"  Avg Reference Reward: {metrics['avg_ref_reward']:.4f}")
    else:
        logger.info("  (Skipped - models not available)")

    # 2. 奖励模型评估
    logger.info("\n--- Reward Model Evaluation ---")
    if reward_model is not None:
        from train_reward_model import create_sample_preference_data

        preference_data = create_sample_preference_data()

        rm_evaluator = RewardModelEvaluator(reward_model, tokenizer, config)
        rm_metrics = rm_evaluator.evaluate_preference_data(preference_data)

        logger.info("\nReward Model Metrics:")
        logger.info(f"  Accuracy: {rm_metrics['accuracy']:.2%}")
        logger.info(f"  Mean Chosen Reward: {rm_metrics['mean_chosen_reward']:.4f}")
        logger.info(f"  Mean Rejected Reward: {rm_metrics['mean_rejected_reward']:.4f}")
        logger.info(f"  Mean Reward Difference: {rm_metrics['mean_reward_diff']:.4f}")
    else:
        logger.info("  (Skipped - reward model not available)")

    # 3. 质量指标
    logger.info("\n--- Quality Metrics ---")
    # 生成一些样例响应用于质量评估
    if policy_model is not None:
        sample_responses = []
        for prompt in tqdm(test_prompts[:5], desc="Generating samples"):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            with torch.no_grad():
                output = policy_model.generate(
                    encoded["input_ids"].to(config.device),
                    max_new_tokens=config.max_new_tokens,
                    temperature=0.9,
                    do_sample=True,
                )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            sample_responses.append(response)

        quality_metrics = QualityMetrics.compute_all_metrics(sample_responses)
        logger.info("\nQuality Metrics:")
        logger.info(
            f"  Mean Response Length: {quality_metrics['mean_char_length']:.1f} chars"
        )
        logger.info(
            f"  Mean Token Count: {quality_metrics['mean_token_length']:.1f} words"
        )
        logger.info(
            f"  Vocabulary Richness: {quality_metrics['vocabulary_richness']:.3f}"
        )
        logger.info(f"  Repetition Rate: {quality_metrics['mean_repetition_rate']:.4f}")
    else:
        logger.info("  (Skipped - model not available)")

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
