"""
RLHF End-to-End Pipeline
========================

RLHF完整Pipeline编排 - 管理SFT、奖励模型、PPO三个阶段

本脚本实现:
1. RLHFPipeline类 - 管理三个阶段
2. 阶段1: SFT监督微调
3. 阶段2: 奖励模型训练
4. 阶段3: PPO强化学习微调
5. 检查点管理 - 阶段间传递
6. 配置化的模型大小

参考课程: 11.1 RLHF完整Pipeline实现 - 完整Pipeline
"""

import os
import sys
import json
import copy
import shutil
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Callable
from enum import Enum
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset as HFDataset
from tqdm import tqdm

from train_reward_model import (
    RewardModel,
    RewardModelConfig,
    PreferenceDataset,
    reward_model_loss,
    collate_fn as rm_collate_fn,
    create_sample_preference_data,
    set_seed,
)
from train_ppo import PPOTrainer, PPOConfig, create_sample_prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLHFStage(Enum):
    """RLHF阶段枚举"""

    SFT = "sft"
    REWARD_MODEL = "reward_model"
    PPO = "ppo"
    COMPLETE = "complete"


@dataclass
class ModelSizeConfig:
    """模型大小配置"""

    name: str
    base_model: str
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


MODEL_CONFIGS = {
    "tiny": ModelSizeConfig(
        name="tiny",
        base_model="microsoft/phi-2",
        use_lora=True,
        lora_r=4,
        lora_alpha=8,
    ),
    "small": ModelSizeConfig(
        name="small",
        base_model="microsoft/phi-2",
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
    ),
    "medium": ModelSizeConfig(
        name="medium",
        base_model="microsoft/phi-2",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
    ),
}


@dataclass
class RLHFPipelineConfig:
    """RLHF Pipeline配置"""

    model_size: str = "tiny"
    output_dir: str = "./rlhf_output"

    # SFT配置
    sft_epochs: int = 3
    sft_batch_size: int = 2
    sft_learning_rate: float = 3e-4
    sft_max_length: int = 512

    # Reward Model配置
    rm_epochs: int = 3
    rm_batch_size: int = 2
    rm_learning_rate: float = 1e-5
    rm_max_length: int = 512

    # PPO配置
    ppo_epochs: int = 2
    ppo_batch_size: int = 2
    ppo_learning_rate: float = 1e-5
    ppo_max_new_tokens: int = 128
    ppo_kl_coef: float = 0.1

    # 其他
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SFTTrainer:
    """
    SFT训练器 - 监督微调预训练模型

    使用人类标注的问答对进行监督微调
    让模型学会按照指令生成回答
    """

    def __init__(
        self,
        base_model_name: str,
        tokenizer: AutoTokenizer,
        config: RLHFPipelineConfig,
    ):
        self.base_model_name = base_model_name
        self.tokenizer = tokenizer
        self.config = config

        # 加载模型
        logger.info(f"Loading SFT base model: {base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        self.model.to(config.device)

    def prepare_dataset(self, data: List[Dict[str, str]]) -> HFDataset:
        """准备SFT数据集"""

        def format_example(example):
            prompt = example["prompt"]
            response = example["response"]
            text = prompt + "\n\n" + response + self.tokenizer.eos_token
            return {"text": text}

        formatted_data = [format_example(d) for d in data]
        dataset = HFDataset.from_list(formatted_data)
        return dataset

    def train(self, train_data: List[Dict[str, str]], output_dir: str):
        """训练SFT模型"""
        logger.info("=" * 60)
        logger.info("Stage 1: SFT Training")
        logger.info("=" * 60)

        # 准备数据集
        dataset = self.prepare_dataset(train_data)

        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM，不需要MLM
        )

        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.sft_epochs,
            per_device_train_batch_size=self.config.sft_batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.config.sft_learning_rate,
            warmup_steps=100,
            max_steps=-1,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            seed=self.config.seed,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # 训练
        trainer.train()

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"SFT model saved to {output_dir}")

        return output_dir


class RewardModelTrainer:
    """奖励模型训练器封装"""

    def __init__(
        self,
        base_model_name: str,
        tokenizer: AutoTokenizer,
        config: RLHFPipelineConfig,
    ):
        self.base_model_name = base_model_name
        self.tokenizer = tokenizer
        self.config = config

        # 奖励模型
        reward_config = RewardModelConfig(
            base_model_name=base_model_name,
            max_length=config.rm_max_length,
            device=config.device,
        )
        self.model = RewardModel(reward_config)

    def train(
        self,
        preference_data: List[Dict[str, str]],
        output_dir: str,
    ):
        """训练奖励模型"""
        logger.info("=" * 60)
        logger.info("Stage 2: Reward Model Training")
        logger.info("=" * 60)

        from train_reward_model import (
            TrainingConfig as RM_TrainingConfig,
            train_reward_model as train_rm,
        )

        rm_config = RM_TrainingConfig(
            base_model_name=self.base_model_name,
            num_epochs=self.config.rm_epochs,
            batch_size=self.config.rm_batch_size,
            learning_rate=self.config.rm_learning_rate,
            max_length=self.config.rm_max_length,
            seed=self.config.seed,
        )

        # 训练
        model, tokenizer = train_rm(rm_config, output_dir)

        return output_dir


class PPOTrainingManager:
    """PPO训练管理器封装"""

    def __init__(
        self,
        policy_model_name: str,
        ref_model_name: str,
        reward_model_path: str,
        tokenizer: AutoTokenizer,
        config: RLHFPipelineConfig,
    ):
        self.policy_model_name = policy_model_name
        self.ref_model_name = ref_model_name
        self.reward_model_path = reward_model_path
        self.tokenizer = tokenizer
        self.config = config

        # PPO配置
        ppo_config = PPOConfig(
            policy_model_name=policy_model_name,
            ref_model_name=ref_model_name,
            reward_model_name=reward_model_path,
            max_new_tokens=config.ppo_max_new_tokens,
            learning_rate=config.ppo_learning_rate,
            batch_size=config.ppo_batch_size,
            mini_batch_size=config.ppo_batch_size,
            num_epochs=config.ppo_epochs,
            kl_coef=config.ppo_kl_coef,
            seed=config.seed,
            device=config.device,
        )

        self.trainer = PPOTrainer(ppo_config)

    def train(self, prompts: List[str], output_dir: str):
        """训练PPO"""
        logger.info("=" * 60)
        logger.info("Stage 3: PPO Training")
        logger.info("=" * 60)

        # 多步训练
        for step in range(3):  # 3步演示
            logger.info(f"PPO Step {step + 1}")
            memory = self.trainer.train_step(prompts)

            # 保存检查点
            checkpoint_dir = os.path.join(output_dir, f"step_{step + 1}")
            self.trainer.save_checkpoint(checkpoint_dir)

        return output_dir


class RLHFPipeline:
    """
    RLHF完整Pipeline编排器

    管理三个阶段的串联:
        1. SFT监督微调
        2. 奖励模型训练
        3. PPO强化学习微调

    支持检查点保存和加载，实现阶段间的无缝衔接
    """

    def __init__(self, config: RLHFPipelineConfig):
        self.config = config

        # 获取模型配置
        model_config = MODEL_CONFIGS.get(config.model_size, MODEL_CONFIGS["tiny"])
        self.model_config = model_config

        # 设置随机种子
        set_seed(config.seed)

        # 输出目录
        self.output_dir = config.output_dir
        self.stage_dirs = {
            RLHFStage.SFT: os.path.join(self.output_dir, "stage1_sft"),
            RLHFStage.REWARD_MODEL: os.path.join(
                self.output_dir, "stage2_reward_model"
            ),
            RLHFStage.PPO: os.path.join(self.output_dir, "stage3_ppo"),
        }

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 当前阶段
        self.current_stage = RLHFStage.SFT

        logger.info(f"RLHF Pipeline initialized with model size: {config.model_size}")
        logger.info(f"Base model: {model_config.base_model}")
        logger.info(f"Output directory: {self.output_dir}")

    def _prepare_sft_data(self) -> List[Dict[str, str]]:
        """准备SFT数据"""
        return [
            {
                "prompt": "解释什么是机器学习？",
                "response": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测，而无需明确编程。机器学习算法会识别数据中的模式和规律，并用这些模式来预测新数据的标签或数值。常见应用包括图像识别、自然语言处理和推荐系统等。",
            },
            {
                "prompt": "什么是量子计算？",
                "response": "量子计算是一种利用量子力学原理进行信息处理的计算方式。它使用量子比特作为基本信息单位，与传统比特的0或1不同，量子比特可以同时处于0和1的叠加状态。这使得量子计算机在处理某些特定问题时具有显著优势。",
            },
            {
                "prompt": "如何学习编程？",
                "response": "学习编程的建议：1) 选择一门入门语言如Python；2) 每天坚持编码练习；3) 完成小项目巩固知识；4) 阅读他人代码学习最佳实践；5) 加入编程社区交流经验。编程是实践性很强的技能，关键是多写多练。",
            },
            {
                "prompt": "为什么天空是蓝色的？",
                "response": "天空呈现蓝色是因为瑞利散射效应。阳光进入大气层时，大气分子对不同波长光线散射程度不同。蓝光波长较短，散射更强烈，因此向四面八方散射的蓝光使我们看到的天空呈现蓝色。日出日落时由于光线穿过更厚的大气层，蓝光被进一步散射，只剩下红橙光。",
            },
            {
                "prompt": "推荐一本好书",
                "response": "我推荐《人类简史》作者尤瓦尔·赫拉利。这本书概述了人类从远古到现代的发展历程，探讨了认知革命、农业革命和科学革命如何塑造了人类文明。内容涵盖历史学、生物学、物理学等多个学科，观点独特见解深刻。",
            },
        ]

    def _check_stage_checkpoint(self, stage: RLHFStage) -> bool:
        """检查阶段检查点是否存在"""
        stage_dir = self.stage_dirs.get(stage)
        if stage_dir is None:
            return False

        # 检查是否存在模型文件
        if stage == RLHFStage.SFT:
            return os.path.exists(
                os.path.join(stage_dir, "model.safetensors")
            ) or os.path.exists(os.path.join(stage_dir, "pytorch_model.bin"))
        elif stage == RLHFStage.REWARD_MODEL:
            return os.path.exists(
                os.path.join(stage_dir, "reward_model.pt")
            ) or os.path.exists(os.path.join(stage_dir, "model.safetensors"))
        elif stage == RLHFStage.PPO:
            return os.path.exists(os.path.join(stage_dir, "step_1", "policy_model"))

        return False

    def run_stage1_sft(self, use_existing_checkpoint: bool = True) -> str:
        """
        运行阶段1: SFT监督微调

        Args:
            use_existing_checkpoint: 如果存在检查点，是否使用

        Returns:
            检查点路径
        """
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Supervised Fine-Tuning (SFT)")
        logger.info("=" * 60)

        stage_dir = self.stage_dirs[RLHFStage.SFT]

        # 检查检查点
        if use_existing_checkpoint and self._check_stage_checkpoint(RLHFStage.SFT):
            logger.info(f"Found existing SFT checkpoint at {stage_dir}, skipping...")
            self.current_stage = RLHFStage.SFT
            return stage_dir

        # 准备数据
        sft_data = self._prepare_sft_data()

        # 创建训练器
        trainer = SFTTrainer(
            base_model_name=self.model_config.base_model,
            tokenizer=self.tokenizer,
            config=self.config,
        )

        # 训练
        output_path = trainer.train(sft_data, stage_dir)

        self.current_stage = RLHFStage.SFT
        return output_path

    def run_stage2_reward_model(
        self,
        sft_checkpoint: str,
        use_existing_checkpoint: bool = True,
    ) -> str:
        """
        运行阶段2: 奖励模型训练

        Args:
            sft_checkpoint: SFT阶段输出的检查点
            use_existing_checkpoint: 如果存在检查点，是否使用

        Returns:
            检查点路径
        """
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: Reward Model Training")
        logger.info("=" * 60)

        stage_dir = self.stage_dirs[RLHFStage.REWARD_MODEL]

        # 检查检查点
        if use_existing_checkpoint and self._check_stage_checkpoint(
            RLHFStage.REWARD_MODEL
        ):
            logger.info(
                f"Found existing Reward Model checkpoint at {stage_dir}, skipping..."
            )
            self.current_stage = RLHFStage.REWARD_MODEL
            return stage_dir

        # 准备偏好数据
        preference_data = create_sample_preference_data()

        # 创建训练器
        # 奖励模型从SFT模型初始化
        trainer = RewardModelTrainer(
            base_model_name=sft_checkpoint,
            tokenizer=self.tokenizer,
            config=self.config,
        )

        # 训练
        output_path = trainer.train(preference_data, stage_dir)

        self.current_stage = RLHFStage.REWARD_MODEL
        return output_path

    def run_stage3_ppo(
        self,
        sft_checkpoint: str,
        reward_model_checkpoint: str,
        use_existing_checkpoint: bool = True,
    ) -> str:
        """
        运行阶段3: PPO强化学习微调

        Args:
            sft_checkpoint: SFT阶段输出的检查点（用于初始化policy和ref）
            reward_model_checkpoint: 奖励模型检查点
            use_existing_checkpoint: 如果存在检查点，是否使用

        Returns:
            检查点路径
        """
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: PPO Reinforcement Learning")
        logger.info("=" * 60)

        stage_dir = self.stage_dirs[RLHFStage.PPO]

        # 检查检查点
        if use_existing_checkpoint and self._check_stage_checkpoint(RLHFStage.PPO):
            logger.info(f"Found existing PPO checkpoint at {stage_dir}, skipping...")
            self.current_stage = RLHFStage.PPO
            return stage_dir

        # 准备prompts
        prompts = create_sample_prompts()

        # 创建PPO管理器
        ppo_manager = PPOTrainingManager(
            policy_model_name=sft_checkpoint,
            ref_model_name=sft_checkpoint,
            reward_model_path=reward_model_checkpoint,
            tokenizer=self.tokenizer,
            config=self.config,
        )

        # 训练
        output_path = ppo_manager.train(prompts, stage_dir)

        self.current_stage = RLHFStage.PPO
        return output_path

    def run(self, use_existing_checkpoints: bool = True) -> Dict[str, str]:
        """
        运行完整的RLHF Pipeline

        三个阶段顺序执行:
            SFT -> Reward Model -> PPO

        Args:
            use_existing_checkpoints: 是否使用已有检查点

        Returns:
            各阶段检查点路径的字典
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING COMPLETE RLHF PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_config.name} ({self.model_config.base_model})")
        logger.info(f"Output: {self.output_dir}")
        logger.info("")

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 阶段1: SFT
        sft_checkpoint = self.run_stage1_sft(use_existing_checkpoints)

        # 阶段2: Reward Model
        rm_checkpoint = self.run_stage2_reward_model(
            sft_checkpoint,
            use_existing_checkpoints,
        )

        # 阶段3: PPO
        ppo_checkpoint = self.run_stage3_ppo(
            sft_checkpoint,
            rm_checkpoint,
            use_existing_checkpoints,
        )

        # 完成
        self.current_stage = RLHFStage.COMPLETE

        logger.info("\n" + "=" * 60)
        logger.info("RLHF PIPELINE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"SFT Model: {sft_checkpoint}")
        logger.info(f"Reward Model: {rm_checkpoint}")
        logger.info(f"PPO Model: {ppo_checkpoint}")

        return {
            "sft": sft_checkpoint,
            "reward_model": rm_checkpoint,
            "ppo": ppo_checkpoint,
        }

    def get_stage_status(self) -> Dict[str, bool]:
        """获取各阶段检查点状态"""
        return {
            "sft": self._check_stage_checkpoint(RLHFStage.SFT),
            "reward_model": self._check_stage_checkpoint(RLHFStage.REWARD_MODEL),
            "ppo": self._check_stage_checkpoint(RLHFStage.PPO),
        }

    def load_checkpoint(self, stage: RLHFStage, path: str):
        """加载特定阶段的检查点"""
        logger.info(f"Loading checkpoint for {stage.value} from {path}")

        if stage == RLHFStage.SFT:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(path)
            return model
        elif stage == RLHFStage.REWARD_MODEL:
            reward_config = RewardModelConfig(
                base_model_name=self.model_config.base_model,
            )
            model = RewardModel(reward_config)
            model.load_state_dict(torch.load(os.path.join(path, "reward_model.pt")))
            return model
        elif stage == RLHFStage.PPO:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                os.path.join(path, "policy_model")
            )
            return model


def main():
    """主函数 - 运行完整RLHF Pipeline"""

    config = RLHFPipelineConfig(
        model_size="tiny",
        output_dir="./rlhf_output",
        sft_epochs=2,
        rm_epochs=2,
        ppo_epochs=1,
        seed=42,
    )

    # 创建Pipeline
    pipeline = RLHFPipeline(config)

    # 打印状态
    logger.info("Initial stage status:")
    for stage, exists in pipeline.get_stage_status().items():
        logger.info(f"  {stage}: {'✓' if exists else '✗'}")

    # 运行Pipeline
    checkpoints = pipeline.run(use_existing_checkpoints=False)

    # 最终状态
    logger.info("\nFinal stage status:")
    for stage, exists in pipeline.get_stage_status().items():
        logger.info(f"  {stage}: {'✓' if exists else '✗'}")

    logger.info("\nPipeline execution completed!")


if __name__ == "__main__":
    main()
