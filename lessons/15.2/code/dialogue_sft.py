"""
Dialogue SFT: Supervised Fine-Tuning for Dialogue Systems

This module handles:
1. Dialogue data preparation and formatting
2. Multi-turn conversation processing
3. System prompt engineering
4. SFT training with LoRA
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class DialogueTurn:
    """Single turn in a conversation"""

    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class DialogueSample:
    """Complete dialogue sample for training"""

    system_prompt: str
    turns: List[DialogueTurn]

    def to_chatml_format(self) -> str:
        """Convert to ChatML format for training"""
        result = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        for turn in self.turns:
            result += f"<|im_start|>{turn.role}\n{turn.content}<|im_end|>\n"
        return result.strip()


class DialogueDataset(Dataset):
    """Dataset for dialogue SFT training"""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        dialogue = self._parse_dialogue(sample)

        # Format dialogue
        formatted = dialogue.to_chatml_format()

        # Tokenize
        encoding = self.tokenizer(
            formatted,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Labels: only compute loss on assistant turns
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask out user turns - find assistant turn boundaries
        # In ChatML: <|im_start|>user ... <|im_end|><|im_start|>assistant ... <|im_end|>
        # Only calculate loss on assistant content
        labels = self._create_labels_for_assistant_only(input_ids, labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _parse_dialogue(self, sample: Dict) -> DialogueSample:
        """Parse raw dialogue data into DialogueSample"""
        return DialogueSample(
            system_prompt=sample.get("system", DEFAULT_SYSTEM_PROMPT),
            turns=[
                DialogueTurn(role=t["role"], content=t["content"])
                for t in sample["turns"]
            ],
        )

    def _create_labels_for_assistant_only(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Create labels that only compute loss on assistant content"""
        tokenizer = self.tokenizer
        user_token_id = (
            tokenizer.additional_special_tokens_ids[
                tokenizer.additional_special_tokens.index("user")
            ]
            if "user" in tokenizer.additional_special_tokens
            else 0
        )

        # Find <|im_start|> tokens and their content
        # Only keep labels for assistant turns
        seq = input_ids.tolist()

        in_assistant = False
        for i, token_id in enumerate(seq):
            # Check if this is an im_start token
            if (
                token_id
                == tokenizer.additional_special_tokens_ids[
                    tokenizer.additional_special_tokens.index("im_start")
                ]
            ):
                # Look at next token to determine role
                if i + 1 < len(seq):
                    # This is a simplified approach - actual implementation
                    # would need proper token matching
                    pass

        return labels


# Default system prompt
DEFAULT_SYSTEM_PROMPT = """你是一个专业、友善的AI助手。

能力范围：
- 回答各类知识性问题
- 帮助分析和技术问题
- 进行创意写作和头脑风暴
- 解释复杂概念

行为规范：
- 回答简洁清晰，避免冗长
- 不确定时承认知识边界
- 主动询问以更好理解需求
- 代码给出实用注释

安全约束：
- 拒绝生成有害内容
- 保护用户隐私
- 提供准确信息"""


class DialogueDataSynthesizer:
    """Synthesize dialogue data from various sources"""

    SCENARIOS = [
        {
            "name": "技术咨询",
            "user_persona": "中级开发者",
            "system_style": "专业、简洁",
            "topics": ["代码优化", "架构设计", "调试技巧", "性能分析"],
        },
        {
            "name": "创意写作",
            "user_persona": "内容创作者",
            "system_style": "富有创意、鼓励性",
            "topics": ["故事构思", "文案撰写", "头脑风暴", "修改建议"],
        },
        {
            "name": "学习辅导",
            "user_persona": "好奇的学习者",
            "system_style": "耐心、启发式",
            "topics": ["概念解释", "学习计划", "知识梳理", "练习反馈"],
        },
        {
            "name": "日常对话",
            "user_persona": "友好交流者",
            "system_style": "轻松、自然",
            "topics": ["兴趣爱好", "生活建议", "日程安排", "情感支持"],
        },
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def synthesize_dialogue(self, scenario: str, num_turns: int = 3) -> Dict[str, Any]:
        """Synthesize a single dialogue sample"""
        scenario_config = next(
            (s for s in self.SCENARIOS if s["name"] == scenario), self.SCENARIOS[0]
        )

        turns = []
        for i in range(num_turns):
            # Generate user message
            topic = random.choice(scenario_config["topics"])
            user_msg = self._generate_user_message(scenario_config, topic, i)
            turns.append({"role": "user", "content": user_msg})

            if i < num_turns - 1:
                # Generate assistant response
                assistant_msg = self._generate_assistant_response(
                    scenario_config, topic, i
                )
                turns.append({"role": "assistant", "content": assistant_msg})

        return {"system": DEFAULT_SYSTEM_PROMPT, "turns": turns, "scenario": scenario}

    def _generate_user_message(self, config: Dict, topic: str, turn_idx: int) -> str:
        """Generate user message for scenario"""
        templates = [
            f"关于{topic}，我有一个问题想请教",
            f"你能帮我解释一下{topic}吗？",
            f"我在处理{topic}时遇到了困难",
            f"对{topic}我很感兴趣，能聊聊吗？",
        ]
        return random.choice(templates)

    def _generate_assistant_response(
        self, config: Dict, topic: str, turn_idx: int
    ) -> str:
        """Generate assistant response for scenario"""
        templates = [
            f"关于{topic}，这是一个很有意思的话题。让我来为你解答...",
            f"好的，关于{topic}，我可以从几个方面来解释...",
            f"在{topic}方面，我有几点建议供你参考...",
        ]
        return random.choice(templates)

    def generate_dataset(
        self, num_samples: int = 1000, scenarios: Optional[List[str]] = None
    ) -> List[Dict]:
        """Generate a complete dataset"""
        if scenarios is None:
            scenarios = [s["name"] for s in self.SCENARIOS]

        dataset = []
        for _ in range(num_samples):
            scenario = random.choice(scenarios)
            num_turns = random.randint(2, 5)
            sample = self.synthesize_dialogue(scenario, num_turns)
            dataset.append(sample)

        return dataset


def prepare_sft_dataset(
    data_path: Optional[str] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    max_length: int = 4096,
    data_fraction: float = 1.0,
) -> DialogueDataset:
    """Prepare dataset for SFT training"""
    if tokenizer is None:
        raise ValueError("tokenizer is required")

    if data_path:
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        # Generate synthetic data for demonstration
        synthesizer = DialogueDataSynthesizer()
        raw_data = synthesizer.generate_dataset(num_samples=1000)

    if data_fraction < 1.0:
        random.shuffle(raw_data)
        raw_data = raw_data[: int(len(raw_data) * data_fraction)]

    return DialogueDataset(raw_data, tokenizer, max_length)


class DialogueSFTTrainer:
    """Trainer for Dialogue SFT"""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./dialogue_sft_output",
        max_seq_length: int = 4096,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        epochs: int = 3,
        warmup_ratio: float = 0.1,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio

        self.tokenizer = None
        self.model = None
        self.trainer = None

    def setup(self):
        """Initialize tokenizer and model"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Setup LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train(
        self,
        train_dataset: DialogueDataset,
        eval_dataset: Optional[DialogueDataset] = None,
    ):
        """Run SFT training"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
            report_to=["tensorboard"],
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        self.trainer.train()

    def save_model(self, path: Optional[str] = None):
        """Save the trained model"""
        save_path = path or self.output_dir
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)


def main():
    """Example usage"""
    # Configuration
    config = {
        "model_path": "meta-llama/Llama-2-7b",
        "output_dir": "./checkpoints/dialogue_sft",
        "max_seq_length": 4096,
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "epochs": 3,
    }

    # Example: Prepare dataset and train
    print("Dialogue SFT Module loaded")
    print("Use DialogueSFTTrainer class to train your model")


if __name__ == "__main__":
    main()
