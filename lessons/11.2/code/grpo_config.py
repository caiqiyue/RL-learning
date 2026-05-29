from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GRPOConfig:
    group_size: int = 16
    kl_coef: float = 0.04
    gamma: float = 1.0
    clip_range: float = 0.2
    clip_range_ratio: float = 2.0
    learning_rate: float = 1e-6
    lr_scheduler_name: str = "cosine"
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    beta: float = 0.004
    ref_model: Optional[str] = None
    max_prompt_length: int = 512
    max_response_length: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    num_episodes: int = 100
    episodes_per_batch: int = 1

    def __post_init__(self):
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO to work")
        if self.kl_coef <= 0:
            raise ValueError("kl_coef must be positive")
        self.beta = self.kl_coef


@dataclass
class DatasetConfig:
    name: str = "gsm8k"
    data_dir: str = "./datasets"
    train_split: str = "train"
    test_split: str = "test"
    max_examples: Optional[int] = None


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_type: str = "qwen2"
    torch_dtype: str = "float32"
    use_flash_attention: bool = False
    trust_remote_code: bool = True
