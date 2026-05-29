"""
QLoRA Configuration Module
============================
This module defines the configuration classes for QLoRA training,
including LoraConfig and BitsAndBytes quantization settings.
"""

from dataclasses import dataclass
from typing import List, Optional
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig


@dataclass
class QLoRAConfig:
    """
    QLoRA configuration container holding all necessary settings for
    quantized LoRA training.

    Attributes:
        model_name: HuggingFace model identifier or local path
        lora_r: LoRA rank dimension (larger = more parameters, better quality)
        lora_alpha: LoRA scaling factor (typically 2x r)
        target_modules: List of module names to apply LoRA to
        lora_dropout: Dropout probability for LoRA layers
        output_dir: Directory for saving checkpoints
        batch_size: Training batch size per device
        gradient_accumulation_steps: Number of steps to accumulate gradients
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        max_seq_length: Maximum sequence length for tokenization
    """

    model_name: str
    lora_r: int = 64
    lora_alpha: int = 16
    target_modules: List[str] = None
    lora_dropout: float = 0.05
    output_dir: str = "./qlora_output"
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 3
    max_seq_length: int = 512

    def __post_init__(self):
        """Set default target modules if not specified."""
        if self.target_modules is None:
            # Default targets for causal language models
            self.target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

    def get_lora_config(self) -> LoraConfig:
        """
        Create a LoraConfig instance for PEFT.

        Returns:
            LoraConfig: Configured PEFT LoraConfig object
        """
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            # Additional stability settings
            inference_mode=False,
            modules_to_save=None,
        )

    def get_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create BitsAndBytesConfig for 4-bit NF4 quantization.

        Key settings explained:
        - load_in_4bit: Enable 4-bit quantization for model weights
        - bnb_4bit_quant_type="nf4": Use NormalFloat4 (optimal for normal distributions)
        - bnb_4bit_compute_dtype: Computation precision (bfloat16 recommended)
        - bnb_4bit_use_double_quant: Quantize scale parameters (saves ~1GB)

        Returns:
            BitsAndBytesConfig: Configured quantization settings
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # 4-bit NormalFloat
            bnb_4bit_compute_dtype="bfloat16",  # BF16 for training stability
            bnb_4bit_use_double_quant=True,  # Double quantization for scale parameters
            bnb_4bit_quant_storage="uint8",  # Store scale params in 8-bit
        )

    def get_training_arguments(self) -> dict:
        """
        Generate training arguments dictionary for Trainer.

        Returns:
            dict: Training configuration arguments
        """
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "bf16": True,  # Use BF16 mixed precision
            "fp16": False,
            "gradient_checkpointing": True,  # Memory optimization
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "optim": "paged_adamw_32bit",  # Paged optimizer for memory
            "logging_steps": 10,
            "save_steps": 100,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "save_total_limit": 2,
            "remove_unused_columns": False,
        }


class MemoryOptimizer:
    """
    Utility class for memory optimization strategies in QLoRA training.

    Provides static methods for configuring various memory-saving techniques.
    """

    @staticmethod
    def enable_gradient_checkpointing(model):
        """
        Enable gradient checkpointing to save memory during backpropagation.

        Trade-off: Uses ~30% more compute time to save ~40% memory.

        Args:
            model: PyTorch model to enable checkpointing on
        """
        model.gradient_checkpointing_enable()
        # Additional settings for better memory management
        model.enable_input_require_grad()

    @staticmethod
    def configure_paged_optimizer():
        """
        Configure paged optimizer settings.

        Paged optimizer keeps optimizer states in CPU memory and
        swaps them to GPU as needed, significantly reducing GPU memory.
        """
        return {"optim": "paged_adamw_32bit", "optim_args": {"page_size": 1}}

    @staticmethod
    def get_memory_breakdown(model_size: str) -> dict:
        """
        Estimate memory breakdown for different model sizes.

        Args:
            model_size: Model size in billions (e.g., "7B", "13B", "65B")

        Returns:
            dict: Memory breakdown estimates
        """
        size_b = int(model_size.replace("B", ""))

        # QLoRA memory breakdown (approximate)
        breakdown = {
            "base_model_weights": size_b * 0.5,  # NF4: 4-bit = 0.5 bytes/param
            "quantization_scales": size_b * 0.008,  # Double quantized scales
            "lora_params": size_b * 0.003,  # LoRA trainable params (r=64)
            "gradients": size_b * 0.003,  # Only LoRA gradients
            "optimizer_states": size_b * 0.006,  # Paged to CPU
            "activations": size_b * 0.12,  # Varies with batch_size
            "kv_cache": size_b * 0.06,  # Varies with seq_length
            "other_overhead": size_b * 0.03,
        }
        breakdown["total_gpu"] = sum(
            v
            for k, v in breakdown.items()
            if k not in ["optimizer_states", "other_overhead"]
        )
        breakdown["total_with_cpu"] = sum(breakdown.values())

        return breakdown


def create_qlora_config(
    model_name: str,
    lora_r: int = 64,
    lora_alpha: int = 16,
    target_modules: List[str] = None,
    lora_dropout: float = 0.05,
    output_dir: str = "./qlora_output",
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    max_seq_length: int = 512,
) -> QLoRAConfig:
    """
    Factory function to create a QLoRAConfig with custom settings.

    Example:
        >>> config = create_qlora_config(
        ...     model_name="meta-llama/Llama-2-7b",
        ...     lora_r=64,
        ...     batch_size=1,
        ...     gradient_accumulation_steps=8
        ... )
    """
    return QLoRAConfig(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        output_dir=output_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_seq_length=max_seq_length,
    )


if __name__ == "__main__":
    # Example: Print memory breakdown for different model sizes
    print("QLoRA Memory Breakdown Estimates:")
    print("=" * 50)

    for size in ["7B", "13B", "33B", "65B"]:
        breakdown = MemoryOptimizer.get_memory_breakdown(size)
        print(f"\n{size} Model:")
        print(f"  GPU Memory: ~{breakdown['total_gpu']:.1f} GB")
        print(f"  Total (with CPU): ~{breakdown['total_with_cpu']:.1f} GB")
