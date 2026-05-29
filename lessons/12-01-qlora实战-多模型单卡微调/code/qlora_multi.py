import argparse
import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Single model configuration"""

    name: str
    model_path: str
    model_type: str
    trust_remote_code: bool = False
    max_seq_length: int = 2048
    dataset_path: str = None
    dataset_split: str = "train"


@dataclass
class LoRAConfig:
    """LoRA configuration"""

    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=list)
    bias: str = "none"


def get_model_specific_lora_targets(model_type: str) -> List[str]:
    """Get LoRA target modules for different model types"""
    targets_map = {
        "llama": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "qwen2": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "cm_proj",
        ],
        "chatglm": ["query_key_value", "dense", "mlp"],
        "deepseek": ["q_proj", "v_proj", "gate_proj", "up_proj"],
        "baichuan": ["W_pack", "gate_proj", "up_proj", "down_proj"],
    }
    return targets_map.get(model_type, targets_map["llama"])


def get_default_lora_targets(model_type: str) -> List[str]:
    """Get default LoRA target modules based on model type"""
    return get_model_specific_lora_targets(model_type)


class MultiModelQLoraTrainer:
    """Trainer for multiple models with QLoRA on single GPU"""

    def __init__(
        self,
        output_dir: str = "./output",
        per_device_batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        warmup_steps: int = 100,
        fp16: bool = True,
        logging_steps: int = 10,
        save_steps: int = 500,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.fp16 = fp16
        self.logging_steps = logging_steps
        self.save_steps = save_steps

        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None

    def estimate_memory(
        self,
        model_path: str,
        model_type: str,
        lora_rank: int = 64,
        max_seq_length: int = 2048,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """Estimate memory requirements for a model"""
        size_map = {
            "7b": 7,
            "8b": 8,
            "13b": 13,
            "14b": 14,
            "33b": 33,
            "65b": 65,
            "70b": 70,
        }

        param_count = 7
        for size_str, count in size_map.items():
            if size_str in model_path.lower():
                param_count = count
                break

        quantized_base_gb = param_count * 0.25 * 2
        lora_params = param_count * (lora_rank * 4) / 1e9
        lora_gb = lora_params * 2
        gradient_gb = lora_params * 4
        optimizer_gb = lora_params * 8
        activation_gb = (
            batch_size * max_seq_length * 4096 * 4 * 1e-9
            if model_type in ["llama", "qwen2"]
            else batch_size * max_seq_length * 4096 * 3 * 1e-9
        )

        return {
            "param_count_b": param_count,
            "quantized_base_gb": quantized_base_gb,
            "lora_gb": lora_gb,
            "gradient_gb": gradient_gb,
            "optimizer_gb": optimizer_gb,
            "activation_gb": activation_gb,
            "total_gb": sum(
                [
                    quantized_base_gb,
                    lora_gb,
                    gradient_gb,
                    optimizer_gb,
                    activation_gb,
                ]
            ),
        }

    def load_model(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
    ):
        """Load model and tokenizer, apply LoRA"""
        self._clear_memory()

        logger.info(f"Loading model: {model_config.name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code,
            padding_side="right",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_path,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            device_map="auto",
            trust_remote_code=model_config.trust_remote_code,
        )

        model = prepare_model_for_kbit_training(model)

        target_modules = lora_config.target_modules
        if not target_modules:
            target_modules = get_default_lora_targets(model_config.model_type)

        lora_cfg = LoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            target_modules=target_modules,
            lora_dropout=lora_config.dropout,
            bias=lora_config.bias,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        self.current_model = model
        self.current_tokenizer = tokenizer
        self.current_model_name = model_config.name

        return model, tokenizer

    def unload_model(self):
        """Unload current model and free memory"""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None

        if self.current_tokenizer is not None:
            del self.current_tokenizer
            self.current_tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
        self.current_model_name = None

        logger.info("Model unloaded, memory cleared")

    def _clear_memory(self):
        """Clear existing model from memory"""
        self.unload_model()

    def _load_dataset(self, path: str, split: str, tokenizer):
        """Load and tokenize dataset"""
        from datasets import load_dataset

        def tokenize_function(examples):
            result = tokenizer(
                examples["text"],
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        dataset = load_dataset("json", data_files=path, split=split)
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

    def _create_trainer(self, model, tokenizer, dataset, output_dir: str):
        """Create trainer instance"""
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=2,
            fp16=self.fp16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        return trainer

    def train_model(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
    ) -> Dict:
        """
        Train a single model with QLoRA

        Args:
            model_config: Model configuration
            lora_config: LoRA configuration

        Returns:
            Training metrics
        """
        logger.info(f"Starting training for model: {model_config.name}")

        model, tokenizer = self.load_model(model_config, lora_config)

        if not model_config.dataset_path:
            raise ValueError("dataset_path is required for training")

        dataset = self._load_dataset(
            model_config.dataset_path, model_config.dataset_split, tokenizer
        )

        output_dir = self.output_dir / "adapters" / model_config.name
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer = self._create_trainer(model, tokenizer, dataset, output_dir)

        metrics = trainer.train()

        adapter_path = output_dir / "adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        self.unload_model()

        logger.info(f"Training completed for {model_config.name}")

        return metrics.metrics

    def train_multiple_models(
        self,
        models: List[Dict],
        lora_config: LoRAConfig,
    ) -> Dict[str, Dict]:
        """
        Train multiple models sequentially

        Args:
            models: List of model configurations
            lora_config: Shared LoRA configuration

        Returns:
            Training metrics for each model
        """
        results = {}

        for model_dict in models:
            model_config = ModelConfig(**model_dict)

            try:
                metrics = self.train_model(model_config, lora_config)
                results[model_config.name] = {
                    "status": "success",
                    "metrics": metrics,
                }
            except Exception as e:
                logger.error(f"Training failed for {model_config.name}: {e}")
                results[model_config.name] = {
                    "status": "failed",
                    "error": str(e),
                }

        return results

    def save_checkpoint(self, model_name: str, step: int, metrics: Dict = None):
        """Save checkpoint for current model"""
        if self.current_model is None:
            raise ValueError("No model currently loaded")

        checkpoint_dir = self.output_dir / "checkpoints" / model_name / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_model.save_pretrained(checkpoint_dir)
        self.current_tokenizer.save_pretrained(checkpoint_dir)

        import json

        metadata = {
            "model_name": model_name,
            "step": step,
            "metrics": metrics or {},
        }
        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return str(checkpoint_dir)


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA multi-model training on single GPU"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--per_device_batch_size", type=int, default=1, help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    args = parser.parse_args()

    import json

    with open(args.config, "r") as f:
        config = json.load(f)

    trainer = MultiModelQLoraTrainer(
        output_dir=args.output_dir,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
    )

    lora_cfg = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target_modules", []),
        bias=config.get("lora_bias", "none"),
    )

    results = trainer.train_multiple_models(
        models=config["models"],
        lora_config=lora_cfg,
    )

    logger.info("=" * 50)
    logger.info("Training Summary:")
    for model_name, result in results.items():
        status = result["status"]
        logger.info(f"  {model_name}: {status}")
    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    main()
