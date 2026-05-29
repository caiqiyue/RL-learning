"""
Domain-Specific Supervised Fine-Tuning (SFT)
Medical and Legal vertical domain fine-tuning with LoRA/QLoRA
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    TaskType,
    PeftModel,
)
from peft import PeftConfig
import bitsandbytes as bnb
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DomainSFTConfig:
    """Configuration for domain-specific SFT"""

    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf"
    output_dir: str = "./output"

    load_in_8bit: bool = True
    load_in_4bit: bool = False
    use_flash_attn: bool = False

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    max_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    fp16: bool = True
    gradient_checkpointing: bool = True

    domain: str = "general"
    use_rag: bool = False
    rag_top_k: int = 5

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Cannot set both load_in_8bit and load_in_4bit")


class DomainSFTDataset(Dataset):
    """Dataset for domain-specific SFT"""

    def __init__(
        self,
        instructions: List[str],
        responses: List[str],
        tokenizer,
        max_length: int = 2048,
        domain: str = "general",
    ):
        self.instructions = instructions
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.domain = domain

    def __len__(self):
        return len(self.instructions)

    def create_instruction_prompt(self, instruction: str, response: str) -> str:
        """Create prompt based on domain"""
        if self.domain == "medical":
            return (
                f"<|system|>\n"
                f"You are a medical AI assistant with expertise in clinical diagnosis, "
                f"treatment recommendations, and healthcare information. "
                f"Provide accurate, evidence-based medical information while emphasizing "
                f"the importance of consulting healthcare professionals."
                f"<|user|>\n{instruction}<|end|>\n"
                f"<|assistant|>\n{response}<|end|>"
            )
        elif self.domain == "legal":
            return (
                f"<|system|>\n"
                f"You are a legal AI assistant with expertise in Chinese law, "
                f"legal analysis, case evaluation, and legal document preparation. "
                f"Provide accurate legal information while emphasizing that this is not "
                f"legal advice and consulting a licensed attorney is recommended."
                f"<|user|>\n{instruction}<|end|>\n"
                f"<|assistant|>\n{response}<|end|>"
            )
        else:
            return f"Instruction: {instruction}\n\nResponse: {response}"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        instruction = self.instructions[idx]
        response = self.responses[idx]

        prompt = self.create_instruction_prompt(instruction, response)

        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )

        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]

        labels = input_ids.copy()

        prompt_len = len(
            self.tokenizer(
                f"Instruction: {instruction}\n\nResponse: ", return_tensors=None
            )["input_ids"]
        )

        labels[:prompt_len] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_and_prepare_model(config: DomainSFTConfig) -> Tuple[Any, Any, Any]:
    """Load and prepare model with LoRA configuration"""

    logger.info(f"Loading model: {config.model_name_or_path}")

    if config.load_in_4bit:
        logger.info("Using QLoRA (4-bit quantization)")
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.load_in_8bit:
        logger.info("Using 8-bit quantization")
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=True, use_fast=False
    )

    tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if not config.load_in_4bit else None,
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    if config.use_flash_attn:
        model_kwargs["use_flash_attention_2"] = True

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, **model_kwargs
    )

    model.config.use_cache = False

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grad()

    logger.info("Setting up LoRA configuration")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer, None


def find_all_linear_layers(model: Any) -> List[str]:
    """Find all linear layers in the model for LoRA targeting"""
    linear_layers = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bit)):
            if "lm_head" not in name:
                linear_layers.append(name.split(".")[-1])

    return list(set(linear_layers))


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for data loader"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class DomainSFTTrainer:
    """Trainer for domain-specific SFT"""

    def __init__(self, config: DomainSFTConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.eval_loader = None

        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        self.train_history = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def setup_model(self):
        """Initialize model, tokenizer, and optimizer"""
        self.model, self.tokenizer, _ = create_and_prepare_model(self.config)
        self.model.to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )

    def setup_training(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ):
        """Setup data loaders and scheduler"""

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.per_device_train_batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_fn,
                pin_memory=True,
            )

        total_steps = (
            len(self.train_loader)
            * self.config.num_train_epochs
            // self.config.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step"""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        if self.config.fp16:
            with autocast():
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
        else:
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

        loss = outputs.loss / self.config.gradient_accumulation_steps

        if self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        train_stats = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

        return train_stats

    def eval_step(self) -> Dict[str, float]:
        """Execute evaluation step"""
        if not self.eval_loader:
            return {}

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if self.config.fp16:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        return {"eval_loss": avg_loss}

    def save_checkpoint(self, is_final: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"

        if is_final:
            checkpoint_dir = self.output_dir / "final_model"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {checkpoint_dir}")

        self.model.save_checkpoint(str(checkpoint_dir))

        train_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "train_history": self.train_history,
            "config": vars(self.config),
        }

        with open(checkpoint_dir / "train_state.json", "w") as f:
            json.dump(train_state, f, indent=2)

    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        """Execute full training loop"""

        logger.info("Setting up model")
        self.setup_model()

        logger.info("Setting up training")
        self.setup_training(train_dataset, eval_dataset)

        if self.config.fp16:
            self.scaler = GradScaler()

        logger.info(f"Starting training for {self.config.num_train_epochs} epochs")
        logger.info(
            f"Total training steps: {len(self.train_loader) * self.config.num_train_epochs}"
        )

        total_steps = len(self.train_loader) * self.config.num_train_epochs

        progress_bar = tqdm(total=total_steps, desc="Training")

        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch

            for step, batch in enumerate(self.train_loader):
                train_stats = self.train_step(batch)

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    progress_bar.update(1)

                    if self.global_step % self.config.logging_steps == 0:
                        logger.info(
                            f"Step {self.global_step} | "
                            f"Loss: {train_stats['loss']:.4f} | "
                            f"LR: {train_stats['learning_rate']:.2e}"
                        )

                    if self.global_step % self.config.eval_steps == 0 and eval_dataset:
                        eval_stats = self.eval_step()
                        logger.info(
                            f"Eval at step {self.global_step} | "
                            f"Eval Loss: {eval_stats.get('eval_loss', 0):.4f}"
                        )

                        if (
                            eval_stats.get("eval_loss", float("inf"))
                            < self.best_eval_loss
                        ):
                            self.best_eval_loss = eval_stats["eval_loss"]
                            self.save_checkpoint()

                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

                self.train_history.append(
                    {"step": self.global_step, "epoch": epoch, **train_stats}
                )

            logger.info(f"Completed epoch {epoch + 1}/{self.config.num_train_epochs}")

        progress_bar.close()

        self.save_checkpoint(is_final=True)

        final_stats = {
            "total_steps": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "train_history": self.train_history,
        }

        logger.info("Training complete!")

        return final_stats


class DomainSFTPredictor:
    """Inference wrapper for domain-specific SFT model"""

    def __init__(self, model_path: str, config: Optional[DomainSFTConfig] = None):
        self.model_path = model_path
        self.config = config or DomainSFTConfig()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name = peft_config.base_model_name_or_path

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()

    def create_prompt(self, instruction: str) -> str:
        """Create prompt based on domain"""
        domain = self.config.domain

        if domain == "medical":
            return (
                f"<|system|>\n"
                f"You are a medical AI assistant. Provide accurate, evidence-based "
                f"medical information while emphasizing consulting healthcare professionals."
                f"<|user|>\n{instruction}<|end|>\n"
                f"<|assistant|>\n"
            )
        elif domain == "legal":
            return (
                f"<|system|>\n"
                f"You are a legal AI assistant. Provide accurate legal information "
                f"while emphasizing this is not legal advice."
                f"<|user|>\n{instruction}<|end|>\n"
                f"<|assistant|>\n"
            )
        else:
            return f"Instruction: {instruction}\n\nResponse:"

    def generate(
        self,
        instruction: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate response for given instruction"""
        prompt = self.create_prompt(instruction)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length - max_new_tokens,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|assistant|>\n" in response:
            response = response.split("<|assistant|>\n")[-1]
        elif "Response:" in response:
            response = response.split("Response:")[-1]

        return response.strip()

    def batch_generate(self, instructions: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple instructions"""
        return [self.generate(instr, **kwargs) for instr in instructions]


def main():
    """Example usage of domain-specific SFT training"""
    import argparse

    parser = argparse.ArgumentParser(description="Domain-Specific SFT Training")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument(
        "--domain", choices=["medical", "legal", "general"], required=True
    )
    parser.add_argument("--output_dir", default="./domain_sft_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)

    args = parser.parse_args()

    config = DomainSFTConfig(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        domain=args.domain,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        lora_r=args.lora_r,
        load_in_8bit=True,
    )

    logger.info(f"Loading dataset from {args.dataset}")
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    instructions = data["instructions"]
    responses = data["responses"]

    logger.info(f"Loaded {len(instructions)} samples")

    trainer = DomainSFTTrainer(config)

    model, tokenizer, _ = create_and_prepare_model(config)

    train_dataset = DomainSFTDataset(
        instructions=instructions,
        responses=responses,
        tokenizer=tokenizer,
        max_length=config.max_length,
        domain=args.domain,
    )

    train_stats = trainer.train(train_dataset)

    logger.info("Training completed!")
    logger.info(f"Final stats: {train_stats}")


if __name__ == "__main__":
    main()
