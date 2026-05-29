#!/usr/bin/env python3
"""
DPO Training Script using TRL's DPOTrainer

Trains a language model using Direct Preference Optimization.
Supports configurable beta, label smoothing, and evaluation.

Usage:
    python train_dpo.py --model_name meta-llama/Llama-2-7b-hf --dataset Anthropic/hh-rlhf
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO Training with TRL")

    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-2-7b-hf",
        help="Base model name or path",
    )
    parser.add_argument(
        "--reference_model",
        default=None,
        help="Reference model (defaults to model_name)",
    )
    parser.add_argument("--dataset", default="Anthropic/hh-rlhf", help="Dataset name")
    parser.add_argument("--dataset_split", default="train", help="Dataset split")
    parser.add_argument("--output_dir", default="./dpo_output", help="Output directory")

    parser.add_argument(
        "--beta", type=float, default=0.3, help="KL penalty coefficient"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0, help="Label smoothing factor"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm"
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Train batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Eval batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=3.0, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")

    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=256, help="Maximum prompt length"
    )

    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Evaluate every N steps"
    )

    parser.add_argument("--fp16", action="store_true", help="Use fp16 precision")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision")
    parser.add_argument("--no_eval", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--use_peft", action="store_true", help="Use PEFT for efficient training"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def prepare_dataset(
    dataset: Dataset, tokenizer: AutoTokenizer, max_prompt_length: int = 256
) -> Dataset:
    """
    Prepare dataset for DPO training by tokenizing prompts and responses.
    """

    def tokenize_split(example):
        prompt = example.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "\n".join(
                [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt]
            )

        chosen = example["chosen"]
        rejected = example["rejected"]

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    dataset = dataset.map(
        tokenize_split, remove_columns=dataset.column_names, desc="Preparing dataset"
    )

    return dataset


def setup_models(args: argparse.Namespace):
    """Initialize model, reference model, and tokenizer."""
    logger.info(f"Loading model: {args.model_name}")

    torch_dtype = torch.float16
    if args.bf16:
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"

    if args.use_peft:
        from peft import LoraConfig, get_peft_model

        logger.info("Applying PEFT LoRA configuration")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.reference_model:
        logger.info(f"Loading reference model: {args.reference_model}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.reference_model,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        logger.info("Using base model as reference model")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return model, ref_model, tokenizer


def compute_reward_margin(model, tokenizer, dataset, num_samples: int = 100) -> float:
    """
    Compute average reward margin (chosen - rejected) on samples.
    """
    model.eval()
    total_margin = 0.0
    samples = min(num_samples, len(dataset))

    for i in range(samples):
        example = dataset[i]
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]

        with torch.no_grad():
            chosen_inputs = tokenizer(
                prompt, chosen, return_tensors="pt", truncation=True, max_length=512
            )
            rejected_inputs = tokenizer(
                prompt, rejected, return_tensors="pt", truncation=True, max_length=512
            )

            chosen_inputs = {k: v.to(model.device) for k, v in chosen_inputs.items()}
            rejected_inputs = {
                k: v.to(model.device) for k, v in rejected_inputs.items()
            }

            chosen_outputs = model(**chosen_inputs)
            rejected_outputs = model(**rejected_inputs)

            chosen_score = chosen_outputs.logits.mean().item()
            rejected_score = rejected_outputs.logits.mean().item()

            total_margin += chosen_score - rejected_score

    return total_margin / samples


class DPOTrainingMonitor:
    """Monitor DPO training metrics."""

    def __init__(self, trainer, eval_dataset):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step = state.global_step

        if self.step % args.logging_steps == 0:
            metrics = {
                "step": self.step,
                "learning_rate": self.trainer.get_learning_rate(),
            }

            if hasattr(self.trainer, "ref_model"):
                try:
                    margin = compute_reward_margin(
                        self.trainer.model,
                        self.trainer.tokenizer,
                        self.eval_dataset,
                        num_samples=20,
                    )
                    metrics["reward_margin"] = margin
                except Exception as e:
                    logger.warning(f"Failed to compute reward margin: {e}")

            self.trainer.log(metrics)


def main():
    args = parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, ref_model, tokenizer = setup_models(args)

    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    logger.info(f"Loaded {len(dataset)} samples")

    dataset = prepare_dataset(dataset, tokenizer, args.max_prompt_length)

    if args.no_eval:
        train_dataset = dataset
        eval_dataset = None
    else:
        split = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        report_to=["tensorboard"],
        seed=args.seed,
        remove_unused_columns=False,
    )

    logger.info("Initializing DPOTrainer")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        label_smoothing=args.label_smoothing if args.label_smoothing > 0 else None,
    )

    logger.info("Starting DPO training")
    dpo_trainer.train()

    final_model_path = Path(args.output_dir) / "final_model"
    logger.info(f"Saving final model to {final_model_path}")
    dpo_trainer.save_model(str(final_model_path))

    if not args.no_eval and eval_dataset:
        logger.info("Running final evaluation")
        eval_results = dpo_trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
