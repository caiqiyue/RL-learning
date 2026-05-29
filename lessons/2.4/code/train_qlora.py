"""
QLoRA Training Script
======================
Complete training script for QLoRA fine-tuning on a single GPU.
Supports both large models (65B) and small test models (1B like TinyLlama).

Usage:
    # Test with tiny model (works on 8GB GPU):
    python train_qlora.py --model_name PY007/TinyLlama-1.1B-step-50K-103k

    # Full 7B model (needs ~12GB VRAM):
    python train_qlora.py --model_name meta-llama/Llama-2-7b --lora_r 64

    # Custom training:
    python train_qlora.py --model_name <model> --num_epochs 3 --batch_size 2
"""

import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from bitsandbytes import BitsAndBytesConfig

from qlora_config import QLoRAConfig, create_qlora_config, MemoryOptimizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="QLoRA Training Script")

    parser.add_argument(
        "--model_name",
        type=str,
        default="PY007/TinyLlama-1.1B-step-50K-103k",
        help="Model name or path (default: TinyLlama for testing)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Dataset name (default: wikitext)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration (default: wikitext-2-raw-v1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qlora_output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (default: 16, use 64 for larger models)",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size (default: 4 for small models)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs (default: 1 for testing)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for memory savings",
    )

    return parser.parse_args()


def print_memory_usage(stage: str):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"[Memory] {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )


def create_quantization_config():
    """
    Create BitsAndBytesConfig for 4-bit quantization.

    Key parameters:
    - load_in_4bit: Enable 4-bit model loading
    - bnb_4bit_quant_type: "nf4" (NormalFloat4) is optimal for neural network weights
    - bnb_4bit_compute_dtype: BF16 for training stability
    - bnb_4bit_use_double_quant: Quantize scale parameters to save ~1GB
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4 - optimal for normal weight distributions
        bnb_4bit_compute_dtype=torch.bfloat16,  # BF16 - better training stability than FP16
        bnb_4bit_use_double_quant=True,  # Double quantization - saves ~1GB for 65B model
        bnb_4bit_quant_storage=torch.uint8,  # Store scale parameters in 8-bit
    )


def create_lora_config(lora_r: int, lora_alpha: int):
    """
    Create LoraConfig for QLoRA training.

    Args:
        lora_r: LoRA rank (higher = more parameters, better quality, more memory)
        lora_alpha: LoRA scaling factor (typically 2x lora_r)

    Returns:
        LoraConfig: Configured PEFT LoraConfig object
    """
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # Target modules for causal language models
        target_modules=[
            "q_proj",
            "v_proj",  # Attention query and value projections
            # "k_proj", "o_proj",         # Uncomment for full attention coverage
            # "gate_proj", "up_proj", "down_proj"  # FFN layers (adds ~0.1GB)
        ],
        lora_dropout=0.05,
        bias="none",  # Don't train bias terms
        task_type=TaskType.CAUSAL_LM,  # Causal language modeling task
        inference_mode=False,  # Training mode
    )


def load_model_and_tokenizer(
    model_name: str, lora_config: LoraConfig, use_gradient_checkpointing: bool
):
    """
    Load quantized model with LoRA adapters.

    This function:
    1. Loads the model in 4-bit NF4 format
    2. Prepares it for kbit training (handles quantized model constraints)
    3. Attaches LoRA adapters
    4. Optionally enables gradient checkpointing

    Args:
        model_name: HuggingFace model identifier
        lora_config: LoRA configuration
        use_gradient_checkpointing: Whether to enable gradient checkpointing

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print_memory_usage("Before model loading")

    # Configure quantization
    bnb_config = create_quantization_config()

    # Load model with quantization
    # trust_remote_code=True allows custom model architectures
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute across devices
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Model computation dtype
    )
    print_memory_usage("After model loading (quantized)")

    # Prepare model for kbit training
    # This handles issues with quantized models (like gradient computation)
    model = prepare_model_for_kbit_training(model)
    print_memory_usage("After prepare_model_for_kbit_training")

    # Attach LoRA adapters
    model = get_peft_model(model, lora_config)
    print_memory_usage("After LoRA attachment")

    # Print trainable parameters
    model.print_trainable_parameters()

    # Enable gradient checkpointing if requested
    # This saves ~40% memory by recomputing activations during backprop
    if use_gradient_checkpointing:
        MemoryOptimizer.enable_gradient_checkpointing(model)
        print("Gradient checkpointing enabled")

    print_memory_usage("After gradient checkpointing setup")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    return model, tokenizer


def prepare_dataset(
    tokenizer, dataset_name: str, dataset_config: str, max_seq_length: int
):
    """
    Prepare dataset for training.

    Loads and tokenizes the dataset, applying appropriate preprocessing
    for causal language modeling.

    Args:
        tokenizer: Tokenizer to use
        dataset_name: Dataset identifier (e.g., "wikitext")
        dataset_config: Dataset configuration (e.g., "wikitext-2-raw-v1")
        max_seq_length: Maximum sequence length for tokenization

    Returns:
        Dataset: Prepared training dataset
    """
    print(f"Loading dataset: {dataset_name}/{dataset_config}")

    # Load raw dataset
    raw_dataset = load_dataset(dataset_name, dataset_config, split="train")

    # Tokenize function
    def tokenize_function(examples):
        """
        Tokenize text with proper truncation and padding.

        For causal LM, we use the same text as both input and label
        (shifted by one position).
        """
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",  # Pad to max_length for efficiency
            return_attention_mask=True,
            return_token_type_ids=False,
        )

    # Map tokenization (batched for efficiency)
    print("Tokenizing dataset...")
    dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,  # Parallel processing
        remove_columns=["text"],  # Remove original text column
    )

    # Filter out examples that are too short (less than 10 tokens)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) >= 10)

    print(f"Dataset prepared: {len(dataset)} samples")
    return dataset


def create_trainer(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_epochs: int,
):
    """
    Create and configure the Trainer.

    Args:
        model: PEFT model with LoRA adapters
        tokenizer: Tokenizer for data collation
        dataset: Prepared training dataset
        output_dir: Directory for checkpoints
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        num_epochs: Number of training epochs

    Returns:
        Trainer: Configured trainer instance
    """
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        # Mixed precision settings
        bf16=True,  # BF16 mixed precision (recommended over FP16)
        fp16=False,
        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False  # Recommended for better memory management
        },
        # Optimizer settings
        optim="paged_adamw_32bit",  # Paged optimizer - keeps states in CPU
        optim_args={"page_size": 1},
        # Logging and saving
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,  # Keep only last 2 checkpoints
        # Learning rate schedule
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        # Other settings
        group_by_length=True,  # Group similar lengths for efficiency
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard integration
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM (not masked LM)
        return_tensors="pt",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    return trainer


def merge_and_export(model, tokenizer, output_dir: str):
    """
    Merge LoRA adapters with base model and export.

    After training, we can merge the LoRA weights with the quantized
    base model to create a standalone model that doesn't require
    LoRA loading.

    Args:
        model: Trained PEFT model
        tokenizer: Tokenizer to save with model
        output_dir: Directory to save merged model
    """
    print("Merging LoRA weights with base model...")

    # Merge adapters
    merged_model = model.merge_and_unload()

    # Save merged model
    print(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Model export complete!")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    print("=" * 60)
    print("QLoRA Training Script")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(
        f"Batch size: {args.batch_size}, Grad accumulation: {args.gradient_accumulation_steps}"
    )
    print(f"Learning rate: {args.learning_rate}, Epochs: {args.num_epochs}")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. QLoRA requires GPU.")
        sys.exit(1)

    print_memory_usage("Start")

    # Create LoRA configuration
    lora_config = create_lora_config(args.lora_r, args.lora_alpha)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, lora_config, args.use_gradient_checkpointing
    )

    # Prepare dataset
    dataset = prepare_dataset(
        tokenizer, args.dataset_name, args.dataset_config, args.max_seq_length
    )

    print_memory_usage("After dataset preparation")

    # Create trainer
    trainer = create_trainer(
        model,
        tokenizer,
        dataset,
        args.output_dir,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.num_epochs,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train()

    print_memory_usage("After training")

    # Save model (PEFT format)
    print(f"\nSaving PEFT model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Print memory summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print_memory_usage("Final")

    # Print training summary
    print(f"""
Training Summary:
    - Model: {args.model_name}
    - LoRA r: {args.lora_r}, alpha: {args.lora_alpha}
    - Final loss: {trainer.state.log_history[-1]["loss"] if trainer.state.log_history else "N/A"}
    - Output saved to: {args.output_dir}
    
To test the trained model, you can:
1. Load the model with LoRA adapters
2. Or merge and export using merge_and_export()
    """)


if __name__ == "__main__":
    main()
