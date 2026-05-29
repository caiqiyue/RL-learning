#!/usr/bin/env python3
"""
Unsloth LoRA Training Script
Unsloth加速的LoRA微调训练脚本，支持消费级GPU高效微调
"""

import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import json
import os


def load_model_and_tokenizer(model_name, max_seq_length=2048, load_in_4bit=True):
    """
    加载Unsloth优化模型和分词器

    Args:
        model_name: HuggingFace模型ID或本地路径
        max_seq_length: 最大序列长度
        load_in_4bit: 是否启用4bit量化

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=load_in_4bit,
    )

    print(f"Model loaded: {model_name}")
    print(f"Model dtype: {model.dtype}")
    print(f"Tokenizer: {tokenizer.__class__.__name__}")

    return model, tokenizer


def configure_lora(model, r=16, lora_alpha=16, target_modules=None):
    """
    配置LoRA参数

    Args:
        model: 已加载的基础模型
        r: LoRA rank，越大表示更强的微调能力
        lora_alpha: LoRA缩放因子
        target_modules: 要应用LoRA的模块列表

    Returns:
        配置好LoRA的模型
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    print(f"Configuring LoRA: r={r}, alpha={lora_alpha}")
    print(f"Target modules: {target_modules}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    model.print_trainable_parameters()

    return model


def prepare_dataset(tokenizer, data_path, max_samples=None):
    """
    准备训练数据集

    Args:
        tokenizer: 分词器
        data_path: JSONL数据文件路径
        max_samples: 最大样本数，None表示全部

    Returns:
        处理好的Dataset
    """
    print(f"Loading dataset from: {data_path}")

    def formatting_prompts_func(example):
        if "messages" in example:
            text = ""
            for msg in example["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"<|{role}|>\n{content}\n"
            text += "<|assistant|>\n"
        elif "text" in example:
            text = example["text"]
        else:
            text = str(example)

        return {"text": text}

    dataset = load_dataset("json", data_files=data_path, split="train")

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    dataset = dataset.map(formatting_prompts_func, remove_columns=dataset.column_names)

    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 8)

    print(f"Dataset prepared: {len(dataset)} samples")

    return dataset


def setup_training_args(
    output_dir,
    per_device_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=2000,
    learning_rate=2e-4,
    warmup_steps=100,
    fp16=True,
    logging_steps=50,
    save_steps=200,
):
    """
    设置训练参数

    Returns:
        TrainingArguments对象
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        fp16=fp16,
        bf16=False,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        learning_rate=learning_rate,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    return training_args


def train(model, tokenizer, train_dataset, eval_dataset=None, training_args=None):
    """
    执行训练流程

    Args:
        model: 配置好LoRA的模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 评估数据集（可选）
        training_args: 训练参数

    Returns:
        Trainer对象
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(
        f"Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}"
    )

    trainer.train()

    return trainer


def save_model(model, tokenizer, output_dir, save_method="merged_16bit"):
    """
    保存训练好的模型

    Args:
        model: 训练好的模型
        tokenizer: 分词器
        output_dir: 输出目录
        save_method: 保存方式 ("lora", "merged_16bit", "merged_4bit")
    """
    print(f"Saving model to: {output_dir}")
    print(f"Save method: {save_method}")

    if save_method == "merged_16bit" or save_method == "merged_4bit":
        model.save_pretrained_merged(output_dir, tokenizer, save_method=save_method)
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    print(f"Model saved to: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Unsloth LoRA Training Script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/llama-3-8b-bnb-4bit",
        help="Model name or path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train.jsonl",
        help="Path to training data (JSONL)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--load_in_4bit",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Load model in 4-bit quantization",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--per_device_batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_steps", type=int, default=2000, help="Maximum training steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum training samples (None for all)",
    )
    parser.add_argument(
        "--save_method",
        type=str,
        default="merged_16bit",
        choices=["lora", "merged_16bit", "merged_4bit"],
        help="How to save the model",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    model = configure_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha)

    train_dataset = prepare_dataset(
        tokenizer, args.data_path, max_samples=args.max_samples
    )

    training_args = setup_training_args(
        output_dir=args.output_dir,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
    )

    trainer = train(model, tokenizer, train_dataset, training_args=training_args)

    save_model(model, tokenizer, args.output_dir, save_method=args.save_method)

    print("Training completed!")


if __name__ == "__main__":
    main()
