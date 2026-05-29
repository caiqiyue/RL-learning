"""
DPO训练脚本

使用 HuggingFace TRL 库的 DPOTrainer 进行直接偏好优化训练。

使用方法：
    # 使用本地偏好数据训练
    python train_dpo.py --preference_data ./preference_data.json --output_dir ./dpo_output

    # 使用 HH-RLHF 数据集训练
    python train_dpo.py --dataset Anthropic/hh-rlhf --output_dir ./dpo_output

    # 使用 LoRA 进行高效训练
    python train_dpo.py --preference_data ./preference_data.json --use_lora --lora_r 64 --output_dir ./dpo_output
"""

import argparse
import json
import os
import math
from typing import Optional, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_dataset

from trl import DPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DPO训练脚本")

    # 数据相关
    parser.add_argument(
        "--preference_data", type=str, default=None, help="本地偏好数据JSON文件路径"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace数据集名称，如 'Anthropic/hh-rlhf'",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="数据集分割，如 'train' 或 'train[:5000]'",
    )

    # 模型相关
    parser.add_argument("--model", type=str, default="gpt2", help="模型名称或本地路径")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="分词器名称或路径，默认使用model的值",
    )
    parser.add_argument(
        "--ref_model",
        type=str,
        default=None,
        help="参考模型名称或路径，默认使用model的值",
    )

    # 训练超参数
    parser.add_argument(
        "--beta", type=float, default=0.1, help="KL温度参数，控制策略偏离参考模型的程度"
    )
    parser.add_argument(
        "--margin_lambda",
        type=float,
        default=0.0,
        help="边际损失权重，默认0表示不使用边际",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=4, help="每个设备的批量大小"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数"
    )
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument(
        "--max_prompt_length", type=int, default=256, help="最大提示长度"
    )

    # LoRA 相关
    parser.add_argument(
        "--use_lora", action="store_true", help="是否使用LoRA进行参数高效微调"
    )
    parser.add_argument(
        "--lora_r", type=int, default=64, help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA alpha parameter"
    )
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,v_proj",
        help="LoRA目标模块，逗号分隔",
    )

    # 输出相关
    parser.add_argument(
        "--output_dir", type=str, default="./dpo_output", help="输出目录"
    )
    parser.add_argument(
        "--save_steps", type=int, default=100, help="保存checkpoint的步数"
    )
    parser.add_argument("--logging_steps", type=int, default=10, help="日志打印步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 其他
    parser.add_argument("--bf16", action="store_true", help="是否使用bfloat16精度")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练本地排名")

    return parser.parse_args()


def load_preference_data_from_json(file_path: str) -> Dataset:
    """从本地JSON文件加载偏好数据"""
    print(f"Loading preference data from {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 转换为 Dataset 格式
    dataset = Dataset.from_list(data)
    print(f"Loaded {len(dataset)} preference pairs")

    return dataset


def load_preference_data_from_hf(dataset_name: str, split: str = "train") -> Dataset:
    """从 HuggingFace 数据集加载偏好数据"""
    print(f"Loading dataset from HuggingFace: {dataset_name}")

    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} samples")

    # TRL 需要特定的字段名
    if "chosen" in dataset.column_names and "rejected" in dataset.column_names:
        # 已经是正确格式
        return dataset
    else:
        # 需要转换格式
        def transform(example):
            return {
                "prompt": example.get("human", example.get("instruction", "")),
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            }

        dataset = dataset.map(transform, remove_columns=dataset.column_names)
        return dataset


def prepare_reference_model(
    model: AutoModelForCausalLM, ref_model_name: Optional[str], use_lora: bool
) -> AutoModelForCausalLM:
    """
    准备参考模型（冻结副本）

    如果使用 LoRA，参考模型应该是原始模型（非 LoRA 适配后的）
    """
    print("Preparing reference model...")

    if ref_model_name:
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)
    else:
        # 使用相同的模型作为参考
        ref_model = AutoModelForCausalLM.from_pretrained(model.name_or_path)

    # 确保参考模型不更新梯度
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


def apply_lora_to_model(model: AutoModelForCausalLM, args) -> AutoModelForCausalLM:
    """为模型应用 LoRA 配置"""
    print("Applying LoRA configuration...")

    target_modules = args.target_modules.split(",")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def setup_training_arguments(args) -> TrainingArguments:
    """设置训练参数"""

    # 计算实际 batch size（考虑梯度累积）
    effective_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    total_steps = (args.num_train_epochs * 100) // effective_batch_size  # 假设100条数据

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        logging_dir=f"{args.output_dir}/logs",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        seed=args.seed,
        data_seed=args.seed,
        fp16=not args.bf16 and torch.cuda.is_available(),
        bf16=args.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
    )

    return training_args


def initialize_dpo_trainer(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    args,
) -> DPOTrainer:
    """初始化 DPOTrainer"""

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        margin_lambda=args.margin_lambda if args.margin_lambda > 0 else None,
        loss_type="sigmoid"
        if args.margin_lambda > 0
        else "bco",  # DPO 默认使用 sigmoid
        dataset_num_proc=4,
        max_grad_norm=1.0,
    )

    return trainer


def print_model_info(model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    print(f"Model precision: {model.dtype}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print("=" * 60 + "\n")


def save_training_config(args, output_dir: str):
    """保存训练配置"""
    config = {
        "model": args.model,
        "beta": args.beta,
        "margin_lambda": args.margin_lambda,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r if args.use_lora else None,
        "seed": args.seed,
    }

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {config_path}")


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("DPO Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Beta: {args.beta}")
    print(f"Learning rate: {args.learning_rate}")
    print(
        f"Batch size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Max length: {args.max_length}")
    print(f"Use LoRA: {args.use_lora}")
    if args.use_lora:
        print(f"LoRA r: {args.lora_r}, alpha: {args.lora_alpha}")
    print("=" * 60 + "\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型和分词器
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    # 确保 pad token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # 2. 应用 LoRA（如果启用）
    if args.use_lora:
        model = apply_lora_to_model(model, args)

    # 打印模型信息
    print_model_info(model, tokenizer)

    # 3. 准备参考模型
    ref_model = prepare_reference_model(model, args.ref_model, args.use_lora)

    # 4. 加载训练数据
    if args.preference_data:
        train_dataset = load_preference_data_from_json(args.preference_data)
    elif args.dataset:
        train_dataset = load_preference_data_from_hf(args.dataset, args.dataset_split)
    else:
        raise ValueError("必须指定 --preference_data 或 --dataset")

    # 5. 设置训练参数
    training_args = setup_training_arguments(args)

    # 6. 初始化 DPOTrainer
    trainer = initialize_dpo_trainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        args=args,
    )

    # 7. 保存训练配置
    save_training_config(args, args.output_dir)

    # 8. 开始训练
    print("Starting DPO training...")
    print(f"Total training steps: {trainer.state.max_steps}")

    train_result = trainer.train()

    # 9. 保存模型和训练结果
    print("\nSaving model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    trainer.save_state()

    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Final model saved to: {os.path.join(args.output_dir, 'final_model')}")
    print(f"Training metrics: {metrics}")


if __name__ == "__main__":
    main()
