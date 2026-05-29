#!/usr/bin/env python3
"""
DPOTrainer示例：使用偏好数据训练GPT-2
演示TRL库DPOTrainer的完整用法，包括数据格式化、beta参数、label_smoothing配置
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model


def prepare_preference_dataset():
    """
    准备DPO训练所需的偏好数据集
    每条数据包含: prompt (用户输入), chosen (偏好响应), rejected (拒绝响应)
    """
    dataset = load_dataset("ym坏的/ultra-chat-mini", split="train[:1000]")

    def to_preference_format(example):
        messages = example.get("messages", [])
        user_content = ""
        assistant_responses = []

        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_responses.append(msg["content"])

        if len(assistant_responses) >= 2:
            chosen = assistant_responses[0]
            rejected = assistant_responses[-1]
        else:
            chosen = assistant_responses[0] if assistant_responses else ""
            rejected = "抱歉，我无法回答这个问题。"

        return {"prompt": user_content, "chosen": chosen, "rejected": rejected}

    dataset = dataset.map(to_preference_format, remove_columns=dataset.column_names)
    return dataset


def setup_models(model_name="gpt2"):
    """初始化策略模型和参考模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="auto"
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="auto"
    )
    ref_model.config.pad_token_id = tokenizer.eos_token_id

    return model, ref_model, tokenizer


def setup_lora_config():
    """配置LoRA微调"""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main():
    print("=" * 60)
    print("DPOTrainer 示例：偏好优化训练GPT-2")
    print("=" * 60)

    model_name = "gpt2"
    max_seq_length = 512
    batch_size = 2
    learning_rate = 5e-6
    num_epochs = 3
    beta = 0.1
    label_smoothing = 0.05

    print(f"\n[配置]")
    print(f"  模型: {model_name}")
    print(f"  最大序列长度: {max_seq_length}")
    print(f"  Batch大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  Beta (KL系数): {beta}")
    print(f"  Label Smoothing: {label_smoothing}")

    print("\n[1/4] 加载偏好数据集...")
    dataset = prepare_preference_dataset()
    print(f"  数据集大小: {len(dataset)}")
    print(f"  样例prompt: {dataset[0]['prompt'][:50]}...")

    print("\n[2/4] 加载模型和参考模型...")
    model, ref_model, tokenizer = setup_models(model_name)

    print("\n[3/4] 配置LoRA微调...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[4/4] 初始化DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        args={
            "output_dir": "./output/dpo",
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "learning_rate": learning_rate,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "beta": beta,
            "label_smoothing": label_smoothing,
            "logging_steps": 10,
            "save_steps": 500,
            "optim": "adamw_torch",
            "fp16": torch.cuda.is_available(),
            "warmup_steps": 100,
        },
        peft_config=lora_config,
    )

    print("\n[训练] 开始DPO训练...")
    print("  注意: DPO不需要奖励模型，直接通过偏好对优化策略")
    trainer.train()

    print("\n[完成] 保存模型...")
    trainer.save_model("./output/dpo-final")
    print(f"  模型已保存至: ./output/dpo-final")


if __name__ == "__main__":
    main()
