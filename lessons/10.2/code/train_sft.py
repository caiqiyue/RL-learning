#!/usr/bin/env python3
"""
SFTTrainer示例：使用LoRA对GPT-2进行监督微调
演示TRL库SFTTrainer的基本用法，包括数据格式化、PEFT配置、训练参数设置
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def prepare_dataset():
    """
    准备示例数据集
    实际应用中替换为你的数据集
    """
    dataset = load_dataset("ym坏的/ultra-chat-mini", split="train[:1000]")

    def format_conversation(example):
        if "messages" in example:
            messages = example["messages"]
        else:
            messages = [
                {"role": "user", "content": example.get("user", "")},
                {"role": "assistant", "content": example.get("assistant", "")},
            ]

        formatted = "<|user|>\n"
        for msg in messages:
            if msg["role"] == "user":
                formatted += msg["content"] + "<|end|>\n"
            elif msg["role"] == "assistant":
                formatted += "<|assistant|>\n" + msg["content"] + "<|end|>\n"

        example["text"] = formatted
        return example

    dataset = dataset.map(format_conversation)
    return dataset


def setup_model_and_tokenizer(model_name="gpt2"):
    """初始化模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def setup_lora_config():
    """配置LoRA参数高效微调"""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "q_proj", "v_proj"],
    )


def main():
    print("=" * 60)
    print("SFTTrainer 示例：使用LoRA微调GPT-2")
    print("=" * 60)

    model_name = "gpt2"
    max_seq_length = 512
    batch_size = 4
    learning_rate = 2e-4
    num_epochs = 3
    gradient_checkpointing = True

    print(f"\n[配置]")
    print(f"  模型: {model_name}")
    print(f"  最大序列长度: {max_seq_length}")
    print(f"  Batch大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  梯度检查点: {gradient_checkpointing}")

    print("\n[1/4] 加载数据集...")
    dataset = prepare_dataset()
    print(f"  数据集大小: {len(dataset)}")

    print("\n[2/4] 加载模型和tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_name)

    print("\n[3/4] 配置LoRA微调...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[4/4] 初始化SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        args={
            "output_dir": "./output/sft",
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "learning_rate": learning_rate,
            "gradient_checkpointing": gradient_checkpointing,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "optim": "adamw_torch",
            "fp16": torch.cuda.is_available(),
        },
        peft_config=lora_config,
        packing=False,
    )

    print("\n[训练] 开始微调...")
    trainer.train()

    print("\n[完成] 保存模型...")
    trainer.save_model("./output/sft-final")
    print(f"  模型已保存至: ./output/sft-final")


if __name__ == "__main__":
    main()
