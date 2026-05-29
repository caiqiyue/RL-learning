"""
LoRA训练脚本 - 使用HuggingFace PEFT库
=====================================

本脚本演示如何使用PEFT库进行LoRA微调，包含完整流程：
1. 模型加载与配置
2. 数据集准备与格式化
3. LoRA包装与可训练参数查看
4. 训练与评估
5. 模型保存与权重合并

依赖：transformers, peft, datasets, accelerate, bitsandbytes

运行示例：
    python train_lora.py \
        --model_name "Qwen/Qwen2.5-0.5B" \
        --dataset_name "yahma/alpaca-cleaned" \
        --output_dir "./lora_output" \
        --rank 8 \
        --batch_size 2 \
        --num_epochs 3
"""

import argparse
import os
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


@dataclass
class TrainingConfig:
    """训练配置"""

    model_name: str = "Qwen/Qwen2.5-0.5B"
    dataset_name: str = "yahma/alpaca-cleaned"
    output_dir: str = "./lora_output"
    rank: int = 8
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 1e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: Optional[int] = None
    max_steps: Optional[int] = -1
    bf16: bool = True
    use_qLoRA: bool = False
    seed: int = 42


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning with PEFT")

    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="模型名称或路径"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="yahma/alpaca-cleaned", help="数据集名称"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./lora_output", help="输出目录"
    )
    parser.add_argument("--rank", "-r", type=int, default=8, help="LoRA秩")
    parser.add_argument(
        "--lora_alpha", type=int, default=None, help="LoRA alpha，默认值是rank*2"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout概率"
    )
    parser.add_argument(
        "--target_modules", type=str, default=None, help="目标模块列表，逗号分隔"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="每设备batch大小")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数"
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.3, help="梯度裁剪最大值"
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=100, help="模型保存步数")
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="最大训练步数，-1表示训练完所有epoch"
    )
    parser.add_argument("--bf16", action="store_true", help="使用BF16混合精度")
    parser.add_argument("--no_bf16", action="store_true", help="不使用BF16混合精度")
    parser.add_argument(
        "--use_qlora", action="store_true", help="使用QLoRA（量化+LoRA）"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    if args.no_bf16:
        args.bf16 = False

    if args.target_modules:
        args.target_modules = [m.strip() for m in args.target_modules.split(",")]

    return args


def print_model_info(model: torch.nn.Module, title: str = "Model Info") -> None:
    """打印模型信息"""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    trainable_params = 0
    all_params = 0

    for param_name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable params: {trainable_params:,}")
    print(f"All params: {all_params:,}")
    print(f"Trainable ratio: {trainable_params / all_params * 100:.3f}%")


def prepare_model(
    model_name: str,
    lora_config: LoraConfig,
    use_qLora: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    准备模型和分词器

    Args:
        model_name: 模型名称或路径
        lora_config: LoRA配置
        use_qLora: 是否使用QLoRA

    Returns:
        (model, tokenizer)
    """
    print(f"\n1. 加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_qLora:
        print("   使用QLoRA配置（INT4量化）")
        bnb_config = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    print(f"   模型加载完成")
    print_model_info(model, "原始模型参数统计")

    print(f"\n2. 应用LoRA配置")
    print(f"   rank={lora_config.r}, alpha={lora_config.lora_alpha}")
    print(f"   target_modules={lora_config.target_modules}")

    model = get_peft_model(model, lora_config)

    print_model_info(model, "LoRA包装后参数统计")

    return model, tokenizer


def create_lora_config(
    rank: int = 8,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    model_name: Optional[str] = None,
) -> LoraConfig:
    """
    创建LoRA配置

    Args:
        rank: LoRA秩
        lora_alpha: 缩放因子，默认是rank*2
        lora_dropout: dropout概率
        target_modules: 目标模块列表
        model_name: 模型名称（用于自动推断目标模块）

    Returns:
        LoraConfig实例
    """
    if lora_alpha is None:
        lora_alpha = rank * 2

    if target_modules is None:
        if model_name:
            model_lower = model_name.lower()
            if "qwen" in model_lower:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "llama" in model_lower:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "chatglm" in model_lower:
                target_modules = ["query_key_value", "dense"]
            else:
                target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]

    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )

    return config


def prepare_dataset(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    max_length: int = 512,
    training_split: str = "train",
) -> Any:
    """
    准备训练数据集

    Args:
        tokenizer: 分词器
        dataset_name: 数据集名称
        max_length: 最大序列长度
        training_split: 训练集分割名称

    Returns:
        处理后的数据集
    """
    print(f"\n3. 加载数据集: {dataset_name}")

    from datasets import load_dataset

    dataset = load_dataset(dataset_name)

    if training_split not in dataset:
        raise ValueError(
            f"Split '{training_split}' not found in dataset. Available: {list(dataset.keys())}"
        )

    train_dataset = dataset[training_split]

    print(f"   原始数据集大小: {len(train_dataset)}")

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化指令数据

        期望的数据格式（Alpaca格式）：
        - instruction: 指令
        - input: 输入（可选）
        - output: 输出
        """
        prompts = []

        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            if input_text and input_text.strip():
                prompt = f"指令: {instruction}\n输入: {input_text}\n输出: {output}"
            else:
                prompt = f"指令: {instruction}\n输出: {output}"

            prompts.append(prompt)

        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    print(f"   进行tokenize处理...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing",
    )

    print(f"   处理后数据集大小: {len(train_dataset)}")
    print(f"   样例: {train_dataset[0]['input_ids'][:20]}...")

    return train_dataset


def setup_training_args(
    config: TrainingConfig,
) -> TrainingArguments:
    """
    设置训练参数

    Args:
        config: 训练配置

    Returns:
        TrainingArguments实例
    """
    print(f"\n4. 配置训练参数")
    print(f"   学习率: {config.learning_rate}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   梯度累积: {config.gradient_accumulation_steps}")

    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    print(f"   有效batch size: {effective_batch_size}")

    args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        fp16=not config.bf16,
        optim="paged_adamw_32bit",
        save_total_limit=3,
        seed=config.seed,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    if config.max_steps > 0:
        args.max_steps = config.max_steps

    if config.eval_steps:
        args.eval_steps = config.eval_steps
        args.eval_strategy = "steps"

    return args


def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Any,
    training_args: TrainingArguments,
) -> Trainer:
    """
    执行训练

    Args:
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练数据集
        training_args: 训练参数

    Returns:
        Trainer实例
    """
    print(f"\n5. 开始训练")
    print(f"   输出目录: {training_args.output_dir}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    print(f"\n   训练完成!")

    return trainer


def save_model(
    trainer: Trainer,
    output_dir: str,
    merge_and_unload: bool = False,
) -> str:
    """
    保存模型

    Args:
        trainer: Trainer实例
        output_dir: 输出目录
        merge_and_unload: 是否合并权重后保存

    Returns:
        保存路径
    """
    print(f"\n6. 保存模型")
    print(f"   保存位置: {output_dir}")

    final_output_dir = os.path.join(output_dir, "final")

    if merge_and_unload:
        print("   执行权重合并...")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(final_output_dir)
        print("   合并后模型已保存 (可用原始模型加载方式加载)")
    else:
        trainer.model.save_pretrained(final_output_dir)
        print("   LoRA适配器已保存 (需使用PeftModel加载)")

    print(f"   模型保存完成: {final_output_dir}")

    return final_output_dir


def load_lora_adapter(
    base_model_name: str,
    adapter_path: str,
) -> AutoModelForCausalLM:
    """
    加载LoRA适配器（用于推理）

    Args:
        base_model_name: 基础模型名称
        adapter_path: 适配器路径

    Returns:
        加载了适配器的模型
    """
    print(f"\n7. 加载LoRA适配器进行推理")
    print(f"   基础模型: {base_model_name}")
    print(f"   适配器: {adapter_path}")

    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("   适配器加载完成")

    return model


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("LoRA Fine-tuning Script")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"数据集: {args.dataset_name}")
    print(f"LoRA rank: {args.rank}")
    print(f"输出目录: {args.output_dir}")

    set_seed(args.seed)

    lora_config = create_lora_config(
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        model_name=args.model_name,
    )

    model, tokenizer = prepare_model(
        args.model_name,
        lora_config,
        use_qLora=args.use_qlora,
    )

    train_dataset = prepare_dataset(
        tokenizer,
        args.dataset_name,
    )

    training_args = setup_training_args(
        TrainingConfig(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            max_steps=args.max_steps,
            bf16=args.bf16,
            use_qLoRA=args.use_qlora,
            seed=args.seed,
        )
    )

    trainer = train(
        model,
        tokenizer,
        train_dataset,
        training_args,
    )

    save_model(
        trainer,
        args.output_dir,
        merge_and_unload=True,
    )

    print("\n" + "=" * 60)
    print("LoRA训练完成!")
    print("=" * 60)

    print("\n使用说明:")
    print("1. 使用LoRA适配器推理:")
    print(f"   from peft import PeftModel")
    print(
        f"   model = PeftModel.from_pretrained(base_model, '{args.output_dir}/final')"
    )
    print(f"   model.generate(**inputs)")
    print("\n2. 合并后模型推理（与原始模型相同）:")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{args.output_dir}/final')")
    print(f"   model.generate(**inputs)")


if __name__ == "__main__":
    main()
