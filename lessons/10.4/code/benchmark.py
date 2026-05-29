#!/usr/bin/env python3
"""
Benchmark Script: Unsloth vs Standard PEFT
对比Unsloth与标准HuggingFace PEFT的训练速度和显存使用
"""

import torch
import time
import gc
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import platform


def get_memory_usage():
    """获取当前GPU显存使用情况（MB）"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        return allocated, reserved
    return 0, 0


def get_system_memory():
    """获取系统内存使用情况（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def print_memory_stats(label=""):
    """打印内存统计信息"""
    gpu_allocated, gpu_reserved = get_memory_usage()
    system_mem = get_system_memory()
    print(
        f"[{label}] GPU Allocated: {gpu_allocated:.1f}MB | GPU Reserved: {gpu_reserved:.1f}MB | System: {system_mem:.1f}MB"
    )


class BenchmarkResult:
    """基准测试结果"""

    def __init__(
        self,
        name,
        model_size,
        batch_size,
        steps,
        total_time,
        tokens_per_sec,
        memory_used,
    ):
        self.name = name
        self.model_size = model_size
        self.batch_size = batch_size
        self.steps = steps
        self.total_time = total_time
        self.tokens_per_sec = tokens_per_sec
        self.memory_used = memory_used


def create_standard_peft_model(model_name, lora_config=None):
    """创建标准PEFT模型（作为对比基准）"""
    print(f"\n{'=' * 50}")
    print(f"Loading Standard PEFT model: {model_name}")
    print_memory_stats("Before loading")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if lora_config is None:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print_memory_stats("After loading")

    return model, tokenizer


def create_unsloth_model(model_name, max_seq_length=2048, load_in_4bit=True):
    """创建Unsloth优化模型"""
    from unsloth import FastLanguageModel

    print(f"\n{'=' * 50}")
    print(f"Loading Unsloth model: {model_name}")
    print_memory_stats("Before loading")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    model.print_trainable_parameters()

    print_memory_stats("After loading")

    return model, tokenizer


def generate_dummy_dataset(tokenizer, num_samples=100, seq_length=512):
    """生成模拟数据集用于基准测试"""
    from datasets import Dataset

    def generate_text(i):
        return f"This is sample number {i}. " * (seq_length // 20)

    texts = [generate_text(i) for i in range(num_samples)]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_length,
            padding="max_length",
        )

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    return dataset


def benchmark_training(
    model, tokenizer, dataset, num_steps=50, per_device_batch_size=2, mode="standard"
):
    """
    基准测试训练性能

    Args:
        model: 模型
        tokenizer: 分词器
        dataset: 数据集
        num_steps: 测试步数
        per_device_batch_size: 每设备batch size
        mode: "standard" 或 "unsloth"

    Returns:
        BenchmarkResult
    """
    print(f"\nRunning benchmark: {num_steps} steps, batch_size={per_device_batch_size}")

    training_args = TrainingArguments(
        output_dir="./benchmark_output",
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=4,
        max_steps=num_steps,
        fp16=True,
        logging_steps=10,
        save_steps=num_steps + 1,
        report_to="none",
        remove_unused_columns=False,
    )

    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print_memory_stats("Before training")

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = num_steps * per_device_batch_size * 4 * 512  # 估算
    tokens_per_sec = total_tokens / total_time

    gpu_allocated, _ = get_memory_usage()

    print_memory_stats("After training")
    print(f"Total time: {total_time:.1f}s")
    print(f"Tokens per second: {tokens_per_sec:.1f}")

    return BenchmarkResult(
        name=f"{mode}_{per_device_batch_size}",
        model_size=sum(p.numel() for p in model.parameters()) / 1e9,
        batch_size=per_device_batch_size,
        steps=num_steps,
        total_time=total_time,
        tokens_per_sec=tokens_per_sec,
        memory_used=gpu_allocated,
    )


def print_comparison(results):
    """打印对比结果表格"""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 80)

    header = (
        f"{'Mode':<20} {'Batch':<6} {'Time(s)':<10} {'Tokens/s':<12} {'GPU(MB)':<10}"
    )
    print(header)
    print("-" * 60)

    for result in results:
        row = f"{result.name:<20} {result.batch_size:<6} {result.total_time:<10.1f} {result.tokens_per_sec:<12.1f} {result.memory_used:<10.1f}"
        print(row)

    if len(results) >= 2:
        baseline = results[0]
        for result in results[1:]:
            speedup = (
                baseline.tokens_per_sec / result.tokens_per_sec
                if result.tokens_per_sec > 0
                else 0
            )
            mem_ratio = (
                baseline.memory_used / result.memory_used
                if result.memory_used > 0
                else 0
            )
            print(f"\n{result.name} vs {baseline.name}:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Memory ratio: {mem_ratio:.2f}x")


def run_benchmark_suite():
    """运行完整的基准测试套件"""
    print("=" * 60)
    print("Unsloth vs Standard PEFT Benchmark")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        )

    results = []

    model_name = "unsloth/llama-3-8b-bnb-4bit"
    num_steps = 50
    per_device_batch_size = 2

    print_memory_stats("Initial")

    # Test Unsloth
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = create_unsloth_model(model_name, max_seq_length=512)
        dataset = generate_dummy_dataset(tokenizer, num_samples=100, seq_length=512)

        result = benchmark_training(
            model,
            tokenizer,
            dataset,
            num_steps=num_steps,
            per_device_batch_size=per_device_batch_size,
            mode="Unsloth",
        )
        results.append(result)

        del model, tokenizer, dataset
        gc.collect()
        torch.cuda.empty_cache()

    except ImportError:
        print("Unsloth not installed, skipping...")
    except Exception as e:
        print(f"Unsloth benchmark failed: {e}")

    # Test Standard PEFT
    try:
        peft_model_name = "unsloth/llama-3-8b-bnb-4bit"
        model, tokenizer = create_standard_peft_model(peft_model_name)
        dataset = generate_dummy_dataset(tokenizer, num_samples=100, seq_length=512)

        result = benchmark_training(
            model,
            tokenizer,
            dataset,
            num_steps=num_steps,
            per_device_batch_size=per_device_batch_size,
            mode="Standard PEFT",
        )
        results.append(result)

        del model, tokenizer, dataset
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Standard PEFT benchmark failed: {e}")

    print_comparison(results)

    return results


if __name__ == "__main__":
    results = run_benchmark_suite()
