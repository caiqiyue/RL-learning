#!/usr/bin/env python3
"""
PPOTrainer示例：使用模拟奖励信号训练GPT-2
演示TRL库PPOTrainer的完整流程，包括策略模型、参考模型、奖励模型配置
包含简化版mock reward用于演示，实际使用时替换为真实奖励模型

注意：PPO训练较为复杂，此示例展示核心流程
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer
from trl.reward_scorer import RewardModel
from peft import LoraConfig, get_peft_model


class MockRewardModel:
    """
    模拟奖励模型
    实际应用中替换为真实训练的奖励模型
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def score(self, prompt, response):
        """
        简化的奖励评分逻辑
        实际中应该使用训练好的Reward Model
        """
        base_score = len(response) / 100.0

        if len(response) > 10:
            base_score += 1.0

        if any(word in response.lower() for word in ["谢谢", "好的", "明白"]):
            base_score += 0.5

        if len(response) > 200:
            base_score += 1.0

        return base_score

    def __call__(self, prompts, responses):
        """批量计算奖励"""
        scores = []
        for prompt, response in zip(prompts, responses):
            score = self.score(prompt, response)
            scores.append(torch.tensor(score))
        return scores


def prepare_prompt_dataset():
    """
    准备PPO训练所需的提示数据集
    PPO需要prompt列表，模型生成response后由奖励模型评分
    """
    dataset = load_dataset("ym坏的/ultra-chat-mini", split="train[:100]")

    def extract_prompts(example):
        messages = example.get("messages", [])
        for msg in messages:
            if msg["role"] == "user":
                return {"prompt": msg["content"]}
        return {"prompt": "默认问题"}

    dataset = dataset.map(extract_prompts, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: x["prompt"] != "默认问题")
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


def generate_responses(model, tokenizer, prompts, max_new_tokens=100):
    """使用策略模型生成响应"""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for prompt, output in zip(prompts, outputs):
        response = tokenizer.decode(output, skip_special_tokens=True)
        response = response[
            len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)) :
        ]
        responses.append(response.strip())

    return responses


def compute_rewards(reward_model, prompts, responses):
    """计算奖励分数"""
    scores = reward_model(prompts, responses)
    return scores


def main():
    print("=" * 60)
    print("PPOTrainer 示例：使用模拟奖励信号训练GPT-2")
    print("=" * 60)

    model_name = "gpt2"
    max_seq_length = 512
    batch_size = 2
    learning_rate = 1e-5
    num_epochs = 2
    kl_beta = 0.1
    max_new_tokens = 100

    print(f"\n[配置]")
    print(f"  模型: {model_name}")
    print(f"  最大序列长度: {max_seq_length}")
    print(f"  Batch大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  KL Beta系数: {kl_beta}")
    print(f"  最大生成token数: {max_new_tokens}")

    print("\n[1/5] 加载提示数据集...")
    dataset = prepare_prompt_dataset()
    prompts = dataset["prompt"][:8]
    print(f"  提示数量: {len(prompts)}")

    print("\n[2/5] 加载模型和tokenizer...")
    model, ref_model, tokenizer = setup_models(model_name)

    print("\n[3/5] 配置模拟奖励模型...")
    reward_model = MockRewardModel(tokenizer)
    print("  使用模拟奖励（实际应用替换为真实奖励模型）")

    print("\n[4/5] 初始化PPOTrainer...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args={
            "output_dir": "./output/ppo",
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "learning_rate": learning_rate,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "kl_beta": kl_beta,
            "max_grad_norm": 1.0,
            "logging_steps": 10,
            "save_steps": 200,
            "optim": "adamw_torch",
            "warmup_steps": 50,
        },
        model_init_kwargs={"max_new_tokens": max_new_tokens},
    )

    print("\n[5/5] 开始PPO训练循环...")
    print("  PPO流程: 生成响应 → 奖励评估 → 优势计算 → 策略更新")

    for epoch in range(num_epochs):
        print(f"\n  --- Epoch {epoch + 1}/{num_epochs} ---")

        for step, example in enumerate(dataset):
            prompt = [example["prompt"]]

            print(f"\n  [Step {step + 1}] 处理提示: {prompt[0][:40]}...")

            response = generate_responses(model, tokenizer, prompt, max_new_tokens)
            print(f"  生成响应长度: {len(response[0])} 字符")

            reward_score = reward_model(prompt, response)
            print(f"  奖励分数: {reward_score[0].item():.3f}")

            stats = trainer.step(prompt, [response[0]], reward_score)

            if step % 5 == 0:
                print(
                    f"  训练统计 - loss: {stats.get('train_loss', 0):.4f}, "
                    f"kl_div: {stats.get('kl_divergence', 0):.4f}"
                )

    print("\n[完成] 保存模型...")
    trainer.save_model("./output/ppo-final")
    print(f"  模型已保存至: ./output/ppo-final")
    print("\n  注意: 实际PPO训练需要:")
    print("    1. 真实训练的奖励模型")
    print("    2. 更复杂的优势函数估计")
    print("    3. 更大的数据集和更长的训练")


if __name__ == "__main__":
    main()
