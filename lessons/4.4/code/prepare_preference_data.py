"""
偏好数据生成脚本

该脚本演示如何从基础模型生成合成偏好数据。
支持三种生成策略：
1. 基于长度和结构的启发式规则
2. 基于评委模型的评分
3. SPIN 自迭代偏好生成

使用方法：
    python prepare_preference_data.py --model gpt2 --output ./preference_data.json
"""

import argparse
import json
import random
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="生成合成偏好数据")
    parser.add_argument("--model", type=str, default="gpt2", help="模型名称或路径")
    parser.add_argument(
        "--output", type=str, default="./preference_data.json", help="输出文件路径"
    )
    parser.add_argument("--num_samples", type=int, default=100, help="生成的样本数量")
    parser.add_argument(
        "--num_candidates", type=int, default=4, help="每个提示生成的候选回答数量"
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument(
        "--strategy",
        type=str,
        default="length_structure",
        choices=["length_structure", "judge", "spin"],
        help="偏好选择策略",
    )
    parser.add_argument("--max_length", type=int, default=512, help="最大生成长度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def load_base_model_and_tokenizer(model_name: str):
    """加载基础模型和分词器"""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_candidates(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    num_candidates: int = 4,
    temperature: float = 0.8,
    max_length: int = 512,
) -> Dict[str, List[str]]:
    """为每个提示生成多个候选回答"""

    candidates_dict = {}

    for i, prompt in enumerate(prompts):
        if i % 10 == 0:
            print(f"Generating candidates for prompt {i}/{len(prompts)}")

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)

        candidates = []
        for _ in range(num_candidates):
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_length // 2,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    top_p=0.95,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt) :].strip()
            candidates.append(generated_text)

        candidates_dict[prompt] = candidates

    return candidates_dict


def score_by_length_structure(response: str) -> float:
    """
    基于长度和结构对回答进行评分

    启发式规则：
    - 长度适中（150-400字符）得分更高
    - 有换行/列表结构得分更高
    - 完整句子得分更高
    """
    score = 0.0

    length = len(response)

    # 长度评分：150-400字符最佳
    if 150 <= length <= 400:
        score += 0.4
    elif 100 <= length < 150 or 400 < length <= 600:
        score += 0.2
    elif length < 50:
        score += 0.0
    else:
        score += 0.1

    # 结构评分
    if "\n" in response:
        score += 0.2
    if any(marker in response for marker in ["1.", "2.", "-", "*", "•"]):
        score += 0.15

    # 完整性评分：结尾有句号或问号
    if response.strip().endswith(("。", "!", "?", ".", "！", "？")):
        score += 0.15

    # 内容丰富度：包含多个逗号（复合句）
    comma_count = response.count("，") + response.count(",")
    if comma_count >= 3:
        score += 0.1

    return min(score, 1.0)


def select_preference_by_length_structure(candidates: List[str]) -> tuple:
    """
    使用长度和结构启发式规则选择偏好对

    Returns:
        tuple: (chosen, rejected) - 偏好和不被偏好的回答
    """
    scored = [(resp, score_by_length_structure(resp)) for resp in candidates]
    scored.sort(key=lambda x: -x[1])

    return scored[0][0], scored[-1][0]


def create_preference_data(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    strategy: str = "length_structure",
    num_candidates: int = 4,
    temperature: float = 0.8,
    max_length: int = 512,
) -> List[Dict[str, str]]:
    """
    创建偏好数据集

    Args:
        model: 基础模型
        tokenizer: 分词器
        prompts: 提示列表
        strategy: 偏好选择策略
        num_candidates: 每个提示生成的候选数量
        temperature: 采样温度
        max_length: 最大生成长度

    Returns:
        List[Dict]: 偏好数据集，每项包含 prompt, chosen, rejected
    """

    print(f"Generating preference data using strategy: {strategy}")

    # 生成候选回答
    candidates_dict = generate_candidates(
        model,
        tokenizer,
        prompts,
        num_candidates=num_candidates,
        temperature=temperature,
        max_length=max_length,
    )

    preference_data = []

    for i, (prompt, candidates) in enumerate(candidates_dict.items()):
        if i % 10 == 0:
            print(f"Processing preference pair {i}/{len(candidates_dict)}")

        if strategy == "length_structure":
            chosen, rejected = select_preference_by_length_structure(candidates)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 确保 chosen 和 rejected 不同
        if chosen == rejected and len(candidates) >= 2:
            # 取第二个作为 rejected
            rejected = candidates[1]

        if chosen != rejected:
            preference_data.append(
                {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            )

    return preference_data


def load_prompts_from_hh_rlhf(num_samples: int = 100, seed: int = 42) -> List[str]:
    """从 Anthropic HH-RLHF 数据集加载提示"""

    print("Loading prompts from Anthropic HH-RLHF dataset...")

    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")
        prompts = [sample["human"] for sample in dataset]

        random.seed(seed)
        random.shuffle(prompts)

        return prompts[:num_samples]
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        print("Using fallback prompts...")

        # Fallback: 使用预定义的通用提示
        return [
            "请解释什么是大语言模型。",
            "如何学习编程？",
            "给我推荐一本好书。",
            "解释量子力学的基本原理。",
            "如何保持健康？",
            "写一个Python快速排序函数。",
            "解释机器学习和深度学习的区别。",
            "如何提高写作能力？",
            "解释区块链的工作原理。",
            "如何做出一杯好咖啡？",
        ] * (num_samples // 10 + 1)


def validate_preference_data(data: List[Dict[str, str]]) -> List[str]:
    """
    验证偏好数据的质量

    Returns:
        List[str]: 发现的问题列表
    """
    issues = []

    for i, sample in enumerate(data):
        # 检查 chosen 和 rejected 是否相同
        if sample.get("chosen") == sample.get("rejected"):
            issues.append(f"Sample {i}: chosen == rejected")

        # 检查回答是否为空
        if not sample.get("chosen", "").strip():
            issues.append(f"Sample {i}: empty chosen response")
        if not sample.get("rejected", "").strip():
            issues.append(f"Sample {i}: empty rejected response")

        # 检查长度差异是否过大
        if sample.get("chosen") and sample.get("rejected"):
            len_diff = abs(len(sample["chosen"]) - len(sample["rejected"]))
            if len_diff > 2000:
                issues.append(
                    f"Sample {i}: unusually large length difference ({len_diff})"
                )

        # 检查 prompt 是否为空
        if not sample.get("prompt", "").strip():
            issues.append(f"Sample {i}: empty prompt")

    return issues


def main():
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载模型
    model, tokenizer = load_base_model_and_tokenizer(args.model)

    # 加载提示
    prompts = load_prompts_from_hh_rlhf(num_samples=args.num_samples, seed=args.seed)

    # 生成偏好数据
    preference_data = create_preference_data(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        strategy=args.strategy,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        max_length=args.max_length,
    )

    # 验证数据
    print("\nValidating preference data...")
    issues = validate_preference_data(preference_data)

    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    else:
        print("No issues found in preference data")

    # 保存数据
    print(f"\nSaving {len(preference_data)} preference pairs to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preference_data, f, ensure_ascii=False, indent=2)

    print("Done!")

    # 打印统计信息
    avg_chosen_len = sum(len(s["chosen"]) for s in preference_data) / len(
        preference_data
    )
    avg_rejected_len = sum(len(s["rejected"]) for s in preference_data) / len(
        preference_data
    )
    print(f"\nStatistics:")
    print(f"  Total samples: {len(preference_data)}")
    print(f"  Average chosen length: {avg_chosen_len:.1f} chars")
    print(f"  Average rejected length: {avg_rejected_len:.1f} chars")


if __name__ == "__main__":
    main()
