"""
Data Augmentation Scripts
数据增强脚本：复述、任务分解、负采样
"""

import json
import random
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict


@dataclass
class ParaphraseTask:
    """复述任务"""

    original_instruction: str
    paraphrased_versions: List[str] = None
    task_type: str = "paraphrase"
    domain: str = "general"

    def __post_init__(self):
        if self.paraphrased_versions is None:
            self.paraphrased_versions = []


class ParaphraseAugmenter:
    """复述增强器 - 改写指令保持意图"""

    TEMPLATES = [
        "请{action}",
        "能否麻烦您{action}",
        "我需要您帮我{action}",
        "请您{action}",
        "麻烦{action}",
        "请帮我{action}",
        "请教您{action}",
        "希望您能{action}",
    ]

    QUESTION_WORDS = ["什么", "怎么", "如何", "为什么", "何时", "何地", "谁", "哪个"]

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def augment(
        self, instructions: List[str], num_variants: int = 4, use_llm: bool = False
    ) -> List[ParaphraseTask]:
        """增强复述数据"""
        results = []

        for instr in tqdm(instructions, desc="Paraphrasing"):
            variants = []

            if use_llm and self.llm_client:
                variants = self._llm_paraphrase(instr, num_variants)
            else:
                variants = self._rule_based_paraphrase(instr, num_variants)

            results.append(
                ParaphraseTask(
                    original_instruction=instr, paraphrased_versions=variants
                )
            )

        return results

    def _rule_based_paraphrase(self, instruction: str, num_variants: int) -> List[str]:
        """基于规则的复述"""
        variants = set()

        if instruction.startswith("请"):
            variants.add(instruction)
            variants.add("麻烦" + instruction[1:])
            variants.add("能否请您" + instruction[1:])
        elif any(instruction.startswith(q) for q in self.QUESTION_WORDS):
            variants.add(instruction)
            variants.add(f"请问{instruction}")
            variants.add(f"我想知道{instruction}")
            variants.add(f"关于{instruction}")
        else:
            variants.add(instruction)
            variants.add(f"请{instruction}")
            variants.add(f"请您{instruction}")
            variants.add(f"麻烦{instruction}")

        base = instruction.replace("。", "").replace("？", "")
        variants.add(f"{base}，可以吗？")
        variants.add(f"请问{base}？")
        variants.add(f"关于{base}，请说明。")

        return list(variants)[:num_variants]

    def _llm_paraphrase(self, instruction: str, num_variants: int) -> List[str]:
        """使用 LLM 进行复述"""
        prompt = f"""请将以下指令改写成 {num_variants} 种不同的表达方式，保持语义不变：

原始指令：{instruction}

请以JSON格式输出，包含字段 paraphrases（字符串列表）："""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("paraphrases", [])


@dataclass
class SubTask:
    """分解后的子任务"""

    instruction: str
    parent_id: str = ""
    step_order: int = 0
    difficulty: str = "simple"

    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "parent_id": self.parent_id,
            "step_order": self.step_order,
            "difficulty": self.difficulty,
        }


class TaskDecomposer:
    """任务分解器 - 将复杂任务拆分为简单子任务"""

    DECOMPOSITION_KEYWORDS = {
        "分析": ["首先分析", "然后评估", "最后总结"],
        "比较": ["一方是", "另一方是", "综合比较结论"],
        "实现": ["设计阶段", "编码阶段", "测试阶段"],
        "解释": ["基本概念", "核心原理", "实际应用"],
    }

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def decompose(
        self, complex_tasks: List[Dict], method: str = "keyword"
    ) -> List[SubTask]:
        """分解复杂任务"""
        results = []

        for task in tqdm(complex_tasks, desc="Decomposing tasks"):
            if method == "keyword":
                sub_tasks = self._keyword_decompose(task)
            elif method == "llm" and self.llm_client:
                sub_tasks = self._llm_decompose(task)
            else:
                sub_tasks = self._keyword_decompose(task)

            results.extend(sub_tasks)

        return results

    def _keyword_decompose(self, task: Dict) -> List[SubTask]:
        """基于关键词的任务分解"""
        instruction = task.get("instruction", "")
        task_id = task.get("id", "unknown")

        sub_tasks = []
        step_num = 0

        for keyword, steps in self.DECOMPOSITION_KEYWORDS.items():
            if keyword in instruction:
                for step_template in steps:
                    sub_tasks.append(
                        SubTask(
                            instruction=step_template,
                            parent_id=task_id,
                            step_order=step_num,
                            difficulty="simple",
                        )
                    )
                    step_num += 1
                break
        else:
            parts = self._split_by_conjunction(instruction)
            for i, part in enumerate(parts):
                sub_tasks.append(
                    SubTask(
                        instruction=part.strip(),
                        parent_id=task_id,
                        step_order=i,
                        difficulty="intermediate",
                    )
                )

        return sub_tasks

    def _split_by_conjunction(self, text: str) -> List[str]:
        """通过连词拆分"""
        separators = ["，并且", "，而且", "，同时", "，此外", "；", "。"]
        parts = [text]

        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts

        return [p for p in parts if p.strip()]

    def _llm_decompose(self, task: Dict) -> List[SubTask]:
        """使用 LLM 进行任务分解"""
        instruction = task.get("instruction", "")
        task_id = task.get("id", "unknown")

        prompt = f"""将以下复杂任务分解为 3-5 个简单的子任务：

任务：{instruction}

要求：
1. 每个子任务应该是独立的、可执行的
2. 子任务之间有逻辑顺序
3. 难度从简单到复杂递进

请以JSON格式输出，包含字段 sub_tasks（对象列表，每个对象包含 instruction, step_order）："""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        result = json.loads(response.choices[0].message.content)
        sub_tasks_data = result.get("sub_tasks", [])

        return [
            SubTask(
                instruction=st["instruction"],
                parent_id=task_id,
                step_order=st.get("step_order", i),
                difficulty=st.get("difficulty", "simple"),
            )
            for i, st in enumerate(sub_tasks_data)
        ]


class NegativeSampler:
    """负采样器 - 生成对比性样本"""

    NEGATIVE_TYPES = [
        "wrong_answer",
        "irrelevant",
        "partially_correct",
        "style_mismatch",
    ]

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def generate_negative_samples(
        self, positive_samples: List[Dict], num_per_type: int = 1
    ) -> List[Dict]:
        """为正样本生成负样本"""
        results = []

        for sample in tqdm(positive_samples, desc="Generating negatives"):
            for neg_type in self.NEGATIVE_TYPES:
                if num_per_type <= 0:
                    continue

                if self.llm_client:
                    negative = self._llm_generate(sample, neg_type)
                else:
                    negative = self._rule_generate(sample, neg_type)

                results.append(
                    {
                        "original_instruction": sample.get("instruction"),
                        "positive_output": sample.get("response"),
                        "negative_output": negative,
                        "negative_type": neg_type,
                    }
                )

        return results

    def _rule_generate(self, sample: Dict, neg_type: str) -> str:
        """基于规则的负样本生成"""
        instruction = sample.get("instruction", "")
        response = sample.get("response", "")

        if neg_type == "wrong_answer":
            return (
                response[: len(response) // 2]
                if len(response) > 20
                else "这不是正确的答案。"
            )

        elif neg_type == "irrelevant":
            return "人工智能是当前技术发展的热点话题。"

        elif neg_type == "partially_correct":
            return (
                response[: len(response) // 3] + "..."
                if len(response) > 30
                else "部分正确。"
            )

        elif neg_type == "style_mismatch":
            return f"答案：{response}"

        return "未知类型的负样本。"

    def _llm_generate(self, sample: Dict, neg_type: str) -> str:
        """使用 LLM 生成负样本"""
        instruction = sample.get("instruction", "")
        positive = sample.get("response", "")

        type_descriptions = {
            "wrong_answer": "生成一个错误但不荒谬的答案",
            "irrelevant": "生成一个看似合理但完全无关的答案",
            "partially_correct": "生成一个包含部分正确但有关键错误的答案",
            "style_mismatch": "生成一个格式/风格不匹配的答案",
        }

        prompt = f"""基于以下问答对，生成一个{type_descriptions[neg_type]}：

指令：{instruction}
正确答案：{positive}

请直接输出负样本答案，不需要额外说明。"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        return response.choices[0].message.content


class ContrastivePairBuilder:
    """对比对构建器 - 构建正负对比训练对"""

    def __init__(self, negative_sampler: NegativeSampler):
        self.negative_sampler = negative_sampler

    def build_pairs(
        self, base_samples: List[Dict], balance_ratio: float = 0.5
    ) -> List[Dict]:
        """
        构建对比训练对
        balance_ratio: 正负样本比例
        """
        pairs = []

        for sample in tqdm(base_samples, desc="Building contrastive pairs"):
            instruction = sample.get("instruction", "")
            positive_response = sample.get("response", "")

            pairs.append(
                {
                    "instruction": instruction,
                    "chosen": positive_response,
                    "rejected": self.negative_sampler._rule_generate(
                        sample, "wrong_answer"
                    ),
                    "pair_type": "quality_contrast",
                }
            )

            pairs.append(
                {
                    "instruction": instruction,
                    "chosen": positive_response,
                    "rejected": self.negative_sampler._rule_generate(
                        sample, "irrelevant"
                    ),
                    "pair_type": "relevance_contrast",
                }
            )

        return pairs

    def format_for_dpo(self, pairs: List[Dict]) -> List[Dict]:
        """格式化为 DPO 训练格式"""
        return [
            {
                "prompt": pair["instruction"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }
            for pair in pairs
        ]


class AugmentationPipeline:
    """增强流水线 - 整合所有增强方法"""

    def __init__(
        self,
        llm_client=None,
        para_augmenter: ParaphraseAugmenter = None,
        decomposer: TaskDecomposer = None,
        neg_sampler: NegativeSampler = None,
    ):
        self.llm_client = llm_client
        self.para_augmenter = para_augmenter or ParaphraseAugmenter(llm_client)
        self.decomposer = decomposer or TaskDecomposer(llm_client)
        self.neg_sampler = neg_sampler or NegativeSampler(llm_client)
        self.contrast_builder = ContrastivePairBuilder(self.neg_sampler)

    def run_full_pipeline(
        self,
        base_instructions: List[Dict],
        paraphrase_ratio: float = 2.0,
        decomposition_ratio: float = 0.3,
        negative_ratio: float = 1.0,
    ) -> Dict[str, List]:
        """运行完整增强流水线"""

        print("Step 1: Paraphrase augmentation...")
        para_results = self.para_augmenter.augment(
            [inst["instruction"] for inst in base_instructions],
            num_variants=int(paraphrase_ratio),
        )

        print("Step 2: Task decomposition...")
        complex_tasks = [
            t
            for t in base_instructions
            if any(
                kw in t.get("instruction", "")
                for kw in ["分析", "比较", "实现", "解释"]
            )
        ]
        sub_tasks = self.decomposer.decompose(complex_tasks)

        print("Step 3: Negative sampling...")
        negative_samples = self.neg_sampler.generate_negative_samples(
            base_instructions, num_per_type=int(negative_ratio)
        )

        print("Step 4: Building contrastive pairs...")
        contrastive_pairs = self.contrast_builder.build_pairs(base_instructions)

        return {
            "paraphrase_tasks": [p.to_dict() for p in para_results],
            "decomposed_tasks": [s.to_dict() for s in sub_tasks],
            "negative_samples": negative_samples,
            "contrastive_pairs": contrastive_pairs,
        }

    def save_augmented_data(self, results: Dict, output_dir: str):
        """保存增强后的数据"""
        for name, data in results.items():
            filepath = f"{output_dir}/{name}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(data)} {name} to {filepath}")


if __name__ == "__main__":
    sample_instructions = [
        {
            "id": "task_001",
            "instruction": "解释什么是机器学习中的监督学习",
            "response": "监督学习是机器学习的一种范式，通过使用标注数据进行训练，模型学习从输入到输出的映射关系。",
            "task_type": "explanation",
            "domain": "technology",
        },
        {
            "id": "task_002",
            "instruction": "对比分析CNN和RNN在处理序列数据时的优劣",
            "response": "CNN擅长并行处理局部特征，计算效率高但难以处理长距离依赖；RNN通过循环结构能够处理任意长度的序列，但存在梯度消失问题。",
            "task_type": "comparison",
            "domain": "technology",
        },
        {
            "id": "task_003",
            "instruction": "用Python实现一个简单的图像分类模型",
            "response": "可以使用PyTorch定义一个卷积神经网络...",
            "task_type": "code_generation",
            "domain": "programming",
        },
    ]

    print("Sample instructions loaded for augmentation demonstration")
    print(f"Total samples: {len(sample_instructions)}")

    augmenter = AugmentationPipeline()

    sample_result = augmenter.run_full_pipeline(sample_instructions[:1])

    for key, value in sample_result.items():
        print(f"\n{key}: {len(value)} items generated")
