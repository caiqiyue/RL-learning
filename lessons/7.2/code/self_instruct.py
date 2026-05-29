"""
Self-Instruct Implementation
实现 Self-Instruct 方法：种子选择、指令生成、多轮过滤
"""

import json
import random
import math
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


@dataclass
class SeedTask:
    """种子任务数据结构"""

    id: str
    instruction: str
    instance: Optional[Dict[str, str]] = None
    task_type: str = "unknown"
    domain: str = "general"
    difficulty: str = "intermediate"

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "instance": self.instance,
            "task_type": self.task_type,
            "domain": self.domain,
            "difficulty": self.difficulty,
        }


@dataclass
class GeneratedTask:
    """生成任务数据结构"""

    instruction: str
    response: str = ""
    task_type: str = "unknown"
    domain: str = "general"
    difficulty: str = "intermediate"
    seed_id: Optional[str] = None
    quality_score: float = 0.0
    is_filtered: bool = False
    filter_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "response": self.response,
            "task_type": self.task_type,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "seed_id": self.seed_id,
            "quality_score": self.quality_score,
            "is_filtered": self.is_filtered,
            "filter_reason": self.filter_reason,
        }


class SelfInstructGenerator:
    """Self-Instruct 核心生成器"""

    DEFAULT_SYSTEM_PROMPT = """你是一位专业的指令数据生成专家，专注于为语言模型微调生成高质量的训练数据。

你的任务是根据给定的种子任务，生成新的指令-响应对。

生成规则：
1. 生成的指令应该清晰、具体，避免歧义
2. 响应应该准确、完整，直接回答指令要求
3. 新生成的任务应该与种子任务有所不同（不同主题、不同表达方式）
4. 任务类型应该多样化：问答、写作、推理、分类、总结等
5. 难度应该有梯度分布：简单、中等、困难

禁止生成：
- 涉及政治、暴力、色情等敏感内容
- 超长指令（超过200字）或超短指令（少于10字）
- 无法回答或无明确答案的问题
- 与已有任务高度重复的内容

请以JSON格式输出，包含以下字段：instruction, response, task_type, difficulty, domain"""

    DEFAULT_FEW_SHOT_PROMPT = """以下是多条生成示例：

示例1：
{
  "instruction": "解释什么是光合作用",
  "response": "光合作用是植物、藻类和某些细菌将光能转化为化学能的过程。在光合作用中，植物利用阳光、二氧化碳和水生成葡萄糖和氧气。这是地球生态系统中能量循环的基础。",
  "task_type": "explanation",
  "difficulty": "simple",
  "domain": "biology"
}

示例2：
{
  "instruction": "对比分析日本和德国在二战后经济复兴策略的异同",
  "response": "共同点：两国都实施了强有力的产业政策，重视技术引进和消化。不同点：日本采用'政府主导型'模式，通过产业政策引导经济发展；德国则更依赖市场机制，强调社会市场经济制度。",
  "task_type": "comparison",
  "difficulty": "intermediate",
  "domain": "history_economics"
}

示例3：
{
  "instruction": "用Python编写一个函数，判断一个字符串是否为回文串",
  "response": "```python\\ndef is_palindrome(s):\\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\\n    return cleaned == cleaned[::-1]\\n```",
  "task_type": "code_generation",
  "difficulty": "intermediate",
  "domain": "programming"
}

现在请根据上述格式，生成新的指令-响应对："""

    def __init__(
        self,
        api_provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        few_shot_prompt: Optional[str] = None,
    ):
        self.api_provider = api_provider
        self.model = model

        if api_provider == "openai" and OpenAI:
            self.client = OpenAI(api_key=api_key)
        elif api_provider == "anthropic" and Anthropic:
            self.client = Anthropic(api_key=api_key)
        else:
            self.client = None

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.few_shot_prompt = few_shot_prompt or self.DEFAULT_FEW_SHOT_PROMPT

    def generate(
        self,
        seed_tasks: List[Dict],
        num_generations: int = 10,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_retries: int = 3,
    ) -> List[GeneratedTask]:
        """从种子任务生成新的指令数据"""
        results = []

        for seed in tqdm(seed_tasks, desc="Generating tasks"):
            for attempt in range(max_retries):
                try:
                    task_dict = self._call_api(seed, temperature, top_p)
                    task = GeneratedTask(
                        instruction=task_dict.get("instruction", ""),
                        response=task_dict.get("response", ""),
                        task_type=task_dict.get("task_type", "unknown"),
                        difficulty=task_dict.get("difficulty", "intermediate"),
                        domain=task_dict.get("domain", "general"),
                        seed_id=seed.get("id"),
                    )

                    if self._passes_basic_filters(task):
                        results.append(task)
                    else:
                        task.is_filtered = True
                        task.filter_reason = "basic_filter_failed"
                    break

                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed after {max_retries} attempts: {e}")
                    continue

        return results

    def _call_api(self, seed: Dict, temperature: float, top_p: float) -> Dict:
        """调用 API 生成任务"""
        prompt = (
            self.few_shot_prompt
            + f"\n\n种子任务: {json.dumps(seed, ensure_ascii=False)}"
        )

        if self.api_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)

        elif self.api_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return json.loads(response.content[0].text)

        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")

    def _passes_basic_filters(self, task: GeneratedTask) -> bool:
        """基本过滤规则"""
        if len(task.instruction) < 10 or len(task.instruction) > 200:
            return False
        if len(task.response) < 20:
            return False
        if any(word in task.instruction.lower() for word in ["暴力", "色情", "政治"]):
            return False
        return True


class DiversityAnalyzer:
    """多样性分析器"""

    TASK_TYPES = [
        "question_answering",
        "writing",
        "reasoning",
        "summarization",
        "code_generation",
        "classification",
        "explanation",
        "comparison",
        "others",
    ]

    DOMAINS = [
        "科技",
        "教育",
        "医疗",
        "金融",
        "法律",
        "历史",
        "文学",
        "艺术",
        "体育",
        "生活",
        "general",
    ]

    def __init__(self):
        self.stats = {}

    def analyze(self, tasks: List[GeneratedTask]) -> Dict[str, Any]:
        """分析数据集的多样性"""
        if not tasks:
            return {}

        self.stats = {
            "total": len(tasks),
            "task_type_distribution": self._analyze_task_types(tasks),
            "domain_distribution": self._analyze_domains(tasks),
            "length_distribution": self._analyze_lengths(tasks),
            "difficulty_distribution": self._analyze_difficulty(tasks),
        }

        return self.stats

    def _analyze_task_types(self, tasks: List[GeneratedTask]) -> Dict[str, int]:
        counter = defaultdict(int)
        for task in tasks:
            counter[task.task_type] += 1
        return dict(counter)

    def _analyze_domains(self, tasks: List[GeneratedTask]) -> Dict[str, int]:
        counter = defaultdict(int)
        for task in tasks:
            counter[task.domain] += 1
        return dict(counter)

    def _analyze_lengths(self, tasks: List[GeneratedTask]) -> Dict[str, Any]:
        response_lengths = [len(t.response) for t in tasks]
        return {
            "min": min(response_lengths),
            "max": max(response_lengths),
            "mean": sum(response_lengths) / len(response_lengths),
            "median": sorted(response_lengths)[len(response_lengths) // 2],
        }

    def _analyze_difficulty(self, tasks: List[GeneratedTask]) -> Dict[str, int]:
        counter = defaultdict(int)
        for task in tasks:
            counter[task.difficulty] += 1
        return dict(counter)

    def print_report(self):
        """打印多样性报告"""
        if not self.stats:
            print("No statistics available. Run analyze() first.")
            return

        print("\n" + "=" * 50)
        print("DIVERSITY ANALYSIS REPORT")
        print("=" * 50)

        print(f"\nTotal tasks: {self.stats['total']}")

        print("\n--- Task Type Distribution ---")
        for ttype, count in sorted(
            self.stats["task_type_distribution"].items(), key=lambda x: -x[1]
        ):
            pct = count / self.stats["total"] * 100
            bar = "█" * int(pct / 2)
            print(f"  {ttype:20s}: {count:4d} ({pct:5.1f}%) {bar}")

        print("\n--- Domain Distribution ---")
        for domain, count in sorted(
            self.stats["domain_distribution"].items(), key=lambda x: -x[1]
        ):
            pct = count / self.stats["total"] * 100
            print(f"  {domain:15s}: {count:4d} ({pct:5.1f}%)")

        print("\n--- Response Length Distribution ---")
        length_stats = self.stats["length_distribution"]
        print(f"  Min: {length_stats['min']:,} chars")
        print(f"  Max: {length_stats['max']:,} chars")
        print(f"  Mean: {length_stats['mean']:,.1f} chars")
        print(f"  Median: {length_stats['median']:,} chars")

        print("\n--- Difficulty Distribution ---")
        for diff, count in sorted(self.stats["difficulty_distribution"].items()):
            pct = count / self.stats["total"] * 100
            print(f"  {diff:15s}: {count:4d} ({pct:5.1f}%)")

        print("=" * 50 + "\n")


class SelfInstructPipeline:
    """Self-Instruct 完整流水线"""

    def __init__(
        self,
        seed_tasks: List[SeedTask],
        generator: SelfInstructGenerator,
        quality_classifier=None,
    ):
        self.seed_tasks = seed_tasks
        self.generator = generator
        self.quality_classifier = quality_classifier
        self.all_tasks: List[GeneratedTask] = []

    def run(
        self,
        num_rounds: int = 5,
        tasks_per_round: int = 100,
        quality_threshold: float = 0.7,
        diversity_threshold: float = 0.3,
    ) -> List[GeneratedTask]:
        """运行完整的 Self-Instruct 流程"""

        seed_dicts = [s.to_dict() for s in self.seed_tasks]

        for round_idx in range(num_rounds):
            print(f"\n{'=' * 40}")
            print(f"Round {round_idx + 1}/{num_rounds}")
            print(f"{'=' * 40}")

            new_tasks = self.generator.generate(
                seed_tasks=seed_dicts, num_generations=tasks_per_round, temperature=0.9
            )

            print(f"Generated: {len(new_tasks)} tasks")

            if self.quality_classifier:
                new_tasks = self._filter_by_quality(new_tasks, quality_threshold)
                print(f"After quality filter: {len(new_tasks)} tasks")

            self.all_tasks.extend(new_tasks)

            self._update_seeds(new_tasks)

            analyzer = DiversityAnalyzer()
            analyzer.analyze(self.all_tasks)
            analyzer.print_report()

        return self.all_tasks

    def _filter_by_quality(
        self, tasks: List[GeneratedTask], threshold: float
    ) -> List[GeneratedTask]:
        """使用质量分类器过滤"""
        filtered = []
        for task in tasks:
            score = self.quality_classifier.predict_quality(task.to_dict())
            task.quality_score = score
            if score >= threshold:
                filtered.append(task)
            else:
                task.is_filtered = True
                task.filter_reason = f"quality_score_{score:.2f}"
        return filtered

    def _update_seeds(self, new_tasks: List[GeneratedTask]):
        """更新种子池，添加通过过滤的高质量任务"""
        for task in new_tasks:
            if not task.is_filtered:
                seed_dicts = [s.to_dict() for s in self.seed_tasks]
                unique = True
                for sd in seed_dicts:
                    if (
                        self._text_similarity(
                            task.instruction, sd.get("instruction", "")
                        )
                        > 0.8
                    ):
                        unique = False
                        break
                if unique:
                    self.seed_tasks.append(
                        SeedTask(
                            id=f"gen_{len(self.seed_tasks)}",
                            instruction=task.instruction,
                            instance={"input": "", "output": task.response},
                            task_type=task.task_type,
                            domain=task.domain,
                            difficulty=task.difficulty,
                        )
                    )

    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)


def load_seeds_from_file(filepath: str) -> List[SeedTask]:
    """从文件加载种子任务"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for item in data:
        tasks.append(
            SeedTask(
                id=item.get("id", f"seed_{len(tasks)}"),
                instruction=item.get("instruction", ""),
                instance=item.get("instance"),
                task_type=item.get("task_type", "unknown"),
                domain=item.get("domain", "general"),
                difficulty=item.get("difficulty", "intermediate"),
            )
        )
    return tasks


def save_tasks_to_json(tasks: List[GeneratedTask], filepath: str):
    """保存生成的任务到 JSON 文件"""
    output = [t.to_dict() for t in tasks]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sample_seeds = [
        SeedTask(
            id="seed_001",
            instruction="解释什么是人工智能中的神经网络",
            task_type="explanation",
            domain="technology",
            difficulty="simple",
        ),
        SeedTask(
            id="seed_002",
            instruction="对比分析传统机器学习与深度学习的优劣",
            task_type="comparison",
            domain="technology",
            difficulty="intermediate",
        ),
        SeedTask(
            id="seed_003",
            instruction="用Python实现一个快速排序算法",
            task_type="code_generation",
            domain="programming",
            difficulty="intermediate",
        ),
    ]

    print("Sample seeds loaded:")
    for seed in sample_seeds:
        print(f"  - {seed.instruction[:50]}...")
