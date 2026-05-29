# 7.2 指令数据合成与多样性增强

## 课程概述

本节课聚焦于大模型微调中至关重要的数据工程环节——指令数据合成与多样性增强。当特定任务的标注数据稀缺时，如何利用大模型自身能力生成高质量的指令数据成为关键课题。我们将深入探讨 Self-Instruct 方法论、数据增强技术、提示工程策略以及质量过滤机制，帮助学员掌握从零构建指令数据集的完整能力。

## 学习目标

- 理解 Self-Instruct 方法的核心原理与执行流程
- 掌握数据增强三大技术：复述、任务分解、负采样
- 学会设计用于合成数据的提示词模板
- 了解多样性指标体系与质量过滤方法
- 能够使用主流 API 或开源模型实现自动化数据合成

## 前置知识

- 了解监督式微调（SFT）的基本原理
- 熟悉大模型 API 的基本调用方式
- 具备 Python 编程基础

---

## 1. Self-Instruct 方法

Self-Instruct 是由 Taori 等人在 2023 年提出的指令数据自动生成方法，其核心思想是**利用强语言模型从少量种子任务出发，通过多轮生成-过滤-评分循环，逐步构建大规模指令数据集**。该方法极大地降低了对人工标注的依赖，是现代微调数据工程的重要基石。

### 1.1 种子任务池构建

种子任务池（Seed Task Pool）是整个 Self-Instruct 流程的起点。一个高质量的种子池应当具备以下特征：

**多样性覆盖**：涵盖多种任务类型（问答、写作、推理、代码等）和多个主题领域（科技、教育、金融、医疗等）。种子量通常在 100-500 条之间，质量和多样性比数量更重要。

**任务格式规范**：每条种子任务包含 instruction（指令）和 instance（示例）两部分。instruction 描述任务要求，instance 可以是 input-output 对，也可以仅有 instruction 而无确定答案的开放型任务。

**构建方法**：可从现有数据集（如 Super-Natural Instructions、FLAN）筛选，也可基于人工设计。关键是要覆盖不同的难度层级和任务范式。

```python
# 种子任务示例结构
seed_task = {
    "id": "seed_001",
    "instruction": "解释什么是机器学习中的梯度下降算法",
    "instance": {
        "input": "",
        "output": "梯度下降是一种优化算法，用于最小化损失函数..."
    },
    "task_type": "explanation",
    "domain": "machine_learning",
    "difficulty": "intermediate"
}
```

### 1.2 生成-过滤-评分多轮循环

Self-Instruct 的核心是一个迭代增强循环，每轮包含以下步骤：

**步骤一：任务生成（Task Generation）**
从种子池或已生成的任务中采样，提示语言模型生成新的指令任务。模型需要判断该任务是否与现有任务足够不同，以保证多样性。

**步骤二：过滤（Filtering）**
去除明显低质量或重复的任务。过滤规则包括：指令过长或过短、包含敏感内容、明显不可回答等。

**步骤三：质量评分（Quality Scoring）**
使用专用评分模型或规则对生成的任务进行质量评估。评分维度包括指令清晰度、任务可解决性、答案正确性等。

```python
# Self-Instruct 多轮循环伪代码
def self_instruct_loop(seed_pool, num_rounds=5, target_size=10000):
    generated_tasks = []
    all_tasks = seed_pool.copy()
    
    for round_idx in range(num_rounds):
        print(f"=== Round {round_idx + 1} ===")
        
        # 1. 任务生成
        new_tasks = []
        for _ in range(batch_size):
            # 采样任务并生成新的
            sampled = sample_tasks(all_tasks, n=3)
            new_task = generate_task(sampled)
            new_tasks.append(new_task)
        
        # 2. 过滤
        filtered_tasks = filter_tasks(new_tasks)
        
        # 3. 评分
        scored_tasks = score_tasks(filtered_tasks)
        high_quality = [t for t in scored_tasks if t["score"] >= threshold]
        
        # 4. 加入结果集
        generated_tasks.extend(high_quality)
        all_tasks.extend(high_quality)
        
        print(f"Generated: {len(new_tasks)}, Passed: {len(high_quality)}")
    
    return generated_tasks
```

### 1.3 多样性增强策略

多样性是 Self-Instruct 的核心挑战之一。以下策略可有效提升数据集多样性：

**任务类型多样化**：确保数据集中包含不同类型的任务——有明确答案的问答、开放式写作、推理分析、代码生成、多步骤任务等。每种类型应有合理占比。

**主题领域扩展**：通过领域词汇提示或主题约束，引导模型生成覆盖不同领域（如法律、医学、教育、科技）的任务。

**难度梯度设计**：在提示中加入难度描述（如"为初学者设计一个问题"vs"设计一个需要深入专业知识的问题"），使数据集包含从简单到复杂的难度梯度。

**反面样本引入**：生成一些容易但答案错误的样本，训练模型识别错误答案的能力。

---

## 2. 数据增强技术

除了 Self-Instruct，还有一类重要的数据增强技术，通过对现有数据进行变换来扩充数据集。核心策略包括复述、任务分解和负采样。

### 2.1 复述增强

复述（Paraphrasing）是指在不改变任务意图的前提下，用不同措辞重新表达指令。这是最基本的数据增强方法，可以显著增加数据多样性，同时保持语义一致。

**实现要点**：
- 保留关键实体和约束条件
- 改变句式结构但保持语法正确
- 保持任务难度大致不变

**应用场景**：当某类指令数据不足时，通过复述生成同一指令的多个变体。

```python
# 复述增强示例
original_instruction = "请总结这篇文章的主要观点，重点关注作者的核心论点。"

paraphrased_variants = [
    "请概括这篇文档的主要内容，特别强调作者想要传达的核心观点。",
    "对这篇文章进行摘要，说明作者的主要论点是什么。",
    "阅读以下文章并提炼其要点，重点阐述作者的核心主张。",
    "请提取这篇文章的主旨，并用简洁的语言概括作者的主要观点。"
]
```

### 2.2 任务分解增强

任务分解（Task Decomposition）是将复杂任务拆解为多个简单子任务的过程。这种方法不仅能扩充数据量，还能帮助模型学习逐步推理的能力。

**分解策略**：
- 按步骤拆分：将多步骤任务拆为独立的单步骤指令
- 按子主题拆分：将涉及多个主题的任务拆为单一主题的子任务
- 按难度递进：从简单问题引导到复杂问题

```python
# 复杂任务分解示例
complex_task = "分析一家科技公司是否值得投资，包括财务指标、市场竞争、技术优势三个方面"

decomposed_tasks = [
    {
        "instruction": "请分析这家科技公司的主要财务指标，包括营收增长、利润率、现金流等",
        "parent_id": "invest_analysis_001"
    },
    {
        "instruction": "评估该公司在所处市场中的竞争地位，分析主要竞争对手及其市场份额",
        "parent_id": "invest_analysis_001"
    },
    {
        "instruction": "分析该公司的核心技术优势，包括专利技术、研发投入、技术团队等",
        "parent_id": "invest_analysis_001"
    }
]
```

### 2.3 负采样增强

负采样（Negative Sampling）是生成对比性样本的技术，训练模型区分正确与错误、相关与不相关的答案。这是提升模型判断能力的重要手段。

**负样本类型**：
- **错误答案型**：指令不变，替换为错误或不完整的答案
- **无关答案型**：答案看似合理但未真正回答问题
- **部分正确型**：答案部分正确但包含关键错误
- **风格对比型**：同一问题的好答案与差答案对比

```python
# 负采样示例：生成对比答案对
original_qa = {
    "instruction": "什么是Python中的列表推导式？",
    "positive_output": "列表推导式是Python的一种简洁语法，用于通过表达式和循环创建新列表。例如 [x**2 for x in range(5)] 会生成 [0, 1, 4, 9, 16]。相比传统循环，代码更简洁高效。"
}

negative_samples = [
    {
        "type": "wrong_answer",
        "output": "列表推导式是Python的一种数据类型，用于存储有序集合。"
    },
    {
        "type": "irrelevant",
        "output": "Python是一种高级编程语言，由Guido van Rossum创建。"
    },
    {
        "type": "partially_correct",
        "output": "列表推导式用于创建列表，语法是 [expr for item in iterable]。它可以让代码更短。"
    }
]
```

---

## 3. 提示工程用于合成数据

合成数据的质量高度依赖于提示词的设计。本节将详细讨论系统提示、小样本示例以及采样参数的设置策略。

### 3.1 系统提示设计

系统提示（System Prompt）定义了生成器模型的角色和行为准则。一个优秀的数据合成系统提示应当包含以下要素：

**角色定义**：明确模型作为"数据生成专家"的身份，提供领域背景。

**生成规则**：清晰列出生成任务时必须满足的条件，如格式要求、质量标准、多样性要求等。

**约束边界**：明确禁止生成的内容类型，如敏感话题、过长回答、重复模式等。

```python
# 数据合成系统提示示例
SYSTEM_PROMPT = """你是一位专业的指令数据生成专家，专注于为语言模型微调生成高质量的训练数据。

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

请以JSON格式输出，字段包括：instruction, response, task_type, difficulty, domain"""
```

### 3.2 小样本示例

小样本示例（Few-shot Examples）在提示中提供 2-5 个示范，可以显著提升生成质量。示例应当覆盖不同任务类型，并展示期望的输出格式。

```python
# 小样本示例在提示中的使用
FEW_SHOT_PROMPT = """你是一位专业的指令数据生成专家。以下是一些生成示例：

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
  "response": "```python\\ndef is_palindrome(s):\\n    # 去除标点和空格，转小写\\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\\n    return cleaned == cleaned[::-1]\\n```",
  "task_type": "code_generation",
  "difficulty": "intermediate",
  "domain": "programming"
}

现在请根据上述格式，生成新的指令-响应对："""
```

### 3.3 采样参数设置

采样参数（Temperature 和 Top-p）控制生成的多样性。参数选择需要根据任务类型和需求进行调整。

**Temperature（温度）**：
- 低温度（0.1-0.3）：生成更确定性，适合需要准确答案的任务
- 中温度（0.5-0.7）：平衡多样性与质量，适合一般生成任务
- 高温度（0.8-1.0）：生成更多样化，适合创意写作和探索性任务

**Top-p（核采样）**：
- 低 top-p（0.5-0.7）：只从最高概率的词汇中采样，生成更保守
- 高 top-p（0.85-0.95）：允许更多低概率词汇进入采样池，增加多样性

```python
# 不同任务的推荐参数设置
GENERATION_CONFIGS = {
    "diverse_task_generation": {
        "temperature": 0.9,
        "top_p": 0.95,
        "system_prompt": SYSTEM_PROMPT,
        "few_shot_prompt": FEW_SHOT_PROMPT
    },
    "high_quality_response": {
        "temperature": 0.3,
        "top_p": 0.8,
        "system_prompt": "你是一位专家，生成准确、详细的回答。"
    },
    "paraphrasing": {
        "temperature": 0.7,
        "top_p": 0.9,
        "system_prompt": "在不改变语义的前提下，用不同的表达方式重写以下文本。"
    }
}
```

---

## 4. 多样性指标体系

评估合成数据质量时，多样性是关键维度之一。一个完善的多样性指标体系应涵盖任务类型分布、主题覆盖度和响应长度分布。

### 4.1 指令类型分布

分析数据集中不同任务类型的占比，确保类型分布均衡。常见任务类型包括：

| 任务类型 | 描述 | 理想占比 |
|---------|------|---------|
| question_answering | 问答类 | 25% |
| writing | 写作类 | 20% |
| reasoning | 推理类 | 15% |
| summarization | 摘要类 | 10% |
| code_generation | 代码类 | 10% |
| classification | 分类类 | 10% |
| others | 其他 | 10% |

```python
# 任务类型分布分析
from collections import Counter
import json

def analyze_task_type_distribution(dataset):
    task_types = [item.get("task_type", "unknown") for item in dataset]
    counter = Counter(task_types)
    total = len(task_types)
    
    print("=== 任务类型分布 ===")
    for task_type, count in counter.most_common():
        percentage = count / total * 100
        bar = "█" * int(percentage / 2)
        print(f"{task_type:20s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    # 计算分布均匀度（熵）
    entropy = -sum((count/total) * math.log(count/total, 2) for count in counter.values())
    max_entropy = math.log(len(counter), 2)
    uniformity = entropy / max_entropy if max_entropy > 0 else 0
    print(f"\n分布均匀度: {uniformity:.2%}")
    return counter
```

### 4.2 主题覆盖度

主题覆盖度衡量数据集覆盖领域的广度。可以使用嵌入模型计算主题相似度，或通过关键词匹配统计领域分布。

```python
# 主题覆盖度分析
DOMAINS = ["科技", "教育", "医疗", "金融", "法律", "历史", "文学", "艺术", "体育", "生活"]

def analyze_domain_coverage(dataset):
    domain_counts = Counter()
    
    for item in dataset:
        instruction = item.get("instruction", "")
        domain = item.get("domain", "unknown")
        if domain == "unknown":
            # 简单的关键词匹配
            for d in DOMAINS:
                if d in instruction:
                    domain_counts[d] += 1
                    break
            else:
                domain_counts["other"] += 1
        else:
            domain_counts[domain] += 1
    
    print("=== 主题覆盖度 ===")
    for domain, count in domain_counts.most_common():
        print(f"{domain}: {count}")
    
    covered_domains = len([d for d, c in domain_counts.items() if c > 0])
    print(f"\n覆盖的主题数: {covered_domains}/{len(DOMAINS)}")
    return domain_counts
```

### 4.3 响应长度分布

响应长度分布反映了数据集对不同详细程度任务的覆盖。理想的数据集应包含从简短回答到详尽分析的各类长度。

```python
import statistics

def analyze_response_length_distribution(dataset):
    lengths = [len(item.get("response", "")) for item in dataset]
    
    print("=== 响应长度分布 ===")
    print(f"最短: {min(lengths):,} 字符")
    print(f"最长: {max(lengths):,} 字符")
    print(f"平均: {statistics.mean(lengths):,.1f} 字符")
    print(f"中位数: {statistics.median(lengths):,.1f} 字符")
    print(f"标准差: {statistics.stdev(lengths):,.1f}")
    
    # 长度区间分布
    buckets = {"<100": 0, "100-300": 0, "300-500": 0, "500-1000": 0, ">1000": 0}
    for length in lengths:
        if length < 100:
            buckets["<100"] += 1
        elif length < 300:
            buckets["100-300"] += 1
        elif length < 500:
            buckets["300-500"] += 1
        elif length < 1000:
            buckets["500-1000"] += 1
        else:
            buckets[">1000"] += 1
    
    print("\n长度区间分布:")
    for bucket, count in buckets.items():
        pct = count / len(lengths) * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>10s}: {count:4d} ({pct:5.1f}%) {bar}")
```

---

## 5. 质量过滤

合成数据中不可避免地存在低质量样本，需要通过质量过滤机制剔除。本节介绍两种主要方法：基于分类器的过滤和人在环验证。

### 5.1 分类器过滤

训练一个轻量级分类器来判断合成数据的质量。分类器可以使用规则特征（如长度、格式），也可以使用 BERT 等预训练模型进行更语义化的判断。

```python
# 质量分类器过滤
import numpy as np

class QualityClassifier:
    def __init__(self, model_path=None):
        self.model = self._load_model(model_path) if model_path else None
    
    def _load_model(self, path):
        # 加载预训练模型
        from transformers import pipeline
        return pipeline("text-classification", model=path)
    
    def extract_features(self, task):
        """提取质量相关特征"""
        features = {}
        
        # 长度特征
        features["instruction_len"] = len(task.get("instruction", ""))
        features["response_len"] = len(task.get("response", ""))
        
        # 格式特征
        features["has_instruction"] = 1 if task.get("instruction") else 0
        features["has_response"] = 1 if task.get("response") else 0
        
        # 结构特征
        instruction = task.get("instruction", "")
        features["question_mark"] = 1 if "?" in instruction else 0
        features["starts_with_please"] = 1 if instruction.startswith("请") else 0
        
        return features
    
    def predict_quality(self, task):
        """预测任务质量分数 (0-1)"""
        if self.model:
            # 使用预训练模型预测
            result = self.model(task.get("instruction", "") + " " + task.get("response", ""))
            return result[0]["score"]
        else:
            # 基于规则的简单评分
            features = self.extract_features(task)
            score = 0.5
            
            # 长度合理性
            if 20 <= features["instruction_len"] <= 200:
                score += 0.1
            if 50 <= features["response_len"] <= 2000:
                score += 0.1
            
            # 格式完整性
            if features["has_instruction"] and features["has_response"]:
                score += 0.2
            
            return min(score, 1.0)

def filter_by_classifier(tasks, classifier, threshold=0.7):
    """使用分类器过滤低质量任务"""
    filtered = []
    
    for task in tqdm(tasks, desc="Quality filtering"):
        score = classifier.predict_quality(task)
        if score >= threshold:
            task["quality_score"] = score
            filtered.append(task)
    
    print(f"过滤前: {len(tasks)}, 过滤后: {len(filtered)}, 通过率: {len(filtered)/len(tasks):.1%}")
    return filtered
```

### 5.2 人在环验证采样

对于高风险或高价值的任务，采用人在环（Human-in-the-Loop）方式进行验证。采样策略包括：

**随机采样**：随机选取一定比例的样本进行人工审核。

**分层层采样**：按任务类型、难度、领域等维度分层，确保各层都有采样。

**高不确定性采样**：对于分类器判断为中等质量的样本（0.4-0.6分），优先进行人工审核。

```python
# 人在环验证采样策略
import random

class HumanInTheLoopSampler:
    def __init__(self, confidence_threshold=(0.4, 0.7)):
        self.low_conf_threshold = confidence_threshold[0]
        self.high_conf_threshold = confidence_threshold[1]
    
    def random_sample(self, tasks, sample_size):
        """随机采样"""
        return random.sample(tasks, min(sample_size, len(tasks)))
    
    def stratified_sample(self, tasks, sample_size):
        """分层层采样"""
        # 按任务类型分层
        by_type = defaultdict(list)
        for task in tasks:
            by_type[task.get("task_type", "unknown")].append(task)
        
        samples = []
        samples_per_type = sample_size // len(by_type)
        
        for task_type, type_tasks in by_type.items():
            if len(type_tasks) <= samples_per_type:
                samples.extend(type_tasks)
            else:
                samples.extend(random.sample(type_tasks, samples_per_type))
        
        return samples
    
    def uncertainty_sample(self, tasks_with_scores, sample_size):
        """高不确定性采样：优先选择分类器得分中等的样本"""
        uncertain = [
            t for t in tasks_with_scores 
            if self.low_conf_threshold <= t.get("quality_score", 0.5) <= self.high_conf_threshold
        ]
        
        if len(uncertain) >= sample_size:
            return random.sample(uncertain, sample_size)
        else:
            # 不足时补充随机采样
            remaining_needed = sample_size - len(uncertain)
            remaining = [t for t in tasks_with_scores if t not in uncertain]
            additional = random.sample(remaining, min(remaining_needed, len(remaining)))
            return uncertain + additional

def human_review_pipeline(tasks, classifier, human_review_budget=100):
    """人工审核流程"""
    sampler = HumanInTheLoopSampler()
    
    # 1. 分类器打分
    for task in tasks:
        task["quality_score"] = classifier.predict_quality(task)
    
    # 2. 采样待审核样本
    uncertain_samples = [t for t in tasks if 0.4 <= t["quality_score"] <= 0.7]
    certain_samples = [t for t in tasks if t["quality_score"] > 0.7 or t["quality_score"] < 0.4]
    
    if len(uncertain_samples) >= human_review_budget:
        review_samples = sampler.uncertain_sample(tasks, human_review_budget)
    else:
        review_samples = uncertain_samples + sampler.random_sample(
            [t for t in tasks if t not in uncertain_samples],
            human_review_budget - len(uncertain_samples)
        )
    
    # 3. 返回待审核列表
    return review_samples
```

---

## 6. 工具使用

本节介绍如何使用主流 API 和开源模型进行指令数据合成。

### 6.1 OpenAI API

```python
# OpenAI API 数据合成
from openai import OpenAI
import os

class OpenAIDataSynthesizer:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate_instruction(self, system_prompt, few_shot_prompt, seed_task, temperature=0.9):
        """生成单条指令数据"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": few_shot_prompt + f"\n\n种子任务: {seed_task}"}
            ],
            temperature=temperature,
            top_p=0.95,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def batch_generate(self, seeds, system_prompt, few_shot_prompt, max_retries=3):
        """批量生成"""
        results = []
        for seed in tqdm(seeds, desc="Generating"):
            for attempt in range(max_retries):
                try:
                    result = self.generate_instruction(
                        system_prompt, few_shot_prompt, seed["instruction"]
                    )
                    result["seed_id"] = seed.get("id")
                    results.append(result)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed after {max_retries} attempts: {e}")
                    continue
        return results
```

### 6.2 Anthropic API

```python
# Anthropic API 数据合成
from anthropic import Anthropic
import os

class AnthropicDataSynthesizer:
    def __init__(self, api_key=None, model="claude-sonnet-4-20250514"):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    def generate_instruction(self, system_prompt, seed_task, temperature=0.9):
        """生成单条指令数据"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"基于以下种子任务生成新的指令数据:\n\n{seed_task}"}
            ],
            temperature=temperature,
        )
        return json.loads(response.content[0].text)
```

### 6.3 开源模型

```python
# 开源模型数据合成（以Qwen为例）
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class OpenSourceSynthesizer:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    def generate(self, prompt, temperature=0.9, max_new_tokens=512):
        """使用开源模型生成"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=max_new_tokens
        )
        
        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response
    
    def batch_generate(self, prompts, temperature=0.9):
        """批量生成"""
        results = []
        for prompt in tqdm(prompts):
            result = self.generate(prompt, temperature=temperature)
            results.append(result)
        return results
```

---

## 总结

本节课我们系统学习了指令数据合成与多样性增强的核心技术：

1. **Self-Instruct 方法**：通过种子任务池出发，经由多轮生成-过滤-评分的迭代循环，逐步构建大规模指令数据集。多样性增强策略确保数据集覆盖多种任务类型和主题领域。

2. **数据增强技术**：复述保持语义改写表达，任务分解将复杂任务拆解为简单子任务，负采样生成对比性样本提升模型判断能力。

3. **提示工程**：系统提示定义角色和规则，小样本示例提供生成示范，Temperature/Top-p 参数控制多样性。

4. **多样性指标**：通过任务类型分布、主题覆盖度、响应长度分布等维度量化评估数据集的多样性。

5. **质量过滤**：结合分类器自动过滤与人在环验证，确保合成数据的整体质量。

6. **工具使用**：支持 OpenAI API、Anthropic API 以及 Hugging Face 开源模型，适配不同资源条件和隐私需求。

这些技术共同构成了现代微调数据工程的核心能力，是构建高质量微调数据集的必经之路。

---

## 延伸阅读

- Taori et al. "Self-Instruct: Aligning Language Models with Self-Generated Instructions" (2023)
- Wang et al. "Super-NaturalInstructions: Generalization via Declarative Instructions" (2022)
- Honovich et al. "UNICON: Universal Consistency via Enhanced In-Context Learning" (2023)
- Ding et al. "Making Reinforcement Learning Fair: Revisiting Reward Specification" (相关内容：数据偏向问题)

---

## 复习题

1. 描述 Self-Instruct 方法的核心流程，并说明种子任务池的多样性对最终数据集质量的影响。

2. 对比三种数据增强技术（复述、任务分解、负采样）的适用场景和局限性。

3. 设计一个用于生成数学推理任务数据的系统提示，说明如何保证生成任务的质量和多样性。

4. 解释为什么仅使用准确率指标不足以评估合成数据集的质量，还需要哪些多样性指标？

5. 讨论人在环验证在数据质量控制中的作用，以及如何在效率和覆盖率之间取得平衡。