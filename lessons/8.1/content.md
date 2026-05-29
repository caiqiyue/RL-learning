# 8.1 偏好数据构建：人类反馈与排序标注

## 课程概述

本节课聚焦 RLHF（基于人类反馈的强化学习）核心环节——偏好数据的构建方法。与监督式微调（SFT）直接指定"正确答案"不同，RLHF 通过收集人类对模型输出的偏好信号来训练奖励模型，使模型能够学习超越简单模仿的复杂偏好行为。本节课将系统讲解偏好数据的本质、格式规范、人工标注流程、质量控制方法以及主流数据集资源。

## 学习目标

- 理解 SFT 与 RLHF 在数据需求层面的本质差异
- 掌握偏好数据的三种主要格式及其适用场景
- 学会设计高效的偏好标注界面和标注流程
- 了解 ELO 评分系统在响应排序中的应用
- 熟悉主流偏好数据集的结构与规模

## 前置知识

- 了解 RLHF 的基本原理与三阶段流程（预训练、SFT、RLHF）
- 熟悉大模型概率生成的基本概念
- 具备 Python 编程基础

---

## 1. 为什么 RLHF 需要偏好数据

### 1.1 SFT 的局限性

监督式微调（SFT）的核心是让模型模仿"正确答案"。给定输入 prompt，标注者直接提供期望的输出 response，模型学习的是一种确定性的映射关系：

```
SFT 数据格式: (prompt, response_groundtruth)
训练目标: 最大化 P(response_groundtruth | prompt)
```

这种方式的本质是**记忆**——模型学到的是在给定情境下应该如何回答，而非理解什么使回答真正有价值。

SFT 的根本局限体现在：

| 局限性 | 具体表现 |
|-------|---------|
| 正确答案不唯一 | "如何解释量子力学"有无数种好的表达方式 |
| 标注者风格偏差 | 标注者的个人风格会被模型强化 |
| 无法学习模糊偏好 | 什么算"有帮助"难以用单一答案定义 |
| 泛化能力弱 | 遇到未见过的问题类型可能完全失效 |

### 1.2 偏好数据的优势

RLHF 采用的偏好数据不告诉模型"怎么说"，而是告诉模型"哪个更好"：

```
偏好数据格式: (prompt, response_A, response_B, preference_label)
训练目标: 学会区分好坏响应的相对关系
```

这种范式带来三大核心优势：

**1. 学习超越模仿的能力**
模型不仅学到如何回答，更学到什么使回答有价值。即使是全新的问题类型，模型也能凭借学到的偏好判断生成高质量内容。

**2. 处理主观评价任务**
对于"是否有帮助"、"是否安全"这类主观指标，偏好比较比正确答案更容易获取标注，且更具一致性。

**3. 避免标注者风格过拟合**
模型学习的是人类偏好的共同特征，而非某位标注者的个人风格。

---

## 2. 偏好数据的格式规范

### 2.1 成对比较格式（Pairwise Comparison）

成对比较是最广泛使用的偏好数据格式，每条样本包含一个 prompt 和两个候选响应，以及人类标注的偏好标签：

```json
{
    "prompt": "请解释什么是机器学习",
    "response_chosen": "机器学习是人工智能的一个分支，让计算机从数据中自动发现模式并做出预测，无需明确编程规则。",
    "response_rejected": "机器学习就是让机器学习",
    "preference_label": 1,
    "annotator_id": "anno_001",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

| 字段 | 说明 |
|-----|------|
| prompt | 输入提示 |
| response_chosen | 人类偏好的响应 |
| response_rejected | 人类不偏好的响应 |
| preference_label | 偏好标记（1 表示 chosen 更好） |
| annotator_id | 标注者 ID（用于一致性分析） |

成对比较的优势在于**标注简单、信息明确**——标注者只需判断"A 比 B 好"或"B 比 A 好"。其局限是每次比较只利用了部分偏好信息，当有多个响应时需要 O(n²) 次比较。

### 2.2 排序格式（Ranking）

排序格式允许一次标注多个响应的相对顺序，适用于评估响应质量差异较大的场景：

```json
{
    "prompt": "写一段关于中秋节的短文",
    "responses": [
        {"id": "A", "text": "中秋节快乐！"},
        {"id": "B", "text": "中秋节是中国传统节日，在农历八月十五，人们赏月吃月饼。"},
        {"id": "C", "text": "月圆之夜，家人团聚，共品香茗，畅谈古今，此乃中秋之趣也。"}
    ],
    "ranking": ["C", "B", "A"],
    "ranking_type": "full",
    "partial_order": null
}
```

**全排序 vs 部分排序**：

| 类型 | 说明 | 适用场景 |
|-----|------|---------|
| 全排序 | 所有响应都有明确先后 | 响应数量少（≤5）且质量差异清晰 |
| 部分排序 | 只保证部分约束（如 C>B>A，中间无约束） | 响应数量多或质量接近难以区分 |

排序格式可以转化为多条文对数据用于奖励模型训练：

```
原始排序: C > B > A
转化文对:
  - (prompt, C, B) → 标签: C 更好
  - (prompt, C, A) → 标签: C 更好
  - (prompt, B, A) → 标签: B 更好
```

### 2.3 评分格式（Scalar Rating）

评分格式直接让标注者给出 1-5 或 1-10 的绝对分数：

```json
{
    "prompt": "解释为什么天空是蓝色的",
    "response": "天空呈现蓝色是因为大气层对阳光的瑞利散射效应，短波长蓝光被散射到各个方向。",
    "rating": 4,
    "dimensions": {
        "helpfulness": 4,
        "honesty": 5,
        "harmlessness": 4
    }
}
```

评分格式的优势是**信息密度高**，一条样本包含多个维度的评分；局限是**标注一致性难保证**——不同标注者对"4分"的含义理解可能差异很大。

### 2.4 格式对比

| 格式 | 数据利用率 | 一致性 | 标注成本 | RLHF 主流程度 |
|-----|-----------|--------|---------|--------------|
| 成对比较 | ★★★☆☆ | ★★★★☆ | 中 | ★★★★★ 最常用 |
| 排序 | ★★★★☆ | ★★★☆☆ | 高 | ★★★☆☆ |
| 评分 | ★★★★★ | ★★☆☆☆ | 低 | ★☆☆☆☆ 少用 |

成对比较因其**标注一致性好、实现简单**成为 RLHF 实际应用中的首选格式。

---

## 3. 人工标注流程

### 3.1 标注界面设计

**Side-by-Side 并排对比界面**

最经典的偏好标注界面，将两个响应并排展示：

```
┌─────────────────────────────────────────────────────────────┐
│  标注任务 #2847                              标注者: 张三    │
├─────────────────────────────────────────────────────────────┤
│  Prompt: 如何在一个月内学会游泳？                          │
├──────────────────────────┬──────────────────────────────────┤
│       Response A         │         Response B               │
│                          │                                  │
│ 要学会游泳，首先要克服    │ 一个月学会游泳是完全可以实现的，  │
│ 对水的恐惧。建议从呼吸    │ 关键在于系统的训练计划。第一周    │
│ 练习开始，逐步过渡到基    │ 每天在泳池练习30分钟，重点是克    │
│ 本的踢腿和划水动作。      │ 服对水的恐惧，学习水下呼吸。第    │
│                          │ 二周开始学习漂浮和踢腿，每天40    │
│                          │ 分钟...                          │
├──────────────────────────┴──────────────────────────────────┤
│  请选择您更偏好的响应：                                     │
│                                                             │
│  ○ A 更好    ○ B 更好    ○ 两者都不错    ○ 两者都差劲      │
│                                                             │
│  [理由（可选）] ________________________________________   │
└─────────────────────────────────────────────────────────────┘
```

**Ranking 排序界面**

当需要排序多个响应时，采用列表式排序界面：

```
┌─────────────────────────────────────────────────────────────┐
│  标注任务 #3921                              标注者: 李四  │
├─────────────────────────────────────────────────────────────┤
│  Prompt: 写一首关于秋天的七言绝句                            │
├─────────────────────────────────────────────────────────────┤
│  请将以下响应按质量从高到低排序（拖拽调整）：                │
│                                                             │
│  ① [ 秋风送爽桂飘香，月照庭院人团圆 ]     📌 最优          │
│  ② [ 叶落知秋意渐浓，霜华初现雁南飞 ]     📌 良好          │
│  ③ [ 秋天来了，天很凉快 ]                 📌 一般          │
│  ④ [ 秋天 ]                               📌 差            │
│                                                             │
│  [备注] ________________________________________________   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 标注者筛选与培训

**筛选标准**：

| 标准 | 要求 | 评估方法 |
|-----|------|---------|
| 领域知识 | 相关领域基础概念理解 | 背景问卷测试 |
| 语言能力 | 母语级语言表达与理解 | 语法改错测试 |
| 一致性 | 相似案例判断一致 | 标注小样本集 |
| 时效性 | 承诺可投入的时间量 | 时间记录跟踪 |

**培训流程**：

1. **理论学习**：阅读标注规范文档，理解三大质量维度
2. **示例解析**：逐条学习 10-20 个标准答案及其理由
3. **模拟标注**：使用标注工具完成 50 个模拟任务
4. **结果校正**：对比标准答案，分析错误原因
5. **考核通过**：模拟任务准确率 ≥ 85% 方可正式标注

### 3.3 ELO 评分系统

对于大规模响应排序，固定 prompt 内的比较顺序可能导致系统性偏差。ELO 评分系统借鉴国际象棋评级机制，通过迭代比较动态估计每个响应的质量分数：

```python
class ELORankingSystem:
    def __init__(self, initial_rating=1500, k_factor=32):
        self.ratings = {}
        self.k_factor = k_factor
        self.initial_rating = initial_rating
    
    def get_rating(self, response_id):
        if response_id not in self.ratings:
            self.ratings[response_id] = self.initial_rating
        return self.ratings[response_id]
    
    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update(self, winner_id, loser_id):
        r_winner = self.get_rating(winner_id)
        r_loser = self.get_rating(loser_id)
        
        e_winner = self.expected_score(r_winner, r_loser)
        e_loser = self.expected_score(r_loser, r_winner)
        
        self.ratings[winner_id] = r_winner + self.k_factor * (1 - e_winner)
        self.ratings[loser_id] = r_loser + self.k_factor * (0 - e_loser)
    
    def compare(self, id_a, id_b):
        rating_a = self.get_rating(id_a)
        rating_b = self.get_rating(id_b)
        prob_a = self.expected_score(rating_a, rating_b)
        return prob_a
```

**ELO 的优势**：
- 自适应调整：多次比较后评分收敛到稳定值
- 处理不完整比较：无需每次都对比所有响应对
- 检测异常响应：ELO 分数明显偏离的响应可能是标注异常

---

## 4. 优质偏好标签的标准

### 4.1 质量评估的三维框架

Anthropic 提出的 HHHL（Hhhonesty、Helpfulness、Harmlessness）框架是事实上的行业标准：

| 维度 | 定义 | 正面示例 | 负面示例 |
|-----|------|---------|---------|
| 有帮助（Helpfulness） | 响应是否真正解决用户问题 | 提供具体可行的步骤 | 模糊笼统的废话 |
| 诚实（Honesty） | 响应是否准确、可信 | 承认不确定性 | 一本正经地胡说八道 |
| 无害（Harmlessness） | 响应是否避免潜在风险 | 拒绝危险请求时温和引导 | 给出危险操作步骤 |

### 4.2 标注指南示例

为保证标注一致性，需提供详细的带有示例的标注指南：

**场景 1：知识问答**

| 情况 | Response A | Response B | 正确偏好 |
|-----|-----------|-----------|---------|
| A 准确 + 完整，B 准确 + 简略 | 量子力学是描述微观粒子行为的物理学理论... | 量子力学是一种物理理论 | A |
| A 准确 + 诚实，B 看似准确 + 错误 | 目前没有科学证据表明... | 吸烟有益健康因为... | A |
| A 坦诚 + 有帮助，B 过度自信 | 我不确定这个问题的答案，建议查阅... | 这件事很简单，只需要... | A |

**场景 2：创意写作**

| 情况 | Response A | Response B | 正确偏好 |
|-----|-----------|-----------|---------|
| A 创意新颖，B 平淡但合格 | 秋夜，月光如水，桂花香里... | 秋天的晚上，月亮很亮 | A（如果创意是目标） |
| A 创意跑题，B 保守正确 | 春天的花朵在雪地上绽放... | 春天是四季之一，花开草长... | B（如果要求正确性） |

### 4.3 标注者间不一致的处理

当多位标注者对同一样本产生分歧时，采用以下策略：

**1. 多数投票（ Majority Voting）**
取多数标注者判断的结果作为最终标签。

**2. 置信度过滤（Confidence Filtering）**
计算标注一致性分数，过滤低置信度样本：

```python
def compute_annotation_confidence(labels):
    """
    labels: list of labels from multiple annotators
    返回: (majority_label, confidence_score)
    """
    from collections import Counter
    counts = Counter(labels)
    total = sum(counts.values())
    
    majority_count = max(counts.values())
    confidence = majority_count / total
    
    majority_label = counts.most_common(1)[0][0]
    return majority_label, confidence

# 过滤低置信度样本
HIGH_CONFIDENCE_THRESHOLD = 0.66

for item in dataset:
    _, confidence = compute_annotation_confidence(item["annotator_labels"])
    if confidence < HIGH_CONFIDENCE_THRESHOLD:
        item["discard_reason"] = "low_annotation_confidence"
```

**3. 专家仲裁（Expert Arbitration）**
对于高价值或高不确定性的样本，交由资深标注专家最终判断。

---

## 5. 数据规模与真实数据集

### 5.1 数据规模估算

根据 InstructGPT 论文，RLHF 所需的数据规模与任务复杂度相关：

| 模型规模 | 偏好对数量 | 说明 |
|---------|-----------|------|
| InstructGPT 小型 | ~5K pairs | 概念验证 |
| InstructGPT 中型 | ~67K pairs | 论文报告的主要规模 |
| InstructGPT 大型 | ~170K+ pairs | GPT-4 级别模型 |

**经验公式**（来自实际项目总结）：

```
推荐偏好对数量 ≈ 10 × 模型参数量（亿） × 任务复杂度系数
```

| 任务类型 | 复杂度系数 |
|---------|-----------|
| 通用对话 | 1.0 |
| 领域专家（医疗/法律） | 2.0-3.0 |
| 代码生成 | 1.5 |
| 创意写作 | 1.2 |

### 5.2 主流偏好数据集

**Anthropic HH-RLHF**

| 属性 | 值 |
|-----|---|
| 规模 | ~161K 偏好对 |
| 来源 | human-written AI assistant conversations |
| 任务类型 | 对话帮助性评估 |
| 特点 | 包含 harmlessness 和 helpfulness 维度的细粒度标注 |
| 获取方式 | HuggingFace: anthropic/hh-rlhf |

```json
{
    "prompt": "How can I pick a lock?",
    "chosen_response": "I'm sorry, but I can't help with that...",
    "rejected_response": "Here's how to pick a lock using a tension wrench..."
}
```

**OpenAI Summarization**

| 属性 | 值 |
|-----|---|
| 规模 | ~137K 偏好对 |
| 来源 | TL;DR summarizing Reddit posts |
| 任务类型 | 文本摘要质量评估 |
| 特点 | 包含多个摘要质量的评分维度 |
| 获取方式 | 请求访问或论文引用获取 |

**Stanford SHP (Stanford Human Preferences)**

| 属性 | 值 |
|-----|---|
| 规模 | ~64K 偏好对 |
| 来源 | Reddit 多个 subreddit |
| 任务类型 |  subreddit 内的回复偏好 |
| 特点 | 领域多样，覆盖 12+ 个 subreddit |
| 获取方式 | HuggingFace: stanfordnlp/SHP |

### 5.3 数据集规模对比

```
数据集规模对比（对数刻度）

HH-RLHF  ████████████████████████████████████  161K
Summarization  ███████████████████████████████  137K
SHP  █████████████████████  64K
自定义中型项目  ██████████  10K-30K
自定义小型实验  ████  1K-5K
```

---

## 总结

本节课我们系统学习了 RLHF 偏好数据构建的核心知识：

1. **SFT vs RLHF 的本质差异**：SFT 让模型模仿正确答案，RLHF 让模型学习区分好坏——后者赋予了模型超越模仿的泛化能力。

2. **三种偏好数据格式**：成对比较（最常用）、排序格式、评分格式，各有权衡。实际应用中成对比较因其一致性好、实现简单而占主导。

3. **人工标注流程要点**：Side-by-Side 界面设计、标注者筛选培训、ELO 评分系统处理响应排序，构成完整的标注pipeline。

4. **优质偏好标签的标准**：HHH 三维评估框架、详细的标注指南示例、以及标注者间不一致的处理策略。

5. **数据规模与数据集**：InstructGPT 使用约 67K 偏好对作为基准，HH-RLHF、OpenAI Summarization、SHP 是三个主流公开数据集。

---

## 延伸阅读

- Ouyang et al. "Training language models to follow instructions with human feedback" (InstructGPT, 2022)
- Bai et al. "Training a Helpful and Harmless Assistant with RLHF" (Anthropic HH-RLHF, 2022)
- Stiennon et al. "Learning to summarize with human feedback" (OpenAI Summarization, 2020)
- Ethayarajh et al. "The Stanford Human Preferences Dataset" (SHP, 2022)
- Lee et al. "Dataset Decomposition in RLHF" (偏好数据的结构化分析, 2023)

---

## 复习题

1. 解释为什么对于主观评价任务（如"是否有帮助"），偏好数据比 SFT 数据更有效。

2. 设计一个针对中文小说推荐场景的偏好标注界面，包含 prompt 设计、响应展示方式、偏好选择逻辑。

3. 对比成对比较与排序格式的优劣，分析在什么场景下应该选择排序格式。

4. 描述当标注者间出现严重分歧（如 Kappa < 0.4）时，应该如何诊断和解决。

5. 查阅 HH-RLHF 数据集，分析其数据分布特点，讨论如果要将 RLHF 应用于中文场景需要做哪些数据适配工作。