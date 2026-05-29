# 8.3 DPO偏好数据准备与训练

## 课程概述

本课程介绍DPO（直接偏好优化）的偏好数据准备与训练实战。DPO通过偏好对（chosen/rejected）直接优化语言模型，无需显式奖励模型。课程涵盖偏好数据来源与格式、数据预处理、质量控制，以及DPOTrainer的完整训练流程与常见问题处理。

## 学习目标

- 掌握DPO偏好数据的标准格式与结构
- 熟悉三类偏好数据来源及其采集方法
- 理解DPO数据预处理的关键步骤
- 掌握TRL库DPOTrainer的完整训练配置
- 了解DPO训练中的常见问题及解决方案

## 前置知识

- 理解DPO的基本原理（参考8.2节）
- 熟悉Python和深度学习基础
- 了解大语言模型微调基本流程

---

## 1. DPO偏好数据格式

### 1.1 标准格式定义

DPO使用的偏好数据是三元组格式：`{prompt, chosen, rejected}`：

```python
# DPO偏好数据标准格式
dpo_example = {
    "prompt": "请解释什么是量子纠缠。",
    "chosen": "量子纠缠是量子力学中的一种现象，当两个或多个粒子相互作用后...",
    "rejected": "量子纠缠就是两个粒子连在一起。"
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| prompt | string | 输入提示/问题 |
| chosen | string | 人类偏好的回答（获胜者） |
| rejected | string | 人类不偏好的回答（失败者） |

### 1.2 数据格式变体

```python
# 变体1：包含多轮对话历史
dpo_with_history = {
    "prompt": [
        {"role": "user", "content": "什么是量子计算？"},
        {"role": "assistant", "content": "量子计算是一种使用量子力学原理的计算方式..."},
        {"role": "user", "content": "它和传统计算有什么区别？"}
    ],
    "chosen": "量子计算与传统计算的主要区别在于...",
    "rejected": "区别很大。"
}

# 变体2：包含元数据
dpo_with_metadata = {
    "prompt": "解释深度学习的工作原理",
    "chosen": "深度学习通过多层神经网络学习数据的层次化表示...",
    "rejected": "深度学习就是很深的机器学习",
    "metadata": {
        "annotator_id": "expert_001",
        "timestamp": "2024-01-15",
        "confidence": 0.95
    }
}
```

### 1.3 格式验证函数

```python
def validate_dpo_format(example):
    required_keys = {"prompt", "chosen", "rejected"}
    if not required_keys.issubset(example.keys()):
        return False, f"Missing keys. Expected: {required_keys}"
    
    if not example["chosen"].strip():
        return False, "chosen response is empty"
    
    if not example["rejected"].strip():
        return False, "rejected response is empty"
    
    if example["chosen"] == example["rejected"]:
        return False, "chosen and rejected are identical"
    
    return True, "valid"

# 使用示例
example = {"prompt": "问题", "chosen": "好答案", "rejected": "差答案"}
is_valid, msg = validate_dpo_format(example)
print(f"Valid: {is_valid}, Message: {msg}")
```

---

## 2. 偏好数据来源

### 2.1 人类标注数据集

**主要公开数据集**：

| 数据集 | 规模 | 特点 |
|--------|------|------|
| HH-RLHF | ~160K | Anthropic发布，包含helpful和harmless两个子集 |
| SHP | ~18K | 斯坦福发布，基于Reddit社区投票偏好 |
| TL;DR | ~120K | Reddit摘要偏好数据 |
| PKU-Alignment | ~55K | 中文偏好数据集 |

**HH-RLHF数据集结构示例**：

```python
# HH-RLHF数据格式
hh_example = {
    "chosen": "我理解你的困惑。这确实是一个复杂的话题，让我来解释...",
    "rejected": "这不是什么大不了的。",
    "human": "A conversation asking about a sensitive topic"
}
```

**数据获取**：

```python
from datasets import load_dataset

# 加载HH-RLHF数据集
hh_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
print(f"数据集大小: {len(hh_dataset)}")

# 查看数据格式
print(hh_dataset[0])
# {'chosen': '...', 'rejected': '...', 'human': '...'}
```

### 2.2 合成数据生成

使用强模型生成候选回答，再通过规则或弱模型选择构造偏好对：

```python
def generate_synthetic_preferences(prompts, strong_model, weak_model, num_candidates=4):
    """
    使用模型生成合成偏好数据
    策略：强模型生成多个候选，弱模型选择或排序
    """
    synthetic_data = []
    
    for prompt in prompts:
        # 1. 用强模型生成多个候选回答
        candidates = strong_model.generate(prompt, num_return=num_candidates)
        
        # 2. 用弱模型对候选排序
        rankings = weak_model.rank(prompt, candidates)
        
        # 3. 构建偏好对：最高分vs最低分
        sorted_by_score = [c for _, c in sorted(zip(rankings, candidates), key=lambda x: x[0])]
        
        synthetic_data.append({
            "prompt": prompt,
            "chosen": sorted_by_score[-1],  # 最好的
            "rejected": sorted_by_score[0]   # 最差的
        })
    
    return synthetic_data
```

**温度采样生成候选**：

```python
def generate_candidates_with_temperature(model, prompt, num_candidates=4, temperature=0.8):
    """使用不同温度采样生成多样化的候选回答"""
    candidates = []
    for temp in np.linspace(0.5, 1.5, num_candidates):
        response = model.generate(
            prompt=prompt,
            temperature=temp,
            top_p=0.95
        )
        candidates.append(response)
    return candidates
```

### 2.3 RLAIF偏好标注

利用AI反馈生成偏好标签，成本低但需注意质量控制：

```python
class RLAIFAnnotator:
    def __init__(self, feedback_model):
        self.model = feedback_model
    
    def annotate_preference(self, prompt, response_a, response_b):
        """让AI模型判断两个回答的偏好"""
        evaluation_prompt = f"""
请判断以下两个AI回复的质量优劣。

用户问题：{prompt}

回复A：{response_a}

回复B：{response_b}

请从以下维度评估：
1. 回答准确性
2. 回答完整性
3. 语言清晰度
4. 有用性

直接给出判断结果：回复A更优 / 回复B更优 / 两者质量相近
理由：[简要说明]
"""
        
        result = self.model.generate(evaluation_prompt)
        
        # 解析结果
        if "A更优" in result or "回复A更优" in result:
            return {"chosen": response_a, "rejected": response_b}
        elif "B更优" in result or "回复B更优" in result:
            return {"chosen": response_b, "rejected": response_a}
        else:
            return None  # 两者相近，丢弃
    
    def batch_annotate(self, pairs):
        """批量标注偏好对"""
        preferences = []
        for prompt, resp_a, resp_b in pairs:
            result = self.annotate_preference(prompt, resp_a, resp_b)
            if result:
                preferences.append(result)
        return preferences
```

---

## 3. 数据预处理

### 3.1 格式转换为DPO格式

```python
def convert_to_dpo_format(raw_data, input_format="rlhf"):
    """
    将不同来源的数据转换为DPO格式
    
    支持格式：
    - "rlhf": 标准RLHF格式 {chosen, rejected}
    - "ranked": 排序格式 {prompt, responses: [r1, r2, ...]}
    - "score": 带分数格式 {prompt, response, score}
    """
    dpo_data = []
    
    if input_format == "rlhf":
        for item in raw_data:
            dpo_data.append({
                "prompt": item.get("prompt", ""),
                "chosen": item["chosen"],
                "rejected": item["rejected"]
            })
    
    elif input_format == "ranked":
        for item in raw_data:
            responses = item["responses"]
            # 排序后选取最好和最差的
            sorted_responses = sorted(responses)
            dpo_data.append({
                "prompt": item["prompt"],
                "chosen": sorted_responses[-1],
                "rejected": sorted_responses[0]
            })
    
    elif input_format == "score":
        for item in raw_data:
            dpo_data.append({
                "prompt": item["prompt"],
                "chosen": item["response"],
                "rejected": item.get("rejected_response", item["response"])  # 占位
            })
    
    return dpo_data
```

### 3.2 质量过滤

```python
def filter_preference_pairs(examples, min_length=10, max_length_ratio=5.0):
    """
    过滤低质量的偏好对
    
    过滤条件：
    1. 回答长度过短
    2. chosen和rejected长度差异过大（长度偏差）
    3. 回答内容重复
    """
    filtered = []
    
    for example in examples:
        chosen = example["chosen"]
        rejected = example["rejected"]
        
        # 长度过滤
        if len(chosen.split()) < min_length or len(rejected.split()) < min_length:
            continue
        
        # 长度比例过滤（避免太长/太短的偏差）
        len_ratio = max(len(chosen), len(rejected)) / (min(len(chosen), len(rejected)) + 1)
        if len_ratio > max_length_ratio:
            continue
        
        # 内容重复检测
        if is_repetitive(chosen) or is_repetitive(rejected):
            continue
        
        filtered.append(example)
    
    return filtered

def is_repetitive(text, n_gram_threshold=3):
    """检测文本是否重复（简单的n-gram检测）"""
    words = text.split()
    if len(words) < n_gram_threshold * 2:
        return False
    
    for n in range(2, n_gram_threshold + 1):
        n_grams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if len(n_grams) != len(set(n_grams)):
            return True
    return False
```

### 3.3 响应长度平衡

```python
def balance_by_length(examples, max_ratio=3.0, max_examples=None):
    """
    长度平衡：限制chosen/rejected长度比例，避免模型学习到长度偏好
    
    问题：如果偏好数据中chosen总是比rejected长，
    DPO可能会学习到"更长=更好"而非"质量更高"
    """
    balanced = []
    
    for example in examples:
        chosen_len = len(example["chosen"].split())
        rejected_len = len(example["rejected"].split())
        
        ratio = max(chosen_len, rejected_len) / (min(chosen_len, rejected_len) + 1)
        if ratio <= max_ratio:
            balanced.append(example)
    
    if max_examples and len(balanced) > max_examples:
        balanced = balanced[:max_examples]
    
    return balanced

# 使用示例
print(f"平衡前: {len(examples)} 条")
balanced_examples = balance_by_length(examples, max_ratio=3.0)
print(f"平衡后: {len(balanced_examples)} 条")
```

### 3.4 处理模糊或平局偏好

```python
def handle_tied_preferences(examples, similarity_threshold=0.95):
    """
    处理模糊或相近的偏好对
    
    策略：
    1. 相似度超过阈值时丢弃（太相似无法判断）
    2. 标记不确定的偏好用于标签平滑
    """
    filtered = []
    uncertain = []
    
    for example in examples:
        chosen = example["chosen"]
        rejected = example["rejected"]
        
        # 计算文本相似度
        similarity = compute_text_similarity(chosen, rejected)
        
        if similarity > similarity_threshold:
            uncertain.append({
                **example,
                "uncertainty": similarity
            })
        else:
            filtered.append(example)
    
    return filtered, uncertain

def compute_text_similarity(text1, text2):
    """简单的词重叠相似度计算"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0
```

---

## 4. DPO训练配置

### 4.1 关键超参数

```python
# DPO训练配置示例
dpo_config = {
    # KL惩罚系数（beta）
    # 控制模型与参考模型的距离
    # 较小值（0.1-0.2）：允许更大偏离，学习更激进
    # 较大值（0.3-0.5）：保持更接近参考模型，更保守
    "beta": 0.3,
    
    # 标签平滑（soft preferences）
    # 将硬标签转换为软标签，提高泛化
    "label_smoothing": 0.1,
    
    # 梯度裁剪
    "max_grad_norm": 1.0,
    
    # 学习率调度
    "learning_rate": 1e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    
    # 训练批次
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 3,
    
    # 其他
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
}
```

### 4.2 beta参数详解

```python
def explain_beta_parameter():
    """
    beta (KL penalty coefficient) 对训练的影响
    """
    explanations = {
        "beta_too_low": """
        beta过低（如0.01）的问题：
        - 模型可能过度偏离参考模型
        - 容易产生幻觉或有害输出
        - 训练初期可能不稳定
        """,
        
        "beta_optimal": """
        适中的beta（0.1-0.3）：
        - 在保持与参考模型接近的同时学习新偏好
        - 平衡安全性和对齐质量
        - 大多数场景推荐使用
        """,
        
        "beta_too_high": """
        beta过高（如0.5+）：
        - 模型几乎不学习，输出接近参考模型
        - DPO优势消失，退化为近似SFT
        - 仅适用于需要保守优化的场景
        """
    }
    
    return explanations

# 不同beta值的训练效果对比
beta_comparison = {
    "beta=0.1": {"learning_aggressiveness": "高", "output_conservatism": "低", "use_case": "快速对齐"},
    "beta=0.3": {"learning_aggressiveness": "中", "output_conservatism": "中", "use_case": "通用推荐"},
    "beta=0.5": {"learning_aggressiveness": "低", "output_conservatism": "高", "use_case": "安全敏感场景"}
}
```

### 4.3 标签平滑（Soft Preferences）

```python
def apply_label_smoothing(labels, smoothing=0.1):
    """
    标签平滑：将硬标签转换为软标签
    
    原始：chosen=1, rejected=0
    平滑后：chosen=1-smoothing, rejected=smoothing
    
    效果：避免模型过度自信，提高泛化能力
    """
    smoothed = {
        "chosen": 1.0 - smoothing,
        "rejected": smoothing
    }
    return smoothed

# 在DPO损失中集成标签平滑
def dpo_loss_with_smoothing(logits, labels, smoothing=0.1):
    """
    软标签版本的DPO损失
    """
    # 原始：logits[:, 0] - logits[:, 1]  # chosen - rejected
    # 平滑：考虑软标签权重
    loss = -labels["chosen_weight"] * logits[:, 0] - labels["rejected_weight"] * logits[:, 1]
    return loss.mean()
```

---

## 5. 使用TRL的DPOTrainer训练

### 5.1 完整训练脚本

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer
from datasets import load_dataset
import torch

def train_dpo_model(
    model_name="meta-llama/Llama-2-7b-hf",
    reference_model_name=None,
    output_dir="./dpo_model_output",
    dataset_path="Anthropic/hh-rlhf",
):
    """
    使用TRL的DPOTrainer进行DPO训练
    """
    # 1. 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 参考模型设置
    if reference_model_name is None:
        reference_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        reference_model = AutoModelForCausalLM.from_pretrained(
            reference_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # 3. 加载数据集
    dataset = load_dataset(dataset_path, split="train")
    
    # 4. 数据预处理
    def prepare_dataset(example):
        return {
            "prompt": example.get("human", ""),
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }
    
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    
    # 5. 训练参数配置
    training_args = {
        "output_dir": output_dir,
        "beta": 0.3,  # KL惩罚系数
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 3,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "save_steps": 500,
        "max_grad_norm": 1.0,
        "fp16": True,
        "report_to": "tensorboard",
    }
    
    # 6. 初始化DPOTrainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=reference_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # 7. 开始训练
    dpo_trainer.train()
    
    # 8. 保存模型
    dpo_trainer.save_model(f"{output_dir}/final_model")
    
    return dpo_trainer
```

### 5.2 数据Collate函数

```python
from torch.utils.data import DataLoader

def dpo_data_collator(batch, tokenizer, max_length=512):
    """
    DPO数据整理器：处理批量偏好数据
    """
    prompts = [item["prompt"] for item in batch]
    chosen = [item["chosen"] for item in batch]
    rejected = [item["rejected"] for item in batch]
    
    # Tokenization
    chosen_enc = tokenizer(
        prompts,
        chosen,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    rejected_enc = tokenizer(
        prompts,
        rejected,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
    }
```

### 5.3 训练监控与评估

```python
class DPOEvaluationMonitor:
    def __init__(self, trainer, eval_dataset):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
    
    def compute_reward_margin(self, model, dataset, num_samples=100):
        """
        计算偏好 margin：chosen - rejected 的奖励差异
        一个好的模型应该对这个差值有正的贡献
        """
        model.eval()
        total_margin = 0
        
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            with torch.no_grad():
                chosen_logits = model(**tokenizer(prompt, chosen))
                rejected_logits = model(**tokenizer(prompt, rejected))
                
                margin = chosen_logits.mean() - rejected_logits.mean()
                total_margin += margin.item()
        
        return total_margin / num_samples
    
    def log_training_metrics(self, step):
        """记录训练指标"""
        metrics = {
            "step": step,
            "learning_rate": self.trainer.get_learning_rate(),
            "reward_margin": self.compute_reward_margin(
                self.trainer.model, 
                self.trainer.train_dataset
            ),
        }
        
        self.trainer.log(metrics)
```

### 5.4 完整评估流程

```python
def evaluate_dpo_model(model, tokenizer, eval_dataset):
    """
    DPO模型评估
    """
    model.eval()
    
    results = {
        "accuracy": 0,
        "avg_reward_margin": 0,
        "num_samples": 0
    }
    
    correct = 0
    total_margin = 0
    
    for example in eval_dataset:
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        
        with torch.no_grad():
            # 计算每个样本的偏好正确率
            chosen_score = get_model_score(model, tokenizer, prompt, chosen)
            rejected_score = get_model_score(model, tokenizer, prompt, rejected)
            
            if chosen_score > rejected_score:
                correct += 1
            
            total_margin += chosen_score - rejected_score
        
        results["num_samples"] += 1
    
    results["accuracy"] = correct / results["num_samples"]
    results["avg_reward_margin"] = total_margin / results["num_samples"]
    
    return results

def get_model_score(model, tokenizer, prompt, response):
    """获取模型对 (prompt, response) 的评分"""
    inputs = tokenizer(prompt, response, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.logits.mean().item()
```

---

## 6. 常见问题与解决方案

### 6.1 长度偏好问题（Reward Hacking）

```python
"""
问题：DPO模型学会通过生成更长但质量不高的回答来获得更高分数
原因：偏好数据中chosen往往比rejected长，模型学到"长=好"

解决方案：
1. 长度平衡过滤
2. 在损失函数中加入长度惩罚
3. 使用响应长度归一化
"""

def length_penalized_dpo_loss(logits, labels, responses, penalty=0.1):
    """
    带长度惩罚的DPO损失
    """
    # 基础DPO损失
    base_loss = -labels["chosen"] * logits[:, 0] - labels["rejected"] * logits[:, 1]
    
    # 长度惩罚
    length_penalty = penalty * (len(responses["chosen"]) - len(responses["rejected"]))
    
    return (base_loss + length_penalty).mean()
```

### 6.2 模式崩塌（Mode Collapse）

```python
"""
问题：模型输出变得单一，丧失多样性
原因：过度优化导致模型只生成"安全"回答

解决方案：
1. 降低beta值，允许更多探索
2. 增加数据多样性
3. 使用对比学习增强多样性信号
"""

def prevent_mode_collapse(dataset, diversity_threshold=0.3):
    """
    通过多样性过滤防止模式崩塌
    """
    diverse_data = []
    
    for i, example in enumerate(dataset):
        if i == 0:
            diverse_data.append(example)
            continue
        
        # 检查与已有样本的相似度
        max_similarity = max(
            compute_text_similarity(example["chosen"], d["chosen"])
            for d in diverse_data
        )
        
        if max_similarity < diversity_threshold:
            diverse_data.append(example)
    
    return diverse_data
```

### 6.3 参考模型过期（Reference Model Staleness）

```python
"""
问题：随着训练进行，参考模型逐渐过时
原因：参考模型固定不变，但策略模型持续更新

解决方案：
1. 定期更新参考模型（每N步）
2. 使用指数移动平均（EMA）更新参考模型
3. 使用迭代式DPO（IPO）方法
"""

class AdaptiveReferenceModel:
    def __init__(self, model, update_interval=500, ema_factor=0.99):
        self.model = model
        self.reference_model = copy.deepcopy(model)
        self.update_interval = update_interval
        self.ema_factor = ema_factor
        self.step = 0
    
    def update_reference(self, current_model):
        """使用EMA更新参考模型"""
        self.step += 1
        
        if self.step % self.update_interval == 0:
            # EMA更新
            for p, c in zip(self.reference_model.parameters(), current_model.parameters()):
                p.data = self.ema_factor * p.data + (1 - self.ema_factor) * c.data
    
    def get_ref_model(self):
        return self.reference_model
```

---

## 7. 总结

1. **DPO数据格式**：标准三元组`{prompt, chosen, rejected}`，支持多种变体格式

2. **数据来源**：人类标注（HH-RLHF、SHP）、合成生成、RLAIF反馈，各有优劣

3. **数据预处理**：格式转换、质量过滤、长度平衡、处理平局偏好是关键步骤

4. **训练配置**：beta参数控制KL惩罚（0.1-0.5），标签平滑可提高泛化

5. **DPOTrainer**：TRL提供完整的DPO训练实现，支持灵活配置

6. **常见问题**：长度偏差通过平衡过滤解决，模式崩塌需多样性增强，参考模型过期可用EMA更新

---

## 延伸阅读

1. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** - Rafailov et al., 2023
2. **The N+1 Problem of DPO: Theoretical Analysis and Practical Solutions** - Chen et al., 2024
3. **HH-RLHF: The Anthropic HH-RLHF Dataset** - Anthropic
4. **TRL Library: Transformer Reinforcement Learning** - von Platen et al., 2024

---

## 复习题

1. **分析DPO偏好数据中chosen/rejected的长度偏差问题，说明这种偏差如何影响DPO训练，并提出至少两种缓解策略。**

2. **对比三种偏好数据来源（人类标注、合成生成、RLAIF）的优缺点，讨论在什么场景下应选择哪种数据来源。**

3. **解释DPO训练中beta参数的作用机制，分析beta值过大或过小分别会导致什么问题。**

4. **讨论DPO训练中可能出现的模式崩塌（Mode Collapse）问题，分析其成因并提出可行的解决方案。**