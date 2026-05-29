# 4.4 DPO实战：偏好数据准备与训练

## 课程概述

本课时是 DPO（Direct Preference Optimization）理论的实践篇。上一课时（4.3）我们已经深入理解了 DPO 的数学原理——如何将 RLHF 的两阶段（奖励模型 + 策略优化）合并为单一监督学习过程。本课时将聚焦于从数据准备到模型训练的全流程实战，包括：偏好数据的格式与来源、如何构建合成偏好数据集、使用 HuggingFace TRL 库训练 DPO 模型、以及训练过程中的关键超参数调试。

**学习目标**
- 掌握偏好数据的标准格式 `{prompt, chosen_response, rejected_response}`
- 理解两种偏好数据构建方式：人工标注 pipeline 与合成偏好生成
- 能够使用 TRL 的 `DPOTrainer` 实现端到端 DPO 训练
- 理解 Reference Model 的作用、内存占用以及为什么不能跳过
- 掌握关键超参数 β（KL 温度）、margin term、length normalization 的作用
- 能够评估 DPO 训练效果（held-out prompts 的 win rate）
- 识别并解决 DPO 训练中的常见问题：reference model drift、mode collapse、preference data noise

**前置知识**：DPO 数学原理（lesson 4.3）、PyTorch 基础、HuggingFace Transformers 基本使用

---

## 1. 偏好数据格式与来源

### 1.1 标准偏好数据格式

DPO 训练的数据是**三元组** `(prompt, chosen_response, rejected_response)`，表示人类对同一提示的两个候选回答的偏好关系：

```python
{
    "prompt": "请解释什么是大语言模型，以及它如何处理文本。",
    "chosen": "大语言模型（Large Language Model）是一种基于深度学习的自然语言处理模型...",
    "rejected": "LLM是一种AI模型。"
}
```

其中：
- `prompt`：输入提示，可以是问题、指令或对话上下文
- `chosen`：人类偏好的回答（标记为正样本）
- `rejected`：人类不偏好的回答（标记为负样本）

**重要约束**：对于同一个 `prompt`，`chosen` 和 `rejected` 必须是不同的回答，且人类的偏好判断应当具有一致性。

### 1.2 主流偏好数据集

| 数据集 | 规模 | 来源 | 典型用途 |
|--------|------|------|----------|
| Anthropic HH-RLHF | ~160K pairs | 人类标注 | 对话assistant训练 |
| OpenAI Summarization | ~290K pairs | 人类标注 | 摘要任务偏好 |
| Stanford SHP | ~90K pairs | Reddit点赞 | 多种指令类型 |
| LMSYS-Chat | ~30K pairs | 人类对战 | 对话偏好 |

### 1.3 Anthropic HH-RLHF 数据格式

HH-RLHF 是最广泛使用的 DPO 训练数据集之一，其格式如下：

```python
{
    "chosen": "Human: 你好\nAssistant: 你好！有什么我可以帮助你的吗？",
    "rejected": "Human: 你好\nAssistant: 你好",
    "human": "你好",
    "assistant": "你好！有什么我可以帮助你的吗？"
}
```

TRL 库提供了直接加载 HH-RLHF 数据集的接口：

```python
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf", split="train")
# dataset 包含 "chosen" 和 "rejected" 字段
```

---

## 2. 构建偏好数据集

### 2.1 人工标注 Pipeline

在真实产品环境中，构建高质量偏好数据的标准流程：

```
┌────────────────────────────────────────────────────────────┐
│              偏好数据标注 Pipeline                           │
│                                                            │
│  Step 1: 提示收集                                          │
│    → 从实际用户query中采样，覆盖目标场景分布                  │
│    → 数量：通常 5K～50K 不等                                 │
│                                                            │
│  Step 2: 候选回答生成                                       │
│    → 使用 SFT 模型或多个基座模型对每个提示生成多个回答       │
│    → 每个提示生成 2～8 个候选回答                           │
│                                                            │
│  Step 3: 人类偏好标注                                       │
│    → 标注者对同一提示的候选回答进行两两比较                  │
│    → 标注界面：显示 prompt + 回答A vs 回答B，选择偏好        │
│    → 每个 pair 可能由 3～5 名标注者独立标注                   │
│                                                            │
│  Step 4: 偏好聚合与质量过滤                                 │
│    → 使用 Bradley-Terry 模型或 Elo 排序聚合多标注者意见       │
│    → 过滤低一致性样本（标注者分歧 > 50%）                    │
│    → 最终输出：(prompt, chosen, rejected) 三元组             │
└────────────────────────────────────────────────────────────┘
```

**标注质量控制**：
- 标注者培训：先在标注指南上通过考核测试
- 交叉验证：10%～20% 的样本由两名标注者同时标注，计算一致性指标
- 不一致样本处理：分歧过大的样本要么删除，要么由高级标注者仲裁

### 2.2 合成偏好数据生成

当没有充足的人工标注预算时，可以使用**合成偏好数据**来启动 DPO 训练。常见方法：

#### 方法一：基于质量的启发式规则

```python
def create_synthetic_preference(model, prompt, temperature=0.8):
    """
    使用模型生成多个回答，根据简单启发式规则判断偏好
    
    启发式规则：
    - 回答长度适中（不太短也不太长）的更优
    - 包含更多专业术语的回答更优
    - 结构更完整的回答更优
    """
    responses = model.generate(prompt, n=4, temperature=temperature)
    
    scores = []
    for resp in responses:
        length_score = min(len(resp) / 200, 1.0)  # 长度适中
        structure_score = 1.0 if "\n" in resp else 0.5  # 有结构
        scores.append(length_score * 0.6 + structure_score * 0.4)
    
    # 排序并构建偏好对
    ranked = sorted(zip(responses, scores), key=lambda x: -x[1])
    return {
        "prompt": prompt,
        "chosen": ranked[0][0],
        "rejected": ranked[-1][0]
    }
```

#### 方法二：基于更弱模型的选择

使用一个较弱/较小的模型作为"裁判"，对强模型生成的候选回答打分：

```python
def create_preference_with_judge(strong_model, judge_model, prompt):
    """
    使用judge模型评判strong模型生成的候选回答
    judge可以是比strong model小但推理能力仍然可靠的模型
    """
    candidates = strong_model.generate(prompt, n=4, temperature=0.7)
    
    scores = []
    for resp in candidates:
        # 使用judge模型打分
        judgment = judge_model.evaluate(prompt, resp)  # 返回0-1分数
        scores.append(judgment)
    
    best_idx = scores.index(max(scores))
    worst_idx = scores.index(min(scores))
    
    return {
        "prompt": prompt,
        "chosen": candidates[best_idx],
        "rejected": candidates[worst_idx]
    }
```

#### 方法三：SPIN 自迭代偏好

参考 lesson 4.3 的 SPIN 方法，模型自我生成偏好数据：

```python
def generate_spin_preference(model, prompt):
    # 生成初始回答
    initial = model.generate(prompt, temperature=0.7)
    
    # 让模型自我批评并生成改进版
    critique = model.generate(
        f"请批评以下回答的不足：'{initial}'",
        temperature=0.5
    )
    improved = model.generate(
        f"基于以下批评改进你的回答：'{critique}'",
        temperature=0.7
    )
    
    return {
        "prompt": prompt,
        "chosen": improved,
        "rejected": initial
    }
```

### 2.3 数据质量检查清单

构建完偏好数据集后，建议进行以下质量检查：

```python
def validate_preference_dataset(dataset):
    issues = []
    
    for i, sample in enumerate(dataset):
        # 检查1：chosen 和 rejected 不能相同
        if sample["chosen"] == sample["rejected"]:
            issues.append(f"Sample {i}: chosen == rejected")
        
        # 检查2：回答不能为空
        if len(sample["chosen"].strip()) == 0 or len(sample["rejected"].strip()) == 0:
            issues.append(f"Sample {i}: empty response")
        
        # 检查3：prompt 不能为空
        if len(sample["prompt"].strip()) == 0:
            issues.append(f"Sample {i}: empty prompt")
        
        # 检查4：长度差异过大可能是标注错误
        len_diff = abs(len(sample["chosen"]) - len(sample["rejected"]))
        if len_diff > 2000:  # 差异超过2000字符可能是问题
            issues.append(f"Sample {i}: unusually large length difference ({len_diff})")
    
    return issues
```

---

## 3. Reference Model 的核心作用

### 3.1 为什么需要 Reference Model

DPO 的损失函数中，Reference Model 扮演着关键角色。在数学上，DPO 隐式地将奖励函数表示为策略与参考策略的概率比：

```
L_DPO = -E_{(x, y_w, y_l) ~ D}[log σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) - β * log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

Reference Model 的作用：
1. **提供基线分布**：使模型知道什么是"中性"的输出，避免极端化
2. **KL 散度的隐式约束**：通过对比当前策略与参考策略的输出比例，确保训练稳定性
3. **防止 reward hacking**：如果没有参考模型约束，模型可能生成在训练数据上得分高但实际质量差的内容

### 3.2 Reference Model 的内存占用

Reference Model 通常是 SFT 训练后的模型副本，与待训练的 Actor Model 参数量相同：

| 模型规模 | 参数量 | FP32 内存 | INT8 内存 | FP16 内存 |
|----------|--------|-----------|-----------|-----------|
| 7B | 70亿 | ~28 GB | ~10 GB | ~14 GB |
| 13B | 130亿 | ~52 GB | ~20 GB | ~26 GB |
| 70B | 700亿 | ~280 GB | ~100 GB | ~140 GB |

**实战建议**：
- 7B 模型：单卡 24GB 可运行（FP16）
- 13B 模型：需要多卡或量化（INT8/QLoRA）
- 70B 模型：必须使用 QLoRA 或多卡并行

### 3.3 Reference Model 的一致性问题

**问题**：训练过程中如果 Reference Model 不冻结，它的输出会与初始状态偏离，导致训练不稳定。

**解决方案**：TRL 默认将 Reference Model 设为冻结状态（`model_ref = reference_model` 不参与梯度更新）。确保：

```python
for param in reference_model.parameters():
    param.requires_grad = False
```

---

## 4. DPO 训练关键超参数

### 4.1 β（KL 温度参数）

β 是 DPO 最关键的超参数，控制着策略偏离参考模型的允许程度：

```python
# TRL DPOTrainer 中的 β 参数
trainer = DPOTrainer(
    model=model,
    ref_model=reference_model,
    beta=0.1,  # 默认值 0.1，范围通常 0.01～0.3
    ...
)
```

**β 的物理意义**：
- β → 0：模型几乎完全忽略参考模型，容易过拟合到偏好数据
- β → ∞：模型几乎不退化，输出与参考模型几乎一样，无法学习新偏好
- β 适中：允许适度偏离参考模型，同时保持基础能力

**β 选择指南**：

| β 范围 | 适用场景 | 效果 |
|--------|----------|------|
| 0.01～0.05 | 偏好数据质量高、多样性好 | 快速学习，但可能不稳定 |
| 0.1（默认） | 通用场景 | 平衡稳定性和学习速度 |
| 0.2～0.3 | 偏好数据有噪声、分布偏移 | 更稳定，但学习速度慢 |

### 4.2 Margin Term（边际项）

Margin term 是 DPO 损失函数的可选扩展，在比较 chosen 和 rejected 时引入边际：

```python
# 带 margin 的 DPO 损失
trainer = DPOTrainer(
    model=model,
    ref_model=reference_model,
    beta=0.1,
    margin_lambda=0.1,  # 边际权重
    ...
)
```

**margin 的作用**：强化对"明确偏好"的监督信号，弱化"边界模糊"样本的影响。

### 4.3 Length Normalization（长度归一化）

偏好数据中，chosen 和 rejected 的回答长度往往不同。如果不进行归一化，模型可能偏向生成特定长度的回答：

```python
trainer = DPOTrainer(
    model=model,
    ref_model=reference_model,
    beta=0.1,
    length_normalization=True,  # TRL 默认开启
    ...
)
```

**为什么需要长度归一化**：
- 训练数据中，chosen 平均长度可能大于 rejected
- 没有归一化时，模型可能学到"长回答更受偏好"的虚假关联
- 开启后，每个回答的概率计算会除以长度，消除长度偏差

---

## 5. 使用 TRL 实现 DPO 训练

### 5.1 完整训练脚本

以下是一个完整的 DPO 训练脚本（见 `code/train_dpo.py`）：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from datasets import load_dataset

def main():
    # 1. 加载模型和分词器
    model_name = "gpt2"  # 替换为你的模型路径
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 确保 pad token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # 2. 准备参考模型（冻结副本）
    reference_model = AutoModelForCausalLM.from_pretrained(model_name)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    # 3. 加载偏好数据
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    
    # 4. 数据预处理
    def preprocess_function(examples):
        return {
            "prompt": examples["human"],
            "chosen": examples["chosen"],
            "rejected": examples["rejected"]
        }
    
    dataset = dataset.map(preprocess_function, batched=False)
    
    # 5. 训练配置
    training_args = TrainingArguments(
        output_dir="./dpo_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,  # 如果GPU支持
        report_to="wandb",  # 可选
    )
    
    # 6. 初始化 DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=reference_model,
        tokenizer=tokenizer,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        max_length=512,
        max_prompt_length=256,
    )
    
    # 7. 开始训练
    trainer.train()
    
    # 8. 保存模型
    trainer.save_model("./dpo_final_model")

if __name__ == "__main__":
    main()
```

### 5.2 使用 LoRA 进行高效 DPO 训练

对于 7B 以上的大模型，建议使用 LoRA 进行参数高效微调：

```python
from peft import LoraConfig, get_peft_model

# LoRA 配置
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 1, 345, 600 || all params: 124, 442, 880 || trainable%: 1.08%
```

---

## 6. DPO 训练效果评估

### 6.1 评估指标

**Win Rate**：在 held-out 测试集上，DPO 训练后的模型输出被人类偏好的比例：

```python
def evaluate_win_rate(model, test_dataset, num_samples=100):
    wins = 0
    total = 0
    
    for sample in test_dataset[:num_samples]:
        chosen_response = sample["chosen"]
        rejected_response = sample["rejected"]
        
        # 使用待评估模型生成回答
        model_response = model.generate(sample["prompt"])
        
        # 简单的自动化评估：比较模型输出与chosen的相似度
        # 实际应用中应使用人类评估或更复杂的自动化指标
        similarity = compute_similarity(model_response, chosen_response)
        
        if similarity > 0.7:
            wins += 1
        total += 1
    
    return wins / total
```

### 6.2 Reference Model 对比评估

另一种有效评估方法是计算训练后模型与 Reference Model 的 KL 散度分布：

```python
def evaluate_kl_distribution(model, ref_model, test_prompts):
    kls = []
    
    for prompt in test_prompts:
        # 编码 prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 计算两模型的输出分布
        with torch.no_grad():
            ref_logits = ref_model(**inputs).logits
        model_logits = model(**inputs).logits
        
        # 计算 KL 散度
        ref_dist = torch.softmax(ref_logits, dim=-1)
        model_dist = torch.softmax(model_logits, dim=-1)
        
        kl = torch.nn.functional.kl_div(
            model_dist.log(), ref_dist, reduction="batchmean"
        )
        kls.append(kl.item())
    
    return {
        "mean_kl": sum(kls) / len(kls),
        "max_kl": max(kls),
        "min_kl": min(kls)
    }
```

---

## 7. 常见问题与调试

### 7.1 Reference Model Drift

**症状**：训练过程中，模型输出逐渐偏离参考模型，且训练loss不下降反而震荡。

**原因**：
- β 设置过低，约束不足
- 参考模型未正确冻结（梯度回传到了ref）

**解决方案**：
```python
# 确保参考模型完全不更新
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# 如果仍然 drift，提高 β
trainer = DPOTrainer(..., beta=0.2)
```

### 7.2 Mode Collapse

**症状**：模型输出变得非常单一，重复生成相似的短回答。

**原因**：
- 偏好数据中 chosen 回答风格过于一致
- β 设置过高，强制模型接近参考模型

**解决方案**：
```python
# 在数据准备阶段增加多样性
# 使用更高温度生成候选回答

# 或者降低 β 以允许更多探索
trainer = DPOTrainer(..., beta=0.05)
```

### 7.3 Preference Data Noise

**症状**：训练 loss 正常下降，但评估时模型质量没有提升甚至下降。

**原因**：偏好数据中存在标注错误或不一致的样本。

**解决方案**：
```python
# 过滤低质量样本
def filter_noisy_preferences(dataset):
    filtered = []
    for sample in dataset:
        # 计算 chosen 和 rejected 的长度差异
        len_diff = abs(len(sample["chosen"]) - len(sample["rejected"]))
        
        # 过滤长度差异过大的样本
        if len_diff < 500:
            # 计算语义相似度，过滤语义过于相近的样本
            similarity = compute_semantic_similarity(
                sample["chosen"], sample["rejected"]
            )
            if similarity < 0.85:  # 确保 chosen 和 rejected 足够不同
                filtered.append(sample)
    
    return filtered
```

---

## 8. 端到端实战示例

以下是一个从数据准备到训练完成的完整示例（伪代码）：

```python
# ========== Step 1: 准备偏好数据 ==========
# 使用 prepare_preference_data.py 生成合成偏好数据
from prepare_preference_data import generate_preference_dataset

prompts = [...]  # 你的提示列表
preference_data = generate_preference_dataset(
    base_model="gpt2",
    prompts=prompts,
    num_candidates=4,
    selection_criteria="length_and_structure"
)

# ========== Step 2: 保存数据 ==========
import json
with open("preference_data.json", "w") as f:
    json.dump(preference_data, f, ensure_ascii=False)

# ========== Step 3: 训练 DPO ==========
from train_dpo import train_dpo_model

trained_model = train_dpo_model(
    model_name="gpt2",
    preference_data_path="preference_data.json",
    output_dir="./dpo_output",
    beta=0.1,
    epochs=3,
    batch_size=4
)

# ========== Step 4: 评估 ==========
from evaluate import evaluate_model

results = evaluate_model(
    trained_model,
    test_data="test_preferences.json",
    metric="win_rate"
)
print(f"Win Rate: {results['win_rate']:.2%}")
```

---

## 本章小结

1. **偏好数据格式**：DPO 训练的核心数据是 `(prompt, chosen, rejected)` 三元组，表示人类对两个候选回答的偏好

2. **数据构建方式**：人工标注（高质量但成本高）和合成偏好（成本低但需要质量过滤）各有优劣，实际应用中常结合使用

3. **Reference Model 的双重作用**：提供训练基线和隐式 KL 约束，防止模型过度优化导致能力退化

4. **β 参数是关键**：控制策略偏离参考模型的程度。β 过小会导致训练不稳定，过大会导致无法学习新偏好

5. **TRL DPOTrainer 封装了完整流程**：从数据加载、模型管理到训练循环，开发者只需关注超参数选择和数据质量

6. **评估不可忽视**：Win rate 是最直接的 DPO 训练效果指标，但需要设计合理的 held-out 测试集

---

## 延伸阅读

-Rafaila et al. 2024: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (DPO 原始论文)
- TRL Documentation: DPOTrainer API reference
- Anthropic HH-RLHF Dataset: http://https://huggingface.co/datasets/Anthropic/hh-rlhf
- DeepSeek-R1 Technical Report: DPO 与 GRPO 的对比分析

---

## 思考题

1. 在构建偏好数据时，为什么需要对同一提示生成多个候选回答，而不是直接比较两个模型的输出？

2. 假设你正在训练一个对话助手，发现 DPO 训练后模型倾向于生成非常长的回答。请分析可能的原因，并提出解决方案。

3. Reference Model 在 DPO 中扮演了类似于 RLHF 中 KL 约束的角色。解释为什么 DPO 可以用单一损失函数替代 RLHF 的两阶段过程。

4. 在合成偏好数据生成中，启发式规则（如长度、结构）可能引入哪些偏差？如何减少这些偏差？

5. 考虑这样一个场景：你的偏好数据中存在 10% 的标注噪声（随机标注错误）。如果不做任何数据清洗直接训练，DPO 可能会学到什么？有什么方法可以缓解这个问题？