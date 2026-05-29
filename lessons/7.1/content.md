# 7.1 SFT数据格式与构建方法

## 课程概述

本课程深入讲解Supervised Fine-Tuning（SFT监督微调）的数据格式规范与构建方法。SFT是让预训练大模型适配特定任务的核心阶段，其效果高度依赖训练数据的质量与格式正确性。本课程从数据结构、格式标准、数据构建方法、质量控制四个维度，系统性掌握SFT数据集的完整生命周期。

## 学习目标

- 掌握SFT数据的基本结构与多轮对话格式
- 理解不同数据格式（JSON/JSONL/CSV）的适用场景
- 熟悉主流对话模板（ChatML、ChatGPT、LLaMA、Alpaca）
- 了解SFT数据集构建的主要方法及优缺点
- 掌握指令模板设计与提示工程技巧
- 建立SFT数据质量控制的方法论

## 前置知识

- 了解大语言模型（LLM）的基本原理
- 熟悉预训练与微调的基本概念
- 具备基础的Python和数据处理经验

---

## 1. SFT数据结构

### 1.1 基础三元组结构

SFT训练数据的基础单元是**指令-输入-输出**三元组，其核心字段定义如下：

| 字段 | 说明 | 必需性 |
|------|------|--------|
| `instruction` | 指令/任务描述 | 必填 |
| `input` | 任务输入上下文 | 可选 |
| `output` | 期望的标准答案 | 必填 |

**单轮对话示例（无输入字段）**：
```json
{
  "instruction": "将以下句子改写成被动语态：",
  "input": "科学家发现了这种新元素。",
  "output": "这种新元素被科学家发现了。"
}
```

**单轮对话示例（有输入字段）**：
```json
{
  "instruction": "判断以下评论的情感是正面还是负面：",
  "input": "这家餐厅的食物非常美味，服务也很周到。",
  "output": "正面"
}
```

**无输入任务的指令**：
```json
{
  "instruction": "解释什么是量子纠缠",
  "input": "",
  "output": "量子纠缠是量子力学中一种现象，当两个或多个粒子处于纠缠态时..."
}
```

### 1.2 多轮对话格式

实际应用场景中，对话系统需要处理**多轮交互上下文**。多轮对话的核心挑战在于：

1. **历史上下文管理**：正确维护对话历史
2. **角色一致性**：区分用户与助手的消息
3. **特殊token处理**：识别消息边界与角色切换

**主流多轮对话模板对比**：

| 模板名称 | 开发者 | 特点 |
|---------|--------|------|
| ChatML | OpenAI | 简洁明确，使用 `<|im_start|>` / `<|im_end|>` |
| ChatGPT | OpenAI | 与ChatML类似，专注GPT系列 |
| LLaMA | Meta | 使用 `<<SYS>>` / `<< /SYS>>` 标记系统消息 |
| Alpaca | Stanford | 简洁格式，适合学术研究 |

**ChatML 格式示例**：
```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业的法律顾问。"},
    {"role": "user", "content": "劳动合同可以口头约定吗？"},
    {"role": "assistant", "content": "根据《劳动合同法》第十条规定，建立劳动关系应当订立书面劳动合同..."},
    {"role": "user", "content": "那口头约定有什么风险？"},
    {"role": "assistant", "content": "口头劳动合同存在以下风险：1. 举证困难... "}
  ]
}
```

**LLaMA 格式示例**：
```json
{
  "text": "<<SYS>>\n你是一个专业的法律顾问。\n<< /SYS>>\n\n[INST] 劳动合同可以口头约定吗？ [/INST] 根据《劳动合同法》第十条规定，建立劳动关系应当订立书面劳动合同... </s>"
}
```

### 1.3 特殊Token体系

大模型通过特殊Token识别消息结构与角色边界：

| Token类型 | 用途 | 示例 |
|----------|------|------|
| `bos` (beginning of sequence) | 序列起始标记 | `<\|beginoftext\|>` |
| `eos` (end of sequence) | 序列结束标记 | `<\|endoftext\|>` |
| `system` | 系统提示标记 | `<\|im_start\|>system` |
| `user` | 用户消息标记 | `<\|im_start\|>user` |
| `assistant` | 助手回复标记 | `<\|im_start\|>assistant` |
| `pad` | 填充标记 | 用于批量训练对齐 |

**特殊Token在训练中的处理**：

```python
# 训练时的输入构建示例
def build_training_input(messages, tokenizer):
    """
    将消息列表转换为模型输入格式
    """
    # 角色标记映射
    role_to_token = {
        "system": "<|im_start|>system\n",
        "user": "<|im_start|>user\n",
        "assistant": "<|im_start|>assistant\n"
    }
    
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += role_to_token[role] + content + "<|im_end|>\n"
    
    # 添加助手回复的结束标记，训练时只预测assistant部分
    text += "<|im_start|>assistant\n"
    return text
```

---

## 2. 数据格式标准

### 2.1 JSON vs JSONL vs CSV

大规模SFT数据集需要选择合适的数据存储格式：

| 特性 | JSON | JSONL | CSV |
|------|------|-------|-----|
| 可读性 | 高（结构化） | 中（每行独立） | 低（表格形式） |
| 流式处理 | 需整体读取 | 支持逐行读取 | 支持逐行读取 |
| 文件大小 | 较大（重复字段名） | 中等 | 最小 |
| 追加写入 | 需整体重写 | 支持append模式 | 支持append模式 |
| 校验能力 | schema校验 | 逐行校验 | 弱类型 |
| 适用规模 | <10万条 | 任意规模 | <100万条 |

**JSONL（JSON Lines）格式推荐用于大规模训练**：

```jsonl
{"instruction": "翻译以下句子", "input": "Hello world", "output": "你好世界"}
{"instruction": "总结以下文章", "input": "...", "output": "..."}
{"instruction": "回答问题", "input": "...", "output": "..."}
```

**优点**：
- 每行独立，支持断点续传
- 批量处理时内存占用低
- streaming训练时无需加载全部数据

### 2.2 主流数据集格式规范

**Alpaca 格式**（Stanford开源数据集格式）：
```json
{
  "instruction": "给出三个保持健康的建议",
  "input": "",
  "output": "1. 保持规律运动...\n2. 均衡饮食...\n3. 充足睡眠..."
}
```

**ShareGPT 格式**（真实对话数据）：
```json
{
  "id": "sg_xxx",
  "conversations": [
    {"from": "human", "value": "用户问题"},
    {"from": "gpt", "value": "助手回答"}
  ]
}
```

**Anthropic 格式**（高安全性场景）：
```json
{
  "prompt": "以下是对话记录：\n\nHuman: 用户问题\n\nAssistant: ",
  "chosen": "助手回答",
  "rejected": "较差回答（用于DPO训练）"
}
```

---

## 3. 构建SFT数据集

### 3.1 数据构建方法对比

| 方法 | 成本 | 质量 | 规模扩展性 | 适用场景 |
|------|------|------|-----------|---------|
| 人工标注 | 高 | 最高 | 低 | 核心数据集、领域专家数据 |
| LLM生成（Self-Instruct） | 低 | 中高 | 高 | 通用指令数据 |
| 众包标注 | 中 | 中 | 中 | 通用任务数据 |
| Back-translation | 中 | 中高 | 中 | 翻译、多语言数据 |
| 数据增强 | 低 | 中 | 高 | 扩充稀缺类别 |

### 3.2 人工标注

人工标注是构建高质量SFT数据的金标准。标注流程包括：

1. **任务定义**：明确标注任务的指令模板与输出规范
2. **标注员培训**：领域知识培训与标注质量一致性校准
3. **标注执行**：按规范生成指令-响应对
4. **质量审核**：交叉验证与一致性检查

**标注质量评估指标**：

| 指标 | 说明 |
|------|------|
| 响应准确率 | 响应内容正确且完整的比例 |
| 指令多样性 | 指令表述方式的多样性程度 |
| 一致性 | 同一指令不同标注者的响应一致性 |
| 拒绝率 | 标注员无法完成标注的比例 |

### 3.3 LLM生成数据（Self-Instruct）

Self-Instruct范式利用强模型（如GPT-4）生成指令数据，显著降低成本：

**核心流程**：
```
1. 种子任务库 → 2. 指令生成 → 3. 输入生成 → 4. 输出生成 → 5. 质量过滤
```

**Self-Instruct论文核心步骤**：

```python
# 简化版Self-Instruct生成流程
def self_instruct_generate(seed_tasks, llm_model, num_generations=100):
    generated_data = []
    
    for _ in range(num_generations):
        # Step 1: 从种子任务中随机选取，生成新指令
        seed = random.choice(seed_tasks)
        new_instruction = llm_model.generate(
            f"请基于以下任务生成一个类似但不同的新指令：\n{seed['instruction']}"
        )
        
        # Step 2: 为新指令生成输入（如果有的话）
        input_text = llm_model.generate(
            f"为以下指令生成一个具体的输入：\n{new_instruction}"
        )
        
        # Step 3: 生成响应
        output_text = llm_model.generate(
            f"请回答以下问题：\n{new_instruction}\n{input_text}"
        )
        
        generated_data.append({
            "instruction": new_instruction,
            "input": input_text,
            "output": output_text
        })
    
    return generated_data
```

**Seed Corpus（种子语料）选择原则**：
- **多样性**：覆盖多种任务类型（问答、写作、分类、推理等）
- **代表性**：选取具有广泛代表性的任务
- **质量**：种子数据本身需高质量

### 3.4 Back-translation与数据增强

**Back-translation（回译法）**：
```python
# 回译增强流程
original = "我爱机器学习"  # 中文

# 翻译成英文
english = translate(original, src="zh", tgt="en")
# "I love machine learning"

# 再翻译回中文
back_translated = translate(english, src="en", tgt="zh")
# "我喜欢机器学习"

# 用于翻译任务的数据增强
augmented_data = {
    "instruction": "将以下中文翻译成英文",
    "input": "我喜欢机器学习",
    "output": "I love machine learning"
}
```

**数据增强技术**：
- 同义词替换（保持语义）
- 指令改写（增加多样性）
- 负采样（生成负样本用于对比学习）

---

## 4. 指令模板与提示工程

### 4.1 Zero-shot vs Few-shot

**Zero-shot训练**：模型仅从指令学习，不提供示例
```json
{
  "instruction": "将以下句子分类为正面或负面：这家餐厅的服务非常差。",
  "input": "",
  "output": "负面"
}
```

**Few-shot训练**：在指令中提供示例帮助模型理解任务
```json
{
  "instruction": "将以下句子分类为正面或负面。\n\n示例1：食物很好吃 → 正面\n示例2：环境很糟糕 → 负面\n\n待分类：这家餐厅的服务非常差。",
  "input": "",
  "output": "负面"
}
```

**训练中的Few-shot策略**：
- 训练时通常使用Zero-shot，Few-shot主要用于推理阶段
- 如需在训练中引入Few-shot，需要在指令中嵌入示例

### 4.2 System Prompt设计

System Prompt在SFT训练中扮演重要角色，影响模型行为：

| Prompt类型 | 效果 |
|-----------|------|
| 角色设定 | "你是一位资深Python工程师" |
| 行为约束 | "回答应简洁明了，避免冗余" |
| 领域专注 | "专注于金融风控领域" |
| 安全边界 | "拒绝回答涉及隐私的问题" |

**System Prompt训练格式**：
```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业、严谨的法律顾问。在回答法律问题时，应引用相关法条，并提醒用户仅供参"},
    {"role": "user", "content": "租房合同最长期限是多久？"},
    {"role": "assistant", "content": "根据《民法典》第七百零五条规定，租赁期限不得超过二十年..."}
  ]
}
```

---

## 5. SFT数据质量控制

### 5.1 质量控制框架

```
数据构建 → 初步筛选 → 质量审核 → 格式校验 → 下采样平衡 → 最终数据集
```

### 5.2 核心质量指标

**指令-响应相关性检验**：
```python
# 伪代码：使用嵌入向量计算相关性
def check_relevance(instruction, response, threshold=0.75):
    instr_embedding = embed_model.encode(instruction)
    resp_embedding = embed_model.encode(response)
    similarity = cosine_similarity(instr_embedding, resp_embedding)
    return similarity >= threshold
```

**幻觉检测要点**：
- 事实性陈述是否可验证
- 数字、日期、人名是否准确
- 逻辑是否自洽

**任务类型平衡**：
```json
{
  "task_distribution": {
    "问答": 0.25,
    "写作": 0.20,
    "摘要": 0.15,
    "翻译": 0.15,
    "代码": 0.15,
    "推理": 0.10
  }
}
```

### 5.3 常见质量问题与处理

| 问题类型 | 表现 | 处理方法 |
|---------|------|---------|
| 指令模糊 | 响应偏离预期 | 重新标注或过滤 |
| 响应不完整 | 回答半途中断 | 补充完整或过滤 |
| 幻觉内容 | 包含错误事实 | fact-checking过滤 |
| 风格不一致 | 同一数据集风格差异大 | 统一格式化或过滤 |
| 长度异常 | 过短或过长 | 长度阈值过滤 |

---

## 6. 总结

本课程系统讲解了SFT数据的格式规范与构建方法：

1. **数据结构**：掌握instruction-input-output三元组与多轮对话格式
2. **格式标准**：JSONL是大规模数据集推荐格式，Alpaca/ShareGPT格式各有适用场景
3. **构建方法**：人工标注质量最高但成本高；Self-Instruct成本低但需质量过滤
4. **模板设计**：Zero-shot为主流，System Prompt影响模型行为
5. **质量控制**：相关性检验、幻觉检测、任务平衡三位一体

高质量SFT数据是模型微调成功的基础，数据工程与模型训练同等重要。

---

## 延伸阅读

1. **Self-Instruct: Aligning Language Models with Self-Generated Instructions** - Wang et al., 2022
2. **Stanford Alpaca: An Instruction-following LLaMA model** - Stanford CRFM
3. **WizardLM: Empowering Large Language Models to Follow Complex Instructions** - Microsoft Research
4. **The Red Team Learning Process** - Anthropic Safety Evaluation

---

## 复习题

1. **比较JSON与JSONL格式的优劣势，说明为何JSONL更适合大规模SFT训练。**

2. **阐述ChatML与LLaMA格式在多轮对话处理上的主要差异，以及它们各自的优势场景。**

3. **分析人工标注与Self-Instruct两种数据构建方法的成本-质量权衡，并说明如何结合使用两种方法。**

4. **设计一个SFT数据质量控制pipeline，说明每个环节的检验标准与过滤策略。**
