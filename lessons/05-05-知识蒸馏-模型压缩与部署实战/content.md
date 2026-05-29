# 5.5 知识蒸馏：模型压缩与部署实战

## 课程概述

本节课聚焦知识蒸馏（Knowledge Distillation，KD）——一种通过让小型"学生"模型学习大型"教师"模型行为来实现模型压缩的技术。与量化直接降低数值精度不同，知识蒸馏通过让学生模型学习教师模型的"知识"来保留更多能力。

本节首先从知识蒸馏的背景与核心思想出发，解释为什么软目标（soft targets）比硬目标（hard targets）携带更多知识。随后深入温度缩放（Temperature Scaling）的原理，以及KL散度损失如何衡量教师与学生输出分布的差异。接着讲解Response-based（基于输出的）和Feature-based（基于中间层的）两种主要蒸馏方法，以及Multi-task蒸馏在LLM中的应用。最后对比蒸馏与量化、蒸馏与LoRA的互补性与各自适用场景，并介绍MiniLM和TinyBERT等经典蒸馏案例。

## 学习目标

- 理解知识蒸馏的核心思想：教师-学生架构、软目标 vs 硬目标
- 掌握温度缩放的原理：T>1时如何软化概率分布，Temperature如何影响蒸馏效果
- 能够实现KL散度蒸馏损失，理解Response-based蒸馏的实现细节
- 理解Feature-based蒸馏：隐藏层匹配与注意力匹配的区别与适用场景
- 了解Multi-task蒸馏在LLM指令微调中的应用
- 对比蒸馏与量化、蒸馏与LoRA的互补性，理解三种方法的适用边界
- 能够根据精度需求和部署约束选择合适的模型压缩策略
- 了解MiniLM、TinyBERT等经典蒸馏案例的设计思路

## 前置知识

- 神经网络基础：前向传播、反向传播、损失函数
- Transformer架构：自注意力、FFN、LayerNorm
- 模型压缩的基本概念：参数量、计算量、内存占用的区别
- PyTorch基础：张量操作、nn.Module、optimizer

---

## 1. 知识蒸馏的核心思想

### 1.1 为什么要蒸馏

大模型虽然性能强大，但推理成本高、延迟大、内存占用大。知识蒸馏是一种"让小模型学习大模型行为"的模型压缩技术——不是直接训练一个小模型，而是让小模型去"模仿"大模型的输出。

**蒸馏的本质**：教师模型（teacher）通常是一个大型高性能模型，学生模型（student）是一个更小的模型。蒸馏的核心是让学生模型学习教师模型的"软输出"，而不是仅仅学习真实标签。

### 1.2 软目标 vs 硬目标

**硬目标（Hard Targets）**：传统的分类任务中，真实标签是one-hot向量，如[0, 0, 1, 0, 0]。这种表示只告诉我们"正确答案是什么"，没有任何关于"错误答案有多接近"的信息。

**软目标（Soft Targets）**：教师模型的输出是一个概率分布，如[0.01, 0.02, 0.85, 0.10, 0.02]。这个分布告诉我们：
- 正确答案的概率是0.85
- 第四名（概率0.10）与正确答案最接近，可能存在某种相似性
- 第二名和第五名（概率0.02）之间有一定距离

**为什么软目标更有价值**：

```
硬目标：[0, 0, 1, 0, 0]     # 只知道正确答案是第3个
软目标：[0.01, 0.02, 0.85, 0.10, 0.02]  # 知道第4个比第2、5个更接近正确
```

软目标包含了**类别之间的相对关系**——这正是大模型学到的深层知识。例如：
- 在1000类分类中，"狗"和"狼"比"狗"和"汽车"更接近
- 在问答中，某些错误答案比其他错误答案"更接近"正确答案

### 1.3 Teacher-Student架构

```
┌─────────────────────────────────────────────────────────┐
│  教师模型 (Teacher)          学生模型 (Student)           │
│  ┌─────────────┐             ┌─────────────┐           │
│  │ Large Model │             │ Small Model  │           │
│  │  (大模型)   │  ──蒸馏──>  │  (小模型)   │           │
│  └─────────────┘             └─────────────┘           │
│        │                           │                    │
│        ▼                           ▼                    │
│  软目标分布                  学习软目标分布              │
│  P_teacher                    P_student                 │
└─────────────────────────────────────────────────────────┘
```

**蒸馏过程**：
1. 教师模型对输入x进行前向传播，输出软目标分布P_teacher
2. 学生模型对同一输入x进行前向传播，输出P_student
3. 计算P_teacher与P_student之间的KL散度作为蒸馏损失
4. 反向传播更新学生模型参数

---

## 2. 温度缩放与KL散度损失

### 2.1 温度缩放的原理

**为什么需要温度**？

在标准softmax中，如果正确答案的概率是0.99，其他类别概率都很小，那么软目标几乎退化为硬目标——失去了软目标的优势。

```
标准 softmax (T=1):
P_i = exp(z_i) / Σexp(z_j)
```

**温度缩放**（Temperature Scaling）通过在softmax中引入温度参数T来软化概率分布：

```
softmax with temperature:
P_i = exp(z_i/T) / Σexp(z_j/T)
```

- **T > 1**：概率分布变得更平滑，类别间差异被放大（更易区分）
- **T < 1**：概率分布变得更尖锐，接近one-hot
- **T = 1**：退化为标准softmax

**温度对分布的影响**：

假设一个模型的logits是[2.0, 1.0, 0.5]：

| 温度T | P分布 | 说明 |
|-------|-------|------|
| T=1 | [0.59, 0.24, 0.17] | 标准分布 |
| T=2 | [0.42, 0.31, 0.27] | 更平滑，各类别概率更接近 |
| T=4 | [0.30, 0.28, 0.25] | 非常平滑，接近均匀分布 |
| T=0.5 | [0.87, 0.11, 0.02] | 更尖锐，接近one-hot |

**蒸馏中的最佳温度**：通常T=2~20之间效果最好。太高的温度（如T=100）会让分布接近均匀，丢失有用信息；太低的温度会让分布接近one-hot，优势全无。

### 2.2 KL散度损失函数

蒸馏的核心损失函数是教师与学生软输出之间的KL散度：

```python
def distillation_loss(student_logits, teacher_logits, temperature):
    """
    计算KL散度蒸馏损失
    
    参数:
        student_logits: 学生模型的输出logits [batch, num_classes]
        teacher_logits: 教师模型的输出logits [batch, num_classes]
        temperature: 温度参数T
    
    返回:
        KL散度损失 (标量)
    """
    # 将logits转换为概率分布（使用温度缩放）
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # 计算KL散度: KL(P || Q) = Σ P * log(P/Q)
    # 注意：KL散度是非对称的，这里是 teacher → student
    kl_loss = F.kl_div(
        student_probs.log(),  # log(Q)
        teacher_probs,         # P
        reduction='batchmean'
    ) * (temperature ** 2)  # 乘以T²补偿缩放效应
    
    return kl_loss
```

**为什么乘以T²？**

KL散度的梯度与1/T²成正比。在反向传播时，如果温度很高，梯度会变得非常小，乘以T²可以补偿这个效应。

### 2.3 组合损失：硬目标 + 软目标

实际蒸馏中，学生模型通常同时学习两个目标：

```
Total Loss = α × Hard Loss + (1-α) × Soft Loss

其中：
- Hard Loss = CrossEntropy(student_logits, true_labels)  # 传统交叉熵
- Soft Loss = KL_divergence(student_soft, teacher_soft)  # 蒸馏KL散度
- α: 硬目标权重，通常0.1~0.5
```

```python
def combined_distillation_loss(
    student_logits,
    teacher_logits,
    true_labels,
    temperature=4.0,
    alpha=0.3
):
    """
    组合蒸馏损失：硬目标 + 软目标
    """
    # 硬损失：学生预测与真实标签的交叉熵
    hard_loss = F.cross_entropy(student_logits, true_labels)
    
    # 软损失：学生与教师软输出的KL散度
    soft_loss = distillation_loss(student_logits, teacher_logits, temperature)
    
    # 组合损失
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
    return total_loss
```

**α的设置建议**：
- α=0：纯软目标蒸馏（仅学习教师）
- α=1：退化为传统训练（仅学习硬目标）
- α=0.3~0.5：通常效果最佳——既学习教师知识，又保持对真实标签的敏感

---

## 3. Response-based 蒸馏

### 3.1 什么是Response-based蒸馏

Response-based蒸馏（也称Logits-based蒸馏）直接让学生学习教师模型的最终输出（logits或预测概率）。这是最简单的蒸馏方法，不需要了解模型内部结构。

```
教师模型:  输入x → [...所有层...] → logits_T → softmax(T) → P_teacher
学生模型:  输入x → [...所有层...] → logits_S → softmax(T) → P_student
                                    ↑
                              计算KL(logits_S, logits_T)
```

### 3.2 蒸馏训练流程

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_student蒸馏(
    teacher_model,
    student_model,
    train_loader,
    temperature=4.0,
    alpha=0.3,
    learning_rate=1e-3,
    epochs=10
):
    """
    完整的蒸馏训练流程
    """
    # 冻结教师模型
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    # 学生模型使用学习率
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 1. 教师前向传播（不计算梯度）
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            
            # 2. 学生前向传播
            student_logits = student_model(inputs)
            
            # 3. 计算组合损失
            loss = combined_distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=temperature,
                alpha=alpha
            )
            
            # 4. 反向传播（仅更新学生）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return student_model
```

### 3.3 Response-based蒸馏的适用场景

**优势**：
- 实现简单，不需要了解教师模型内部结构
- 适用于任何有logits输出的模型
- 计算开销小

**局限**：
- 只学习最终输出，无法学习教师的中间表示能力
- 对于复杂任务（如文本生成），最终输出的信息量可能不足

**适用场景**：
- 分类任务（BERT-style模型）
- 简单回归任务
- 当学生模型结构与教师相似时效果更好

---

## 4. Feature-based 蒸馏

### 4.1 为什么要特征蒸馏

Response-based蒸馏只学习教师模型的最终输出。但大模型的真正"知识"不仅体现在最终预测上，更体现在中间层的表示能力上——比如理解语法结构、捕捉语义关系等。

**Feature-based蒸馏**让学生学习教师中间层的表示：

```
教师模型:  input → ... → Layer_N → hidden_T → 输出
学生模型:  input → ... → Layer_M → hidden_S → 输出
                 ↑                    ↑
            学习匹配              学习匹配
            hidden_T              hidden_S
```

### 4.2 隐藏层匹配

**Adapter-based匹配**：在学生模型的隐藏层后添加一个adapter，将学生表示映射到与教师相同维度，然后计算MSE损失：

```python
def feature_distillation_loss(
    student_hidden,   # [batch, seq_len, hidden_dim_S]
    teacher_hidden,   # [batch, seq_len, hidden_dim_T]
    projector=None    # 可选：将学生维度映射到教师维度
):
    """
    特征蒸馏损失：MSE between projected student and teacher hidden states
    """
    if projector is not None:
        # 投影到相同维度
        student_hidden = projector(student_hidden)
    
    # MSE损失
    loss = F.mse_loss(student_hidden, teacher_hidden)
    return loss
```

**维度不匹配的处理**：

如果教师hidden_dim=1024，学生hidden_dim=256，需要投影：

```python
class HiddenProjector(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.projection = nn.Linear(student_dim, teacher_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(self.projection(x))

# 使用
projector = HiddenProjector(student_dim=256, teacher_dim=1024).to(device)
```

### 4.3 注意力蒸馏

**注意力匹配**让学生学习教师的注意力分布。BERT的self-attention包含丰富的语法和语义信息，这些信息可以被蒸馏：

```python
def attention_distillation_loss(
    student_attention,  # [batch, heads, seq_len, seq_len]
    teacher_attention,   # [batch, heads, seq_len, seq_len]
    temperature=2.0
):
    """
    注意力蒸馏损失：KL divergence between attention distributions
    """
    # 将attention视为概率分布
    student_attn = F.softmax(student_attention / temperature, dim=-1)
    teacher_attn = F.softmax(teacher_attention / temperature, dim=-1)
    
    # 计算KL散度（对所有头取平均）
    kl_loss = F.kl_div(
        student_attn.log(),
        teacher_attn,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    return kl_loss
```

### 4.4 完整的Feature蒸馏训练

```python
def train_with_feature_distillation(
    teacher_model,
    student_model,
    train_loader,
    layer_id=12,        # 学生模型要匹配的层
    temperature=4.0,
    alpha=0.5
):
    """
    完整的特征蒸馏训练
    """
    # 冻结教师，初始化学生
    teacher_model.eval()
    student_model.train()
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            # 教师前向传播（获取中间层输出）
            with torch.no_grad():
                teacher_output = teacher_model(inputs, output_hidden_states=True)
                teacher_hidden = teacher_output.hidden_states[layer_id]
            
            # 学生前向传播（获取中间层输出）
            student_output = student_model(inputs, output_hidden_states=True)
            student_hidden = student_output.hidden_states[layer_id]
            
            # 特征蒸馏损失
            feat_loss = feature_distillation_loss(student_hidden, teacher_hidden)
            
            # Response蒸馏损失（可选）
            response_loss = F.cross_entropy(student_output.logits, labels)
            
            # 组合
            total_loss = alpha * feat_loss + (1 - alpha) * response_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

---

## 5. 多任务蒸馏与LLM应用

### 5.1 Multi-task蒸馏的挑战

大语言模型（LLM）通常需要完成多种任务：文本生成、摘要、翻译、问答等。Multi-task蒸馏面临以下挑战：

1. **不同任务的知识结构不同**：一个教师模型在不同任务上的行为可能不同
2. **任务间干扰**：某些任务的知识可能互相干扰
3. **软目标的质量问题**：教师在某些任务上可能表现不佳

### 5.2 Task-specific蒸馏策略

**分任务蒸馏**：对每个任务单独蒸馏一个学生模型，然后将多个学生模型的能力合并：

```python
def multi_task_distillation(
    teacher_model,
    student_models,  # 多个学生模型，每个对应一个任务
    task_data_loaders,  # 每个任务的数据加载器
    tasks=['summarization', 'translation', 'qa']
):
    for task, student_model, data_loader in zip(tasks, student_models, task_data_loaders):
        print(f"蒸馏任务: {task}")
        
        for epoch in range(epochs):
            for batch in data_loader:
                # 教师在当前任务上的输出
                with torch.no_grad():
                    teacher_output = teacher_model(batch, task=task)
                
                # 学生学习
                student_output = student_model(batch)
                loss = F.kl_div(
                    student_output.logits / temperature,
                    teacher_output.logits / temperature
                ) * (temperature ** 2)
                
                loss.backward()
                optimizer.step()
```

### 5.3 任务相关蒸馏

**Task-related Distillation**：识别与目标任务最相关的教师层，然后重点蒸馏这些层：

```python
def task_related_distillation(
    teacher_model,
    student_model,
    task_data,
    relevance_scores  # 预先计算的任务相关性分数 [num_layers]
):
    """
    根据任务相关性分数选择重要层进行蒸馏
    """
    for layer_id in range(num_layers):
        if relevance_scores[layer_id] > threshold:
            # 蒸馏这一层
            teacher_hidden = get_teacher_layer(teacher_model, layer_id)
            student_hidden = get_student_layer(student_model, layer_id)
            
            loss = feature_distillation_loss(student_hidden, teacher_hidden)
            loss.backward()
```

---

## 6. 蒸馏 vs 量化 vs LoRA

### 6.1 三种压缩方法的对比

| 维度 | 知识蒸馏 | 量化 | LoRA |
|------|----------|------|------|
| **原理** | 学生学习教师行为 | 降低数值精度 | 注入低秩矩阵 |
| **模型结构** | 改变模型结构（更少的层/神经元） | 不变 | 不变 |
| **权重变化** | 完全重新训练学生模型 | 精度降低 | 大多数权重冻结，仅训练adapter |
| **压缩比** | 可控（根据学生架构） | 2-8x | 有限压缩，效率来自共享 |
| **精度损失** | 取决于学生架构选择 | INT4可能较大 | 最小（通过微调恢复） |
| **计算成本** | 高（需要完整训练） | 低（校准） | 低（仅训练少量参数） |
| **适用场景** | 需要显著减小模型规模 | 资源受限部署 | 高效微调+部署 |

### 6.2 蒸馏与量化的互补性

**蒸馏和量化可以叠加使用**：

```
大型教师模型 (FP16)
       ↓ 蒸馏
中型学生模型 (FP16)
       ↓ 量化
小型学生模型 (INT8)
```

**叠加效果**：
- LLaMA-7B (FP16) → 蒸馏 → LLaMA-2B (FP16) → 量化 → LLaMA-2B (INT8)
- 最终模型体积从13GB降至3.3GB（约4x压缩）

**叠加顺序**：通常先蒸馏后量化，因为：
1. 蒸馏后的学生模型已经学会了关键知识，对量化误差更鲁棒
2. 量化主要影响权重，蒸馏后模型的中间表示更稳定

### 6.3 蒸馏与LoRA的不同哲学

**蒸馏**：从零训练一个小型学生模型，完全重新学习
**LoRA**：保持原模型大部分权重不变，只训练少量适配参数

```
蒸馏思路：
"训练一个全新的小模型，让它模仿大模型的行为"
→ 需要大量训练数据和时间
→ 彻底改变模型结构

LoRA思路：
"保持大模型不变，通过adapter让它适应新任务"
→ 训练成本极低
→ 保持原模型结构不变
```

**何时用蒸馏**：需要显著减小模型规模（参数数量）时
**何时用LoRA**：需要在原模型基础上高效微调时

---

## 7. 蒸馏实战：MiniLM与TinyBERT案例

### 7.1 MiniLM：蒸馏BERT到更小更深

**MiniLM**是Microsoft提出的蒸馏方法，它的核心洞察是：BERT的最终层隐藏状态包含了最丰富的语义信息，可以通过蒸馏这一层来实现高效压缩。

**MiniLM的设计原则**：

1. **深度学生**：不是减少层数，而是减少每层的隐藏维度
   - 教师：BERT-base (12层, 768维)
   - 学生：MiniLM (6层, 384维, 但更深)
   
2. **自注意力蒸馏**：让学生学习教师的自注意力分布
   ```python
   # MiniLM的注意力蒸馏损失
   def miniLM_attention_loss(student_attn, teacher_attn, num_heads):
       total_loss = 0
       for h in range(num_heads):
           s_attn = student_attn[h]
           t_attn = teacher_attn[h]
           total_loss += F.kl_div(s_attn.log(), t_attn)
       return total_loss / num_heads
   ```

3. **隐藏状态蒸馏**：让学生学习教师的最终层输出
   ```python
   # MiniLM的隐藏状态蒸馏损失
   def miniLM_hidden_loss(student_hidden, teacher_hidden):
       return F.mse_loss(student_hidden, teacher_hidden)
   ```

**MiniLM的效果**：

| 模型 | 参数量 | GLUE分数 |
|------|--------|---------|
| BERT-base | 110M | 84.3 |
| MiniLM | 22M | 81.2 |
| 压缩比 | 5x | 下降3% |

### 7.2 TinyBERT：两阶段蒸馏

**TinyBERT**（华为诺亚方舟实验室）提出了两阶段蒸馏方法：

```
阶段1：通用蒸馏
  教师：BERT-base
  学生：TinyBERT (4层, 768→312)
  使用大型语料库进行通用知识蒸馏

阶段2：任务特定蒸馏
  使用下游任务数据微调
  额外加入任务特定层蒸馏
```

**TinyBERT的蒸馏内容**：

1. **嵌入层蒸馏**：词嵌入 + embedding layer输出
2. **注意力蒸馏**：self-attention scores + attention probabilities
3. **隐藏层蒸馏**：transformer层输出
4. **预测层蒸馏**：最终logits输出

**TinyBERT的效果**：

| 模型 | 参数量 | SST-2 | MRPC |
|------|--------|-------|------|
| BERT-base | 110M | 93.2 | 85.4 |
| TinyBERT | 14.5M | 91.3 | 82.1 |
| 压缩比 | 7.5x | 下降2% | 下降3% |

### 7.3 蒸馏策略总结

**设计学生模型的原则**：

1. **深度优先**：增加层数比增加宽度更能保留表达能力
2. **维度匹配**：投影层要足够表达力，避免成为瓶颈
3. **层次选择**：确定要蒸馏哪些层，通常是中间层
4. **损失权重**：合理分配response loss和feature loss的权重

**训练技巧**：

1. **预热**：先用硬目标训练学生几个epoch，再切换到蒸馏
2. **渐进式**：从粗粒度（response）到细粒度（attention）逐步添加
3. **数据增强**：KD需要大量无标签数据，与自监督结合效果好

---

## 8. 蒸馏的实践考虑

### 8.1 选择学生架构

**经验法则**：

```
学生参数 ≈ 教师参数 / 压缩比
压缩比通常2-10x之间
```

**学生架构设计检查表**：

- [ ] 学生模型的层数：通常是教师的1/4到1/2
- [ ] 隐藏维度：按比例缩减，但保留足够表达能力
- [ ] 注意力头数：至少保留4头以保证多样性
- [ ] FFN维度：通常与隐藏维度保持一定比例
- [ ] 词汇表：可以共享，也可以裁剪（需重新训练embedding）

### 8.2 蒸馏的温度选择

**温度的影响**：

| 温度范围 | 效果 | 适用场景 |
|----------|------|----------|
| T=1-2 | 接近硬目标，分布尖锐 | 学生能力较强时 |
| T=2-4 | 适中软化，最常用 | 通用蒸馏 |
| T=4-8 | 更平滑的分布 | 学生能力较弱，需要更多"暗知识" |
| T>10 | 分布接近均匀，丢失信息 | 不推荐 |

**自适应温度**：

```python
def adaptive_temperature(epoch, total_epochs):
    """
    训练过程中逐渐降低温度
    初期：让学生学习更平滑的分布（探索）
    后期：让分布更尖锐（精确）
    """
    start_temp = 8.0
    end_temp = 2.0
    return start_temp - (start_temp - end_temp) * (epoch / total_epochs)
```

### 8.3 蒸馏的评估

**评估指标**：

1. **下游任务性能**：在任务数据集上测试准确率/F1
2. **知识保留率**：相对教师性能的百分比
3. **压缩效率**：性能损失与压缩比的比值

```python
def evaluate_distillation(teacher, student, test_loader):
    """评估蒸馏效果"""
    teacher_acc = evaluate_model(teacher, test_loader)
    student_acc = evaluate_model(student, test_loader)
    
    compression_ratio = count_params(teacher) / count_params(student)
    knowledge_retention = student_acc / teacher_acc * 100
    
    return {
        "teacher_accuracy": teacher_acc,
        "student_accuracy": student_acc,
        "compression_ratio": compression_ratio,
        "knowledge_retention_pct": knowledge_retention
    }
```

---

## 总结

本节课围绕知识蒸馏展开，主要内容：

1. **知识蒸馏核心思想**：通过教师-学生架构，让学生学习教师的软目标
2. **软目标 vs 硬目标**：软目标包含类别间的相对关系，携带更多"暗知识"
3. **温度缩放**：T>1软化概率分布，让蒸馏更有效
4. **KL散度损失**：衡量教师与学生软输出分布的差异
5. **Response-based蒸馏**：学习教师的最终输出logits
6. **Feature-based蒸馏**：学习教师的中间层表示（隐藏状态、注意力）
7. **Multi-task蒸馏**：对不同任务分别蒸馏，识别任务相关层
8. **蒸馏与量化的互补**：可以叠加使用实现更大压缩比
9. **蒸馏 vs LoRA**：蒸馏改变结构，LoRA保持结构+添加adapter
10. **MiniLM与TinyBERT**：经典蒸馏案例，深度学生+注意力蒸馏

---

## 扩展阅读

- Hinton et al. (2015). *Distilling the Knowledge in a Neural Network* — 知识蒸馏的开山之作
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT* — BERT蒸馏的经典案例
- Sun et al. (2019). *TinyBERT: Distilling BERT for Natural Language Understanding* — 两阶段蒸馏方法
- Wang et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression* — 深度学生+注意力蒸馏
- Mirzadeh et al. (2020). *Improved Knowledge Distillation via Teacher Assistant* — 多阶段蒸馏缓解能力差距

---

## 复习题

1. **软目标的数学解释**：设教师模型对输入x的logits为[2.0, 1.0, 0.0, -1.0]，计算T=1和T=4时的softmax概率分布，并解释为什么T=4时的分布对蒸馏更有价值。

2. **KL散度推导**：解释为什么蒸馏损失中需要乘以T²（温度的平方）。从KL散度的梯度和链式法则的角度说明。

3. **蒸馏与量化的叠加**：假设LLaMA-7B（FP16）需要部署在只有6GB显存的GPU上。设计一个完整的压缩方案，说明先蒸馏还是先量化，以及为什么。

4. **Feature vs Response蒸馏**：对比Response-based蒸馏和Feature-based蒸馏的适用场景。什么情况下应该使用Feature-based蒸馏？为什么？

5. **MiniLM设计分析**：MiniLM选择"更窄但更深"的学生架构（6层×384维 vs 12层×768维），而不是"更浅"的架构（4层×768维）。从信息容量和层次表示的角度分析这种设计选择的优劣。