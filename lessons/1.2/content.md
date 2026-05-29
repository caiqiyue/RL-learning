# 1.2 微调策略选择：全量/LoRA/QLoRA/适配器

## 课程概述

本课时深入对比各种大模型微调方法的原理、适用场景与工程实现，帮助工程师根据硬件资源和任务需求做出最优技术决策。

**学习目标**
- 理解全量微调、LoRA、QLoRA、适配器微调的原理差异
- 掌握不同场景下的技术选型决策树
- 了解各方法的显存占用与计算成本对比

**前置知识**：第一章1.1大模型概述

---

## 1. 微调方法全景图

### 1.1 四类核心方法对比

```
预训练模型权重
    │
    ├── ① 全量微调 (Full Fine-tuning)
    │       └── 更新100%参数，显存占用最大
    │
    ├── ② LoRA / QLoRA
    │       └── 仅更新低秩适配器，参数<1%
    │
    ├── ③ 适配器微调 (Adapter Tuning)
    │       └── 插入瓶颈层，保留原参数冻结
    │
    └── ④ 部分参数微调 (Freeze/P-Tuning)
            └── 冻结大部分，仅更新顶层或新增参数
```

### 1.2 显存占用公式

**核心公式：模型显存 ≈ 参数数量 × 精度字节数**

| 精度格式 | 字节/参数 | 7B模型 | 70B模型 | 130B模型 |
|---------|----------|---------|---------|---------|
| FP32 | 4B | 28GB | 280GB | 520GB |
| FP16/BF16 | 2B | 14GB | 140GB | 260GB |
| INT8 | 1B | 7GB | 70GB | 130GB |
| INT4 | 0.5B | 3.5GB | 35GB | 65GB |

**训练显存附加项（以7B-FP16为例）**
```
基础模型权重：7B × 2B = 14GB
梯度：7B × 2B = 14GB
优化器状态(AdamW)：7B × 2 × 2B = 28GB  ← 最大项
激活值(视batch size而定)：~8-16GB
总计：64GB+ （仅模型+训练开销）
```

---

## 2. 全量微调 (Full Fine-tuning)

### 2.1 原理

全量微调是在预训练模型基础上，对所有参数进行反向传播更新。

```python
# 全量微调示例（PyTorch）
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

# 需要加载到GPU的所有参数
for name, param in model.named_parameters():
    param.requires_grad = True  # 全部开启梯度

# 训练时所有参数都会更新
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

### 2.2 优缺点分析

| 优点 | 缺点 |
|------|------|
| 任务适配性最强 | 显存占用大（≥65GB for 7B） |
| 全参数优化，灵活性最高 | 计算资源消耗巨大 |
| 适合数据充足的场景 | 容易过拟合，尤其是小数据集 |
| 可充分学习任务特性 | 每个任务需独立完整训练 |

### 2.3 适用场景

```python
# 适合全量微调的情况
if (len(training_data) > 100_000 and 
    available_gpu_memory > 200GB and
    task_is_highly_specialized):
    
    choose_full_finetuning()
```

**典型案例**
- 医疗：百万级临床数据微调
- 法律：千万级判决文书训练
- 金融：大规模交易数据建模

---

## 3. LoRA (Low-Rank Adaptation)

### 3.1 核心思想

LoRA假设大模型的权重更新（ΔW）具有低 intrinsic rank，可以通过两个小矩阵的乘积来近似。

**数学原理**
```
原始全量微调： W_new = W_0 + ΔW   （ΔW 与 W_0 同维度）

LoRA核心洞察：
    ΔW 可以分解为 B×A，其中 B∈R^(d×r)，A∈R^(r×k)
    且 r << min(d, k)
    
LoRA更新：W_new = W_0 + B×A
```

**直观理解**

```
假设 W_0 ∈ R^(1000×1000)，共1M参数

全量微调：更新全部1M参数

LoRA（r=8）：
    A ∈ R^(8×1000)：8K参数
    B ∈ R^(1000×8)：8K参数
    总共 16K 参数
    参数量减少 98.4%
```

### 3.2 代码实现

```python
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# 基础模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

# LoRA配置
lora_config = LoraConfig(
    r=8,                    # rank，低秩维度，通常8~64
    lora_alpha=16,          # 缩放因子，通常2×rank
    target_modules=[         # 应用LoRA的模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 包装模型
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# 输出：trainable params: 16,XXXX || all params: 7,XXX,XXX,XXX || 0.XX% trainable
```

### 3.3 显存对比

| 配置 | 可训练参数 | 训练显存 | 推理显存 |
|------|-----------|---------|---------|
| 全量FP16 | 7B | ~65GB | 14GB |
| LoRA(r=8) | ~16M (~0.2%) | ~23GB | 14GB |
| LoRA(r=64) | ~134M (~1.9%) | ~35GB | 14GB |

### 3.4 优缺点分析

| 优点 | 缺点 |
|------| 缺点 |
|------|------|
| 显存需求大幅降低 | 表达能力受限于低秩假设 |
| 训练速度快 | 部分任务可能不如全量微调 |
| 多任务可共享原模型 | 超参(rank)需要调优 |
| 无推理延迟（可合并权重） | 对极端复杂任务可能不足 |

### 3.5 适用场景

```python
# LoRA最佳场景
if (available_gpu_memory < 80GB and 
    len(training_data) < 100_000 and
    need_multi_task_deployment):
    choose_lora()
```

---

## 4. QLoRA (量化版LoRA)

### 4.1 核心创新

QLoRA = LoRA + 4-bit量化 + 分页优化器

```python
# QLoRA三件套
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

# 1. 4-bit NF4量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # NormalFloat4，非线性量化
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True   # 双量化：量化参数本身再量化
)

# 2. 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=quantization_config,
    device_map="auto"
)

# 3. 应用LoRA
model = get_peft_model(model, lora_config)
```

### 4.2 NF4量化原理

传统INT4使用均匀量化：
```
INT4量化点：[-7, -5, -3, -1, 1, 3, 5, 7]  # 线性分布
```

NF4针对神经网络权重正态分布优化：
```
NF4量化点：[-1, -0.7, -0.3, -0.1, 0, 0.1, 0.3, 0.7, 1]  
# 在0附近更密集，在两端较稀疏
```

### 4.3 显存对比（65B模型）

| 方法 | 基础模型显存 | LoRA显存 | 总显存 |
|------|------------|---------|-------|
| 全量FP16 | 130GB | - | 650GB |
| LoRA FP16 | 130GB | 0.4GB | ~150GB |
| QLoRA INT4 | 32GB | 0.4GB | ~48GB |

**QLoRA可以让65B模型在单张RTX 4090(24GB)上运行！**

### 4.4 分页优化器

QLoRA使用分页优化器处理GPU显存溢出：
```python
# 分页优化器配置
optimizer = bnb.optim.PagedAdamW32bit(
    model.parameters(),
    lr=1e-4,
    optim_args={"page_size": 1}  # 优化器状态分页
)
# 当GPU显存不足时，自动将优化器状态分页到CPU内存
```

---

## 5. 适配器微调 (Adapter Tuning)

### 5.1 原理

在Transformer层之间插入小型适配器模块，保持原模型参数冻结。

```
原始Transformer层：
    Input → Self-Attention → Add&Norm → FFN → Output

Adapter插入后：
    Input → Self-Attention → Add&Norm → [Adapter] → FFN → [Adapter] → Output
```

**适配器结构（ bottleneck architecture）**
```python
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)   # d_model → bottleneck
        self.activation = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, d_model)  # bottleneck → d_model
        self.skip = nn.Identity()  # 跳跃连接
        
    def forward(self, x):
        # 跳跃连接保证至少不会变差
        return self.skip(x) + self.up(self.activation(self.down(x)))
```

### 5.2 多适配器共存

适配器的独特优势：同一模型可加载多个适配器处理不同任务
```python
# 加载多个适配器
model.load_adapter("adapter_medical", "medical_adapter")
model.load_adapter("adapter_legal", "legal_adapter")
model.load_adapter("adapter_code", "code_adapter")

# 动态切换
model.set_adapter("medical_adapter")  # 处理医疗咨询
```

---

## 6. Freeze / P-Tuning

### 6.1 Freeze微调

仅微调模型的最后几层或新增的少量参数：

```python
# Freeze微调示例
for name, param in model.named_parameters():
    if "layer.final_layer_norm" in name or "lm_head" in name:
        param.requires_grad = True   # 仅顶层可训练
    else:
        param.requires_grad = False  # 冻结

# 可训练参数：~1-5% of total
```

### 6.2 P-Tuning v2

在每个Transformer层的前馈网络中插入可训练的前缀：

```python
# P-Tuning v2 配置
from peft import PromptTuningConfig, get_peft_model

ptv2_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,        # 虚拟token数量
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="这是一个关于XX领域的问题：" 
)

model = get_peft_model(base_model, ptv2_config)
```

---

## 7. 技术选型决策树

```
┌──────────────────────────────────────────────┐
│         你的硬件配置是什么？                  │
└────────────────────┬───────────────────────┘
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
  RTX 4090      A100 40GB      A100 80GB×8+
  (24GB)        (80GB)         (640GB+)
      │              │              │
      ▼              ▼              ▼
   QLoRA        LoRA          全量微调
   INT4          FP16         或
   (65B模型)    (70B模型)     LoRA+
                            多节点
```

### 详细决策规则

| 硬件 | 数据量 | 方法 | 预期效果 |
|------|--------|------|---------|
| RTX 4090 (24GB) | < 10K | QLoRA INT4 | ★★★☆☆ |
| RTX 4090 (24GB) | 10K-50K | QLoRA INT4 + 延长训练 | ★★★★☆ |
| A100 (40GB) | < 50K | LoRA FP16 | ★★★★☆ |
| A100 (40GB) | 50K-100K | LoRA + 更大rank | ★★★★★ |
| A100 (80GB) | > 100K | 全量微调或LoRA | ★★★★★ |
| 多卡集群 | ANY | DeepSpeed ZeRO-3 | ★★★★★ |

---

## 8. 进阶：混合使用策略

### 8.1 LoRA + 全量微调混合

针对不同层级使用不同策略：
```python
# 底层：LoRA（保持通用能力）
# 顶层：全量微调（学习任务特定模式）

lora_config = LoraConfig(r=8, target_modules=["q_proj", "k_proj", "v_proj"])

# + 仅对最后2层做全量微调
for name, param in model.named_parameters():
    if "model.layers.30" in name or "model.layers.31" in name:
        param.requires_grad = True
```

### 8.2 梯度累积突破显存限制

```python
# 用小batch_size + 大gradient_accumulation_steps 实现等效大批量
effective_batch_size = batch_size * gradient_accumulation_steps
# 例：batch_size=1, steps=32 → 等效batch_size=32
```

---

## 本章小结

| 方法 | 显存需求 | 计算量 | 表达能力 | 适用场景 |
|------|---------|--------|---------|---------|
| 全量微调 | 65GB+ (7B) | 最高 | 最高 | 大数据+强硬件 |
| LoRA | 23GB (7B) | 中等 | 较高 | 中等数据+单卡 |
| QLoRA | 10GB (7B) | 较低 | 中等 | 低资源+快速实验 |
| 适配器 | 15GB (7B) | 低 | 中等 | 多任务切换 |
| Freeze | 12GB (7B) | 最低 | 较低 | 小样本+快速冷启动 |

---

## 思考题

1. 为什么LoRA在保持<1%可训练参数的情况下，仍能取得接近全量微调的效果？这与神经网络「本征维度」概念有何关联？
2. QLoRA的双量化（Double Quantization）在工程上是如何实现的？它对训练稳定性有何影响？
3. 如果你有一个4张RTX 4090（每张24GB）的服务器集群，应该选择哪种微调策略？为什么？
4. 适配器微调与LoRA在本质上有何区别？它们各自的优缺点在什么场景下会更明显？