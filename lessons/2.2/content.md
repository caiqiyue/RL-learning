# 2.2 LoRA代码实现与参数配置

## 课程概述

本课时深入讲解LoRA的工程实现，从底层PyTorch实现到高层PEFT库封装，涵盖参数配置、训练流程、权重合并等完整链路。通过具体代码示例和配置模板，帮助工程师掌握LoRA微调的实战技能。

**学习目标**
- 理解LoRA的PyTorch底层实现原理
- 掌握PEFT库中LoraConfig的核心参数
- 能够构建完整的LoRA训练与推理流程
- 了解不同模型规模下的最优配置策略

**前置知识**：第二章2.1 LoRA原理解析

---

## 1. LoRA的PyTorch实现

### 1.1 为什么从底层实现开始

理解LoRA的底层实现有助于：
- 排查训练中的梯度异常问题
- 根据任务需求自定义LoRA层
- 优化推理性能时进行权重合并

### 1.2 Linear层子类实现

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    LoRA线性层的PyTorch实现
    
    核心公式: y = Wx + (scaling * BA)x
    其中 BA = lora_B @ lora_A
    """
    
    def __init__(self, 
                 original_layer: nn.Linear,
                 r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.0,
                 merge_weights: bool = False):
        """
        初始化LoRA线性层
        
        Args:
            original_layer: 原始预训练层，权重被冻结
            r: LoRA秩，决定低秩矩阵的维度
            lora_alpha: 缩放因子，用于调整BA的贡献度
            lora_dropout: LoRA分支的dropout概率
            merge_weights: 是否在训练前合并权重
        """
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights
        
        # 计算缩放因子：alpha / r，这决定了LoRA分支的权重
        self.scaling = self.lora_alpha / self.r
        
        # 仅当r > 0时创建LoRA参数
        if r > 0:
            # 原始层的输入输出维度
            in_features = original_layer.in_features
            out_features = original_layer.out_features
            
            # 初始化LoRA-A和LoRA-B矩阵
            # A: 随机初始化（正态分布）
            # B: 零初始化
            self.lora_A = nn.Parameter(torch.randn(r, in_features) / math.sqrt(r))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            
            # 可选的dropout层
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
            
            # 冻结原始层权重
            self.original_layer.weight.requires_grad = False
            if original_layer.bias is not None:
                self.original_layer.bias.requires_grad = False
        else:
            # r=0时不使用LoRA，直接透传
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：y = Wx + scaling * (BA)x
        """
        # 原始层输出
        original_output = self.original_layer(x)
        
        if self.r == 0:
            return original_output
        
        # LoRA分支输出
        # x首先通过dropout，然后与lora_A相乘，再与lora_B相乘
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        
        # 返回原始输出 + 缩放后的LoRA输出
        return original_output + self.scaling * lora_output
    
    def merge(self):
        """
        将LoRA权重合并到原始层中，用于推理加速
        
        计算: W_merged = W + scaling * (B @ A)
        合并后无需额外计算LoRA分支
        """
        if self.r == 0:
            return
            
        # 计算合并后的权重
        delta_w = self.lora_B @ self.lora_A
        merged_weight = self.original_layer.weight + self.scaling * delta_w
        
        # 更新原始层权重
        self.original_layer.weight = nn.Parameter(merged_weight)
        
        # 清空LoRA参数（可选，释放显存）
        self.lora_A = None
        self.lora_B = None
```

### 1.3 初始化细节解析

```python
# LoRA初始化的关键设计

# 1. 矩阵A初始化（随机正态分布）
self.lora_A = nn.Parameter(torch.randn(r, in_features) / math.sqrt(r))

# 目的：确保初始梯度有效传播
# 公式中的除以sqrt(r)是Xavier初始化的变体

# 2. 矩阵B初始化（零矩阵）
self.lora_B = nn.Parameter(torch.zeros(out_features, r))

# 目的：训练初期BA=0，使输出完全等于原始层输出
# 这样LoRA对预训练模型的影响从零开始逐渐增加

# 3. 为什么这样设计？
# 如果B初始为非零值，训练初期会破坏预训练模型的能力
# 零初始化让模型先保持原有能力，再逐渐学习新知识
```

---

## 2. PEFT库高层封装

### 2.1 LoraConfig参数详解

```python
from peft import LoraConfig

# 基础配置
lora_config = LoraConfig(
    # ===== 核心参数 =====
    r=8,                          # rank，低秩维度
    lora_alpha=16,                # 缩放因子
    lora_dropout=0.05,            # dropout概率
    
    # ===== 目标模块 =====
    target_modules=[              # 应用LoRA的模块列表
        "q_proj",                 # Query投影
        "k_proj",                 # Key投影  
        "v_proj",                 # Value投影
        "o_proj",                 # Output投影
        "gate_proj",              # 门控投影 (FFN入口)
        "up_proj",                # 上投影 (FFN中间)
        "down_proj",              # 下投影 (FFN出口)
    ],
    
    # ===== 偏差配置 =====
    bias="none",                  # 是否训练偏置项: "none" | "lora_only" | "all"
    
    # ===== 任务类型 =====
    task_type="CAUSAL_LM",        # 任务类型: "CAUSAL_LM" | "SEQ_CLS" | etc.
)
```

### 2.2 核心参数说明

| 参数 | 默认值 | 说明 | 选择建议 |
|------|--------|------|---------|
| `r` | 8 | LoRA秩，决定低秩矩阵的中间维度 | 8-64，效果随r增大提升但不显著 |
| `lora_alpha` | 16 | 缩放因子，实际影响力为 `alpha/r` | 通常设为 `2*r` |
| `lora_dropout` | 0.0 | LoRA分支的dropout | 0.05-0.1 可防止过拟合 |
| `target_modules` | None | 目标模块列表，None时自动检测 | 至少包含 q/v_proj |
| `bias` | "none" | 是否训练偏置项 | "none"最省显存 |

### 2.3 模型包装与可训练参数查看

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    device_map="auto",
    torch_dtype=torch.float16
)

# 创建LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 包装模型
model = get_peft_model(base_model, lora_config)

# 查看可训练参数比例
model.print_trainable_parameters()
# 输出示例:
# trainable params: 16,777,216 || all params: 7,611,485,952 || 0.220% trainable
```

### 2.4 目标模块选择策略

```python
# 不同目标模块组合的参数量对比（7B模型）

target_configs = {
    # 仅注意力层（最常用）
    "q_k_v_proj": ["q_proj", "k_proj", "v_proj"],
    # 含输出投影
    "attention_full": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # 含FFN层（最大参数量）
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
}

# 推荐策略：
# - 资源受限：仅 q_proj, v_proj
# - 常规场景：q_proj, k_proj, v_proj, o_proj  
# - 最大效果：全部target_modules
```

---

## 3. 完整训练流程

### 3.1 训练配置与优化器设置

```python
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model

# LoRA训练的关键区别：学习率通常更高
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # 梯度累积，等效batch=16
    learning_rate=1e-4,                 # LoRA推荐学习率: 1e-4 ~ 3e-4
    max_grad_norm=0.3,                  # 梯度裁剪
    warmup_ratio=0.03,                  # 学习率预热
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    bf16=True,                          # 使用BF16混合精度
    optim="paged_adamw_32bit",          # 分页AdamW，节省显存
)

# 获取PEFT模型
model = get_peft_model(base_model, lora_config)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
trainer.train()
```

### 3.2 常见坑：学习率差异

```python
# ⚠️ 重要：LoRA与全量微调的学习率差异

# 全量微调的学习率通常: 1e-5 ~ 3e-5
# LoRA微调的学习率通常: 1e-4 ~ 3e-4

# 为什么？
# 1. LoRA参数极少，需要更大的步长来有效学习
# 2. 全量微调时大参数空间会自然衰减学习效果
# 3. LoRA的scaling因子也影响有效学习率

# 错误示例：使用全量微调的学习率训练LoRA
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # ❌ 太低

# 正确示例
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # ✅ 合理
```

### 3.3 保存与加载LoRA适配器

```python
# 保存LoRA适配器（仅保存LoRA参数，不含原始模型）
model.save_pretrained("./lora_adapter")
# 生成 adapter_config.json 和 adapter_model.safetensors

# 加载LoRA适配器
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
model = PeftModel.from_pretrained(model, "./lora_adapter")

# 推理
output = model.generate(**inputs)
```

---

## 4. 权重合并与推理优化

### 4.1 merge_and_unload方法

```python
from peft import PeftModel

# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# 合并权重：LoRA权重合并到原始层，推理时无需额外计算
merged_model = model.merge_and_unload()

# 推理（与原始模型调用方式完全相同）
output = merged_model.generate(**inputs)
```

### 4.2 合并原理

```python
# 合并操作的数学本质
# 原始: y = Wx + scaling * (B @ A) @ x
# 合并后: y = (W + scaling * B @ A) @ x

# 权重合并步骤：
# 1. 计算 delta_w = scaling * (B @ A)
# 2. W_merged = W + delta_w  
# 3. 用W_merged替换W
# 4. 删除A、B矩阵，释放显存

# 合并后：
# - 模型体积不变
# - 推理速度与原始模型相同
# - 无法再切换回原始状态（如需保留，应在合并前保存checkpoint）
```

### 4.3 Unload vs Merge

| 方法 | 适用场景 | 优缺点 |
|------|---------|--------|
| `merge_and_unload()` | 最终部署，需要最快推理速度 | 无法切换适配器，合并不可逆 |
| `unermerge()` | 需要在多个适配器间切换 | 推理时需额外计算，可动态切换 |

---

## 5. 不同模型规模的配置模板

### 5.1 小模型 (1B-3B)

```python
# 推荐配置：r=16, 全部层应用
lora_config_small = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# 显存: ~8GB (FP16) 可训练参数量: ~1-2%
```

### 5.2 中模型 (7B)

```python
# 推荐配置：r=8, 注意力层
lora_config_medium = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none", 
    task_type="CAUSAL_LM",
)
# 显存: ~23GB (FP16) 可训练参数量: ~0.2%
```

### 5.3 大模型 (13B)

```python
# 推荐配置：r=8, 注意力层 + 部分FFN
lora_config_large = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj"],  # 不含down_proj减少参数
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# 显存: ~40GB (FP16) 可训练参数量: ~0.3%
```

### 5.4 超大模型 (70B+)

```python
# 推荐配置：r=4, 仅QKV
lora_config_xlarge = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],  # 最精简配置
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# 显存: ~160GB (FP16) or ~50GB (QLoRA INT4)
# 可训练参数量: ~0.05%
```

### 5.5 配置速查表

| 模型规模 | 参数量 | 推荐r | 目标模块 | 显存(FP16) | 可训练参数 |
|---------|--------|-------|---------|-----------|-----------|
| 1B | 1B | 16 | 全部 | ~6GB | ~1.5% |
| 7B | 7B | 8 | Attention | ~23GB | ~0.2% |
| 13B | 13B | 8 | Attn+部分FFN | ~40GB | ~0.3% |
| 70B | 70B | 4 | 仅QKV | ~160GB | ~0.05% |

---

## 6. 完整代码示例

### 6.1 端到端训练脚本

```python
"""
LoRA微调完整脚本
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# 1. 加载模型与分词器
model_name = "Qwen/Qwen2.5-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 配置LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. 包装模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 16,777,216 || all params: 7,611,485,952 || 0.220% trainable

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="./lora_qwen2.5_7b",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    optim="paged_adamw_32bit",
)

# 5. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()

# 6. 保存适配器
model.save_pretrained("./lora_adapter")

# 7. （可选）合并权重用于推理
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./lora_merged_model")
```

---

## 本章小结

| 主题 | 关键要点 |
|------|---------|
| PyTorch实现 | LoRA通过两个矩阵A、B实现低秩分解，零初始化保证训练稳定性 |
| 核心参数 | r（秩）、alpha（缩放）、target_modules（目标层） |
| PEFT封装 | `LoraConfig` + `get_peft_model` 简化训练流程 |
| 学习率 | LoRA需要更高的学习率（1e-4 vs 1e-5全量微调） |
| 权重合并 | `merge_and_unload()` 实现无损推理加速 |
| 配置选择 | 小模型用大r+全层，大模型用小r+仅注意力 |

---

## 延伸阅读

1. **原始论文**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685) - 微软2021
2. **PEFT官方文档**: https://huggingface.co/docs/peft
3. **HuggingFace PEFT示例**: https://github.com/huggingface/peft
4. **QLoRA论文**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

---

## 思考题

1. 在LoRALinear的实现中，为什么矩阵B初始化为零矩阵，而矩阵A使用随机初始化？这种设计对训练稳定性有何影响？

2. 假设你有一个7B模型，显存限制为40GB，应该如何配置LoRA参数（r值、target_modules）？请给出具体配置并计算预估显存占用。

3. `merge_and_unload()`后原始LoRA参数是否还存在？为什么在实际部署中推荐使用合并后的模型？

4. 如果训练过程中发现模型几乎不学习（loss下降极慢），可能的原因是什么？如何在代码中排查和解决？