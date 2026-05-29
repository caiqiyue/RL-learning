# 2.4 QLoRA实战：单卡微调65B模型

## 课程概述

本课时将QLoRA理论转化为实践，演示如何在单张消费级GPU上微调65B参数的大语言模型。我们将详细分析显存占用，构建完整的训练流程，并提供可运行的代码示例。课程最后会使用小型模型（1B参数）进行测试，让读者在有限硬件条件下验证整个流程。

**学习目标**
- 掌握QLoRA环境配置（bitsandbytes, peft, transformers）
- 理解单卡65B模型的显存分解与优化策略
- 学会配置NF4量化与双量化参数
- 实现带梯度检查点（gradient checkpointing）的训练循环
- 掌握分页优化器与批处理策略
- 完成模型导出与评估流程

**前置知识**：第二章2.3 QLoRA原理：NF4量化与双量化

---

## 1. 环境配置

### 1.1 核心依赖

QLoRA训练需要以下核心库：

```
bitsandbytes      # 4-bit量化与分页优化器
peft              # LoRA/QLora适配器管理
transformers      # 模型加载与训练
accelerate        # 分布式训练与设备管理
datasets          # 数据集处理
torch             # PyTorch深度学习框架
```

### 1.2 安装命令

```bash
pip install bitsandbytes peft transformers accelerate datasets torch
```

### 1.3 环境验证

```python
import torch
import bitsandbytes as bnb
import peft
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"bitsandbytes: {bnb.__version__}")
print(f"peft: {peft.__version__}")
print(f"transformers: {transformers.__version__}")
```

### 1.4 硬件要求

| 模型规模 | 最低GPU显存 | 推荐GPU | 可用性 |
|---------|------------|--------|--------|
| 7B | 8GB | RTX 3060 12GB | 消费级 |
| 13B | 16GB | RTX 4090 24GB | 消费级 |
| 33B | 24GB | RTX 4090 × 1 | 高端消费级 |
| 65B | 48GB | A100 40GB/80GB | 数据中心 |

---

## 2. QLoRA配置详解

### 2.1 LoraConfig核心参数

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=64,                          # LoRA秩，越大效果越好但显存更多
    lora_alpha=16,                 # 缩放因子，通常是r的2倍
    target_modules=[               # 要应用LoRA的模块
        "q_proj", "v_proj",        # Attention的Q和V投影
        "k_proj", "o_proj",       # K和O投影（可选）
        "gate_proj", "up_proj", "down_proj"  # FFN层
    ],
    lora_dropout=0.05,             # Dropout概率
    bias="none",                   # 不训练bias
    task_type=TaskType.CAUSAL_LM   # 任务类型：因果语言模型
)
```

### 2.2 量化配置（BitsAndBytes）

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,             # 加载模型为4-bit
    bnb_4bit_quant_type="nf4",     # 量化类型：NF4
    bnb_4bit_compute_dtype="bfloat16",  # 计算精度
    bnb_4bit_use_double_quant=True,    # 启用双量化
    bnb_4bit_quant_storage="uint8"     # 量化参数存储格式
)
```

### 2.3 量化参数详解

```
load_in_4bit: 启用4-bit加载，模型权重将被量化

bnb_4bit_quant_type:
  - "nf4": NormalFloat4，针对正态分布优化
  - "fp4": Float4，均匀量化点

bnb_4bit_compute_dtype:
  - "bfloat16": 推荐，训练更稳定
  - "float16": 传统16位浮点
  - "float32": 最高精度但慢

bnb_4bit_use_double_quant:
  - True: 对scale参数再量化（节省约1GB）
  - False: scale用FP16存储

bnb_4bit_quant_storage:
  - "uint8": 8-bit存储量化参数
  - "float16": 16-bit存储（更精确但占更多显存）
```

---

## 3. 单卡65B显存分解

### 3.1 完整模型显存占用

```
QLoRA训练时65B模型显存分解（NF4 + BF16计算）：

┌─────────────────────────────────────────────────────────────┐
│ 组件                    │ 精度   │ 显存占用 │ 说明          │
├─────────────────────────────────────────────────────────────┤
│ 基座模型权重            │ NF4   │ 32.5GB  │ 65B × 0.5B    │
│ 基座模型量化参数(scale) │ INT8   │ ~0.5GB  │ 双量化压缩     │
│ LoRA A矩阵              │ BF16   │ ~0.1GB  │ 可训练         │
│ LoRA B矩阵              │ BF16   │ ~0.1GB  │ 可训练         │
│ 梯度                    │ BF16   │ ~0.2GB  │ 仅LoRA参数    │
│ 优化器状态              │ FP32   │ ~0.4GB  │ 分页到CPU     │
│ 激活值                  │ BF16   │ ~8GB    │ batch_size=1  │
│ KV Cache               │ BF16   │ ~4GB    │ 序列长度1024  │
│ 其他开销                │ -      │ ~2GB    │ 临时缓冲等    │
├─────────────────────────────────────────────────────────────┤
│ 总计                    │        │ ~48GB   │              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 为什么能装下

关键优化：
1. **NF4量化**：130GB → 32.5GB（75%压缩）
2. **双量化**：scale参数从2GB → 0.5GB
3. **分页优化器**：优化器状态主要存CPU
4. **梯度冻结**：只存LoRA参数的梯度
5. **梯度检查点**：用计算换显存

### 3.3 不同GPU配置对比

| GPU型号 | 显存 | 可加载模型 | 备注 |
|---------|------|-----------|------|
| RTX 3060 | 12GB | 7B | 需极致优化 |
| RTX 4090 | 24GB | 13B | 推荐入门 |
| A100 40GB | 40GB | 33B | 入门级 |
| A100 80GB | 80GB | 65B | 标准配置 |

---

## 4. 模型加载与适配器绑定

### 4.1 完整加载流程

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from bitsandbytes import BitsAndBytesConfig

def load_qlora_model(model_name, lora_config):
    # Step 1: 配置量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True
    )
    
    # Step 2: 加载模型（量化格式）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Step 3: 配置LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model
```

### 4.2 device_map策略

```python
# 策略1：自动分布（多卡）
device_map="auto"      # 自动分配到多卡

# 策略2：单卡统一
device_map={"": 0}     # 全部在GPU 0

# 策略3：自定义分布
device_map={
    "model.embed_tokens": 0,
    "model.layers": "sequential",  # 顺序分配
    "lm_head": 0
}
```

### 4.3 可训练参数打印

```
trainable params: 16,874,496 || all params: 6,738,415,872 || trainable%: 0.25%
```

这意味着对于65B模型，只有约0.25%的参数是可训练的！

---

## 5. 训练流程实现

### 5.1 完整训练脚本结构

```python
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class QLoRATrainer:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        
    def setup_training(self):
        # Step 1: 准备量化模型训练
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Step 2: 添加LoRA适配器
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)
        
    def train(self, output_dir, batch_size, epochs):
        # Step 3: 配置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_steps=100,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,  # 关键优化
            optim="paged_adamw_32bit",    # 分页优化器
            save_total_limit=2
        )
        
        # Step 4: 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Step 5: 开始训练
        trainer.train()
```

### 5.2 关键训练参数详解

| 参数 | 值 | 说明 |
|------|-----|------|
| `per_device_train_batch_size` | 1-4 | 根据显存调整，65B通常用1 |
| `gradient_accumulation_steps` | 4-16 | 用计算换显存 |
| `learning_rate` | 1e-4 ~ 3e-4 | LoRA推荐稍高 |
| `bf16` | True | BF16训练更稳定 |
| `gradient_checkpointing` | True | 节省约40%显存 |
| `optim` | `paged_adamw_32bit` | 分页优化器 |
| `lr_scheduler_type` | `cosine` | 余弦学习率 |

### 5.3 梯度检查点原理

```python
# 梯度检查点：反向传播时重新计算前向激活
# 节省显存：激活值从 ~20GB → ~8GB

model.gradient_checkpointing_enable()

# 工作原理：
# 前向传播：只存储部分激活值
# 反向传播：重新计算需要的激活值
# 权衡：用额外计算换显存
```

---

## 6. 内存优化技巧

### 6.1 梯度累积

```python
# 梯度累积允许小batch_size模拟大批量
gradient_accumulation_steps = 16
effective_batch_size = batch_size * gradient_accumulation_steps

# 例如：单卡只能batch_size=1，通过累积16步
# 实现 effective_batch_size = 16 的效果
```

### 6.2 梯度检查点

```python
# 启用梯度检查点（显存节省约40%）
model.enable_gradient_checkpointing()

# 原理：
# 不存储所有中间激活
# 反向传播时重新计算
# 计算时间增加约30%，显存减少约40%
```

### 6.3 分页优化器

```python
# 使用分页优化器（节省优化器状态显存）
training_args = TrainingArguments(
    optim="paged_adamw_32bit",  # 分页AdamW
    optim_args={"page_size": 1}  # 分页大小
)

# 原理：
# 优化器状态存储在CPU
# 按需调度到GPU
# 减少GPU显存占用约50%
```

### 6.4 序列长度截断

```python
# 减少最大序列长度（显著降低激活值显存）
max_seq_length = 512  # 1024 vs 512，显存差2倍

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = dataset.map(
    lambda examples: tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length
    ),
    batched=True
)
```

### 6.5 混合精度训练

```python
# BF16混合精度（推荐）
training_args = TrainingArguments(
    bf16=True,      # BF16计算
    fp16=False      # 不使用FP16
)

# 或FP16（传统选择）
training_args = TrainingArguments(
    fp16=True,
    bf16=False
)
```

---

## 7. 模型评估与导出

### 7.1 训练完成后合并权重

```python
from peft import PeftModel

def merge_lora_weights(base_model, lora_adapter_path, output_path):
    # 加载量化基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # 合并权重（量化格式）
    merged_model = model.merge_and_unload()
    
    # 保存合并后的模型
    merged_model.save_pretrained(output_path)
    
    return merged_model
```

### 7.2 模型评估指标

```python
from transformers import pipeline

def evaluate_model(model_path, test_dataset):
    # 创建推理pipeline
    generator = pipeline(
        "text-generation",
        model=model_path,
        device_map="auto"
    )
    
    # 评估困惑度（Perplexity）
    def calculate_perplexity(model, dataset):
        total_loss = 0
        num_tokens = 0
        
        for batch in dataset:
            inputs = tokenizer(batch["text"], return_tensors="pt")
            outputs = model(**inputs)
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].shape[1]
            num_tokens += inputs["input_ids"].shape[1]
        
        perplexity = torch.exp(torch.tensor(total_loss / num_tokens))
        return perplexity.item()
    
    perplexity = calculate_perplexity(model, test_dataset)
    return {"perplexity": perplexity}
```

### 7.3 推理测试

```python
def test_generation(model_path, prompt, max_new_tokens=100):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
```

---

## 8. 实战：运行小型模型测试

### 8.1 为什么先用小模型测试

```
原因：
1. 65B模型需要约48GB显存，门槛高
2. 小模型测试可以验证代码逻辑
3. 调通后再上大模型，成功率更高
4. 小模型训练快（分钟级 vs 小时级）

推荐测试模型：
- TinyLlama (1.1B): ~2GB显存
- Llama 2 7B: ~8GB显存
```

### 8.2 使用TinyLlama测试

```python
# 小模型配置
model_name = "PY007/TinyLlama-1.1B-step-50K-103k"  # 1.1B参数

lora_config = LoraConfig(
    r=16,                      # 小模型用小秩
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./tinyllama-qlora",
    num_train_epochs=1,         # 测试用1个epoch
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10
)
```

### 8.3 数据准备

```python
from datasets import load_dataset

def prepare_dataset(tokenizer, dataset_name="wikitext", split="train"):
    # 加载原始数据集
    raw_dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return dataset
```

### 8.4 完整测试流程

```bash
# 1. 安装依赖
pip install bitsandbytes peft transformers accelerate datasets torch

# 2. 运行训练
python train_qlora.py --model_name PY007/TinyLlama-1.1B-step-50K-103k

# 3. 查看训练日志
# 应显示：trainable params, loss, learning rate

# 4. 测试推理
python test_generation.py --model_path ./output/model
```

---

## 9. 代码文件说明

本课时代码目录结构：

```
lessons/2.4/code/
├── requirements.txt      # 依赖列表
├── qlora_config.py       # QLoRA配置类
└── train_qlora.py        # 完整训练脚本
```

### 9.1 requirements.txt

包含所有必需的Python包及其版本。

### 9.2 qlora_config.py

定义QLoRA训练的配置文件，包括：
- LoraConfig参数
- BitsAndBytesConfig量化配置
- 显存优化参数

### 9.3 train_qlora.py

完整可运行的训练脚本，包含：
- 模型加载（支持量化）
- 数据预处理
- 训练循环
- 模型保存与导出

---

## 本章小结

| 概念 | 核心要点 |
|------|----------|
| **QLoRA环境** | bitsandbytes + peft + transformers |
| **LoraConfig** | r/alpha/target_modules三要素 |
| **量化配置** | NF4 + BF16计算 + 双量化 |
| **65B显存分解** | 32.5GB权重 + 0.5GB scale + 8GB激活 + ~7GB其他 |
| **梯度检查点** | 用计算换显存，节省约40% |
| **分页优化器** | 优化器状态分页到CPU |
| **小模型测试** | TinyLlama 1.1B验证流程 |

**关键经验**：
1. 先用小模型（1B）验证代码流程
2. batch_size从1开始，逐步增加
3. 梯度累积可以弥补小batch_size
4. 保存checkpoint以便恢复训练

---

## 扩展阅读

1. **QLoRA原论文**：Tim Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

2. **bitsandbytes文档**：https://huggingface.co/docs/bitsandbytes

3. **PEFT库文档**：https://huggingface.co/docs/peft

4. **TinyLlama模型**：https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-103k

---

## 思考题

1. **显存优化**：假设你有一张24GB显存的RTX 4090，想微调一个13B模型，列出至少3种优化策略，并估算每种能节省多少显存。

2. **配置选择**：解释为什么QLoRA训练时推荐`bnb_4bit_compute_dtype="bfloat16"`而不是`"float16"`？两者有什么区别？

3. **分页优化器**：分析分页优化器如何影响训练速度？如果训练速度下降20%，但换来了50%的显存节省，这个交换值得吗？

4. **小模型测试**：为什么先用TinyLlama测试而不是直接用65B模型？从工程角度分析这种做法的好处和潜在风险。