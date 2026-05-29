# 10.1 LLaMA-Factory：多模型配置与训练

## 课程概述

本节课介绍 LLaMA-Factory 这一开源微调框架，探讨如何通过统一配置管理多模型的微调流程。LLaMA-Factory 极大简化了从模型注册、数据配置到训练启动的完整链路，支持全参数微调、LoRA、QLoRA 等多种训练模式。

**学习目标**
- 理解 LLaMA-Factory 的设计理念与架构优势
- 掌握多模型的注册与配置方法
- 学会编写 YAML 训练配置文件
- 熟悉 LoRA/QLoRA 训练的参数调优
- 掌握模型导出与推理流程

**前置知识**
- 大语言模型基础（Transformer, Attention）
- 常见的参数高效微调方法（LoRA, QLoRA）
- Python 和 YAML 基础
- 命令行操作能力

---

## 1. LLaMA-Factory 概述

### 1.1 什么是 LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 是一个开源的大模型微调框架，由北京大学 NLP 实验室开发。其核心设计理念是**统一配置、一键启动**：通过声明式的 YAML 配置，用户可以无需编写代码即可完成模型微调。

```
LLaMA-Factory 核心能力：
├── 多模型支持：LLaMA, Qwen, ChatGLM, Baichuan, Mistral, DeepSeek...
├── 多种训练模式：Full-tune, LoRA, QLoRA, ArrowATTN, GaLore...
├── 统一数据格式：Instruction Dataset Format
├── 高效计算：Flash Attention, Gradient Checkpoint, DeepSpeed
├── 监控与可视化：TensorBoard, MLflow, LocalUI
└── 导出工具：合并 LoRA 权重，导出 HuggingFace 格式
```

### 1.2 为什么选择 LLaMA-Factory

| 痛点 | 传统方式 | LLaMA-Factory |
|------|----------|---------------|
| 模型配置 | 需写大量代码加载模型 | YAML 声明即可 |
| 数据格式 | 各模型格式不统一 | 统一 Instruction 格式 |
| 多模型切换 | 代码耦合，更换模型麻烦 | 配置切换，零代码改动 |
| 参数调优 | 需要理解训练框架细节 | 预设参数 + 文档说明 |
| 训练监控 | 手动集成日志 | 内置 TensorBoard/LocalUI |

### 1.3 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      LLaMA-Factory                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ Model Zoo   │   │ Data Hub    │   │ Trainer Hub │       │
│  │ 模型注册    │   │ 数据集配置   │   │ 训练器选择   │       │
│  └─────────────┘   └─────────────┘   └─────────────┘       │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          ↓                                  │
│              ┌───────────────────────┐                    │
│              │    YAML Config         │                    │
│              │    统一配置入口         │                    │
│              └───────────────────────┘                    │
│                          ↓                                  │
│              ┌───────────────────────┐                    │
│              │    Training Engine    │                    │
│              │  PyTorch + DeepSpeed   │                    │
│              └───────────────────────┘                    │
│                          ↓                                  │
│              ┌───────────────────────┐                    │
│              │   Export & Inference   │                    │
│              │   导出 + 推理服务       │                    │
│              └───────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 支持的模型与注册

### 2.1 支持的模型列表

LLaMA-Factory 支持 100+ 开源模型的微调，主要包括：

| 模型家族 | 代表模型 | 特性 |
|----------|----------|------|
| **LLaMA** | LLaMA-2, LLaMA-3, LLaMA-3.1 | 通用基座 |
| **Qwen** | Qwen-2, Qwen-2.5, Qwen-MoE | 阿里通义千问 |
| **ChatGLM** | ChatGLM-3, ChatGLM-4 | 清华大学 |
| **Baichuan** | Baichuan-2 | 百川智能 |
| **Mistral** | Mistral-7B, Mixtral-8x7B | 专家混合 |
| **DeepSeek** | DeepSeek-7B, DeepSeek-MoE | 深度求索 |
| **Yi** | Yi-6B, Yi-34B | 零一万物 |
| **Gemma** | Gemma-2B, Gemma-7B | Google |
| **Phi** | Phi-3-mini, Phi-3-medium | Microsoft |

### 2.2 模型注册机制

模型注册是让 LLaMA-Factory 识别你的模型的关键步骤：

```yaml
# examples/models/llama3_8b.yaml
### model registry example
huggingface_repo_id: meta-llama/Meta-Llama-3-8B-Instruct
output_model_type: llama3        # 模型架构类型
model_name: LLaMA-3-8B-Instruct  # 注册名称

# 本地模型路径示例
# huggingface_repo_id: /path/to/local/model
```

常用 `output_model_type` 值：
- `llama`, `llama3`, `llama3.1` - LLaMA 系列
- `qwen2`, `qwen2.5` - Qwen 系列
- `chatglm3`, `chatglm4` - ChatGLM 系列
- `baichuan2` - 百川系列
- `mistral`, `mixtral` - Mistral 系列
- `deepseek` - DeepSeek 系列

### 2.3 模型名称映射

在训练配置中使用 `model_name` 引用已注册的模型：

```yaml
# train_config.yaml
model_name: LLaMA-3-8B-Instruct  # 引用已注册的模型
```

---

## 3. 数据集配置

### 3.1 统一数据格式

LLaMA-Factory 使用统一的 Instruction 格式数据集：

```json
{
    "system": "你是一个有帮助的AI助手",
    "instruction": "请解释什么是大语言模型",
    "input": "",
    "output": "大语言模型（Large Language Model）是一种..."
}
```

多轮对话格式：

```json
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮助你的吗？"},
        {"role": "user", "content": "解释一下机器学习"},
        {"role": "assistant", "content": "机器学习是..."}
    ]
}
```

### 3.2 数据集注册

```yaml
# examples/datasets/train_demo.yaml
### dataset register example
dataset: my_dataset            # 数据集标识
file_name: data/train.json    # 数据文件路径
formatting: llama3             # 格式类型：llama3, chatglm3, qwen等
```

### 3.3 数据格式化类型

不同的模型架构需要不同的数据格式：

| formatting | 说明 | 适用模型 |
|------------|------|----------|
| `llama3` | `<\|begin_of_text\|><\|start_header_id\|>...` | LLaMA-3 |
| `chatglm3` | `[gmask]`, `[sop]` token | ChatGLM-3/4 |
| `qwen` | `<\|im_start\|>`, `<\|im_end\|>` | Qwen 系列 |
| `baichuan2` | `<reserved_106>`, `<reserved_107>` | 百川 |
| `generic` | 通用格式 | 其他模型 |

---

## 4. 训练配置 YAML

### 4.1 基础训练配置

```yaml
# train_config.yaml
### 基本信息
model_name: LLaMA-3-8B-Instruct
dataset: my_dataset
output_dir: ./output/llama3-8b-lora

### 模型配置
model_type: llama3
torch_dtype: float16           # float32, float16, bfloat16
use_flash_attention: true       # 加速attention计算

### 训练参数
learning_rate: 5.0e-5
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4  # effective batch = 16
max_grad_norm: 1.0
warmup_ratio: 0.05
weight_decay: 0.01

### LoRA 配置
use_lora: true
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
loraplus_lr_ratio: 1.0        # LoRA+ 专用学习率比例
use_rslora: false              # 使用随机LoRA秩

### 量化配置 (QLoRA)
quantization_bit: 4            # 4bit, 8bit, or none
bnb_4bit_compute_dtype: float16
bnb_4bit_use_double_quant: true

### 其他
logging_steps: 10
save_steps: 500
bf16: true                     # 使用 bf16 精度
deepspeed: examples/deepspeed/ds_config.json
```

### 4.2 关键参数详解

#### 4.2.1 学习率相关

```yaml
# 学习率配置
learning_rate: 5.0e-5          # 基础学习率

# LoRA+ 特有参数
loraplus_lr_ratio: 16.0        # LoRA+ 中 B 的学习率 = lr * ratio

# 随机 LoRA
use_rslora: true               # 自动选择随机秩（通常 8-64）
rs_lora_alpha: 16              # 等价于 lora_alpha
```

#### 4.2.2 批处理相关

```yaml
# 批处理配置
per_device_train_batch_size: 4    # 每个GPU的batch size
per_device_eval_batch_size: 4

gradient_accumulation_steps: 4    # 梯度累积步数
# 实际 batch_size = 4 * 4 * num_gpus = 16 (单卡)

# 混合精度
fp16: false                       # old way
bf16: true                        # 推荐，精度更高
```

#### 4.2.3 LoRA 专项

```yaml
# LoRA 配置
use_lora: true
lora_rank: 8                      # 秩，越大参数量越多
lora_alpha: 16                    # 缩放因子，通常 = 2 * rank
lora_dropout: 0.05
target_modules:                   # 要应用LoRA的模块
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

#### 4.2.4 QLoRA 量化

```yaml
# QLoRA 配置
quantization_bit: 4              # 量化位数
bnb_4bit_compute_dtype: float16  # 计算精度
bnb_4bit_use_double_quant: true  # 双重量化
bnb_4bit_quant_type: nf4         # 量化类型：nf4, fp4
```

### 4.3 完整配置示例

```yaml
# train_config.yaml
### 模型与数据
model_name: Qwen2.5-7B-Instruct
dataset: my_instruct_data
output_dir: ./output/qwen2.5-7b-lora

### 训练模式
stage: sft                        # sft, pt, rm, rl
torch_dtype: bfloat16
use_flash_attention: true

### 训练超参
learning_rate: 3.0e-4
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
max_grad_norm: 1.0
warmup_ratio: 0.1
scheduler: cosine

### LoRA
use_lora: true
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

### 量化 (可选)
# quantization_bit: 4
# bnb_4bit_compute_dtype: bfloat16

### 保存与监控
logging_steps: 10
save_steps: 500
eval_steps: 500
save_total_limit: 2

### DeepSpeed (可选)
# deepspeed: examples/deepspeed/ds_config.json

### 其他
report_to: tensorboard
output_dir: ./output/qwen2.5-7b-lora
```

---

## 5. 多模型训练工作流

### 5.1 完整工作流

```
Step 1: 模型注册 (Model Registration)
        定义 model_name → huggingface_repo_id 映射

Step 2: 数据准备 (Dataset Preparation)
        准备 Instruction 格式 JSON 数据集

Step 3: 配置训练 YAML
        编写包含模型、数据、训练参数的配置文件

Step 4: 启动训练
        调用 LLaMA-Factory CLI 或 Python API 启动训练

Step 5: 导出合并模型
        将 LoRA 权重合并回基座模型

Step 6: 推理验证
        使用合并后的模型进行推理
```

### 5.2 Step 1: 模型注册

创建或编辑 `examples/models/` 下的模型配置文件：

```yaml
# examples/models/qwen2_7b.yaml
huggingface_repo_id: Qwen/Qwen2.5-7B-Instruct
output_model_type: qwen2
model_name: Qwen2.5-7B-Instruct
```

### 5.3 Step 2: 数据准备

准备 Instruction 数据集：

```json
[
    {
        "instruction": "你是一个有用的AI助手",
        "input": "解释量子计算",
        "output": "量子计算是一种利用量子力学原理进行信息处理的计算方式..."
    }
]
```

注册数据集：

```yaml
# examples/datasets/my_data.yaml
dataset: my_instruct_data
file_name: data/my_data.json
formatting: qwen
```

### 5.4 Step 3 & 4: 配置与启动训练

```bash
# 使用命令行启动训练
llamafactory-cli train examples/train_full.yaml

# 或使用指定配置
llamafactory-cli train \
    --config train_config.yaml
```

Python API 方式：

```python
from llamafactory import LLaMAFactory

# 创建工厂
factory = LLaMAFactory.from_config("train_config.yaml")

# 启动训练
factory.train()
```

### 5.5 Step 5: 导出与合并

训练完成后，需要将 LoRA 权重合并到基座模型：

```bash
# 导出合并模型
llamafactory-cli export \
    --config examples/export.yaml \
    --merge_lora true

# HuggingFace 格式导出
llamafactory-cli export \
    --config examples/export.yaml \
    --save_safetensors true
```

### 5.6 Step 6: 推理验证

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载合并后的模型
model = AutoModelForCausalLM.from_pretrained(
    "./output/merged_model",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./output/merged_model")

# 推理
prompt = "解释什么是机器学习"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 6. 训练模式详解

### 6.1 全参数微调 (Full Fine-tuning)

```yaml
# full_finetune.yaml
use_lora: false                    # 关闭 LoRA
stage: sft

# 全参数训练需要更多显存
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

特点：
- 更新所有参数
- 显存需求大（至少 80G）
- 训练效果最好，但最容易过拟合

### 6.2 LoRA 微调

```yaml
# lora_train.yaml
use_lora: true
lora_rank: 8
lora_alpha: 16
target_modules:
  - q_proj
  - v_proj
```

特点：
- 只更新 LoRA 参数
- 显存需求中等（20-40G）
- 参数效率高，适合垂直领域适配

### 6.3 QLoRA 微调

```yaml
# qlora_train.yaml
use_lora: true
quantization_bit: 4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

lora_rank: 64
lora_alpha: 128
```

特点：
- 4bit 量化基座模型 + LoRA
- 显存需求最低（6-24G）
- 适合消费级 GPU

---

## 7. 高级功能

### 7.1 多模态扩展 LLaMA-Factory

LLaMA-Factory 支持多模态模型的微调（如 LLaVA）：

```yaml
# examples/train_vision.yaml
model_name: llava-1.5-7b
dataset: llava_instruct
model_type: llava

# 视觉模块
freeze_vision_layer: true
train_vision: false               # 只训练文本 adapter
```

### 7.2 RLHF 集成

LLaMA-Factory 支持完整的 RLHF 流程：

```yaml
# Reward Model 训练
stage: rm
learning_rate: 1.0e-6

# PPO 训练
stage: rl
learning_rate: 1.0e-6
gamma: 0.99
lam: 0.95
```

### 7.3 其他训练策略

| 策略 | 说明 | 配置 |
|------|------|------|
| **GaLore** | 梯度低秩投影，减少显存 | `use_galore: true` |
| **ArrowATTN** | 高效注意力机制 | `use_arrow_attn: true` |
| **LISA** | 层级化稀疏注意力 | `use_lisa: true` |

---

## 8. 常见问题与调优

### 8.1 显存不足

| 方案 | 显存降低 | 代价 |
|------|----------|------|
| 降低 batch_size | ~30% | 训练速度下降 |
| 开启 gradient checkpointing | ~40% | 反向传播变慢 |
| 使用 QLoRA 4bit | ~70% | 可能有精度损失 |
| 使用 DeepSpeed ZeRO | ~60% | 多卡通信开销 |

### 8.2 训练不稳定

```yaml
# 稳定训练配置
warmup_ratio: 0.1              # 预热步数
max_grad_norm: 0.5             # 梯度裁剪
fp32: false                    # bf16 混合精度
clip_range: 0.2                # PPO 中使用
```

### 8.3 效果不佳

检查清单：
- [ ] 数据质量：清洗异常值
- [ ] 学习率：尝试 1e-5 到 5e-5
- [ ] LoRA 秩：增大 rank（如 16→64）
- [ ] epoch 数：过拟合可减少，欠拟合可增加
- [ ] 格式化：检查 template 是否匹配模型

---

## 总结

本节课我们系统学习了 LLaMA-Factory 的核心功能：

1. **框架优势**：统一配置、多模型支持、开箱即用

2. **模型注册**：通过 YAML 声明式注册，支持 HuggingFace 和本地路径

3. **数据集格式**：统一的 Instruction 格式，支持多轮对话

4. **训练配置**：详细讲解了学习率、LoRA、QLoRA、量化等核心参数

5. **工作流**：模型注册 → 数据准备 → 配置 → 训练 → 导出 → 推理

6. **训练模式**：全参数微调、LoRA、QLoRA 各有适用场景

---

## 扩展阅读

- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory) - 官方代码库
- [LLaMA-Factory 文档](https://llamafactory.readthedocs.io/) - 详细使用指南
- [LoRA 论文](https://arxiv.org/abs/2106.09685) - LoRA 原始论文
- [QLoRA 论文](https://arxiv.org/abs/2305.14314) - QLoRA 量化原理

---

## 复习题

1. **LLaMA-Factory 如何实现"零代码"微调？其核心设计理念是什么？**

2. **在 LoRA 训练中，`lora_rank` 和 `lora_alpha` 分别控制什么？为什么通常设置 `alpha = 2 * rank`？**

3. **对比全参数微调、LoRA、QLoRA 三种模式的显存占用、参数量和适用场景。**

4. **假设你需要微调一个 Qwen-7B 模型用于客服场景，请设计训练配置（考虑使用单卡 3090, 24G 显存）。**

5. **在数据格式上，llama3、chatglm3、qwen 三种格式有什么本质区别？为什么需要不同的格式化模板？**