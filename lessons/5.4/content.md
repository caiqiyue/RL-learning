# 5.4 INT8/INT4部署：TensorRT与vLLM量化推理

## 课程概述

本节课聚焦大模型量化后的实际部署问题。在完成模型量化（PTQ/QAT）后，如何将量化模型高效部署到生产环境是核心问题。TensorRT和vLLM是当前两个最重要的量化部署框架：TensorRT通过INT8核实现极低延迟，适合在线推理；vLLM通过PagedAttention实现高吞吐批量推理，适合离线批处理场景。

本节首先从量化模型的服务化挑战出发，阐明为什么需要专门的推理引擎。随后深入TensorRT的量化工作流（ONNX导出→校准→INT8引擎构建→推理），并给出FP16与INT8的性能基准对比。接着介绍vLLM的量化推理能力（AWQ/GPTQ集成、量化模型加载、批处理），并从延迟和吞吐量两个维度对比两种方案。最后讲解INT4部署（GGUF格式、llama.cpp）以及完整的端到端部署工作流。

## 学习目标

- 掌握TensorRT量化工作流的全流程：ONNX导出→TRT量化校准→INT8引擎构建→推理验证
- 理解TensorRT INT8校准的原理，了解KL散度校准与最小最大化校准的区别
- 能够使用vLLM加载量化模型（AWQ/GPTQ格式）并进行批量推理
- 理解TensorRT（最佳延迟）与vLLM（最佳吞吐）的适用场景差异，能根据场景选型
- 了解INT4量化的GGUF格式、llama.cpp部署方式以及INT4的精度衰减基准
- 掌握从训练/微调到量化部署的完整端到端工作流
- 能够根据精度需求和资源限制对比FP16/INT8/INT4的不同精度级别

## 前置知识

- 量化基本原理：INT8/INT4的量化公式、scale/zero_point概念（详见5.1-5.3节）
- 神经网络模型导出概念：PyTorch模型到ONNX格式的转换
- 深度学习推理优化的基本概念：Batching、Kernel Fusion、Memory Bound
- Linux命令行操作基础，以及对NVIDIA GPU的基本了解

---

## 1. 为什么需要专门的推理引擎

### 1.1 量化模型服务化的挑战

模型量化后直接使用PyTorch加载推理存在几个严重问题：

**动态量化 vs 静态量化**：PyTorch的`quantize_dynamic`只量化权重，激活值在推理时动态量化——这意味着每次推理都有额外的量化/反量化开销，且无法利用INT8矩阵乘法的硬件加速。

**缺乏Kernel Fusion**：PyTorch推理会将量化操作拆散到多个独立Kernel执行，量化→矩阵乘法→反量化各自独立调用CUDA核，内存带宽浪费严重。专门推理引擎会将Quantize→Matmul→Dequantize fuse成单一INT8 Kernel。

**内存布局不优化**：通用框架使用泛化的内存布局，量化后模型权重虽然变小，但内存访问模式仍然低效。TensorRT/vLLM会将权重重新排列为硬件友好的格式（如NCHW vs NHWC）。

**缺乏批处理优化**：PyTorch的batch推理是"sequential"的——一个batch处理完再处理下一个，无法掩盖数据加载延迟。vLLM的PagedAttention通过分页内存管理实现真正的并行批处理。

### 1.2 TensorRT vs vLLM：双雄定位

当前量化部署有两个主流选择，定位截然不同：

| 维度 | TensorRT | vLLM |
|------|----------|------|
| **核心优化** | Kernel Fusion + INT8硬件加速，最小化延迟 | PagedAttention + 连续批处理，最大化吞吐 |
| **擅长场景** | 在线推理，延迟敏感，单请求流 | 离线批处理，吞吐为王，多请求并发 |
| **模型格式** | 需要转换为TensorRT Engine (.engine) | 原生支持HuggingFace格式 + AWQ/GPTQ量化权重 |
| **量化方式** | 训练后量化（需要校准数据） | 自动加载量化权重（AWQ/GPTQ），无需校准 |
| **部署难度** | 高（需要CUDA环境、TRT API） | 低（pip install即可，HuggingFace无缝对接） |
| **硬件绑定** | NVIDIA GPU（CUDA生态） | NVIDIA GPU（通过PyTorch CUDA） |

**选择原则**：延迟敏感（在线服务）选TensorRT，吞吐敏感（批处理）选vLLM。两者并非互斥——可以用vLLM做离线大批量推理，用TensorRT做在线实时推理。

---

## 2. TensorRT INT8量化部署

### 2.1 TensorRT量化工作流总览

TensorRT的INT8量化部署分为四个阶段：

```
PyTorch模型 → ONNX导出 → TRT校准 → INT8 Engine构建 → 推理
     ↓                                    ↓
  .pt文件                          .engine文件（二进制）
```

**为什么需要ONNX作为中间格式？**

TensorRT不直接支持PyTorch模型，需要通过ONNX作为桥接协议。PyTorch→ONNX→TRT的流程让TensorRT能够访问模型的完整计算图。ONNX作为标准格式，保证了不同框架间的互操作性。

### 2.2 ONNX导出

PyTorch模型导出为ONNX格式的核心要点：

```python
import torch
import torch.onnx

# 假设有一个加载好的PyTorch模型
model = load_your_model()
model.eval()

# 准备一个代表性输入（用于追踪计算图）
dummy_input = torch.randn(1, 512, hidden_size).cuda()

# 导出为ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,          # ONNX算子集版本
    do_constant_folding=True,  # 常量折叠优化
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }
)
```

**注意事项**：

- `export_params=True`会将模型权重一起导出，否则只导出结构
- `opset_version=17`是较新的版本，支持更多算子
- `dynamic_axes`允许不同batch_size和sequence_length的输入
- 对于含LayerNorm等结构敏感的模型，ONNX导出可能需要特殊处理

### 2.3 TensorRT量化校准

INT8量化需要"校准"——通过一批代表性数据确定最佳的量化参数（scale）。TensorRT提供两种校准器：

#### 2.3.1 KL散度校准器（默认）

KL散度校准通过最小化量化前后的信息熵差异来确定scale：

```python
import tensorrt as trt

# 创建TRT builder
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << (trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)

# 解析ONNX模型
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# 获取校准数据（代表性数据集）
calibration_dataset = load_calibration_data()  # 建议100-500个样本

# 创建INT8校准器
calibrator = trt.Calibrator.create(
    name="INT8 calibrator",
    data_reader=calibration_dataset,
    num_inputs=len(calibration_dataset),
    quant_proj=trt.QuantizationProjecting.PROJECT_WEIGHTS
)

# 配置builder为INT8模式
builder.int8_mode = True
builder.int8_calibrator = calibrator

# 构建引擎
engine = builder.build_serialized_network(network, config)
```

#### 2.3.2 校准数据的准备

校准数据需要满足以下要求：

- **代表性**：能代表生产环境中的数据分布（不要用异常值或极端值）
- **数量**：通常100-500个样本足够，过多不会提升精度
- **格式**：与模型输入格式完全一致（batch_size=1）
- **内容**：建议从验证集中均匀采样

```python
def load_calibration_data():
    dataset = []
    data_loader = get_validation_loader(batch_size=1)
    for i, (input_ids, attention_mask) in enumerate(data_loader):
        if i >= 300:  # 取300个样本
            break
        dataset.append((input_ids.numpy(), attention_mask.numpy()))
    return dataset
```

### 2.4 TensorRT INT8推理

引擎构建完成后，加载并推理：

```python
import tensorrt as trt
import numpy as np

# 加载引擎
with open("model.engine", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

# 创建执行上下文
context = engine.create_execution_context()

# 准备输入数据（INT8输入需要整数）
input_ids = np.random.randint(0, 32000, (1, 512)).astype(np.int32)

# 分配GPU内存
d_input = cuda.mem_alloc(input_ids.nbytes)
d_output = cuda.mem_alloc(1024 * 512 * 4)  # logits

# 创建CUDA流
stream = cuda.Stream()

# 推理
cuda.memcpy_htod_async(d_input, input_ids, stream)
context.execute_async_v3(
    bindings=[int(d_input), int(d_output)],
    stream_handle=stream.handle
)
output = np.empty((1, 512), dtype=np.float32)
cuda.memcpy_dtoh_async(output, d_output, stream)
stream.synchronize()
```

### 2.5 FP16 vs INT8性能基准对比

以下是一个典型的LLaMA-7B在不同精度下的性能对比（单请求，sequence_length=512）：

| 精度 | 权重大小 | 内存占用 | 延迟 | 吞吐量(tokens/s) |
|------|---------|---------|------|----------------|
| FP32 | 26GB | ~32GB | ~800ms | ~640 |
| FP16 | 13GB | ~16GB | ~450ms | ~1137 |
| INT8 | 6.5GB | ~9GB | ~280ms | ~1830 |
| INT4 | 3.3GB | ~5GB | ~180ms | ~2844 |

**数据说明**：

- 测试环境：A100 40GB，BS=1
- 内存占用包含模型权重 + KV Cache + 激活值
- 实际数值因模型结构和硬件而异，仅供参考

**核心结论**：

- INT8相比FP16有约1.6x的加速，内存减少约50%
- INT4进一步加速但精度损失显著（见第4节）
- 延迟降低的核心原因是INT8矩阵乘法吞吐翻倍 + 内存带宽压力减半

---

## 3. vLLM量化推理

### 3.1 vLLM的量化定位

vLLM的核心优势不是"更低的延迟"，而是"更高的吞吐量"。它通过PagedAttention实现虚拟批处理——即使硬件BS=4的实际批处理，也能通过分页内存管理近乎无缝地处理更大batch。

vLLM原生支持多种量化格式，最常用的是**AWQ**（Activation-aware Weight Quantization）和**GPTQ**（Generative Pretrained Transformer Quantization）。两者都是训练后量化（PTQ），不需要额外校准数据。

### 3.2 AWQ量化原理

AWQ（Activation-aware Weight Quantization）观察到：**权重的重要性不是均匀的——与较大激活值对应的权重通道更重要，应该保留更高精度**。

AWQ的核心思想：

```
对于权重 W 和激活值 X
重要性 ∝ |W| × |X| （按通道计算）
重要的权重通道保留FP16，其他通道量化到INT4/INT8
```

**与GPTQ的区别**：GPTQ需要完整的校准数据集来确定量化参数；AWQ只需少量数据（甚至不需要），因为它依赖"激活值越大权重越重要"的观察。

### 3.3 vLLM加载AWQ量化模型

```python
from vllm import LLM, SamplingParams

# 加载AWQ量化模型（自动识别量化权重）
# 模型来源：LMDeploy、AutoAWQ等工具产出的AWQ格式模型
llm = LLM(
    model="your-awq-model-path",
    quantization="AWQ",           # 指定量化方式
    tensor_parallel_size=1,        # 多GPU并行数
    max_model_len=4096,           # 最大序列长度
    dtype="float16",             # 激活值精度
)

# 推理
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

outputs = llm.generate(["Hello, world!", "The capital of France is"], sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")
```

### 3.4 vLLM批量推理

vLLM的批量推理是其最强特性——它通过分页内存管理实现近乎零浪费的批处理：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="your-model-path", tensor_parallel_size=2)

# 批量请求（vLLM自动处理 batching）
sampling_params = SamplingParams(temperature=0.8, max_tokens=512)

prompts = [
    "Explain the theory of relativity in simple terms:",
    "Write a Python function to calculate fibonacci numbers:",
    "What are the main differences between SQL and NoSQL databases?",
    "Describe the process of photosynthesis:",
    "How does neural network backpropagation work?",
    "What is the meaning of life according to philosophy?",
    "Explain quantum computing basics:",
    "Write a haiku about programming:",
]

# 自动批处理，vLLM会尽可能填满GPU
outputs = llm.generate(prompts, sampling_params)

# 处理结果
for output in outputs:
    print(f"[{output.request_id}] {output.outputs[0].text[:100]}...")
```

**vLLM的批量优化机制**：

- **Continuous Batching**：迭代级调度，新请求动态加入正在处理的batch
- **PagedAttention**：将KV Cache分页管理，避免显存碎片
- **Prefix Caching**：共享前缀的请求复用计算结果
- **Speculative Decoding**：用小模型预测大模型输出，加速生成

### 3.5 vLLM与TensorRT的关键对比

| 维度 | vLLM | TensorRT |
|------|------|----------|
| **延迟** | 中等（延迟>TensorRT INT8） | 极低（延迟最优） |
| **吞吐** | 极高（连续批处理优化） | 中等（固定batch） |
| **内存效率** | 高（PagedAttention省显存） | 中等 |
| **量化支持** | AWQ/GPTQ自动加载 | INT8（需要校准） |
| **多模态** | 支持 | 受限 |
| **易用性** | 高（HuggingFace无缝对接） | 中（需要TRT构建流程） |
| **适用场景** | 离线批处理、高并发在线服务 | 低延迟在线推理 |

**实测参考**（LLaMA-7B，A100 40GB）：

- vLLM FP16：~1800 tokens/s（BS=32时）
- vLLM INT8：~3200 tokens/s（BS=32时）
- TensorRT INT8：~280ms 首次token延迟（BS=1）

---

## 4. INT4部署：GGUF格式与llama.cpp

### 4.1 INT4的极致压缩

INT4可以做到4x的压缩比——相比FP32的26GB，LLaMA-7B的INT4版本只需要约3.3GB。这使得模型可以部署在消费级GPU甚至移动设备上。

**INT4的内存节省原理**：

```
FP32:  1个数字 = 4字节
FP16:  1个数字 = 2字节
INT8:  1个数字 = 1字节  (4x vs FP32)
INT4:  1个数字 = 0.5字节 (8x vs FP32)
```

但INT4的精度损失是显著的。LLaMA-7B在MMLU上：FP16=58.3%，INT8=56.8%，INT4=51.2%。INT4的精度损失约7%，对于某些任务可能不可接受。

### 4.2 GGUF格式

GGUF（Generic Gradient Univeral Format）是llama.cpp团队设计的量化格式，专门针对大语言模型的INT4/INT5/INT6/INT8量化进行了优化。

**GGUF的设计目标**：

- 包含量化权重和必要的元数据（模型结构、词汇表等），单文件分发
- 支持mmap（内存映射）加载——模型可以不用全部加载到内存，按需读取
- 向前兼容，新模型结构只需更新格式版本号

**GGUF的量化方式**：GGUF采用混合精度量化策略——对权重分块（block），每个block有独立的scale。大部分权重用INT4存储，对精度敏感的某些层（如embedding输出）保留更高精度。

### 4.3 llama.cpp量化与部署

使用llama.cpp将HuggingFace格式模型转换为GGUF：

```bash
# 克隆并编译llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release

# 下载原始模型（以LLaMA-7B为例）
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf

# 量化模型（FP16 → Q4_K_M INT4）
./build/bin/llama-quantize Llama-2-7b-hf/model.safetensors Llama-2-7b-q4_k_m.gguf Q4_K_M

# 推理测试
./build/bin/llama-cli \
    -m Llama-2-7b-q4_k_m.gguf \
    -n 256 \
    -p "The capital of France is" \
    --temp 0.7
```

**GGUF量化级别说明**：

| 级别 | 说明 | 内存节省 | 精度 |
|------|------|---------|------|
| Q8_0 | 完整INT8 | 75% | ~FP16 |
| Q4_K_M | 中等INT4 | 87.5% | ~85% FP16 |
| Q4_K_S | 小型INT4 | 87.5% | ~83% FP16 |
| Q3_K_M | INT3 | 91% | ~80% FP16 |
| Q2_K | INT2 | 94% | ~75% FP16 |

### 4.4 INT4精度衰减基准

不同任务下INT4相对FP16的精度保持率（参考值）：

| 任务 | FP16 | INT8 | INT4 (Q4_K_M) |
|------|------|------|--------------|
| MMLU (5-shot) | 58.3% | 56.8% | 51.2% |
| HellaSwag (10-shot) | 72.4% | 71.9% | 69.8% |
| Winogrande | 73.2% | 72.5% | 68.1% |
| ARC-Challenge | 53.2% | 52.1% | 46.8% |
| HumanEval (pass@1) | 29.9% | 28.7% | 24.6% |

**观察**：

- INT4在知识密集型任务（MMLU、ARC）上损失较大
- 在常识推理任务（HellaSwag、Winogrande）上损失相对较小
- 代码生成任务（HumanEval）对精度极为敏感，INT4损失约5%

**何时用INT4**：当内存极度受限（如单卡RTX 3080 10GB），且任务对精度要求不高时，INT4是唯一可行方案。精度敏感任务不应使用INT4。

---

## 5. 端到端部署工作流

完整的量化部署工作流分为5个阶段：

```
阶段1: 训练/微调   →  阶段2: 量化   →  阶段3: 导出   →  阶段4: 部署
```

### 阶段1：训练/微调

```python
# 使用标准流程训练模型（参考第4-5章内容）
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("base-model")
tokenizer = AutoTokenizer.from_pretrained("base-model")

# 指令微调
train_dataset = load_dataset("your-instruction-data", split="train")
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
model.save_pretrained("./fine-tuned-model")
```

### 阶段2：量化（GPTQ/AWQ）

```python
# AWQ量化（推荐，更简单）
from transformers import AutoModelForCausalLM
from awq import AutoAWQForCausalLM

model = AutoModelForCausalLM.from_pretrained("./fine-tuned-model")
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4}

awq_model = AutoAWQForCausalLM.from_pretrained(model)
awq_model.quantize(tokenizer, quant_config=quant_config)
awq_model.save_pretrained("./awq-quantized-model")
```

### 阶段3：导出到服务格式

**TensorRT路径**：

```bash
# PyTorch → ONNX → TensorRT
python export_onnx.py --model_path ./fine-tuned-model --output model.onnx
python build_trt_engine.py --onnx model.onnx --output model.engine --int8
```

**vLLM路径**：直接使用，无需额外导出。

**llama.cpp路径**：

```bash
./llama-quantize ./fine-tuned-model/model.safetensors ./quantized.gguf Q4_K_M
```

### 阶段4：部署

**TensorRT部署**（延迟敏感场景）：

```python
# deploy_trt.py
import tensorrt as trt

class TensorRTEngine:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime().deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
    
    def infer(self, input_ids):
        # 执行推理
        output = ...  # 见2.4节
        return output

# 启动服务
engine = TensorRTEngine("model.engine")
```

**vLLM部署**（高吞吐场景）：

```python
# deploy_vllm.py
from vllm import LLM, SamplingParams

llm = LLM(
    model="./awq-quantized-model",
    quantization="AWQ",
    tensor_parallel_size=1,
    max_model_len=4096,
)

# REST API服务（可用vLLM内置server）
# vLLM提供OpenAI兼容的API
# python -m vllm.entrypoints.openai.api_server --model ./awq-quantized-model
```

### 完整流程决策树

```
你的场景？
    │
    ├─ 延迟敏感 + 在线推理 → TensorRT INT8
    │       │
    │       └→ PyTorch → ONNX → TRT校准 → INT8 Engine → 部署
    │
    ├─ 吞吐为王 + 高并发 → vLLM
    │       │
    │       └→ AWQ/GPTQ量化 → 直接加载 → PagedAttention推理
    │
    └─ 极致压缩 + 边缘部署 → llama.cpp + GGUF INT4
            │
            └→ 转换为GGUF → llama.cpp推理
```

---

## 6. 内存/吞吐综合对比

| 精度 | 权重大小 | 内存占用 | 延迟(BS=1) | 吞吐(BS=32) | 精度损失 |
|------|---------|---------|------------|-------------|---------|
| FP32 | 26GB | ~32GB | ~800ms | - | 0% |
| FP16 | 13GB | ~16GB | ~450ms | ~1800 tok/s | ~0% |
| INT8 | 6.5GB | ~9GB | ~280ms | ~3200 tok/s | ~1-2% |
| INT4 | 3.3GB | ~5GB | ~180ms | ~4500 tok/s | ~5-7% |

**选型建议**：

- **FP16**：精度基准，所有量化方案与之对比
- **INT8**：最佳平衡点，50%内存节省 + 1.6x加速，精度损失小——生产环境首选
- **INT4**：极致压缩场景，精度损失较大——仅在内存极度受限时考虑

**内存计算公式**：

```
推理内存 ≈ 权重大小 + KV_cache + 激活值
KV_cache ≈ 2 × num_layers × batch_size × seq_len × hidden_size × 2(bytes for FP16)

例如 LLaMA-7B (BF16, seq_len=2048):
  权重 = 13GB (FP16)
  KV_cache = 2 × 32 × 1 × 2048 × 4096 × 2 / 1024³ ≈ 2GB
  总计 ≈ 15GB
```

---

## 总结

本节课围绕INT8/INT4量化部署展开，主要内容：

1. **推理引擎的必要性**：量化模型需要专门的推理引擎来融合量化Kernel、优化内存布局、实现高效批处理
2. **TensorRT INT8工作流**：PyTorch → ONNX导出 → TRT校准 → INT8引擎构建 → 推理，延迟最优
3. **TensorRT校准原理**：KL散度校准通过最小化量化前后的信息熵差异确定scale，需要代表性校准数据
4. **vLLM量化推理**：原生支持AWQ/GPTQ量化权重加载，通过PagedAttention实现高吞吐批处理
5. **TensorRT vs vLLM**：TensorRT最佳延迟，vLLM最佳吞吐——根据场景选型
6. **INT4与GGUF**：llama.cpp的GGUF格式实现极致压缩，但精度损失显著（约5-7%）
7. **端到端工作流**：训练/微调 → 量化（AWQ/GPTQ） → 导出 → TensorRT或vLLM部署
8. **选型决策**：INT8是生产环境首选，INT4仅在内存极度受限时考虑

---

## 扩展阅读

- TensorRT INT8文档：https://docs.nvidia.com/deeplearning/tensorrt/ — NVIDIA官方TensorRT文档
- vLLM文档：https://vllm.readthedocs.io/ — vLLM官方文档，包含量化模型加载指南
- AWQ论文：Lin et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM* — INT4量化的重要工作
- GPTQ论文：Frantar et al. (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — INT8量化经典方法
- llama.cpp：https://github.com/ggerganov/llama.cpp — GGUF格式和llama.cpp实现
- GGUF规范：https://github.com/ggerganov/ggml/blob/master/docs/gguf.md — GGUF格式详细规范

---

## 复习题

1. **TensorRT量化工作流**：描述PyTorch模型通过TensorRT INT8部署的完整工作流程，并解释为什么需要ONNX作为中间格式。

2. **校准的作用**：TensorRT的INT8校准器需要代表性数据集。解释KL散度校准的原理，以及为什么校准数据必须具有代表性。

3. **TensorRT vs vLLM选型**：一个在线客服系统需要平均延迟<300ms，另一个是离线文档分析系统需要日处理10万份文档。分别选择哪个部署方案，并说明理由。

4. **INT4精度损失分析**：给定MMLU基准测试中INT4相比FP16下降约7%的情况，分析哪些类型任务受影响最大，哪些受影响最小。从量化误差的角度解释这一现象。

5. **端到端部署设计**：设计一个完整的部署工作流，处理以下约束：单卡A100 40GB，需要同时服务延迟敏感（在线对话）和吞吐敏感（文档摘要）两种请求。需要用到哪些技术组合？