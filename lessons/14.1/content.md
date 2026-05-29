# 14.1 vLLM批量推理与量化部署

## 课程概述

本课程介绍如何使用vLLM实现高效的批量推理和量化模型部署。vLLM是由伯克利大学LMSYS实验室开源的高吞吐量LLM推理引擎，通过PagedAttention和连续批处理技术显著提升推理效率。本课程涵盖vLLM的核心架构、量化技术、批量推理模式以及性能优化策略。

## 学习目标

- 理解vLLM的核心技术：PagedAttention、连续批处理、KV缓存管理
- 掌握vLLM服务器的启动与OpenAI兼容API调用
- 学会使用AWQ、GPTQ、FP8等量化方法压缩模型
- 实现高效的批量推理流程
- 估算和优化vLLM的内存使用

## 前置知识

- Python编程基础
- 深度学习模型推理基本概念
- Transformer架构理解
- 至少一块GPU的运行经验

---

## 1. vLLM核心架构

### 1.1 PagedAttention

PagedAttention是vLLM的核心创新，灵感来自操作系统的虚拟内存分页管理。传统LLM推理中，KV缓存按序列存储为连续内存块，导致内存碎片化和浪费。PagedAttention将KV缓存划分为固定大小的"页"，像内存分页一样动态分配和管理。

传统方式的问题：
```
序列1: [Token0, Token1, Token2, ..., TokenN] → 预分配 (N+P) 个token空间
序列2: [Token0, Token1, Token2]             → 预分配 (N+P) 个token空间
```

PagedAttention的改进：
```
物理内存页:
Page 0: [KV of Token0, Token1, ..., Token31]    → 由序列A和B共享
Page 1: [KV of Token32, Token33, ..., Token63]  → 由序列A独享
Page 2: [KV of Token0, Token1, ..., Token31]     → 由序列B独享
```

通过分页管理，vLLM将内存利用率提升至接近100%，显著降低显存碎片化。在长上下文场景下，PagedAttention可减少高达60%的显存使用。

### 1.2 连续批处理（Continuous Batching）

传统的静态批处理需要等待整个批次所有序列完成才能处理新请求，导致GPU空闲。连续批处理（也称"迭代级批处理"）在每个时间步动态添加新序列到批次、移除已完成的序列，实现请求级的动态批处理。

批处理流程：
```
时间步 T0: [请求A, 请求B]        → GPU处理中
时间步 T1: [请求A完成, 新请求C]  → 动态替换
时间步 T2: [请求C, 请求D]        → 持续轮转
```

vLLM的连续批处理配合分页注意力，使GPU利用率常年在90%以上，吞吐量比HuggingFace Transformers高10-30倍。

### 1.3 自动前缀缓存

当多个请求共享系统提示（System Prompt）或用户提示前缀时，vLLM自动识别并缓存这些共享前缀。后续请求只需计算各自唯一的补全部分，复用已缓存的前缀KV状态。

```
请求A: [System] + [User1] + [Completion1]
请求B: [System] + [User2] + [Completion2]
        ↑共享前缀↑    ↑需各自计算↑
```

该特性对RAG和多轮对话场景特别有价值，可将首token延迟降低50%以上。

---

## 2. vLLM服务部署

### 2.1 启动vLLM服务器

vLLM提供开箱即用的OpenAI兼容API服务器。一行命令即可启动推理服务端点：

```bash
# 基础启动（FP16精度）
vllm serve Qwen/Qwen2.5-7B-Instruct

# 带量化的启动
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --quantization awq \
    --tensor-parallel-size 2 \
    --max-model-len 8192

# 自定义端口和GPU
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 32768
```

关键启动参数：

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--quantization` | 量化方法（awq/gptq/fp8/None） | 根据硬件选择 |
| `--tensor-parallel-size` | 张量并行GPU数量 | 1-8 |
| `--max-model-len` | 最大上下文长度 | 2048-131072 |
| `--gpu-memory-utilization` | GPU显存使用比例 | 0.85-0.95 |
| `--max-num-batched-tokens` | 单批次最大token数 | 32768-131072 |

### 2.2 OpenAI兼容API

vLLM服务器暴露与OpenAI API完全兼容的接口，支持Chat Completions和Completions端点：

```bash
# Chat Completions API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "你是一位专业助手"},
      {"role": "user", "content": "解释量子计算的基本原理"}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": false
  }'

# Completions API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "prompt": "量子计算的基本原理是",
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### 2.3 请求格式与参数详解

```python
# 完整请求示例
{
    "model": "Qwen2.5-7B-Instruct",     # 模型名称
    "messages": [                        # 对话消息
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "max_tokens": 512,                   # 最大生成token数
    "temperature": 0.7,                  # 采样温度（0=贪婪）
    "top_p": 0.9,                        # 核采样阈值
    "stop": ["\n\n", "User:"],           # 停止词
    "stream": false,                      # 是否流式输出
    "presence_penalty": 0.0,             # 存在惩罚
    "frequency_penalty": 0.0,            # 频率惩罚
    "repeat_penalty": 1.1,               # 重复惩罚
    "seed": 42                           # 随机种子
}
```

### 2.4 流式响应

vLLM支持Server-Sent Events（SSE）流式输出：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

stream = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "写一首诗"}],
    max_tokens=200,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 3. vLLM量化技术

### 3.1 AWQ量化（Activation-Aware Weight Quantization）

AWQ是一种针对LLM设计的4比特权重量化方法，通过考虑激活值分布而非仅权重分布来保护关键权重通道。AWQ假设权重中只有约1%的通道对模型精度贡献最大，这些通道保持高精度。

```bash
# 启动AWQ量化模型
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ
```

AWQ的核心思想：
```
传统量化: 所有权重 → 均匀映射到INT4
AWQ: 权重 × 激活分布权重 → 保护重要通道
```

### 3.2 GPTQ量化

GPTQ是一种后训练量化方法，使用反向Hessian信息指导量化，在4-8比特量化上表现优异：

```bash
# GPTQ量化启动
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ
```

### 3.3 FP8量化

FP8（8位浮点）是NVIDIA H100/H200上的原生格式，vLLM支持FP8权重和激活的混合量化：

```bash
# FP8量化启动
vllm serve Qwen/Qwen2.5-7B-Instruct-FP8
```

FP8优势：
- H100硬件原生支持，性能接近FP16
- 相比INT8，无需复杂校准数据
- 量化损失小，精度保持率高

### 3.4 INT4/INT8服务策略

| 量化方法 | 精度 | 内存压缩 | 速度 | 适用场景 |
|---------|------|---------|------|---------|
| FP16 | 16bit | 1x | 1.0x | 基线 |
| INT8 | 8bit | 2x | 1.3x | 通用场景 |
| FP8 | 8bit | 2x | 1.4x | H100/H200 |
| AWQ/INT4 | 4bit | 4x | 1.8x | 极致压缩 |
| GPTQ/INT4 | 4bit | 4x | 1.7x | 本地部署 |

---

## 4. 批量推理模式

### 4.1 OpenAI API批量请求

OpenAI Batch API允许在单次请求中提交最多1000个任务，24小时内返回结果：

```python
from openai import OpenAI

client = OpenAI(api_key="OPENAI_API_KEY")

# 构建批量请求
tasks = []
for i in range(100):
    tasks.append({
        "custom_id": f"request_{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"任务{i}的描述"}],
            "max_tokens": 256
        }
    })

# 提交批量请求
batch = client.batches.create(
    input_file=tasks,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# 轮询查询结果
result = client.batches.retrieve(batch.id)
```

### 4.2 异步批量处理

使用asyncio实现本地异步批量推理：

```python
import asyncio
from openai import AsyncOpenAI

async def process_requests(prompts: list[str]):
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    
    async def single_request(prompt: str, idx: int):
        response = await client.chat.completions.create(
            model="Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256
        )
        return idx, response.choices[0].message.content
    
    # 并发控制：最多同时N个请求
    semaphore = asyncio.Semaphore(10)
    
    async def bounded_request(prompt: str, idx: int):
        async with semaphore:
            return await single_request(prompt, idx)
    
    tasks = [bounded_request(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    
    return sorted(results, key=lambda x: x[0])
```

### 4.3 Chunked Prefill长上下文处理

处理超长上下文时，vLLM的Chunked Prefill将长序列分块预填充，避免一次性占用过多显存：

```python
# 长上下文请求（>8192 tokens）
long_prompt = "..." * 10000  # 模拟长上下文

# vLLM自动分块处理
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": long_prompt}],
    max_tokens=512,
    max_completion_tokens=2048  # 限制生成长度
)
```

---

## 5. 性能优化

### 5.1 张量并行（Tensor Parallelism）

当模型过大无法在单GPU容纳时，张量并行将模型权重分割到多GPU：

```bash
# 启动4卡并行
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9
```

张量并行工作原理：
```
GPU0: [W0_0, W0_1]  ← 保存权重分片0
GPU1: [W1_0, W1_1]  ← 保存权重分片1
GPU2: [W2_0, W2_1]  ← 保存权重分片2
GPU3: [W3_0, W3_1]  ← 保存权重分片3

前向传播：各GPU计算部分结果 → AllReduce汇总
```

### 5.2 投机解码（Speculative Decoding）

投机解码使用小型"draft"模型快速生成候选token，再由大型"target"模型验证：

```python
# 投机解码配置
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "解释量子计算"}],
    extra_body={
        "guided_decoding_chain": ["draft_model_name"]  # 需服务器配置
    }
)
```

投机解码加速比通常为2-4倍，适合高吞吐场景。

### 5.3 Beam Search与采样策略

```python
# 贪婪解码（最快）
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": prompt}],
    stream=False,
    max_tokens=100
)

# 核采样（推荐用于生成）
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

# Beam Search（质量最高，最慢）
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": prompt}],
    extra_body={"解码方式": "beam", "beam_size": 4},
    max_tokens=100
)
```

---

## 6. 内存估算

### 6.1 模型大小计算

```python
def estimate_model_size(num_parameters: int, precision: str = "fp16") -> float:
    """估算模型内存占用（GB）"""
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
        "fp8": 1,
    }
    return num_parameters * bytes_per_param.get(precision, 2) / 1e9

# 示例：Qwen2.5-7B (FP16)
params = 7_000_000_000  # 70亿参数
print(f"FP16: {estimate_model_size(params, 'fp16'):.2f} GB")   # ~14GB
print(f"INT8: {estimate_model_size(params, 'int8'):.2f} GB")   # ~7GB
print(f"INT4: {estimate_model_size(params, 'int4'):.2f} GB")   # ~3.5GB
```

### 6.2 KV缓存大小计算

```python
def estimate_kv_cache_size(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_seq_len: int,
    batch_size: int,
    precision: str = "fp16"
) -> float:
    """估算KV缓存内存（GB）"""
    bytes_per_token = num_layers * 2 * num_heads * head_dim * 2  # K和V
    bytes_per_param = {"fp16": 2, "bf16": 2, "int8": 1, "fp8": 1}
    
    total_bytes = bytes_per_token * max_seq_len * batch_size * bytes_per_param.get(precision, 2)
    return total_bytes / 1e9

# 示例：Llama3-8B
print(f"KV Cache (1024 seq, batch=32, fp16): {estimate_kv_cache_size(32, 32, 128, 1024, 32):.2f} GB")
print(f"KV Cache (4096 seq, batch=16, fp16): {estimate_kv_cache_size(32, 32, 128, 4096, 16):.2f} GB")
```

### 6.3 vLLM内存管理策略

vLLM采用分层内存管理：

```
显存分配:
┌─────────────────────────────────────────┐
│ 模型权重    │ 14GB (FP16, 7B模型)        │
├─────────────────────────────────────────┤
│ KV缓存     │ 可变 (按需分配/释放)        │
├─────────────────────────────────────────┤
│ 临时激活值 │ ~2-4GB                     │
├─────────────────────────────────────────┤
│ CUDA内核   │ ~1GB                       │
└─────────────────────────────────────────┘

--gpu-memory-utilization 控制用于模型+KV缓存的比例
剩余显存保留给激活值和临时缓冲
```

---

## 总结

本课程涵盖了vLLM批量推理与量化部署的核心知识点：

1. **PagedAttention**：通过虚拟内存分页思想管理KV缓存，将显存利用率提升至90%以上，显著减少内存碎片

2. **连续批处理**：动态添加/移除请求，保持GPU持续高效运转，吞吐量提升10-30倍

3. **量化技术**：AWQ、GPTQ、FP8等多级量化方案，在精度与内存间取得平衡

4. **批量推理**：通过OpenAI Batch API和异步并发实现高吞吐量批量处理

5. **性能优化**：张量并行、投机解码、Beam Search等技术进一步提升推理效率

---

## 延伸阅读

- [vLLM Official Documentation](https://docs.vllm.ai/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [AWQ: Activation-Aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [Continuous Batching for LLM Serving](https://www.usenix.org/conference/osdi22/presentation yu)

---

## 复习题

1. **解释PagedAttention如何减少LLM推理的显存碎片化**

2. **比较连续批处理与传统静态批处理的优缺点**

3. **在什么场景下AWQ量化优于GPTQ？反之呢？**

4. **计算：7B模型使用FP16精度，配置max_model_len=8192、batch_size=16，估算总显存需求**

5. **简述投机解码的工作原理及其适用场景**