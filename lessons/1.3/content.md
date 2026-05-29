# 1.3 硬件选择与显存估算

## 课程概述

本课时系统讲解大模型训练与推理的硬件选型核心知识，包括GPU架构原理、显存估算公式、多卡扩展策略，以及云端与本地部署的成本效益分析。

**学习目标**
- 理解GPU核心架构（CUDA Cores、Tensor Cores、HBM）与选型关系
- 掌握各精度格式下的显存占用公式与估算方法
- 学会计算不同规模模型的训练显存需求
- 了解多卡并行策略（Tensor Parallel、Pipeline Parallel）的适用场景
- 建立云端与本地部署的决策框架

**前置知识**：第一章1.1大模型概述、1.2微调方法

---

## 1. GPU架构基础

### 1.1 GPU与CPU的核心差异

```
CPU设计理念：低延迟（Low Latency）
- 少量强大核心（High IPC）
- 大容量L3缓存
- 复杂分支预测
- 适合串行任务

GPU设计理念：高吞吐（High Throughput）
- 大量小型核心（ Thousands of Cores）
- 共享显存带宽
- 适合并行计算
- 深度学习优化的Tensor Core
```

### 1.2 CUDA Cores与Tensor Cores

**CUDA Cores（流式多处理器）**
- 基础计算单元，进行常规FP32/FP64运算
- Volta之前：CUDA Cores + FP64单元分离
- Turing/Ampere：INT32/FP32/FP64共享单元

**Tensor Cores（张量核心）**
- 专为深度学习矩阵乘法设计
- 支持混合精度计算（FP16输入，FP32累加）
- 每个Tensor Core每时钟周期完成 4×4×4 矩阵运算

| 核心类型 | FP32 TOPS | FP16 TOPS | BF16 TOPS | 主要用途 |
|---------|-----------|-----------|-----------|---------|
| CUDA Core | 1× | 2× | 2× | 通用计算 |
| Tensor Core | - | 8× | 8× | 矩阵乘法加速 |

### 1.3 HBM与GDDR显存

| 显存类型 | 带宽 | 容量 | 功耗 | 典型应用 |
|---------|------|------|------|---------|
| **HBM2e** | ~2TB/s | 40-80GB | 高 | A100/H100数据中心卡 |
| **HBM3** | ~3.35TB/s | 80-188GB | 极高 | H200/B200 |
| **GDDR6X** | ~1TB/s | 24GB | 中 | RTX 4090消费级 |

**HBM（High Bandwidth Memory）优势**
```
传统GDDR：
    GPU核心 ← PCB走线 → 显存颗粒
    带宽受限于引脚数量和信号完整性

HBM：
    GPU核心 ← 通过TSV（硅通孔）→ 堆叠显存
    3D堆叠，极短连接距离，超高带宽
```

**为什么数据中心GPU选择HBM**
1. 带宽需求：700亿参数模型训练需要 ~2TB/s 带宽
2. 能效比：HBM每比特传输能耗比GDDR6X低50%
3. 面积效率：HBM节省PCB空间，支持更大显存容量

---

## 2. NVIDIA GPU产品线对比

### 2.1 消费级vs数据中心级

| 型号 | 定位 | FP32 | FP16 (Tensor) | 显存 | 带宽 | TDP | 价格(参考) |
|------|------|------|---------------|------|------|-----|----------|
| RTX 4090 | 消费旗舰 | 82 TFLOPS | 660 TFLOPS | 24GB GDDR6X | 1TB/s | 450W | ¥15,000 |
| RTX 3090 Ti | 消费旗舰 | 40 TFLOPS | 320 TFLOPS | 24GB GDDR6X | 1TB/s | 450W | ¥10,000 |
| A10G | 数据中心 | 31 TFLOPS | 125 TFLOPS | 24GB | 200GB/s | 150W | ¥25,000 |
| A100 40GB | 数据中心 | 19.5 TFLOPS | 312 TFLOPS | 40GB HBM2 | 1.6TB/s | 400W | ¥80,000 |
| A100 80GB | 数据中心 | 19.5 TFLOPS | 624 TFLOPS | 80GB HBM2 | 2TB/s | 400W | ¥150,000 |
| H100 SXM | 数据中心 | 51 TFLOPS | 989 TFLOPS | 80GB HBM3 | 3.35TB/s | 700W | ¥250,000 |
| H200 | 数据中心 | 51 TFLOPS | 989 TFLOPS | 188GB HBM3 | 4.8TB/s | 700W | ¥400,000 |

### 2.2 关键规格解读

**RTX 4090（Ada Lovelace）**
```python
# RTX 4090 核心规格
config = {
    "architecture": "Ada Lovelace",
    "cuda_cores": 16384,
    "tensor_cores": 512,
    "memory": "24GB GDDR6X",
    "memory_bandwidth": "1TB/s",
    "l2_cache": "72MB",
    "pcie": "Gen4 x16",
}
```
- 适合：单卡微调（7B模型FP16 / 13B模型QLoRA）
- 不适合：70B以上模型全量微调

**A100 80GB（ Ampere）**
```python
# A100 核心规格
config = {
    "architecture": "Ampere",
    "cuda_cores": 6912,
    "tensor_cores": 432,
    "memory": "80GB HBM2",
    "memory_bandwidth": "2TB/s",
    "l2_cache": "40MB",
    "multi_instance_gpu": True,  # MIG支持
}
```
- 适合：70B模型LoRA微调、130B模型QLoRA
- 优势：MIG技术可分割成7个独立实例

**H100/H200（Hopper）**
```python
# H100 核心规格
config = {
    "architecture": "Hopper",
    "cuda_cores": 16896,
    "tensor_cores": 528,
    "memory": "80GB HBM3",
    "memory_bandwidth": "3.35TB/s",
    "new_features": [
        "Transformer Engine",      # FP8加速
        "DPX",                      # 动态编程加速
        "distributed lmem",         # 分布式共享内存
    ]
}
```
- 适合：千亿参数模型训练、RLHF PPO阶段
- H200独有：188GB显存，适配超大Context

---

## 3. 显存估算公式

### 3.1 精度格式与存储大小

| 精度格式 | 字节/参数 | 主要用途 | 精度损失 |
|---------|----------|---------|---------|
| **FP32** | 4B | 梯度、优化器状态 | 无 |
| **FP16** | 2B | 混合精度训练 | 轻微 |
| **BF16** | 2B | 混合精度训练（更宽动态范围） | 轻微 |
| **INT8** | 1B | 量化推理 | 中等 |
| **INT4** | 0.5B | 量化微调（QLoRA） | 较明显 |
| **INT2** | 0.25B | 实验性极致压缩 | 严重 |

**BF16 vs FP16**
```
FP16: 1bit符号 + 5bit指数 + 10bit尾数 → 动态范围窄，溢出风险高
BF16: 1bit符号 + 8bit指数 + 7bit尾数 → 动态范围与FP32相近
```

### 3.2 推理显存估算

**核心公式**：`推理显存 ≈ 参数数量 × 精度字节数`

| 模型规模 | FP16 | INT8 | INT4 |
|---------|------|------|------|
| 7B | 14GB | 7GB | 3.5GB |
| 13B | 26GB | 13GB | 6.5GB |
| 30B | 60GB | 30GB | 15GB |
| 70B | 140GB | 70GB | 35GB |
| 130B | 260GB | 130GB | 65GB |
| 671B | 1.3TB | 671GB | 335GB |

### 3.3 训练显存估算

**训练显存组成**
```
总训练显存 =
    ├── 模型权重（Frozen）: P × 2B（FP16）
    ├── 模型权重（Trainable）: P × 2B
    ├── 梯度: P × 2B
    ├── 优化器状态（AdamW）: P × 2 × 4B（FP32副本）
    ├── 激活值: B × S × L × H × 2B（混合精度）
    └── 临时缓冲区: ~2GB
```

**简化公式（全量微调FP16）**
```
训练显存 ≈ P × 20B（对于百亿参数以下模型）
```

**精确公式**
```python
def estimate_training_vram(
    params: int,           # 模型参数量（单位：Billion）
    precision: str,       # "fp16", "bf16", "fp32"
    batch_size: int,
    seq_len: int,
    layers: int,
    hidden_size: int,
    use_lora: bool = False,
    lora_rank: int = 0
) -> dict:
    bytes_per_param = {
        "fp32": 4, "bf16": 2, "fp16": 2, "int8": 1, "int4": 0.5
    }[precision]
    
    # 模型权重
    model_weights = params * 1e9 * bytes_per_param / (1024**3)  # GB
    
    # 梯度（仅Trainable部分）
    trainable_params = params
    if use_lora:
        trainable_params = params * (2 * lora_rank * 4096) / 1e9  # 简化估算
    gradients = trainable_params * bytes_per_param / (1024**3)
    
    # 优化器状态（AdamW: 2×FP32副本）
    optimizer_states = trainable_params * 4 * 2 / (1024**3)
    
    # 激活值（估算）
    # 简化的激活值估算：batch_size * seq_len * layers * hidden_size * 4 * bytes_per_param
    activations = (batch_size * seq_len * layers * hidden_size * 4 * bytes_per_param) / (1024**3)
    
    total = model_weights + gradients + optimizer_states + activations + 2  # +2 buffer
    
    return {
        "model_weights": model_weights,
        "gradients": gradients,
        "optimizer_states": optimizer_states,
        "activations": activations,
        "total_gb": total
    }
```

### 3.4 各规模模型显存需求速查

**全量微调（FP16）**
| 模型 | 模型权重 | 梯度 | 优化器状态 | 激活值 | 总计 |
|------|---------|------|-----------|--------|------|
| 7B | 14GB | 14GB | 56GB | 8-16GB | 92-100GB |
| 13B | 26GB | 26GB | 104GB | 12-24GB | 168-180GB |
| 70B | 140GB | 140GB | 560GB | 40-80GB | 920-1000GB |
| 130B | 260GB | 260GB | 1040GB | 80-160GB | 1700-1800GB |

**LoRA微调（FP16）**
| 模型 | 基础权重 | LoRA参数 | 训练显存 |
|------|---------|---------|---------|
| 7B | 14GB | ~0.02GB (r=8) | ~24GB |
| 13B | 26GB | ~0.04GB (r=8) | ~36GB |
| 70B | 140GB | ~0.1GB (r=8) | ~160GB |

**QLoRA微调（INT4）**
| 模型 | 量化权重 | LoRA参数 | 训练显存 |
|------|---------|---------|---------|
| 7B | 3.5GB | ~0.02GB | ~6GB |
| 13B | 6.5GB | ~0.04GB | ~10GB |
| 65B | 32GB | ~0.1GB | ~48GB |
| 70B | 35GB | ~0.1GB | ~52GB |

---

## 4. 多GPU扩展策略

### 4.1 数据并行（Data Parallelism）

最简单的并行方式，适合大多数场景。

```python
# DeepSpeed ZeRO-DP
configs = {
    "ZeRO-1": {
        "description": "分片优化器状态",
        "memory_per_gpu": "~50% reduction"
    },
    "ZeRO-2": {
        "description": "分片优化器状态 + 梯度",
        "memory_per_gpu": "~75% reduction"
    },
    "ZeRO-3": {
        "description": "分片所有状态（参数+梯度+优化器）",
        "memory_per_gpu": "~90% reduction"
    }
}
```

### 4.2 张量并行（Tensor Parallelism, TP）

将单一层的参数分割到多卡，适合超大模型单层。

```
       Layer N
         │
    ┌────┴────┐
    ▼         ▼
  GPU 0     GPU 1
  W[0:k]    W[k:d]
    │         │
    ▼         ▼
  MatMul    MatMul
    │         │
    └────┬────┘
         ▼
      AllReduce
```

**TP通信开销**：每层两次AllReduce
**适用场景**：单卡无法容纳单层（如70B的Attention层）

### 4.3 流水线并行（Pipeline Parallelism, PP）

将不同层分配到不同设备，减少通信量。

```
GPU 0: Layer 1-18    [====forward====]
GPU 1: Layer 19-36  [==forward==][==backward==]
GPU 2: Layer 37-54  [==forward==][==backward==]
GPU 3: Layer 55-72  [====backward====]
        │    │    │    │
        └────┴────┴────┘
           bubble
```

**PP气泡问题**：通过micro-batch填充

### 4.4 3D并行组合

| 并行策略 | 分割维度 | 通信模式 | 典型场景 |
|---------|---------|---------|---------|
| DP-only | Batch | AllReduce | 单机多卡 |
| TP-only | Layer inner | AllReduce | 超大单层 |
| PP-only | Layers | P2P | 多节点 |
| TP+DP | Layer×Batch | AllReduce+P2P | 70B单节点4卡 |
| TP+DP+PP | All 3D | AllReduce+P2P | 100B+集群 |

---

## 5. 成本效益分析

### 5.1 硬件性价比对比

| 硬件配置 | 每元能买到的TFLOPS(FP16) | 每GB显存成本 |
|---------|------------------------|-------------|
| RTX 4090 | ~44 | ~¥625 |
| A100 40GB | ~0.24 | ~¥2,000 |
| A100 80GB | ~4.2 | ~¥1,875 |
| H100 SXM | ~4.0 | ~¥2,500 |
| H200 | ~2.5 | ~¥2,128 |

**结论**：消费级GPU性价比约是数据中心的10-20倍

### 5.2 按模型规模选型

| 模型规模 | 最低成本方案 | 推荐方案 |
|---------|-------------|---------|
| 7B | RTX 4090 (24GB) | RTX 4090 |
| 13B | RTX 4090 + QLoRA | A100 40GB 或 L40 |
| 30B | A100 40GB (QLoRA) | A100 80GB (LoRA) |
| 70B | A100 80GB (QLoRA) | 2×A100 80GB (LoRA) |
| 130B | H100 80GB (QLoRA) | 4×H100 (TP+PP) |

### 5.3 云端vs本地部署

| 因素 | 云端 (AWS/GCP/Azure) | 本地 |
|------|---------------------|------|
| **初始成本** | 低（按需付费） | 高（采购硬件） |
| **灵活性** | 随时扩展 | 固定容量 |
| **延迟** | 受网络影响 | 本地无延迟 |
| **数据安全** | 需评估合规 | 完全可控 |
| **大规模长期训练** | 成本高 | 成本可控 |
| **小规模实验** | 成本低 | 资源浪费 |
| **主流场景推荐** | 概念验证、小规模实验 | 生产级训练、敏感数据 |

**成本临界点估算**
```
假设：
- RTX 4090电费：¥0.5/度，450W，连续运行
- A100 80GB云端：¥15/小时

计算：
- RTX 4090 月电费：450W × 24h × 30天 × ¥0.5 = ¥162
- A100 月租（24h运行）：¥15 × 24 × 30 = ¥10,800

临界点：云端成本 ≈ 本地电费时，本地更具成本优势
对于高强度训练，本地约3个月回本
```

### 5.4 云厂商选择参考

| 厂商 | GPU型号 | 性价比 | 特色服务 |
|------|--------|--------|---------|
| AutoDL | A100/H100 | ★★★★☆ | 算力市场，按需租用 |
| 阿里云 | A10G/A100 | ★★★☆☆ | 弹性计算，配套完善 |
| 腾讯云 | L40/A100 | ★★★☆☆ | 吡哔云计算 |
| GCP | A100/H100 | ★★★★☆ | TPU选择多 |
| CoreWeave | H100/A100 | ★★★★★ | 深度学习优化 |

---

## 6. 实战决策框架

### 6.1 硬件选择流程

```
问题1：你的模型规模是？
    │
    ├── ≤7B → 问题2A
    ├── 7B-70B → 问题2B
    └── >70B → 问题2C
    │
问题2A（消费级）：你的显存预算是？
    ├── 24GB (RTX 4090) → QLoRA INT4
    └── 24GB × 2+ → LoRA FP16
    │
问题2B（中等规模）：你的训练方式是？
    ├── 全量微调 → 需要80GB+显存，多卡
    ├── LoRA → A100 80GB或H100
    └── QLoRA → A100 40GB或双RTX 4090
    │
问题2C（大规模）：你有集群吗？
    ├── 有 → TP+PP+DP 3D并行
    └── 无 → 考虑云端或QLoRA+gradient checkpointing
```

### 6.2 显存优化技巧

```python
# 1. 梯度检查点（Gradient Checkpointing）
# 用计算换显存，减少激活值存储
model.gradient_checkpointing_enable()

# 2. 混合精度训练（Mixed Precision）
# FP16计算，FP32优化器状态
trainer = Trainer(
    model=model,
    training_args=TrainingArguments(
        fp16=True,           # 或 bf16=True
        optim="adamw_torch_fused",
    )
)

# 3. 梯度累积（Gradient Accumulation）
# 用小batch模拟大批量
per_device_train_batch_size=1,
gradient_accumulation_steps=32,  # 等效batch=32

# 4. DeepSpeed ZeRO
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
    }
}
```

### 6.3 快速估算工具

```python
def quick_estimator(model_size_billions, method, hardware):
    """快速显存估算（单位：GB）"""
    
    base = {
        "full_ft_fp16": model_size_billions * 20,
        "full_ft_bf16": model_size_billions * 20,
        "lora_fp16": 14 + model_size_billions * 2,
        "qlora_int8": 7 + model_size_billions * 0.3,
        "qlora_int4": 3.5 + model_size_billions * 0.1,
    }
    
    return base.get(method, "unknown_method")
```

---

## 本章小结

| 主题 | 核心要点 |
|------|---------|
| **GPU架构** | Tensor Core是深度学习加速关键；HBM提供超高带宽 |
| **NVIDIA产品线** | RTX 4090性价比最高；H100/H200适合大规模训练 |
| **显存公式** | 训练显存 ≈ 参数×20B（FP16全量微调） |
| **多卡策略** | ZeRO适合通用场景；TP适合超大单层；PP适合多节点 |
| **成本决策** | 小规模QLoRA+消费卡；大规模考虑集群或云端 |

---

## 延伸阅读

- NVIDIA A100 Whitepaper: Ampere Architecture
- NVIDIA H100 Whitepaper: Hopper Architecture
- DeepSpeed ZeRO: Memory Optimizations For Large Model Training
- Microsoft: Deep Learning Training Guide
- 知乎：RTX 4090 vs A100 深度学习性能实测

---

## 思考题

1. 为什么HBM显存比GDDR6X更适合数据中心GPU？从带宽和能效两个角度分析。
2. 假设你需要微调一个130B参数的模型，仅有单张RTX 4090（24GB），请设计一个可行的方案（可能需要组合QLoRA + 梯度检查点 + 优化技巧）。
3. 在选择云端vs本地部署时，除了硬件成本，还需要考虑哪些因素？建立一个完整的决策框架。
4. 如果使用DeepSpeed ZeRO-3进行70B模型的训练，理论上可以将显存降低到原来的约10%。请估算在ZeRO-3下，单卡A100 80GB能容纳多大的模型进行全量微调？