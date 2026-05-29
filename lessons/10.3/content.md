# 10.3 DeepSpeed ZeRO优化与分布式训练

## 课程概述

本节课介绍DeepSpeed ZeRO（Zero Redundancy Optimizer）优化技术与分布式训练实践。随着大语言模型参数规模从数十亿增长到数千亿，单GPU显存已无法容纳完整的模型、优化器状态和梯度数据。ZeRO通过分片（sharding）技术将训练状态分布到多个GPU，有效解决显存瓶颈问题。我们将深入讲解ZeRO的三个Stage级别、DeepSpeed配置、多GPU启动方法，以及在实际项目中的应用指南。

## 学习目标

- 理解分布式训练的必要性及其与单GPU训练的核心差异
- 掌握ZeRO-1/2/3三个Stage的分片策略与内存节省效果
- 理解ZeRO-Offload如何利用CPU/NVMe扩展GPU显存容量
- 能够配置DeepSpeed的ds_config.json并启动多GPU训练
- 掌握梯度检查点与ZeRO的组合使用技巧
- 能够根据模型规模选择合适的ZeRO配置方案

## 前置知识

- 了解深度学习训练的基本原理，包括前向传播、反向传播与梯度更新
- 熟悉PyTorch的DataParallel和DistributedDataParallel
- 理解混合精度训练（FP16/BF16）的基本概念
- 有使用transformers库加载预训练模型的经验

---

## 1. 为什么需要分布式训练

### 1.1 大模型时代的显存挑战

当前大语言模型的参数规模呈指数级增长：

| 模型 | 参数量 | FP16权重大小 | 全量训练显存需求 |
|-----|-------|------------|----------------|
| LLaMA-7B | 70亿 | ~14GB | ~28GB |
| LLaMA-70B | 700亿 | ~140GB | ~280GB |
| GPT-175B | 1750亿 | ~350GB | ~700GB |
| GPT-530B | 5300亿 | ~1060GB | ~2120GB |

单张NVIDIA A100 GPU只有80GB显存，H100有80GB或甚至更高配置但成本巨大。即便是最新的H100，单卡也难以容纳70B参数模型的完整训练状态。

训练显存的主要消耗来源：

- **模型权重**：FP16下每10亿参数约2GB
- **优化器状态**：Adam优化器需要存储一阶矩和二阶矩，每参数约8字节
- **梯度**：存储梯度值，每参数约2字节（FP16）
- **激活值**：前向传播中间结果，取决于序列长度和batch size

以LLaMA-70B为例，仅优化器状态就需要约560GB，远超单卡能力。

### 1.2 传统数据并行的问题

传统的DistributedDataParallel（DDP）虽然能将数据并行到多卡，但每个GPU都需要复制完整的模型副本：

```
DDP架构（4 GPU示例）：
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ GPU 0   │  │ GPU 1   │  │ GPU 2   │  │ GPU 3   │
│ 模型副本│  │ 模型副本│  │ 模型副本│  │ 模型副本│
│ 优化器  │  │ 优化器  │  │ 优化器  │  │ 优化器  │
│ 梯度副本│  │ 梯度副本│  │ 梯度副本│  │ 梯度副本│
└─────────┘  └─────────┘  └─────────┘  └─────────┘
```

这种方式的问题是：每张卡都存储完整模型和优化器状态，显存效率极低。当模型增大时，连模型权重都无法放入单卡。

### 1.3 ZeRO的核心思想

ZeRO（Zero Redundancy Optimizer）的核心思想是**分片**：将训练状态（优化器状态、梯度、模型参数）分割存储在多个GPU上，每个GPU只保存完整状态的一部分，消除显存冗余。

```
ZeRO架构（4 GPU，ZeRO-3示例）：
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ GPU 0   │  │ GPU 1   │  │ GPU 2   │  │ GPU 3   │
│ 1/4参数 │  │ 1/4参数 │  │ 1/4参数 │  │ 1/4参数 │
│ 1/4优化器│  │ 1/4优化器│  │ 1/4优化器│  │ 1/4优化器│
│ 1/4梯度 │  │ 1/4梯度 │  │ 1/4梯度 │  │ 1/4梯度 │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
```

在需要时，通过通信聚合完整状态，完成计算后再释放。这实现了显存的线性扩展。

---

## 2. ZeRO三阶段详解

### 2.1 ZeRO-1：优化器状态分片

ZeRO-1（Stage 1）仅对优化器状态进行分片。假设有N个GPU，每个GPU只存储1/N的优化器状态（Adam的momentum和variance）。

```
ZeRO-1显存分布（4 GPU）：
- 模型参数：完整副本（每卡）
- 梯度：完整副本（每卡）
- 优化器状态：1/4分片（每卡）

内存节省：约4倍（主要节省优化器状态）
通信量：与DDP基本持平
```

ZeRO-1的配置简单，兼容性最好，适合以下场景：
- 混合精度训练时优化器状态占用大
- 模型参数不大，但batch size需要较大
- 作为入门级ZeRO配置

### 2.2 ZeRO-2：优化器状态 + 梯度分片

ZeRO-2在ZeRO-1基础上增加梯度分片。反向传播时，每个梯度片只存在于对应GPU上，不再广播到所有卡。

```
ZeRO-2显存分布（4 GPU）：
- 模型参数：完整副本（每卡）
- 梯度：1/4分片（每卡）
- 优化器状态：1/4分片（每卡）

内存节省：约4倍（同时节省梯度和优化器状态）
通信量：略有增加（all-reduce变为reduce-scatter）
```

ZeRO-2的通信模式从`all-reduce`变为`reduce-scatter`，总通信量与DDP相当，但每卡显存负担大幅降低。

### 2.3 ZeRO-3：完整状态分片

ZeRO-3对模型参数也进行分片。这是最激进的ZeRO配置，允许训练超大规模模型。

```
ZeRO-3显存分布（4 GPU）：
- 模型参数：1/4分片（每卡）
- 梯度：1/4分片（每卡）
- 优化器状态：1/4分片（每卡）

内存节省：约4倍（模型参数+梯度+优化器状态全部节省）
通信量：显著增加（需要参数片通信）
```

ZeRO-3的挑战在于通信量大幅增加。由于前向和反向传播需要完整模型，每次操作都需要动态聚合参数分片。这对网络带宽要求较高。

### 2.4 ZeRO阶段对比

| 特性 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|-----|--------|--------|--------|
| 分片内容 | 优化器状态 | 优化器状态+梯度 | 优化器+梯度+参数 |
| 显存节省 | ~4x | ~4x | ~4x |
| 通信模式 | all-reduce | reduce-scatter | all-gather+reduce-scatter |
| 通信量 | 基准 | 基准 | 显著增加 |
| 兼容性 | 最好 | 良好 | 需要模型支持 |
| 适用场景 | 入门级 | 通用 | 超大模型 |

---

## 3. ZeRO-Offload与内存扩展

### 3.1 CPU Offload技术

ZeRO-Offload是ZeRO的扩展技术，将部分数据 Offload 到CPU内存或NVMe存储，从而在有限GPU显存中训练更大模型。

```
ZeRO-Offload策略：
- 优化器状态：始终保留在GPU（如有足够显存）
- 梯度：如GPU不足，Offload到CPU
- 参数：计算时在GPU，不计算时可在CPU/NVMe
```

### 3.2 ZeRO-Offload配置

DeepSpeed提供两种主要的Offload策略：

**ZeRO-2 + CPU Offload**：
- 梯度Offload到CPU
- 计算时将梯度 Fetch 回GPU
- 优化器状态保留在GPU

**ZeRO-3 + CPU Offload**：
- 参数和梯度都 Offload
- 每个计算步骤需要频繁 CPU-GPU 传输
- 适用于超大规模模型（数百B参数）

### 3.3 NVMe Offload

对于超大规模模型（如175B+），即使CPU内存也不够用。ZeRO-3支持NVMe Offload，将数据存储在高速SSD上：

```
NVMe Offload架构：
GPU ←→ CPU ←→ NVMe SSD
        高速通道
```

NVMe Offload的带宽要求高，通常需要使用CPU-GPU互连（如PCIe 4.0 x16）才能保证效率。

### 3.4 适用场景

| 配置 | 模型规模 | 速度 | 适用场景 |
|-----|---------|------|---------|
| ZeRO-3 | 1B-70B | 最快 | 多卡、高带宽 |
| ZeRO-2 + Offload | 70B-400B | 中等 | CPU内存充足 |
| ZeRO-3 + NVMe | 400B+ | 慢 | 超大规模模型 |

---

## 4. DeepSpeed配置详解

### 4.1 基础配置结构

DeepSpeed通过JSON配置文件（通常命名为`ds_config.json`）控制训练行为：

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 4,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

### 4.2 关键配置参数

**Batch Size相关**：

| 参数 | 说明 | 计算方式 |
|-----|------|---------|
| `train_batch_size` | 单GPU的batch size × GPU数 × gradient_accumulation | 全局batch size |
| `gradient_accumulation_steps` | 梯度累积步数 | 有效batch = train_batch_size × grad_accum |
| `per_device_batch_size` | 每张卡单个设备的batch size | 通常设置为最大支持值 |

**混合精度配置**：

```json
"fp16": {
  "enabled": true,
  "loss_scale": 0,
  "loss_scale_window": 1000,
  "initial_scale_power": 16
}
```

或使用BF16（更宽动态范围，推荐用于A100/H100）：

```json
"bf16": {
  "enabled": true
}
```

**ZeRO优化配置**：

```json
"zero_optimization": {
  "stage": 3,
  "stage3_param_persistence_threshold": 1e4,
  "stage3_gather_16bit_weights_on_model_save": true,
  "stage3_prefetch_bucket_size": 1e6,
  "stage3_max_live_parameters": 1e9,
  "stage3_max_reuse_distance": 1e9
}
```

### 4.3 ZeRO-3详细配置

对于ZeRO-3，重要参数包括：

```json
"zero_optimization": {
  "stage": 3,
  "stage3_gather_16bit_weights_on_model_save": true,
  "stage3_param_persistence_threshold": 1e4,
  "stage3_max_live_parameters": 1e9,
  "stage3_max_reuse_distance": 1e9,
  "stage3_prefetch_bucket_size": 1e6,
  "stage3_minimal_scatter_gather": true
}
```

- `stage3_gather_16bit_weights_on_model_save`：保存模型时聚合分片权重
- `stage3_prefetch_bucket_size`：预取参数片的缓冲区大小
- `stage3_minimal_scatter_gather`：优化通信，使用更小的参数片通信

### 4.4 CPU Offload配置

```json
"zero_optimization": {
  "stage": 3,
  "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
  },
  "offload_param": {
    "device": "nvme",
    "pin_memory": true
  }
}
```

### 4.5 通信配置优化

对于多节点训练，网络带宽至关重要：

```json
"communication": {
  "data_type": "fp16",
  "allgather_bucket_size": 1e7,
  "reduce_scatter_bucket_size": 1e7
}
```

---

## 5. 多GPU训练启动

### 5.1 deepspeed命令行

使用DeepSpeed启动训练的基本命令：

```bash
deepspeed --num_gpus=8 train_deepspeed.py \
    --deepspeed ds_config.json \
    --model_name_or_path facebook/opt-1.3b
```

### 5.2 多节点配置

多节点训练时，需要指定节点信息和通信方式：

```bash
deepspeed --num_gpus=8 \
    --num_nodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    train_deepspeed.py
```

### 5.3 环境变量配置

也可以使用环境变量配置：

```bash
export WORLD_SIZE=16
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500

deepspeed --num_gpus=8 train_deepspeed.py
```

### 5.4 通信带宽注意事项

分布式训练的性能瓶颈往往在通信带宽：

- **同一节点内**：使用NVLink可达300GB/s，无需特别优化
- **跨节点**：需要高速网络（InfiniBand 100G+），或使用CPU-GPU通信优化
- **跨节点ZeRO-3**：通信量显著增加，带宽成为瓶颈

---

## 6. 梯度检查点与ZeRO组合

### 6.1 梯度检查点回顾

梯度检查点（Gradient Checkpointing）是一种用计算换内存的技术。前向传播时不保存所有激活值，而是在反向传播时重新计算需要的激活值。

显存节省约60-70%，代价是增加约30%的计算时间。

### 6.2 组合使用效果

ZeRO和梯度检查点是互补技术，可以叠加：

| 优化技术 | 显存节省 | 计算开销 |
|---------|---------|---------|
| ZeRO-3 | ~4x | 少量通信增加 |
| 梯度检查点 | ~2x | ~30% |
| ZeRO-3 + 检查点 | ~8x | 通信+计算双重开销 |

组合使用的配置示例：

```python
training_args = TrainingArguments(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# DeepSpeed会自动与ZeRO协调
```

### 6.3 注意事项

- 梯度检查点在ZeRO-3下效果更好，因为ZeRO-3本身显存压力最大
- 检查点与offload组合时，反向传播需要CPU-GPU传输，注意带宽
- 某些模型（如Mamba）可能不适合梯度检查点

---

## 7. 实践指南：选择合适的ZeRO配置

### 7.1 配置选择决策表

根据模型规模和硬件条件选择ZeRO配置：

| 模型规模 | GPU数量 | 推荐配置 | 备注 |
|---------|--------|---------|------|
| 7B | 1xA100 | ZeRO-2 | 足够放下7B模型 |
| 7B | 多卡 | ZeRO-3 | 加速训练 |
| 13B | 4xA100 | ZeRO-2或ZeRO-3 | 根据batch需求选择 |
| 70B | 8xA100 | ZeRO-3 | 必须ZeRO-3 |
| 70B | 4xA100 | ZeRO-3 + Offload | 内存不足 |
| 175B+ | 多节点 | ZeRO-3 + Offload | 超大规模 |

### 7.2 常见问题与解决方案

**问题1：通信成为瓶颈**

当GPU数量增加但网络带宽受限时，ZeRO-3的通信开销可能超过计算收益。

解决方案：
- 考虑ZeRO-2，通信量较小
- 使用通信重叠技术（`overlap_comm`）
- 升级网络硬件

**问题2：模型保存问题**

ZeRO-3下模型参数是分片的，直接保存只会保存分片。

解决方案：
- 设置`stage3_gather_16bit_weights_on_model_save: true`
- 使用DeepSpeed的模型保存接口
- 或在保存时手动聚合

**问题3：调试困难**

ZeRO-3的分片机制使单卡调试变得复杂。

解决方案：
- 先在ZeRO-2下验证代码正确性
- 使用单卡测试基本逻辑
- 使用DeepSpeed的日志功能

### 7.3 性能调优建议

1. **Batch Size**：尽量使用较大的batch size，ZeRO通信可以有效隐藏
2. **通信重叠**：启用`overlap_comm`让计算和通信并行
3. **混合精度**：使用BF16代替FP16，A100+H100上性能更好
4. **梯度累积**：增大gradient_accumulation_steps，弥补小batch size的不足

---

## 8. 总结

### 8.1 核心要点

- **显存瓶颈**：单GPU无法容纳70B+参数模型的完整训练状态
- **ZeRO分片**：通过分片优化器状态、梯度、参数消除显存冗余
- **三阶段选择**：ZeRO-1/2/3逐步增加分片力度，适合不同场景
- **Offload扩展**：ZeRO-Offload利用CPU/NVMe扩展显存容量
- **配置关键**：batch size、ZeRO stage、混合精度是核心配置项
- **组合优化**：梯度检查点+ZeRO可进一步降低显存

### 8.2 实践要点

- 从ZeRO-2开始，兼容性好，调试简单
- 70B+模型必须使用ZeRO-3
- 超大模型考虑ZeRO-3 + Offload
- 注意网络带宽对ZeRO-3性能的影响
- 使用DeepSpeed官方工具监测和调试

### 8.3 进阶方向

- 深入理解ZeRO-Infinity：NVMe Offload的完整实现
- 探索Pipeline Parallelism与ZeRO的结合
- 学习DeepSpeed MoE（Mixture of Experts）支持
- 研究分布式训练中的梯度压缩技术

---

## 延伸阅读

1. **DeepSpeed官方文档**：https://www.deepspeed.ai/
2. **ZeRO论文**：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
3. **DeepSpeedExamples**：GitHub上的官方示例仓库
4. **LLaMA训练博客**：Meta官方的大模型训练技术分享
5. **HuggingFace DeepSpeed集成**：https://huggingface.co/docs/transformers/main/en/deepspeed

---

## 复习题

1. **问题一**：解释ZeRO-1、ZeRO-2、ZeRO-3在分片策略上的核心差异。如果有8张GPU，理论上分别能节省多少显存？

2. **问题二**：为什么ZeRO-3的通信量显著高于ZeRO-1和ZeRO-2？在什么场景下这会成为性能瓶颈？

3. **问题三**：假设你有4张A100 GPU（每张80GB），需要训练一个130B参数的模型。请设计一个可行的ZeRO配置方案，并说明理由。

4. **问题四**：梯度检查点和ZeRO都是节省显存的技术，它们能否同时使用？组合使用时可能会遇到什么挑战？