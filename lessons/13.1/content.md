# 13.1 LLaMA/Qwen/ChatGLM/DeepSeek架构差异

## 课程概述

本课程深入分析当前主流开源大语言模型的核心架构差异，重点关注LLaMA、Qwen、ChatGLM和DeepSeek四种架构的技术特点。这些架构差异直接影响模型的能力表现、训练稳定性和对各种微调技术（尤其是LoRA）的适配效果。理解这些底层差异是进行有效模型微调的前提。

## 学习目标

- 掌握四种主流模型架构的核心组件及其功能
- 理解RMSNorm、RoPE、SwiGLU等关键技术的原理
- 比较不同架构在归一化、注意力机制、激活函数上的差异
- 分析架构差异对LoRA微调效果的影响
- 为后续课程中选择合适的微调策略奠定基础

## 先修知识

- 了解Transformer架构的基本组成（编码器/解码器、注意力机制）
- 熟悉自回归语言模型的工作原理
- 具备基础的深度学习知识（Norm、激活函数等）

---

## 1. 为什么要理解架构差异

在进行模型微调之前，理解目标模型的架构设计至关重要。不同的模型架构对微调技术有着显著的适配差异。

### 1.1 对LoRA Targeting的影响

LoRA（Low-Rank Adaptation）通过在模型的线性层旁边添加低秩矩阵来实现参数高效微调。然而，不同架构的attention层实现方式不同，决定了LoRA应该针对哪些权重矩阵进行适配：

- **标准LLaMA架构**：LoRA通常作用于Q和V的投影矩阵，部分实现也会同时作用于K和O
- **ChatGLM的多-query注意力**：K和V被多个query头共享，这意味着需要调整LoRA targeting策略
- **DeepSeek的MLA架构**：通过低秩压缩减少了KV cache，LoRA的适配方式需要相应调整

### 1.2 对功能支持的影响

某些模型架构特性决定了它们天然更适合某些任务：

| 特性 | 适用模型 | 微调考量 |
|------|---------|---------|
| 长上下文支持 | Qwen、DeepSeek | 需要注意RoPE外推的微调策略 |
| 高效推理 | ChatGLM、DeepSeek | MLA等注意力变体需要特定优化 |
| 社区生态丰富 | LLaMA | 有更多现成的LoRA变体可参考 |

---

## 2. LLaMA架构详解

LLaMA（Large Language Model Meta AI）是Meta发布的开源大语言模型基础架构，基于标准的decoder-only transformer，但在多个组件上进行了改进。

### 2.1 核心组件

**RMSNorm（Root Mean Square Normalization）**

LLaMA使用RMSNorm替代传统的LayerNorm。RMSNorm只计算均方根而不去均值，计算公式为：

```
RMSNorm(x) = x / RMS(x) * γ
RMS(x) = sqrt(mean(x^2))
```

相比LayerNorm，RMSNorm减少了约7%的计算量，同时在训练稳定性和模型性能上表现相当。

**Rotary Position Embedding（RoPE）**

LLaMA采用旋转位置编码来编码位置信息。与可学习的绝对位置编码或相对位置编码不同，RoPE通过将位置信息旋转到Query和Key向量中来实现。核心思想是将位置编码为旋转矩阵，乘以Q和K向量后，相对位置信息自然蕴含在它们的点积结果中。

```python
# RoPE的核心思想（简化版）
def rotate_qk(q, k, position_ids):
    # 将位置信息编码为旋转角度
    positions = position_ids.unsqueeze(-1)
    # 对每对(q,k)应用旋转
    cos_pos = torch.cos(positions * theta)
    sin_pos = torch.sin(positions * theta)
    # 旋转Q和K
    q_rot = q * cos_pos + rotate_half(q) * sin_pos
    k_rot = k * cos_pos + rotate_half(k) * sin_pos
    return q_rot, k_rot
```

**SwiGLU激活函数**

LLaMA使用SwiGLU（Swish-Gated Linear Unit）作为激活函数，定义为：

```
SwiGLU(x) = Swish(W₁x) * (W₂x) = (x * σ(W₁x)) * (W₂x)
```

与标准的ReLU或GELU相比，SwiGLU通过门控机制引入了可学习的非线性，提升了模型的表达能力。

### 2.2 LLaMA架构的层结构

```
LLaMA Layer:
├── Input LayerNorm (RMSNorm)
├── Attention
│   ├── QKV Linear Projection
│   ├── RoPE Position Encoding
│   ├── Scaled Dot-Product Attention
│   └── Output Linear
├── Post Attention Norm (RMSNorm)
└── Feed Forward (SwiGLU)
    ├── Gate Linear (W₁)
    ├── Up Linear (W₃)
    └── Down Linear (W₂)
```

### 2.3 在微调中的意义

LLaMA架构的简洁性和模块化设计使其成为众多下游模型的基座。Alibaba的Qwen系列、DeepSeek系列等都是在LLaMA架构基础上进行改进的。理解LLaMA是理解其他衍生架构的基础。

---

## 3. Qwen架构详解

Qwen是Alibaba发布的开源大语言模型，在LLaMA架构基础上进行了多项优化，主要体现在训练稳定性和计算效率上的提升。

### 3.1 与LLaMA的核心差异

**Pre-RMSNorm（前置归一化）**

Qwen采用了Pre-RMSNorm策略，即在attention和FFN之前应用归一化，而不是像标准LLaMA那样在之后应用。

```
LLaMA: x → Attention → Add → Norm → FFN → Add → Output
Qwen:  x → Norm → Attention → Add → Norm → FFN → Add → Output
```

Pre-RMSNorm通过将归一化前置，提高了训练初期的稳定性，特别是在大批量训练场景下。

**注意力机制的改进**

Qwen集成了Flash Attention优化，通过分块计算和IO-aware调度，显著降低了attention计算的内存复杂度（从O(N²)降至接近O(N)）。

### 3.2 架构对比

| 组件 | LLaMA | Qwen |
|------|-------|------|
| 归一化位置 | Post RMSNorm | Pre RMSNorm |
| 注意力优化 | 标准实现 | Flash Attention |
| 位置编码 | RoPE | RoPE + NTK外推 |
| FFN实现 | SwiGLU | SwiGLU |

### 3.3 对微调的影响

Qwen的Pre-RMSNorm设计意味着在应用LoRA时需要注意残差连接的位置。LoRA模块插入的位置和方式可能需要根据PreLN的特性进行微调，以确保训练的稳定性。

---

## 4. ChatGLM架构详解

ChatGLM（General Language Model）是清华大学提出的模型架构，采用了独特的GLM（General Language Model）设计理念，与标准decoder-only transformer有显著差异。

### 4.1 GLM的核心设计

GLM的核心思想是将自然语言理解任务统一建模为**自回归填孔任务（Auto-regressive Blank Infilling）**。不同于传统的语言模型，GLM：

1. 将输入文本中的连续token片段替换为\[MASK\]标记
2. 使用双向注意力对被mask的片段进行编码
3. 以自回归方式生成被mask的内容

这种设计使得ChatGLM在保持生成能力的同时，获得了类似BERT的双向理解能力。

### 4.2 ChatGLM的注意力机制

**第一层：Bi-directional Attention**

ChatGLM在前几层使用双向注意力，这与标准的decoder-only transformer（全自回归）有本质区别。双向注意力使得模型能够更好地理解输入prompt的语义。

**PostLN vs PreLN**

ChatGLM采用了**PostLN**（在残差连接之后进行归一化），这与Qwen的PreLN形成对比：

```
PostLN:     x = x + Sublayer(x); x = Norm(x)
PreLN:      x = Norm(x + Sublayer(x))
```

PostLN的结构在训练稳定性上更具挑战性，但也提供了更好的梯度流动特性。

**Multi-Query Attention（MQA）**

ChatGLM的部分变体采用MQA，其中多个Query头共享同一个Key和Value头：

- 标准MHA：每个头有独立的Q、K、V
- MQA：多个Q头共享K、V，大幅减少KV cache
- GQA（Grouped-Query Attention）：介于MHA和MQA之间

```python
# MQA vs MHA对比
MHA:  q_proj → [h_q * d],  k_proj → [h_k * d],  v_proj → [h_v * d]
MQA:  q_proj → [h_q * d],  k_proj → [1  * d],  v_proj → [1  * d]
```

### 4.3 架构差异总结表

| 特性 | LLaMA | Qwen | ChatGLM | DeepSeek |
|------|-------|------|---------|----------|
| 归一化方式 | Post RMSNorm | Pre RMSNorm | Post LN | Pre RMSNorm |
| 注意力类型 | MHA | MHA + Flash | MQA/GQA | MHA + MLA |
| 位置编码 | RoPE | RoPE + NTK | RoPE | RoPE + NTK |
| 激活函数 | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| Attention层 | 单向 | 单向 | 混合双向 | 单向 |

---

## 5. DeepSeek架构详解

DeepSeek是深度求索发布的开源大语言模型系列，其架构设计引入了多项创新技术，特别是在推理效率方面。

### 5.1 DeepSeekMoE：混合专家架构

DeepSeek采用MoE（Mixture of Experts）架构来大幅提升模型容量同时控制计算成本：

- **专家分割**：将FFN分割为多个"专家"子网络
- **Top-K路由**：每个token只激活少数专家（如Top-2）
- **细粒度专家**：将专家分割为更细粒度的子专家

```
Standard FFN:     x → FFN(x)
MoE FFN:          x → Σᵢ Gateᵢ(x) * Expertᵢ(x)
                  其中 Gateᵢ 只在 expertᵢ 属于 Top-K 时非零
```

### 5.2 MLA：Multi-head Latent Attention

MLA是DeepSeek的核心创新之一，通过低秩矩阵分解来压缩KV cache：

**标准MHA的KV cache问题**：
- 假设有h个attention头，每个头维度为d
- KV cache需要存储2 × N × h × d个参数（N为序列长度）

**MLA的压缩方案**：
```
KV_cache_MLA = W_kv @ down_project(x)  # 压缩后的低秩表示
Q = W_q @ x                              # Q保持完整维度
K = W_kv @ down_project(x)               # K来自压缩的KV
```

通过低秩投影，MLA将KV cache的存储需求从O(hd)降低到O(d')，其中d' << hd。

### 5.3 架构特性对微调的影响

DeepSeek的MoE架构意味着：

1. **专家级别的LoRA**：可以针对特定专家进行LoRA微调，而不是整个FFN
2. **路由一致性**：需要确保微调后专家的路由模式保持稳定
3. **负载均衡**：微调可能影响专家的激活分布，需要监控

MLA的低秩特性意味着注意力层的LoRA targeting需要考虑压缩后的表示维度。

---

## 6. 四种架构的综合对比

### 6.1 核心组件对比

| 维度 | LLaMA | Qwen | ChatGLM | DeepSeek |
|------|-------|------|---------|----------|
| **归一化** | RMSNorm | Pre RMSNorm | LayerNorm | Pre RMSNorm |
| **Attention** | MHA | MHA+Flash | MQA/GQA | MLA |
| **位置编码** | RoPE | RoPE+NTK | RoPE | RoPE+NTK |
| **FFN** | SwiGLU | SwiGLU | SwiGLU | SwiGLU+MoE |
| **架构类型** | Decoder-only | Decoder-only | GLM混合 | Decoder-only+MoE |

### 6.2 对LoRA微调的适配性

| 模型 | LoRA Targeting建议 | 特殊考量 |
|------|-------------------|---------|
| **LLaMA** | Q, V (可加K, O) | 成熟社区，参考最多 |
| **Qwen** | Q, V (PreLN需注意) | Flash Attention兼容性好 |
| **ChatGLM** | Q, V (MQA需调整) | 多query特性影响适配位置 |
| **DeepSeek** | Q, V (专家级可选) | MoE需要特殊的专家级LoRA |

### 6.3 推理效率对比

| 模型 | 推理内存效率 | 计算效率 | 备注 |
|------|------------|---------|------|
| LLaMA | ★★★☆☆ | ★★★☆☆ | 标准实现 |
| Qwen | ★★★☆☆ | ★★★★☆ | Flash Attention优化 |
| ChatGLM | ★★★★☆ | ★★★★☆ | MQA减少KV cache |
| DeepSeek | ★★★★★ | ★★★☆☆ | MLA+MoE显著优化 |

---

## 7. 架构差异对微调策略的启示

### 7.1 如何选择目标架构进行微调

基于架构特性，选择建议如下：

**追求社区生态丰富**：选择LLaMA系
- 大量开源LoRA可参考
- 社区支持完善
- 文档和教程丰富

**追求长上下文能力**：选择Qwen或DeepSeek
- NTK-aware RoPE外推支持更长上下文
- 适合长文本分析和摘要任务

**追求推理效率**：选择ChatGLM或DeepSeek
- MQA/MLA显著减少内存占用
- 适合边缘设备部署

### 7.2 跨模型LoRA迁移的局限

由于架构差异，直接将为一个模型训练的LoRA迁移到另一个模型往往效果不佳：

1. **归一化位置差异**：PostLN vs PreLN导致权重分布不同
2. **注意力维度差异**：MHA vs MQA vs MLA的头配置不同
3. **FFN结构差异**：标准FFN vs MoE FFN结构完全不同

建议在新模型上从头训练LoRA，即使从相似架构迁移，也需要考虑上述差异进行必要调整。

---

## 总结

本课程系统分析了LLaMA、Qwen、ChatGLM和DeepSeek四种主流开源大模型架构的核心差异：

1. **LLaMA**以简洁的模块化设计成为众多衍生架构的基座，RMSNorm、RoPE、SwiGLU的组合经过验证且效果好

2. **Qwen**在LLaMA基础上引入Pre-RMSNorm和Flash Attention优化，提高了训练稳定性和计算效率

3. **ChatGLM**采用独特的GLM设计，前几层的双向注意力和PostLN结构提供了不同的能力组合

4. **DeepSeek**通过MoE和MLA两项创新技术，在保持高性能的同时显著优化了推理效率

理解这些架构差异是制定有效微调策略的前提。下一课程将基于这些知识，学习针对不同架构的具体LoRA微调技术。

---

## 延伸阅读

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360)
- [DeepSeekMoE: Mixture-of-Experts Scaling with Hardware-aware Communication](https://arxiv.org/abs/2401.06066)
- [Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

---

## 复习题

1. **比较题**：LLaMA的Post RMSNorm和Qwen的Pre RMSNorm有什么区别？这种差异对LoRA微调有什么影响？

2. **分析题**：ChatGLM的多-query注意力（MQA）和标准的多头注意力（MHA）相比，在LoRA targeting时需要做哪些调整？

3. **简答题**：DeepSeek的MLA（Multi-head Latent Attention）是如何减少KV cache的？这一特性对微调有什么启示？

4. **综合题**：如果你需要在边缘设备上部署一个微调后的模型，应该如何在LLaMA、Qwen、ChatGLM、DeepSeek之间选择？考虑哪些因素？

5. **应用题**：某同学想将一个针对LLaMA训练的LoRA权重直接应用到Qwen模型上，你认为可行吗？需要考虑哪些架构差异？