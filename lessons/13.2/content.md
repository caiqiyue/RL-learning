# 13.2 跨模型适配器与权重转换

## 课程概述

本课程探讨跨模型适配器的转换与合并技术。由于不同模型架构的权重维度、注意力机制和 tokenizer 存在差异，直接将在一个模型上训练的 LoRA 适配器应用到另一个模型往往不可行。本课程介绍权重转换的原理、适配器合并策略（Task Vector、TIES-Merging、DARE、WARM），以及使用 PEFT 和 mergekit 工具实现跨模型适配器迁移的完整工作流程。

## 学习目标

- 理解 LoRA 适配器在不同模型间不可直接迁移的原因
- 掌握线性投影矩阵法进行跨模型权重转换的原理
- 熟悉四种适配器合并策略的算法思想与适用场景
- 能够使用 PEFT 和 mergekit 实现适配器的转换与合并
- 了解跨模型迁移的局限性及其应对方法

## 先修知识

- 理解 LoRA 微调的基本原理（参考 13.1 课程）
- 熟悉 Transformer 架构的权重矩阵结构（Q/K/V/O 投影）
- 了解不同模型架构的归一化位置差异（PostLN vs PreLN）
- 具备 Python 深度学习开发经验

---

## 1. 适配器可移植性的挑战

### 1.1 为什么 LoRA 适配器不能直接跨模型使用

LoRA 适配器的核心是在预训练模型的特定权重矩阵旁添加低秩矩阵（通常记为 A 和 B），通过 `ΔW = BA` 来近似更新。当源模型和目标模型的权重维度不一致时，适配器无法直接应用。

**维度不匹配示例**：

```
源模型 LLaMA 7B:  Q投影 [4096 x 4096], 注意力头数 h=32, 头维度 d=128
目标模型 LLaMA 13B: Q投影 [5120 x 5120], 注意力头数 h=40, 头维度 d=128
```

如果直接加载 LLaMA 7B 的 LoRA 适配器到 13B 模型，矩阵维度不兼容，会导致运行时错误。

### 1.2 架构差异的影响

除了维度差异，以下架构特性也会影响适配器迁移：

| 架构特性 | 影响 | 示例 |
|---------|------|------|
| **归一化位置** | PreLN vs PostLN 导致残差连接权重分布不同 | LLaMA vs Qwen |
| **注意力类型** | MHA vs MQA vs MLA 的 K/V 头配置不同 | ChatGLM MQA |
| **FFN 结构** | 标准 FFN vs MoE FFN 专家结构完全不同 | DeepSeek MoE |
| **Tokenizer** | 不同 tokenizer 的词表大小和 ID 映射不同 | 输出质量下降 |

### 1.3 Tokenizer 差异的特殊影响

即使成功将权重迁移到新模型，如果 tokenizer 不同，输出的 token 序列会完全乱码：

```
源 tokenizer: {"你好": 1024, "世界": 2048}
目标 tokenizer: {"你好": 512, "世界": 4096}
```

模型生成的 ID 如果直接用目标 tokenizer 解码，会产生无意义的文本。因此跨模型迁移时，tokenizer 必须保持一致或进行映射。

---

## 2. 跨模型权重转换原理

### 2.1 适用场景：相似架构间的转换

当源模型和目标模型拥有**相似的架构设计**（相同的归一化位置、相同的注意力类型、相同的 FFN 结构），只是规模不同（如 LLaMA 7B → LLaMA 13B）时，可以通过**线性投影矩阵**进行权重转换。

**核心思想**：

```
源模型权重矩阵 W_s [d_s x d_s]
目标模型权重矩阵 W_t [d_t x d_t]
投影矩阵 P [d_t x d_s]

目标域的等效权重更新: ΔW_t = P @ ΔW_s @ P^T
```

### 2.2 Q/K/V 投影的转换

对于注意力层的 Q、K、V 投影，转换需要考虑头维度对齐：

```python
# 简化版投影计算
def project_lora_weights(source_adapter, target_config, projection_matrix):
    """
    将源模型的 LoRA 权重投影到目标模型空间
    
    Args:
        source_adapter: 源模型的 LoRA 权重 (A, B 矩阵)
        target_config: 目标模型配置
        projection_matrix: Q/K/V 投影矩阵
    """
    # 对于 Q 投影: ΔW_q_t = P_q @ ΔW_q_s @ P_k^T
    # 实际实现中需要分别计算 A 和 B 的投影
    
    A_proj = projection_matrix @ source_adapter.A  # [d_t x r]
    B_proj = source_adapter.B @ projection_matrix.T  # [r x d_t]
    
    return A_proj, B_proj.T
```

### 2.3 LLaMA → Qwen 转换的特殊处理

LLaMA 和 Qwen 虽然架构相似，但存在关键差异：

1. **Pre-RMSNorm vs Post RMSNorm**：Qwen 在注意力前进行归一化
2. **RoPE 实现细节**：两者 RoPE 的 theta 参数和编码方式可能有差异

转换时需要额外处理归一化层的权重缩放：

```python
def convert_llama_to_qwen(lora_weights, model_configs):
    """
    将 LLaMA LoRA 转换为 Qwen 格式
    """
    converted = {}
    
    for name, weight in lora_weights.items():
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            # 投影到 Qwen 的维度
            proj_matrix = build_projection(name, model_configs['llama'], model_configs['qwen'])
            converted[name] = project_to_qwen_dimension(weight, proj_matrix)
        elif 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name:
            # FFN 层的维度转换
            converted[name] = project_ffn_weights(weight, model_configs)
        elif 'input_layernorm' in name or 'post_layernorm' in name:
            # 归一化层权重需要缩放
            converted[name] = scale_layernorm_weights(weight, model_configs)
    
    return converted
```

### 2.4 转换损失与质量衰减

投影转换是一个有损操作，转换后的适配器与从头训练的适配器相比会有一定性能损失：

- **投影误差**：`P @ P^T ≠ I`（当 d_s ≠ d_t 时）
- **信息丢失**：低秩近似的误差会累积
- **分布偏移**：不同归一化位置导致激活分布差异

经验表明，相似架构间的转换通常保留 70-90% 的原始性能。

---

## 3. 适配器合并技术

### 3.1 Task Vector：权重空间中的方向

Task Vector 理论将微调后的权重视为预训练权重空间中的一个向量：

```
Task Vector TV = W_finetuned - W_pretrained
```

合并的本质是找到多个任务向量的合适组合方式。

**任务向量合并的几何直觉**：

```
W_merged = W_pretrained + f(TV_1, TV_2, ..., TV_n)

其中 f 是合并函数：
- 简单平均: f = α * ΣTV_i / n
- 加权平均: f = Σα_i * TV_i
```

### 3.2 TIES-Merging：三步冲突消解

TIES-Merging（Task Vector Izzy Ensemble）专门解决多个任务向量方向冲突的问题。

**步骤 1：选择参考向量**
选取与所有任务向量总体方向最接近的作为参考。

**步骤 2：符号消解**
对于每个维度，比较各任务向量的符号，选择占多数的符号作为合并方向：

```
冲突场景:
TV_1: [+1, -2, +3]
TV_2: [+2, +1, -1]
TV_3: [+1, -1, +2]

合并符号: [+1, ?, +1]  (第二维度冲突最多)
```

**步骤 3：幅度消解**
对于符号相同的维度，选择幅度最大的任务向量作为代表。

```python
def ties_merging(task_vectors, weights=None):
    """
    TIES-Merging 实现
    
    Args:
        task_vectors: List[dict], 每个任务的权重更新
        weights: 可选的权重系数
    """
    if weights is None:
        weights = [1.0] * len(task_vectors)
    
    # Step 1: 选择参考方向（所有向量的均值方向）
    mean_direction = sum(w * tv for w, tv in zip(weights, task_vectors))
    
    # Step 2: 符号消解
    merged_sign = torch.sign(mean_direction)
    
    # Step 3: 幅度消解 - 选择变化最显著的任务向量
    merged_weights = {}
    for key in task_vectors[0].keys():
        stacked = torch.stack([tv[key] for tv in task_vectors])
        sign_votes = torch.sign(stacked.sum(dim=0))
        merged_weights[key] = sign_votes * stacked.abs().max(dim=0)[0]
    
    return merged_weights
```

### 3.3 DARE：Drop And Rescale

DARE（Drop And Rescale）通过稀疏化来实现无损合并。

**核心思想**：

1. **随机丢弃**：以一定概率 p 随机丢弃每个任务向量的权重
2. **重新缩放**：对保留下来的权重进行缩放补偿

```
α = 1 / (1 - p)  # 缩放因子

保留权重: TV_i * α
丢弃权重: 0
```

**理论保证**：当 p 选择适当值时，DARE 可以在期望意义上保持合并后模型的表达能力。

```python
def dare_merging(task_vectors, drop_prob=0.5, seed=42):
    """
    DARE 合并策略
    
    Args:
        task_vectors: 任务向量列表
        drop_prob: 丢弃概率
    """
    torch.manual_seed(seed)
    merged = {}
    scale_factor = 1 / (1 - drop_prob)
    
    for key in task_vectors[0].keys():
        stacked = torch.stack([tv[key] for tv in task_vectors])
        mask = torch.rand_like(stacked) > drop_prob
        scaled = stacked * mask.float() * scale_factor
        merged[key] = scaled.mean(dim=0)
    
    return merged
```

### 3.4 WARM：Fisher 加权平均

WARM（Weighted Averaging by Fisher Information）利用 Fisher 信息矩阵来指导权重分配。

**核心思想**： Fisher 信息反映了每个参数对模型性能的重要程度，重要的参数应该有更大的权重。

```
Fisher 信息矩阵 F ≈ E[(∇log p(y|x,θ))²]

WARM 合并: W = Σ_i (F_i / Σ_j F_j) * W_i
```

**实际实现中的简化**：通常使用对角 Fisher 信息或 batch 统计量来近似。

```python
def warm_merging(task_vectors, fisher_weights):
    """
    WARM 合并策略
    
    Args:
        task_vectors: 任务向量列表
        fisher_weights: 每个任务向量的 Fisher 对角权重 [n_tasks x n_params]
    """
    normalized_fisher = fisher_weights / fisher_weights.sum(dim=0, keepdim=True)
    
    merged = {}
    for key in task_vectors[0].keys():
        stacked = torch.stack([tv[key] for tv in task_vectors])
        weights = normalized_fisher[:, None].expand_as(stacked)
        merged[key] = (weights * stacked).sum(dim=0)
    
    return merged
```

### 3.5 四种合并策略对比

| 策略 | 适用场景 | 计算复杂度 | 理论基础 |
|------|---------|-----------|---------|
| **简单平均** | 任务向量方向一致 | O(n) | 经验 |
| **Task Vector** | 方向基本一致，幅度不同 | O(n) | 几何直觉 |
| **TIES-Merging** | 任务向量存在冲突 | O(n × d) | 投票机制 |
| **DARE** | 需要稀疏化合并结果 | O(n) | 概率论 |
| **WARM** | 有 Fisher 信息可用 | O(n × d) | 信息几何 |

---

## 4. 实际转换工作流程

### 4.1 完整工作流程概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     跨模型适配器转换流程                          │
├─────────────────────────────────────────────────────────────────┤
│  1. 提取源模型 LoRA 的 delta 权重 (ΔW = BA)                      │
│  2. 分析源模型和目标模型的架构差异                               │
│  3. 构建投影矩阵映射不同维度的权重                               │
│  4. 处理归一化层和 FFN 层的权重转换                              │
│  5. 验证转换后权重的维度一致性                                    │
│  6. 可选：使用合并策略与其他适配器组合                            │
│  7. 在目标模型上加载并测试                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 步骤详解

**步骤 1：提取 delta 权重**

```python
from peft import PeftModel

def extract_delta_weights(peft_model):
    """从 PEFT 模型提取 delta 权重"""
    delta_weights = {}
    for name, param in peft_model.named_parameters():
        if 'lora_' in name:
            # 提取 LoRA A, B 矩阵
            delta_weights[name] = param.data
    return delta_weights
```

**步骤 2：分析架构差异**

```python
def analyze_architecture_diff(source_model, target_model):
    """分析两个模型的架构差异"""
    diff_report = {
        'hidden_dim': source_model.config.hidden_size != target_model.config.hidden_size,
        'num_heads': source_model.config.num_attention_heads != target_model.config.num_attention_heads,
        'head_dim': source_model.config.head_dim != target_model.config.head_dim,
        'num_layers': source_model.config.num_hidden_layers != target_model.config.num_hidden_layers,
        'norm_position': source_model.config.post_layer_norm_prefix != target_model.config.post_layer_norm_prefix,
    }
    return diff_report
```

**步骤 3：构建投影矩阵**

```python
def build_projection_matrix(source_dim, target_dim, mode='linear'):
    """构建维度投影矩阵"""
    if mode == 'linear':
        # 线性投影
        projection = nn.Linear(source_dim, target_dim)
    elif mode == 'identity':
        # 同维度直接返回单位阵
        projection = nn.Identity()
    return projection
```

**步骤 4：合并到目标模型**

```python
def merge_into_target_model(target_model, converted_weights):
    """将转换后的权重合并到目标模型"""
    for name, weight in converted_weights.items():
        if name in target_model.state_dict():
            target_model.state_dict()[name].copy_(weight)
    return target_model
```

### 4.3 使用 mergekit 简化流程

mergekit 提供了更高层的抽象来简化适配器合并：

```python
# mergekit YAML 配置示例
# merge_llama_qwen.yaml
sources:
  - model: ./llama-7b-lora
    layer_range: [0, 32]
  - model: ./qwen-7b-lora
    layer_range: [0, 32]

merge_method: ties
density: 0.5
weight: 0.5

output_model_type: QwenForCausalLM
output_dir: ./merged-qwen-model
```

```bash
# 使用 mergekit 命令行
mergekit-yaml merge_llama_qwen.yaml
```

---

## 5. 局限性分析

### 5.1 架构差异的硬性限制

以下情况下，跨模型适配器转换**本质上不可行**：

| 限制类型 | 说明 | 示例 |
|---------|------|------|
| **注意力机制完全不同** | MHA vs MLA 的 K/V 存储结构不同 | LLaMA vs DeepSeek MLA |
| **MoE vs 标准 FFN** | 专家结构无法对应 | LLaMA vs DeepSeekMoE |
| ** tokenizer 完全不同** | 词表大小差异 > 20% | 中文模型 vs 英文模型 |
| **归一化类型不同** | RMSNorm vs LayerNorm | LLaMA vs ChatGLM |

### 5.2 软性限制（可处理但有损失）

| 限制类型 | 影响 | 处理方法 |
|---------|------|---------|
| **Tokenizer 差异较小** | 10-20% 词汇重叠 | 重新训练 embedding 层 |
| **头数不同但维度相同** | 投影误差累积 | 使用投影矩阵 |
| **层数不同** | 浅层可部分映射 | 选择对应层 |

### 5.3 性能衰减的预期

基于经验，跨模型迁移后性能的预期：

| 迁移类型 | 相似度 | 预期性能保留 |
|---------|--------|-------------|
| 同架构不同规模 | 95%+ | 85-95% |
| LLaMA ↔ Qwen | 85% | 70-85% |
| LLaMA ↔ ChatGLM | 60% | 40-60% |
| 跨家族（中文↔英文） | 30% | 不推荐 |

---

## 6. 工具生态

### 6.1 PEFT（Parameter Efficient Fine-Tuning）

PEFT 是 Hugging Face 提供的参数高效微调库，支持 LoRA、Prefix Tuning、Prompt Tuning 等多种方法。

**核心功能**：

- 加载预训练模型 + LoRA 适配器
- 保存和加载适配器权重
- 权重合并与卸载

```python
from peft import PeftModel, get_peft_model, LoraConfig

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# 添加 LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

# 合并 LoRA 权重到基础模型
merged_model = model.merge_and_unload()
```

### 6.2 mergekit

mergekit 是专门用于模型权重合并的工具，支持：

- 多模型/多适配器的合并
- 多种合并策略（TIES、DARE、WARM、简单平均）
- YAML 配置驱动的合并流程

**安装**：

```bash
pip install mergekit
```

**基本用法**：

```python
import mergekit

# 使用配置字典
config = {
    'models': ['./model1', './model2'],
    'merge_method': 'ties',
    'parameters': {'density': 0.5}
}

mergekit.merge(config, output_path='./merged_model')
```

### 6.3 其他工具

| 工具 | 用途 | 特点 |
|------|------|------|
| ** axle** | 权重合并 | 支持更复杂的图结构合并 |
| **LLM-Finetuning** | 微调框架 | 内置多种合并策略 |
| **FastChat** | 对话框架 | 支持多模型切换 |

---

## 总结

本课程系统介绍了跨模型适配器转换与合并的技术体系：

1. **可移植性挑战**：LoRA 适配器不能跨模型直接使用，原因在于权重维度不同、架构特性差异和 tokenizer 不兼容

2. **权重转换原理**：对于相似架构（如 LLaMA 7B → LLaMA 13B），可以通过线性投影矩阵将 delta 权重映射到目标模型空间，核心公式为 `ΔW_t = P @ ΔW_s @ P^T`

3. **四种合并策略**：
   - Task Vector：将微调视为权重空间的向量操作
   - TIES-Merging：通过符号投票消解冲突
   - DARE：通过随机丢弃和重新缩放实现稀疏化合并
   - WARM：利用 Fisher 信息加权平均

4. **工作流程**：提取 delta 权重 → 分析架构差异 → 构建投影矩阵 → 合并到目标模型 → 验证测试

5. **局限性**：架构差异越大，性能损失越大。完全不同架构（MHA vs MLA、MoE vs 标准 FFN）之间无法迁移

---

## 延伸阅读

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Task Vectors: Efficient Steering of LLMs](https://arxiv.org/abs/2404.07996)
- [TIES-Merging: Task Vector Ensemble with Intelligent Sign Resolution](https://arxiv.org/abs/2309.10500)
- [WARM: Weight Averaging meets Fisher for Out-of-Distribution Generalization](https://arxiv.org/abs/2310.10046)
- [mergekit: A Toolkit for Merging Large Language Models](https://github.com/arcee-ai/mergekit)
- [PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models](https://github.com/huggingface/peft)

---

## 复习题

1. **概念题**：解释为什么 LLaMA 7B 的 LoRA 适配器不能直接应用到 LLaMA 13B 模型上？如果要转换，需要哪些步骤？

2. **计算题**：假设源模型 Q 投影维度为 4096，目标模型 Q 投影维度为 5120，请说明如何构建投影矩阵 P 并计算转换后的 delta 权重。

3. **分析题**：TIES-Merging 的核心思想是什么？它是如何处理多个任务向量之间方向冲突的？请举例说明。

4. **比较题**：DARE 和 WARM 两种合并策略在原理上有什么本质区别？各自适合什么场景使用？

5. **应用题**：你有一个在 Qwen-7B 上微调的情感分类 LoRA 适配器，现在需要将其应用到 LLaMA-7B 模型上。请列出详细的转换步骤，并说明哪些地方可能需要特殊处理。