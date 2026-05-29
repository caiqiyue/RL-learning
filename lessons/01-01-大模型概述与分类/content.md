# 1.1 大模型概述与分类

## 课程概述

本课时系统梳理大语言模型(LLM)的发展脉络、核心能力与分类体系，为后续微调技术学习建立顶层认知框架。

**学习目标**
- 理解通用大模型的核心定义与能力边界
- 掌握主流模型的分类维度（架构/规模/模态/开源属性）
- 了解GPT/LLaMA/Qwen/ChatGLM/DeepSeek等代表模型的特色

**前置知识**：深度学习基础（Transformer架构）

---

## 1. 什么是通用大模型

### 1.1 核心定义

通用大模型（General Purpose Large Model）是一类基于大规模数据和海量参数训练的深度学习模型，具备跨任务、跨领域的通用能力，无需为每个任务单独设计模型结构。

**三大核心特征**

| 特征 | 说明 | 示例 |
|------|------|------|
| **大规模参数** | 通常数十亿到万亿级别 | GPT-3: 175B, GPT-4: ~1.8T |
| **预训练-微调范式** | 海量无监督预训练 + 少量有监督微调 | Pre-training → SFT → RLHF |
| **涌现能力(Emergence)** | 参数量突破临界点后出现不可预测的新能力 | 思维链(Chain-of-Thought)、上下文学习 |

### 1.2 与窄域模型的关键区别

```
窄域模型（如BERT-base）        通用大模型（如GPT-4）
├── 单一任务能力               ├── 多任务统一处理
├── 参数量固定                ├── 参数量弹性扩展
├── 依赖人工特征工程           ├── 自动学习语义表示
└── 迁移能力有限              └── 涌现能力与强泛化
```

---

## 2. 大模型分类体系

### 2.1 按技术架构划分

#### Transformer架构模型（绝对主流）

**Encoder-only**（理解为主）
- 代表：BERT、RoBERTa、ERNIE
- 特点：双向注意力，适合理解任务（分类、NER、问答）
- 局限：生成能力弱

**Decoder-only**（生成为主）
- 代表：GPT系列、LLaMA、Qwen、DeepSeek
- 特点：单向自回归，生成能力强
- 优势：Scaling Law效果显著，工程实现简单
- 注意：现代大模型几乎全部采用此架构

**Encoder-Decoder**
- 代表：T5、BART、GLM-130B
- 特点：双向上下文 + 自回归生成
- 场景：翻译、摘要、序列到序列任务

#### MoE架构模型（效率革新）

**混合专家模型（Mixture of Experts）**
```
传统Dense模型：所有参数每次都激活
MoE模型：每次只激活部分"专家"网络

┌─────────────────────────────────┐
│  输入 → Router → 选择Top-K专家   │
│              ├── Expert 1 ──┐   │
│              ├── Expert 2 ──┼──→ Output
│              └── Expert N ──┘   │
└─────────────────────────────────┘
```

- 代表：Mistral MoE、Qwen-MoE、DeepSeek-V2
- 优势：稀疏激活，参数量巨大但推理成本可控
- 挑战：负载均衡、训练稳定性

### 2.2 按模态划分

| 模态类型 | 代表模型 | 核心能力 |
|---------|---------|---------|
| **纯文本模型** | GPT-4、Claude、LLaMA、Qwen | 文本生成、理解、推理 |
| **多模态模型** | GPT-4V、Gemini、Qwen-VL | 图文理解、视频分析 |
| **代码模型** | GPT-4、DeepSeek-Coder、CodeLLaMA | 代码生成、Debug |
| **Embedding模型** | text-embedding-ada-002、SGPT | 向量检索、语义匹配 |

### 2.3 按规模划分

| 规模级别 | 参数量 | 显存需求(FP16) | 代表模型 | 典型场景 |
|---------|--------|----------------|---------|---------|
| **小模型** | < 7B | < 14GB | LLaMA-3.2-3B, Qwen2.5-1.5B | 端侧部署、消费级GPU |
| **中模型** | 7B ~ 13B | 14GB ~ 26GB | LLaMA-3.1-8B, Qwen2.5-7B | 单卡微调、研究 |
| **大模型** | 30B ~ 70B | 60GB ~ 140GB | LLaMA-3.1-70B, Qwen2.5-72B | 多卡推理、企业级 |
| **超大模型** | > 100B | > 200GB | GPT-4、Claude-3、DeepSeek-V3 | 云端API、复杂推理 |

### 2.4 按开源属性划分

#### 开源模型生态

| 模型系列 | 机构 | 典型版本 | 特点 |
|---------|------|---------|------|
| **LLaMA** | Meta | LLaMA-3.1-70B | 生态最丰富，衍生模型众多 |
| **Qwen** | 阿里云 | Qwen2.5-72B | 中文最强，开源最积极 |
| **ChatGLM** | 智谱AI | GLM-4-9B | 中文对话优化，清华系 |
| **DeepSeek** | 深度求索 | DeepSeek-V3-671B | MoE架构，性价比高 |
| **Yi** | 零一万物 | Yi-1.5-34B | 中英双语优秀 |
| **Baichuan** | 百川智能 | Baichuan2-13B | 中文医疗/法律优化 |

#### 闭源模型API

| 模型 | 提供商 | API形式 | 特点 |
|------|-------|--------|------|
| GPT-4o / o1 / o3 | OpenAI | 云端API | 能力最强，成本高 |
| Claude 3.5 | Anthropic | 云端API | 长上下文、安全性强 |
| Gemini 2.0 | Google | 云端API | 多模态领先 |
| 文心一言4.0 | 百度 | 云端API | 中文增强 |

---

## 3. 主流模型架构详解

### 3.1 GPT系列（Decoder-only演进）

```
GPT-1 (117M) → GPT-2 (1.5B) → GPT-3 (175B) → GPT-4 (推测1.8T)
     │              │              │              │
  预训练+微调    更多数据       Scale Law       多模态+推理
```

**GPT-3核心架构**
```python
# GPT-3 简化伪代码
class GPT3(nn.Module):
    def __init__(self, vocab_size=50257, ctx_len=2048, n_layers=96, n_heads=96, d_model=12288):
        self.transformer = nn.Sequential(
            nn.Embedding(vocab_size, d_model),           # 词嵌入
            nn.ModuleList([DecoderBlock(n_heads, d_model) for _ in range(n_layers)]),
            nn.LayerNorm(d_model),
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 语言模型头
    
    def forward(self, input_ids):
        h = self.transformer(input_ids)
        logits = self.lm_head(h)
        return logits
```

**GPT-4关键能力**
- 扩展上下文：128K tokens
- 多模态：支持图像输入
- 复杂推理：Chain-of-Thought显著提升
- 指令遵循：InstructGPT后的人类偏好对齐

### 3.2 LLaMA系列（开源生态基石）

Meta发布的LLaMA是开源模型最重要的里程碑，催生了大量衍生模型。

**LLaMA-3.1 架构创新**
```python
# LLaMA-3 关键配置（LLaMA-3.1-70B）
config = {
    "vocab_size": 128_256,
    "hidden_size": 8192,
    "intermediate_size": 28672,    # SwiGLU FFN
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,      # GQA（组查询注意力）
    "rope_theta": 500000.0,       # RoPE远程衰减调优
}
```

**核心组件**
1. **SwiGLU激活函数**：SiLU + GLU，Transformer FFN中替代ReLU
2. **GQA（Group Query Attention）**：减少KV缓存，推理加速
3. **RoPE旋转位置编码**：支持超长上下文

### 3.3 Qwen系列（中文最强开源）

阿里巴巴通义千问系列在中英双语、代码、数学方面表现突出。

**Qwen2.5 特色技术**
```python
# Qwen2.5 关键组件

# 1. RoPE + YaRN（远程注意力调优）
# 支持32K/128K上下文
rope_freq = base / (theta ** (i / n_heads_dim))

# 2. SMP（滑动窗口注意力 + 稀疏模式）
attention_mask = create_sliding_window_mask(window_size=4096)

# 3. Qwen2.5的Tokenizer（BPE + 中文优化）
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-72B")
# 词表128K，中文token效率比GPT4提升30%
```

### 3.4 DeepSeek系列（性价比之王）

**DeepSeek-V2：MoE架构创新**
```python
# DeepSeek-V2 MoE配置
moe_config = {
    "num_experts": 8,           # 每个token激活8个专家（共128个专家）
    "num_experts_per_tok": 8,  # Top-8路由
    "shared_experts": 2,        # 共享专家（始终激活）
    "model_size": "671B",
    "active_params": "37B",    # 实际激活参数量
}
```

**DeepSeek-V3 关键创新**
- **FP8混合精度训练**：首次在超大模型验证
- **Multi-head Latent Attention (MLA)**：减少推理显存
- **DeepSeekMoE-E**：细粒度专家分解

---

## 4. 预训练与微调范式

### 4.1 预训练阶段

**大规模无监督学习**
```python
# 经典预训练目标：下一个Token预测
def pretrain_loss(model, input_ids):
    # input_ids: [batch, seq_len]
    logits = model(input_ids)           # [batch, seq_len, vocab_size]
    
    # 移位：预测next_token
    shift_logits = logits[:, :-1, :]    # [batch, seq_len-1, vocab_size]
    shift_labels = input_ids[:, 1:]      # [batch, seq_len-1]
    
    loss = F.cross_entropy(shift_logits.reshape(-1, vocab_size),
                          shift_labels.reshape(-1))
    return loss
```

**预训练数据规模（参考GPT-3）**
| 数据来源 | 规模 |
|---------|------|
| Common Crawl | 4100亿 tokens |
| WebText2 | 190亿 tokens |
| Books | 670亿 tokens |
| Wikipedia | 30亿 tokens |
| **总计** | ~5000亿 tokens |

### 4.2 微调阶段

```
预训练模型（通用能力）
    │
    ├── SFT（监督微调）：学习任务格式
    │       └── 使用少量高质量指令数据
    │
    ├── RLHF（人类反馈强化学习）：对齐人类偏好
    │       ├── Reward Model训练
    │       └── PPO优化
    │
    └── 领域微调（可选）：垂直领域适配
            └── 医疗/法律/金融等专业数据
```

---

## 5. 涌现能力(Emergence)

大模型在参数量突破临界点后，会涌现出在小模型上不存在的能力。

### 5.1 典型涌现能力

| 能力 | 小模型表现 | 大模型表现 | 涌现临界 |
|------|----------|-----------|---------|
| **思维链推理** | 简单直推 | 复杂多步推理 | ~100B |
| **上下文学习** | 不支持 | Few-shot学习 | ~10B |
| **代码生成** | 片段补全 | 完整项目生成 | ~10B |
| **多语言翻译** | 单向翻译 | 跨语言泛化 | ~50B |
| **数学推理** | 基础计算 | 奥赛级别证明 | ~100B |

### 5.2 能力地图

```
参数量 ──────────────────────────────────────────→
  │
1B  ├── 简单问答
  │   ├── 基础文本生成
  │   └── 短文本分类
  │
10B ├── 上下文Few-shot学习
  │   ├── 多语言基础翻译
  │   └── 简单代码补全
  │
70B ├── 复杂推理（CoT）
  │   ├── 完整代码生成
  │   ├── 数学证明
  │   └── 多模态理解
  │
700B+ ├── 科研级推理
      ├── 复杂系统设计
      └── 开放式创意生成
```

---

## 6. 主流开源模型对比

| 模型 | 参数量 | 上下文 | 中文能力 | 代码能力 | 开源协议 |
|------|--------|--------|---------|---------|---------|
| LLaMA-3.1-70B | 70B | 128K | ★★★ | ★★★★ | Llama3.1 |
| Qwen2.5-72B | 72B | 128K | ★★★★★ | ★★★★ | Apache 2.0 |
| DeepSeek-V3 | 671B(37B激活) | 128K | ★★★★ | ★★★★★ | MIT |
| ChatGLM4-9B | 9B | 128K | ★★★★ | ★★★ | 清北协议 |
| Yi-1.5-34B | 34B | 200K | ★★★★ | ★★★★ | Yi License |

---

## 本章小结

1. **通用大模型**的核心特征：大规模参数、预训练-微调范式、涌现能力
2. **分类维度**：架构（Decoder-only主导）、规模（7B/13B/70B/100B+）、模态（文本/多模态）、开源属性
3. **主流架构**：GPT的Decoder-only、LLaMA的SwiGLU+Qwen的GQA、DeepSeek的MoE
4. **范式演进**：Pretraining → SFT → RLHF → 领域微调

---

## 延伸阅读

- GPT-3论文：Language Models are Few-Shot Learners
- LLaMA论文：Open and Efficient Foundation Language Models
- DeepSeek-V3论文：DeepSeek-V3 Technical Report
- GPT-4架构揭秘：[OpenAI官方文档](https://openai.com/index/gpt-4-research/)

---

## 思考题

1. 为什么Decoder-only架构成为大模型主流？与Encoder-Decoder相比各有什么优劣？
2. MoE架构的"稀疏激活"特性如何在保持模型能力的同时降低推理成本？
3. 从涌现能力的角度，分析为什么规模(Scaling)是GPT系列成功的关键因素？
4. 开源模型(LLaMA/Qwen/DeepSeek)与闭源模型(GPT-4/Claude)在能力和生态上各有什么优劣势？