# 12.1 QLoRA实战：多模型单卡微调

## 课程概述

本节课介绍如何使用 QLoRA 在单张 GPU 上微调多个不同架构的大语言模型。QLoRA 结合了 4-bit NF4 量化与 LoRA 低秩适配器技术，使得在消费级 GPU（如 24GB 显存）上同时微调多个模型成为可能。我们将深入探讨不同模型家族（LLaMA、Qwen、ChatGLM、DeepSeek）的特殊处理方式，以及多模型场景下的工程实践。

**学习目标**
- 掌握 QLoRA 在不同模型家族上的应用方法
- 学会在单卡上管理多个模型的内存使用
- 理解不同模型的 LoRA 目标层配置差异
- 掌握多模型适配器的注册、管理与合并流程
- 解决多模型微调中的常见问题（OOM、tokenization 差异等）

**前置知识**
- QLoRA 量化原理（详见第11.2节或第3章）
- LoRA 基本概念与配置
- PyTorch 基础
- transformers 库使用经验

---

## 1. QLoRA 与不同模型家族

### 1.1 模型架构多样性概述

不同大模型家族在架构上存在显著差异，这直接影响到 QLoRA 微调的配置方式：

```
┌─────────────────────────────────────────────────────────────┐
│                     主流模型家族架构差异                      │
├───────────────┬───────────────┬───────────────┬───────────────┤
│    LLaMA      │     Qwen      │    ChatGLM    │   DeepSeek   │
├───────────────┼───────────────┼───────────────┼───────────────┤
│  RMSNorm      │  RMSNorm      │   RMSNorm     │  RMSNorm     │
│  RoPE         │  RoPE         │   RoPE        │  RoPE        │
│  SwigLU       │  SwigLU       │   SwigLU      │  SwigLU      │
│  —            │  —            │   GLM层归一化  │  MoE变体     │
│  —            │  长上下文支持  │  注意力兼容   │  DeepSeekMoe │
└───────────────┴───────────────┴───────────────┴───────────────┘
```

### 1.2 LLaMA 系列：标准应用

LLaMA 是 QLoRA 微调最成熟的模型家族，生态完善，社区支持好。

**已知良好配置**：

```python
# LLaMA QLoRA 标准配置
LLAMA_QLORA_CONFIG = {
    # 模型配置
    "model_name": "meta-llama/Llama-2-7b-hf",  # 或 Llama-3-8B
    "model_type": "llama",
    
    # 量化配置（NF4）
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    
    # LoRA 配置
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "v_proj",    # 核心：注意力投影
        "k_proj", "o_proj",    # 可选：完整注意力
        "gate_proj", "up_proj", "down_proj"  # MLP 层
    ],
    
    # 训练配置
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "warmup_steps": 100,
}
```

**LLaMA 的 LoRA 层级建议**：

```python
def get_llama_lora_target_modules(layer_type: str = "full") -> list[str]:
    """
    LLaMA 模型的 LoRA 目标模块配置
    
    Args:
        layer_type: "core" | "full" | "minimal"
    
    Returns:
        需要添加 LoRA 的模块名称列表
    """
    targets = {
        "core": ["q_proj", "v_proj"],  # 最少参数，效果好
        "full": ["q_proj", "v_proj", "k_proj", "o_proj"],  # 完整注意力
        "minimal": ["q_proj"],  # 极简配置
    }
    return targets.get(layer_type, targets["core"])
```

### 1.3 Qwen 系列：特殊考虑

Qwen（通义千问）在 tokenizer 和架构上有一些独特之处：

**特殊考虑点**：

1. **不同 Tokenizer**：
   - Qwen 使用 BPE tokenizer，词汇表更大（约 151,936 tokens）
   - 特殊 token 较多（如 `<|im_start|>`、`<|im_end|>` 等）

2. **长上下文支持**：
   - Qwen 2 支持高达 128K 上下文
   - 需要特别配置 `model.config.max_position_embeddings`

3. **Attention 实现**：
   - 使用改进的 RoPE 位置编码
   - 需要配置 `use_sliding_window` 参数

**Qwen QLoRA 配置**：

```python
# Qwen QLoRA 配置
QWEN_QLORA_CONFIG = {
    "model_name": "Qwen/Qwen2-7B",
    "model_type": "qwen2",
    
    # 量化配置
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",  # Qwen 推荐 BF16
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    
    # LoRA 配置（Qwen 特有）
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "cm_proj"  # Qwen 特有的 embedding 投影层
    ],
    
    # 特殊配置
    "use_sliding_window": True,  # 长上下文优化
    "tokenizer.pad_token_id": 151643,
}
```

**Qwen Tokenizer 处理**：

```python
def setup_qwen_tokenizer(tokenizer):
    """
    配置 Qwen tokenizer 的特殊 token
    """
    # 添加特殊 token
    special_tokens = {
        "pad_token": "<|extra_0|>",
        "eos_token": "<|endoftext|>",
    }
    
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = special_tokens["pad_token"]
    
    # 设置合理的最大长度
    tokenizer.model_max_length = 128 * 1024  # 128K
    
    return tokenizer
```

### 1.4 ChatGLM 系列：架构差异

ChatGLM（智谱清言）有独特的架构设计：

**架构差异**：

1. **RMSNorm 替代 LayerNorm**：
   - 使用 RMSNorm 而非标准 LayerNorm
   - 归一化方式不同

2. **旋转位置编码变体**：
   - 使用 RoPE 但实现略有不同
   - 需要特别配置 `rope_ratio`

3. **注意力机制**：
   - 使用 `ChatGLMAttention` 而非标准 attention
   - `attention_mapping` 需要特别处理

**ChatGLM QLoRA 配置**：

```python
# ChatGLM QLoRA 配置
CHATGLM_QLORA_CONFIG = {
    "model_name": "THUDM/chatglm3-6b",
    "model_type": "chatglm",
    
    # 量化配置
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    
    # LoRA 配置（ChatGLM 特有）
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "query_key_value",  # ChatGLM 独特的 QKV 合并层
        "dense",            # 输出投影
        "mlp",              # FFN 层（不是 gate/up/down 分离）
    ],
    
    # ChatGLM 特殊配置
    "add_bias_linear": False,  # ChatGLM 不使用 bias
    "rmsnorm": True,          # 明确使用 RMSNorm
}
```

**ChatGLM 层名称映射**：

```python
def get_chatglm_lora_target_modules() -> list[str]:
    """
    ChatGLM 模型的 LoRA 目标模块
    注意：ChatGLM 的层名称与其他模型不同
    """
    return [
        "query_key_value",  # 合并的 QKV 投影层
        "dense",           # 输出投影 (o_proj 的等价层)
    ]
```

### 1.5 DeepSeek 系列：MoE 架构

DeepSeek 系列包含 MoE（Mixture of Experts）架构变体，需要特别处理：

**MoE 特殊考虑**：

1. **稀疏激活**：
   - 虽然参数总量大，但每次前向只激活部分专家
   - 实际显存需求取决于 `num_experts_per_tok`

2. **专家并行**：
   - 多专家可能分布在不同设备
   - 需要配置 `expert_parallel_size`

3. **辅助损失**：
   - MoE 需要额外的负载均衡损失
   - 需要配置 `aux_loss_coef`

**DeepSeek MoE QLoRA 配置**：

```python
# DeepSeek MoE QLoRA 配置
DEEPSEEK_MOE_CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-MoE-16B",
    "model_type": "deepseek",
    
    # 量化配置
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    
    # LoRA 配置（针对 MoE）
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "v_proj",           # 注意力
        "gate_proj", "up_proj",       # MoE FFN
        # 注意：down_proj 不在 MoE 中存在
    ],
    
    # MoE 特殊配置
    "num_experts": 16,               # 总专家数
    "num_experts_per_tok": 2,        # 每个 token 激活的专家数
    "aux_loss_coef": 0.01,           # 辅助损失系数
}
```

---

## 2. 单卡多模型策略

### 2.1 内存估算

在单卡上加载多个模型前，必须先估算内存需求：

```python
def estimate_qlora_memory(
    model_name: str,
    model_type: str,
    quantization: dict = None,
    lora_config: dict = None
) -> dict:
    """
    估算 QLoRA 微调的显存需求
    
    Returns:
        包含各项内存需求的字典
    """
    import subprocess
    import torch
    
    # 基本参数（从模型名推断）
    param_count = infer_model_size(model_name)
    
    # 量化后基座模型大小（NF4 约为 FP16 的 1/4）
    quantized_base = param_count * 0.25 * 2  # bytes per param (4-bit = 0.5 byte)
    
    # LoRA 适配器大小
    lora_params = calculate_lora_params(
        model_type=model_type,
        rank=lora_config.get("lora_rank", 64),
        alpha=lora_config.get("lora_alpha", 128),
    )
    lora_memory = lora_params * 2  # FP16
    
    # 梯度与优化器状态（LoRA 只更新适配器）
    gradient_memory = lora_params * 4  # FP32
    optimizer_memory = lora_params * 8  # Adam 状态
    
    # 激活值与临时缓存
    activation_memory = estimate_activation_memory(
        batch_size=1,
        seq_length=512,
        hidden_size=get_hidden_size(model_type)
    )
    
    total_memory_gb = (
        quantized_base + 
        lora_memory + 
        gradient_memory + 
        optimizer_memory + 
        activation_memory
    ) / (1024 ** 3)
    
    return {
        "param_count_b": param_count / (1024 ** 3),
        "quantized_base_gb": quantized_base / (1024 ** 3),
        "lora_gb": lora_memory / (1024 ** 3),
        "gradient_gb": gradient_memory / (1024 ** 3),
        "optimizer_gb": optimizer_memory / (1024 ** 3),
        "activation_gb": activation_memory / (1024 ** 3),
        "total_gb": total_memory_gb,
        "safety_margin_gb": 4.0,  # 预留 4GB 安全边际
        "recommended_gpu_gb": total_memory_gb + 4.0,
    }


def infer_model_size(model_name: str) -> int:
    """从模型名推断参数量"""
    size_map = {
        "7b": 7 * 1024 ** 3,
        "8b": 8 * 1024 ** 3,
        "13b": 13 * 1024 ** 3,
        "14b": 14 * 1024 ** 3,
        "33b": 33 * 1024 ** 3,
        "65b": 65 * 1024 ** 3,
        "70b": 70 * 1024 ** 3,
    }
    
    for size_str, count in size_map.items():
        if size_str in model_name.lower():
            return count
    return 7 * 1024 ** 3  # 默认 7B
```

### 2.2 顺序加载策略

当单卡无法同时加载多个模型时，采用顺序加载策略：

```python
import gc
import torch

class SequentialModelLoader:
    """
    单卡多模型顺序加载器
    
    流程：load → train → save adapter → unload → repeat
    """
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir
        self.torch = torch
        self.current_model = None
        self.current_tokenizer = None
        
    def load_model(
        self, 
        model_config: dict,
        lora_config: dict
    ) -> tuple:
        """
        加载模型和 tokenizer
        
        Returns:
            (model, tokenizer)
        """
        # 清理之前的模型
        self._clear_current_model()
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_name"],
            trust_remote_code=model_config.get("trust_remote_code", False)
        )
        
        # 加载模型（量化）
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name"],
            load_in_4bit=model_config.get("load_in_4bit", True),
            load_in_8bit=model_config.get("load_in_8bit", False),
            bnb_4bit_compute_dtype=self._get_dtype(
                model_config.get("bnb_4bit_compute_dtype", "float16")
            ),
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=model_config.get(
                "bnb_4bit_use_double_quant", True
            ),
            device_map="auto",
            trust_remote_code=model_config.get("trust_remote_code", False),
        )
        
        # 准备量化模型进行训练
        model = prepare_model_for_kbit_training(model)
        
        # 应用 LoRA
        lora_config_obj = LoraConfig(
            r=lora_config["lora_rank"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["lora_target_modules"],
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config_obj)
        
        self.current_model = model
        self.current_tokenizer = tokenizer
        
        return model, tokenizer
    
    def unload_model(self):
        """卸载当前模型，释放显存"""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            
        if self.current_tokenizer is not None:
            del self.current_tokenizer
            self.current_tokenizer = None
        
        # 清理 CUDA 缓存
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.synchronize()
        
        # Python 垃圾回收
        gc.collect()
    
    def _clear_current_model(self):
        """清理现有模型"""
        self.unload_model()
    
    def _get_dtype(self, dtype_str: str):
        """转换 dtype 字符串到 torch dtype"""
        dtype_map = {
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "float32": self.torch.float32,
        }
        return dtype_map.get(dtype_str, self.torch.float16)
```

### 2.3 检查点管理

在多模型训练之间保存和加载检查点：

```python
import os
from pathlib import Path

class CheckpointManager:
    """管理多个模型的检查点"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查点元数据
        self.metadata_file = self.checkpoint_dir / "checkpoints.json"
        
    def save_checkpoint(
        self,
        model,           # PEFT model
        tokenizer,
        model_name: str,
        step: int,
        metrics: dict = None
    ) -> str:
        """
        保存模型检查点
        
        Returns:
            检查点路径
        """
        checkpoint_path = self.checkpoint_dir / model_name / f"step_{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 PEFT 适配器
        model.save_pretrained(checkpoint_path)
        
        # 保存 tokenizer
        tokenizer.save_pretrained(checkpoint_path)
        
        # 保存元数据
        metadata = {
            "model_name": model_name,
            "step": step,
            "metrics": metrics or {},
        }
        import json
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # 更新总索引
        self._update_index(model_name, step, checkpoint_path)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, model_name: str, step: int = None):
        """
        加载指定检查点
        
        Args:
            model_name: 模型名称
            step: 步骤数，None 则加载最新的
        """
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if step is None:
            step = self._get_latest_step(model_name)
        
        checkpoint_path = self.checkpoint_dir / model_name / f"step_{step}"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return checkpoint_path
    
    def _update_index(self, model_name: str, step: int, path: Path):
        """更新检查点索引"""
        import json
        
        index = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                index = json.load(f)
        
        index[model_name] = {
            "latest_step": step,
            "path": str(path),
        }
        
        with open(self.metadata_file, "w") as f:
            json.dump(index, f, indent=2)
    
    def _get_latest_step(self, model_name: str) -> int:
        """获取某个模型的最新步骤"""
        import json
        
        if not self.metadata_file.exists():
            return 0
            
        with open(self.metadata_file, "r") as f:
            index = json.load(f)
        
        return index.get(model_name, {}).get("latest_step", 0)
```

---

## 3. 模型特定的 LoRA 目标层

### 3.1 注意力层的差异处理

不同模型的注意力层架构不同，需要针对性配置：

```python
def get_model_specific_lora_targets(model_type: str) -> dict:
    """
    根据模型类型返回 LoRA 目标模块配置
    
    Args:
        model_type: "llama" | "qwen" | "chatglm" | "deepseek"
    
    Returns:
        包含各层配置的字典
    """
    configs = {
        "llama": {
            # 标准 attention：q, k, v, o 分开
            "attention_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp_modules": ["gate_proj", "up_proj", "down_proj"],
            "other_modules": [],
            "notes": "LLaMA 使用标准 SwigLU FFN"
        },
        
        "qwen2": {
            # Qwen2：包含 cm_proj 用于 embedding 投影
            "attention_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp_modules": ["gate_proj", "up_proj", "down_proj"],
            "other_modules": ["cm_proj"],  # Qwen 特有
            "notes": "Qwen2 需要配置滑动窗口 attention"
        },
        
        "chatglm": {
            # ChatGLM：QKV 合并为一个层
            "attention_modules": ["query_key_value"],  # 合并的 QKV
            "mlp_modules": ["mlp"],  # 整体 MLP，而非分离的门控层
            "other_modules": ["dense"],  # 输出投影
            "notes": "ChatGLM 使用 RMSNorm 和不同的层归一化"
        },
        
        "deepseek": {
            # DeepSeek MoE：专家结构
            "attention_modules": ["q_proj", "v_proj"],  # 标准注意力
            "mlp_modules": ["gate_proj", "up_proj"],  # MoE FFN（无 down_proj）
            "other_modules": ["moe_gate"],  # MoE 门控
            "notes": "DeepSeek MoE 需要配置专家数量和辅助损失"
        },
        
        "baichuan": {
            # 百川：类似的架构但层名称略有不同
            "attention_modules": ["W_pack"],  # 合并的 QKV
            "mlp_modules": ["gate_proj", "up_proj", "down_proj"],
            "other_modules": [],
            "notes": "百川使用不同的注意力实现"
        },
    }
    
    return configs.get(model_type, configs["llama"])


def build_lora_target_list(
    model_type: str,
    include_mlp: bool = True,
    include_other: bool = False
) -> list[str]:
    """
    根据配置构建完整的 LoRA 目标模块列表
    """
    config = get_model_specific_lora_targets(model_type)
    
    targets = []
    targets.extend(config["attention_modules"])
    
    if include_mlp:
        targets.extend(config["mlp_modules"])
    
    if include_other:
        targets.extend(config["other_modules"])
    
    return targets
```

### 3.2 Query-Key-Value vs MLP 层的选择

并非所有层都需要 LoRA，需要根据任务类型选择：

```python
class LoRATargetSelector:
    """
    LoRA 目标层选择器
    
    根据训练目标和模型特性，推荐合适的 LoRA 层配置
    """
    
    # 不同任务的推荐配置
    TASK_CONFIGS = {
        "instruction_following": {
            "description": "指令遵循任务",
            "recommend_attention": "full",   # 全部注意力层
            "recommend_mlp": "gate_up",       # 只加 gate 和 up_proj
            "rank_boost": 1.0,
        },
        "code_generation": {
            "description": "代码生成任务",
            "recommend_attention": "qkvo",     # 包含 k 和 o
            "recommend_mlp": "full",          # 完整 MLP
            "rank_boost": 1.5,               # 代码任务提升 rank
        },
        "math_reasoning": {
            "description": "数学推理任务",
            "recommend_attention": "qv",       # 核心 q, v
            "recommend_mlp": "none",          # 数学不需要太多 MLP
            "rank_boost": 1.2,
        },
        "chat_conversation": {
            "description": "对话任务",
            "recommend_attention": "qvo",     # q, v, o
            "recommend_mlp": "up",            # 只加 up_proj
            "rank_boost": 1.0,
        },
    }
    
    @classmethod
    def get_recommended_config(
        cls,
        model_type: str,
        task_type: str,
        base_rank: int = 64
    ) -> dict:
        """
        获取推荐配置
        
        Args:
            model_type: 模型类型
            task_type: 任务类型
            base_rank: 基础 rank 值
        
        Returns:
            推荐配置字典
        """
        task_config = cls.TASK_CONFIGS.get(
            task_type, 
            cls.TASK_CONFIGS["instruction_following"]
        )
        
        # 构建目标层列表
        attention_map = {
            "core": ["q_proj", "v_proj"],
            "qv": ["q_proj", "v_proj"],
            "qvo": ["q_proj", "v_proj", "o_proj"],
            "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "full": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        
        mlp_map = {
            "none": [],
            "gate": ["gate_proj"],
            "up": ["up_proj"],
            "gate_up": ["gate_proj", "up_proj"],
            "full": ["gate_proj", "up_proj", "down_proj"],
        }
        
        attention_targets = attention_map.get(
            task_config["recommend_attention"], 
            attention_map["qv"]
        )
        mlp_targets = mlp_map.get(
            task_config["recommend_mlp"],
            mlp_map["gate_up"]
        )
        
        # 过滤不存在的层（针对特定模型）
        model_config = get_model_specific_lora_targets(model_type)
        attention_targets = [
            t for t in attention_targets 
            if t in model_config["attention_modules"] + model_config.get("other_modules", [])
        ]
        mlp_targets = [
            t for t in mlp_targets 
            if t in model_config["mlp_modules"]
        ]
        
        # 计算最终 rank
        final_rank = int(base_rank * task_config["rank_boost"])
        
        return {
            "lora_target_modules": attention_targets + mlp_targets,
            "lora_rank": final_rank,
            "lora_alpha": final_rank * 2,
            "task_type": task_type,
            "model_type": model_type,
            "notes": task_config["description"],
        }
```

### 3.3 配置示例

```python
# 不同模型的完整 LoRA 配置示例

MODELS_LORA_CONFIG = {
    "llama-2-7b": {
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    
    "qwen2-7b": {
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "cm_proj"
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    
    "chatglm3-6b": {
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "query_key_value", "dense", "mlp"
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    
    "deepseek-moe-16b": {
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj", "v_proj",
            "gate_proj", "up_proj",
            "moe_gate"
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
}
```

---

## 4. 实战工作流程

### 4.1 多模型训练流程

```
┌─────────────────────────────────────────────────────────────┐
│              单卡多模型 QLoRA 训练工作流程                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  阶段 1：准备阶段                                             │
│  ├── 估算各模型内存需求                                        │
│  ├── 确认 GPU 显存足够                                        │
│  └── 创建输出目录结构                                         │
│                                                             │
│  阶段 2：模型 A 训练                                          │
│  ├── 加载模型 A + tokenizer                                   │
│  ├── 配置 LoRA                                               │
│  ├── 训练 N steps                                           │
│  ├── 保存 adapter checkpoint                                 │
│  └── 卸载模型 A（释放显存）                                    │
│                                                             │
│  阶段 3：模型 B 训练                                          │
│  ├── 加载模型 B + tokenizer                                   │
│  ├── 配置 LoRA                                               │
│  ├── 训练 N steps                                           │
│  ├── 保存 adapter checkpoint                                 │
│  └── 卸载模型 B                                              │
│                                                             │
│  阶段 4：批量导出与合并                                        │
│  ├── 合并多个模型的 adapter                                   │
│  ├── 评估合并效果                                            │
│  └── 导出最终模型                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 完整训练脚本

```python
# train_multi_model.py
import argparse
import logging
from dataclasses import dataclass
from typing import List, Dict
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """单个模型配置"""
    name: str
    model_path: str
    model_type: str
    trust_remote_code: bool = False
    max_seq_length: int = 2048


@dataclass  
class LoRAConfig:
    """LoRA 配置"""
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = None


class MultiModelQLoraTrainer:
    """多模型 QLoRA 训练器"""
    
    def __init__(
        self,
        output_dir: str,
        per_device_batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        warmup_steps: int = 100,
    ):
        self.output_dir = output_dir
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
    
    def train_model(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        dataset_path: str,
        dataset_split: str = "train",
    ) -> dict:
        """
        训练单个模型
        
        Returns:
            训练历史字典
        """
        logger.info(f"开始训练模型: {model_config.name}")
        
        # 1. 加载模型和 tokenizer
        model, tokenizer = self._load_model(model_config, lora_config)
        
        # 2. 准备数据集
        dataset = self._load_dataset(dataset_path, dataset_split, tokenizer)
        
        # 3. 训练
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/{model_config.name}",
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            warmup_steps=self.warmup_steps,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            fp16=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        
        trainer = self._create_trainer(
            model, tokenizer, dataset, training_args
        )
        
        history = trainer.train()
        
        # 4. 保存 adapter
        adapter_path = f"{self.output_dir}/adapters/{model_config.name}"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        
        # 5. 卸载模型
        self._unload_model()
        
        return history.metrics
    
    def _load_model(
        self, 
        model_config: ModelConfig, 
        lora_config: LoRAConfig
    ):
        """加载量化模型并应用 LoRA"""
        # 清理之前的模型
        self._unload_model()
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_path,
            trust_remote_code=model_config.trust_remote_code,
            padding_side="right",
        )
        
        # 确保有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载量化模型
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_path,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            device_map="auto",
            trust_remote_code=model_config.trust_remote_code,
        )
        
        # 准备 k-bit 训练
        model = prepare_model_for_kbit_training(model)
        
        # 配置 LoRA
        lora_target_modules = lora_config.target_modules or self._get_default_lora_targets(
            model_config.model_type
        )
        
        lora_cfg = LoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_config.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        
        self.current_model = model
        self.current_tokenizer = tokenizer
        self.current_model_name = model_config.name
        
        return model, tokenizer
    
    def _load_dataset(self, path: str, split: str, tokenizer):
        """加载数据集"""
        dataset = load_dataset("json", data_files=path, split=split)
        
        def tokenize_function(examples):
            result = tokenizer(
                examples["text"],
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
    
    def _create_trainer(self, model, tokenizer, dataset, training_args):
        """创建训练器"""
        from transformers import Trainer
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 因果语言模型
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        return trainer
    
    def _unload_model(self):
        """卸载模型释放显存"""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
        
        if self.current_tokenizer is not None:
            del self.current_tokenizer
            self.current_tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        self.current_model_name = None
    
    def _get_default_lora_targets(self, model_type: str) -> List[str]:
        """获取默认的 LoRA 目标模块"""
        defaults = {
            "llama": ["q_proj", "v_proj", "k_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"],
            "qwen2": ["q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", "cm_proj"],
            "chatglm": ["query_key_value", "dense", "mlp"],
            "deepseek": ["q_proj", "v_proj", "gate_proj", "up_proj"],
        }
        return defaults.get(model_type, defaults["llama"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="模型配置文件路径")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()
    
    import json
    with open(args.config, "r") as f:
        config = json.load(f)
    
    trainer = MultiModelQLoraTrainer(
        output_dir=args.output_dir,
        per_device_batch_size=config.get("per_device_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 16),
    )
    
    # 训练每个模型
    for model_cfg in config["models"]:
        model_config = ModelConfig(**model_cfg)
        lora_config = LoRAConfig(**config["lora"])
        
        metrics = trainer.train_model(
            model_config=model_config,
            lora_config=lora_config,
            dataset_path=model_cfg["dataset_path"],
        )
        
        logger.info(f"模型 {model_config.name} 训练完成: {metrics}")


if __name__ == "__main__":
    main()
```

---

## 5. 适配器注册与管理

### 5.1 适配器注册表

```python
# adapter_registry.py
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class AdapterInfo:
    """适配器信息"""
    name: str
    model_name: str           # 原始模型名
    model_type: str          # 模型类型
    adapter_path: str        # 适配器保存路径
    base_model_path: str     # 基座模型路径
    created_at: str          # 创建时间
    lora_rank: int
    lora_alpha: int
    task_type: str           # 任务类型
    dataset: str             # 训练数据集
    metrics: dict            # 训练指标


class AdapterRegistry:
    """
    适配器注册表
    
    管理多个模型的适配器，支持注册、查询、加载等操作
    """
    
    def __init__(self, registry_path: str = "./adapter_registry.json"):
        self.registry_path = Path(registry_path)
        self.adapters: Dict[str, AdapterInfo] = {}
        self._load()
    
    def _load(self):
        """从文件加载注册表"""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                self.adapters = {
                    name: AdapterInfo(**info) 
                    for name, info in data.get("adapters", {}).items()
                }
    
    def _save(self):
        """保存注册表到文件"""
        data = {
            "adapters": {
                name: asdict(info) 
                for name, info in self.adapters.items()
            },
            "updated_at": datetime.now().isoformat(),
        }
        
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def register(
        self,
        name: str,
        model_name: str,
        model_type: str,
        adapter_path: str,
        base_model_path: str,
        lora_rank: int,
        lora_alpha: int,
        task_type: str,
        dataset: str,
        metrics: dict = None
    ):
        """
        注册新的适配器
        
        Args:
            name: 适配器名称（唯一标识）
            model_name: 原始模型名
            model_type: 模型类型
            adapter_path: 适配器路径
            base_model_path: 基座模型路径
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            task_type: 任务类型
            dataset: 训练数据集
            metrics: 训练指标
        """
        info = AdapterInfo(
            name=name,
            model_name=model_name,
            model_type=model_type,
            adapter_path=adapter_path,
            base_model_path=base_model_path,
            created_at=datetime.now().isoformat(),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            task_type=task_type,
            dataset=dataset,
            metrics=metrics or {},
        )
        
        self.adapters[name] = info
        self._save()
        
        return info
    
    def unregister(self, name: str):
        """取消注册适配器"""
        if name in self.adapters:
            del self.adapters[name]
            self._save()
    
    def get(self, name: str) -> Optional[AdapterInfo]:
        """获取适配器信息"""
        return self.adapters.get(name)
    
    def list_adapters(
        self,
        model_type: str = None,
        task_type: str = None,
    ) -> List[AdapterInfo]:
        """
        列出适配器
        
        Args:
            model_type: 按模型类型筛选
            task_type: 按任务类型筛选
        
        Returns:
            适配器列表
        """
        results = list(self.adapters.values())
        
        if model_type:
            results = [a for a in results if a.model_type == model_type]
        
        if task_type:
            results = [a for a in results if a.task_type == task_type]
        
        return results
    
    def find_compatible(
        self, 
        base_model_path: str,
        task_type: str = None
    ) -> List[AdapterInfo]:
        """
        查找兼容的适配器
        
        Args:
            base_model_path: 基座模型路径
            task_type: 任务类型（可选）
        
        Returns:
            兼容的适配器列表
        """
        results = [
            a for a in self.adapters.values()
            if a.base_model_path == base_model_path
        ]
        
        if task_type:
            results = [a for a in results if a.task_type == task_type]
        
        return results
    
    def load_adapter_model(
        self,
        name: str,
        device: str = "cuda"
    ):
        """
        加载带有适配器的模型
        
        Returns:
            (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        info = self.get(name)
        if not info:
            raise ValueError(f"Adapter not found: {name}")
        
        # 加载基座模型
        base_model = AutoModelForCausalLM.from_pretrained(
            info.base_model_path,
            load_in_4bit=True,
            device_map=device,
        )
        
        # 加载适配器
        model = PeftModel.from_pretrained(base_model, info.adapter_path)
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(info.adapter_path)
        
        return model, tokenizer
    
    def export_config(self, name: str, output_path: str = None) -> dict:
        """
        导出适配器配置
        
        用于分享或部署
        """
        info = self.get(name)
        if not info:
            raise ValueError(f"Adapter not found: {name}")
        
        config = {
            "name": info.name,
            "model_name": info.model_name,
            "model_type": info.model_type,
            "base_model_path": info.base_model_path,
            "adapter_path": info.adapter_path,
            "lora_config": {
                "r": info.lora_rank,
                "alpha": info.lora_alpha,
            },
            "task_type": info.task_type,
            "dataset": info.dataset,
            "created_at": info.created_at,
        }
        
        if output_path:
            with open(output_path, "w") as f:
                json.dump(config, f, indent=2)
        
        return config
```

### 5.2 批量操作

```python
class BatchAdapterOperations:
    """批量适配器操作"""
    
    def __init__(self, registry: AdapterRegistry):
        self.registry = registry
    
    def merge_all(
        self,
        output_dir: str,
        base_model_path: str,
        merge_strategy: str = "average"
    ) -> Dict[str, str]:
        """
        合并所有兼容的适配器
        
        Args:
            output_dir: 输出目录
            base_model_path: 基座模型路径
            merge_strategy: "average" | "weighted"
        
        Returns:
            {adapter_name: merged_path}
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        os.makedirs(output_dir, exist_ok=True)
        merged_paths = {}
        
        compatible = self.registry.find_compatible(base_model_path)
        
        if not compatible:
            raise ValueError("No compatible adapters found")
        
        # 加载基座模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            load_in_4bit=True,
            device_map="cpu",  # 合并在 CPU 上进行
        )
        
        for adapter_info in compatible:
            # 加载单个适配器
            model = PeftModel.from_pretrained(
                base_model, 
                adapter_info.adapter_path
            )
            
            # 合并并卸载适配器
            merged_model = model.merge_and_unload()
            
            # 保存
            merged_path = f"{output_dir}/{adapter_info.name}"
            merged_model.save_pretrained(merged_path)
            merged_paths[adapter_info.name] = merged_path
        
        return merged_paths
    
    def batch_evaluate(
        self,
        base_model_path: str,
        adapters: List[str],
        eval_dataset,
        eval_metric_fn
    ) -> Dict[str, dict]:
        """
        批量评估多个适配器
        
        Returns:
            {adapter_name: metrics}
        """
        results = {}
        
        for adapter_name in adapters:
            info = self.registry.get(adapter_name)
            if not info or info.base_model_path != base_model_path:
                continue
            
            model, tokenizer = self.registry.load_adapter_model(adapter_name)
            
            # 评估
            metrics = eval_metric_fn(model, tokenizer, eval_dataset)
            results[adapter_name] = metrics
            
            # 清理
            del model
            del tokenizer
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
```

---

## 6. 评估与合并

### 6.1 合并策略

```python
# merge_and_eval.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AdapterMerger:
    """
    适配器合并器
    
    支持多种合并策略
    """
    
    STRATEGIES = ["average", "weighted", "task_vector"]
    
    def merge(
        self,
        base_model_path: str,
        adapter_paths: List[str],
        output_path: str,
        strategy: str = "average",
        weights: List[float] = None,
        base_model_kwargs: dict = None
    ):
        """
        合并多个适配器到基座模型
        
        Args:
            base_model_path: 基座模型路径
            adapter_paths: 适配器路径列表
            output_path: 输出路径
            strategy: 合并策略
            weights: 加权平均的权重
            base_model_kwargs: 加载基座模型的额外参数
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        base_model_kwargs = base_model_kwargs or {}
        
        # 加载基座模型（CPU，FP32）
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            load_in_4bit=False,
            torch_dtype=torch.float32,
            device_map="cpu",
            **base_model_kwargs
        )
        
        # 根据策略合并
        if strategy == "average":
            self._merge_average(base_model, adapter_paths, output_path)
        elif strategy == "weighted":
            self._merge_weighted(base_model, adapter_paths, weights, output_path)
        elif strategy == "task_vector":
            self._merge_task_vector(base_model, adapter_paths, output_path)
    
    def _merge_average(
        self,
        base_model,
        adapter_paths: List[str],
        output_path: str
    ):
        """简单平均合并"""
        logger.info("Merging adapters with simple average...")
        
        # 加载第一个适配器作为基础
        merged_model = PeftModel.from_pretrained(
            base_model,
            adapter_paths[0]
        )
        
        # 累加其他适配器
        for adapter_path in adapter_paths[1:]:
            adapter_model = PeftModel.from_pretrained(
                base_model,
                adapter_path
            )
            
            # 累加 LoRA 层权重
            for name, param in adapter_model.named_parameters():
                if "lora_" in name:
                    merged_param = merged_model.state_dict().get(name)
                    if merged_param is not None:
                        merged_param.add_(param)
        
        # 平均
        num_adapters = len(adapter_paths)
        for name, param in merged_model.named_parameters():
            if "lora_" in name:
                param.div_(num_adapters)
        
        # 保存
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        
        # 保存 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_paths[0])
        tokenizer.save_pretrained(output_path)
    
    def _merge_weighted(
        self,
        base_model,
        adapter_paths: List[str],
        weights: List[float],
        output_path: str
    ):
        """加权平均合并"""
        if weights is None:
            weights = [1.0 / len(adapter_paths)] * len(adapter_paths)
        
        assert len(weights) == len(adapter_paths)
        
        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        logger.info(f"Using weights: {weights}")
        
        # 类似平均合并，但使用加权
        ...
    
    def _merge_task_vector(
        self,
        base_model,
        adapter_paths: List[str],
        output_path: str
    ):
        """Task Vector 合并（基于方向）"""
        logger.info("Merging with Task Vector strategy...")
        
        # 获取基座模型参数方向
        base_params = {
            name: param.clone() 
            for name, param in base_model.named_parameters()
        }
        
        # 累加方向向量
        for adapter_path in adapter_paths:
            adapter_model = PeftModel.from_pretrained(
                base_model,
                adapter_path
            )
            
            for name, param in adapter_model.named_parameters():
                if "lora_" in name:
                    base_name = name.replace("lora_", "")
                    if base_name in base_params:
                        base_params[base_name].add_(param)
        
        # 应用方向
        for name, param in base_model.named_parameters():
            if name in base_params:
                param.copy_(base_params[name])
        
        base_model.save_pretrained(output_path)
```

### 6.2 评估框架

```python
class MultiModelEvaluator:
    """多模型评估器"""
    
    def __init__(self, registry):
        self.registry = registry
    
    def evaluate(
        self,
        adapter_name: str,
        eval_tasks: List[str],
        eval_dataset: dict,
        batch_size: int = 4
    ) -> dict:
        """
        评估适配器在多个任务上的表现
        
        Args:
            adapter_name: 适配器名称
            eval_tasks: 评估任务列表
            eval_dataset: 数据集字典 {task_name: dataset}
            batch_size: 批次大小
        
        Returns:
            评估结果
        """
        # 加载模型
        model, tokenizer = self.registry.load_adapter_model(adapter_name)
        
        results = {}
        for task_name, dataset in eval_tasks:
            metrics = self._evaluate_task(
                model, tokenizer, dataset, batch_size
            )
            results[task_name] = metrics
        
        return results
    
    def _evaluate_task(
        self,
        model,
        tokenizer,
        dataset,
        batch_size: int
    ) -> dict:
        """评估单个任务"""
        # 通用的评估逻辑
        # 根据任务类型选择评估指标
        raise NotImplementedError("Implement task-specific evaluation")
    
    def compare_adapters(
        self,
        adapter_names: List[str],
        eval_tasks: List[str],
        eval_dataset: dict
    ) -> dict:
        """
        比较多个适配器
        
        Returns:
            比较结果表格
        """
        all_results = {}
        
        for adapter_name in adapter_names:
            try:
                results = self.evaluate(
                    adapter_name, 
                    eval_tasks, 
                    eval_dataset
                )
                all_results[adapter_name] = results
            except Exception as e:
                logger.warning(f"Failed to evaluate {adapter_name}: {e}")
                all_results[adapter_name] = {"error": str(e)}
        
        # 生成比较表格
        comparison = self._generate_comparison_table(all_results, eval_tasks)
        
        return comparison
    
    def _generate_comparison_table(
        self,
        results: dict,
        tasks: List[str]
    ) -> dict:
        """生成比较表格"""
        table = {}
        
        for task in tasks:
            task_results = {}
            for adapter_name, adapter_results in results.items():
                if "error" not in adapter_results.get(task, {}):
                    task_results[adapter_name] = adapter_results[task]
            
            # 找最佳
            if task_results:
                best_adapter = max(
                    task_results.keys(),
                    key=lambda k: task_results[k].get("score", 0)
                )
                table[task] = {
                    "results": task_results,
                    "best": best_adapter,
                    "best_score": task_results[best_adapter].get("score", 0)
                }
        
        return table
```

---

## 7. 常见问题与解决方案

### 7.1 OOM（显存不足）

```python
class OOMHandler:
    """OOM 处理器"""
    
    @staticmethod
    def handle_oom(model_config: dict, lora_config: dict) -> dict:
        """
        根据当前 OOM 情况调整配置
        
        Returns:
            调整后的配置
        """
        suggestions = []
        
        # 建议 1：降低序列长度
        if "max_seq_length" not in model_config:
            suggestions.append("reduce max_seq_length to 1024 or 512")
            model_config["max_seq_length"] = 1024
        
        # 建议 2：增加 gradient accumulation
        suggestions.append("increase gradient_accumulation_steps")
        model_config["gradient_accumulation_steps"] = (
            model_config.get("gradient_accumulation_steps", 1) * 2
        )
        
        # 建议 3：减少 LoRA rank
        if lora_config["lora_rank"] > 32:
            suggestions.append("reduce lora_rank from {} to 32".format(
                lora_config["lora_rank"]
            ))
            lora_config["lora_rank"] = 32
            lora_config["lora_alpha"] = 64
        
        # 建议 4：减少 batch size
        suggestions.append("reduce per_device_batch_size to 1")
        model_config["per_device_batch_size"] = 1
        
        return model_config, lora_config, suggestions


def safe_load_model(model_path: str, **kwargs):
    """
    安全加载模型，带 OOM 重试
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **kwargs
            )
            return model
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM on attempt {attempt + 1}, retrying with reduced settings...")
                # 清理
                import gc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # 减少批次大小重试
                kwargs["max_seq_length"] = kwargs.get("max_seq_length", 2048) // 2
            else:
                raise
    raise RuntimeError("Failed to load model after retries")
```

### 7.2 适配器合并冲突

```python
class MergeConflictResolver:
    """解决合并冲突"""
    
    @staticmethod
    def resolve_conflict(
        adapter_paths: List[str],
        base_model_path: str,
        conflict_strategy: str = "latest"
    ) -> List[str]:
        """
        解决适配器之间的冲突
        
        Args:
            adapter_paths: 冲突的适配器路径列表
            base_model_path: 基座模型路径
            conflict_strategy: "latest" | "best_metric" | "majority"
        
        Returns:
            解决后的适配器列表
        """
        if conflict_strategy == "latest":
            # 选择最新的
            import os
            paths_with_mtime = [
                (p, os.path.getmtime(p)) 
                for p in adapter_paths
            ]
            paths_with_mtime.sort(key=lambda x: x[1], reverse=True)
            return [p for p, _ in paths_with_mtime]
        
        elif conflict_strategy == "best_metric":
            # 选择指标最好的
            # 需要从 metadata 读取指标
            raise NotImplementedError("Implement best_metric resolution")
        
        elif conflict_strategy == "majority":
            # 选择在多数任务上表现好的
            raise NotImplementedError("Implement majority resolution")
        
        return adapter_paths
```

### 7.3 Tokenization 差异处理

```python
class TokenizerNormalizer:
    """
    处理不同模型 tokenizer 差异
    
    确保多模型训练时使用一致的 tokenization
    """
    
    @staticmethod
    def normalize_for_multi_model(
        tokenizers: dict,  # {model_name: tokenizer}
        common_special_tokens: list = None
    ) -> dict:
        """
        规范化多个模型的 tokenizer
        
        添加共同的 special tokens，确保一致的 tokenization 行为
        """
        common_special_tokens = common_special_tokens or [
            "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"
        ]
        
        # 收集所有 special tokens
        all_special_tokens = set()
        for tokenizer in tokenizers.values():
            all_special_tokens.update(tokenizer.special_tokens_map.keys())
        
        # 为每个 tokenizer 添加缺失的 special tokens
        for model_name, tokenizer in tokenizers.items():
            for token in common_special_tokens:
                if token not in tokenizer.special_tokens_map:
                    # 添加一个占位符
                    tokenizer.add_special_tokens({
                        token: f"[{token.strip('<|>').upper()}]"
                    })
            
            tokenizer.model_max_length = min(
                tokenizer.model_max_length for tokenizer in tokenizers.values()
            )
        
        return tokenizers
    
    @staticmethod
    def compute_tokenization_overhead(
        text: str,
        tokenizer_a,
        tokenizer_b
    ) -> dict:
        """
        计算两个 tokenizer 的 tokenization 开销差异
        
        用于评估不同 tokenizer 对训练效率的影响
        """
        tokens_a = tokenizer_a.encode(text)
        tokens_b = tokenizer_b.encode(text)
        
        return {
            "text_length": len(text),
            "tokens_a": len(tokens_a),
            "tokens_b": len(tokens_b),
            "overhead_ratio": len(tokens_a) / max(len(tokens_b), 1),
            "difference": len(tokens_a) - len(tokens_b),
        }
```

---

## 总结

本节课我们系统学习了 QLoRA 在多模型场景下的实战应用：

1. **不同模型家族的 QLoRA 配置**：
   - LLaMA：标准配置，最成熟稳定
   - Qwen：特殊 tokenizer 和长上下文支持
   - ChatGLM：RMSNorm 和合并 QKV 层
   - DeepSeek MoE：稀疏专家结构的特殊处理

2. **单卡多模型策略**：
   - 内存估算确保不会 OOM
   - 顺序加载：train → save → unload → next
   - 检查点管理实现训练恢复

3. **LoRA 目标层配置**：
   - 不同模型有不同的 attention 层结构
   - QKV 投影 vs 合并层
   - MLP 层配置的差异

4. **适配器注册与管理**：
   - 注册表跟踪所有适配器
   - 批量操作支持
   - 兼容性检查

5. **合并与评估**：
   - 多种合并策略（平均、加权、Task Vector）
   - 多任务评估框架
   - OOM 和冲突处理

6. **常见问题处理**：
   - OOM 的配置调整
   - 合并冲突解决
   - Tokenization 差异规范化

---

## 扩展阅读

- [QLoRA 论文](https://arxiv.org/abs/2305.14314) - QLoRA 官方论文
- [LLaMA 架构解析](https://arxiv.org/abs/2302.13971) - LLaMA 论文
- [Qwen2 技术报告](https://arxiv.org/abs/2309.16609) - Qwen2 技术报告
- [ChatGLM 技术报告](https://arxiv.org/abs/2303.18223) - ChatGLM 技术报告
- [DeepSeek MoE 论文](https://arxiv.org/abs/2401.06066) - DeepSeek MoE 论文
- [bitsandbytes 文档](https://github.com/bitsandbytes-foundation/bitsandbytes) - 量化库文档
- [PEFT 库文档](https://github.com/huggingface/peft) - LoRA 适配器库

---

## 复习题

1. **在单卡上训练 7B、13B、33B 三个不同规模的模型，显存分别是多少？如何规划训练顺序？**

2. **为什么 ChatGLM 的 `query_key_value` 层需要单独处理，而 LLaMA 的 `q_proj`、`k_proj`、`v_proj` 可以分开处理？**

3. **在多模型适配器合并时，"平均合并"和"Task Vector 合并"各有什么优缺点？适用场景是什么？**

4. **假设 Qwen 和 LLaMA 训练同一个数据集，如何处理它们的 tokenizer 差异？请设计一个规范化方案。**

5. **在 DeepSeek MoE 上应用 LoRA 时，为什么只需要 targeting `gate_proj` 和 `up_proj`，而不需要 `down_proj`？如果加入 `down_proj` 会发生什么问题？**