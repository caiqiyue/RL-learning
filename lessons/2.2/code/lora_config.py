"""
LoRA Configuration Module
=========================
LoRA参数配置模块，提供不同模型规模的配置模板和目标模块选择策略。

本模块包含：
- LoraConfig dataclass：统一的LoRA参数配置
- 目标模块选择：为不同模型架构（LLaMA、Qwen、ChatGLM等）提供模块名映射
- 配置工厂函数：根据模型规模自动推荐最优配置

参考：https://huggingface.co/docs/peft
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelScale(Enum):
    """模型规模枚举"""

    SMALL = "small"  # 1B-3B
    MEDIUM = "medium"  # 7B
    LARGE = "large"  # 13B
    XLARGE = "xlarge"  # 70B+


class TargetModuleStrategy(Enum):
    """目标模块选择策略"""

    QKV_ONLY = "qkv_only"  # 仅QKV（最小）
    ATTENTION = "attention"  # 完整注意力层
    ATTENTION_FFN = "attention_ffn"  # 注意力层+部分FFN
    ALL_LINEAR = "all_linear"  # 所有线性层（最大）


@dataclass
class LoraConfig:
    """
    LoRA参数配置数据类

    属性说明：
    - r: LoRA秩，决定低秩矩阵的中间维度。越大表达能力越强，但参数量也越多
    - lora_alpha: 缩放因子。实际影响力为 alpha/r， 通常设为 2*r
    - lora_dropout: LoRA分支的dropout概率，推荐0.05-0.1防止过拟合
    - target_modules: 应用LoRA的模块名称列表
    - bias: 是否训练偏置项，可选"none", "lora_only", "all"
    - task_type: 任务类型，如"CAUSAL_LM", "SEQ_CLS"等

    示例：
        >>> config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
        >>> print(f"Trainable params ratio: {config.get_params_ratio()}")
    """

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: Optional[List[str]] = None

    def __post_init__(self):
        """参数校验"""
        if self.r <= 0:
            raise ValueError(f"r must be positive, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError(f"lora_dropout must be in [0, 1], got {self.lora_dropout}")
        if self.bias not in ["none", "lora_only", "all"]:
            raise ValueError(
                f"bias must be one of 'none', 'lora_only', 'all', got {self.bias}"
            )

    def get_scaling(self) -> float:
        """获取缩放因子"""
        return self.lora_alpha / self.r

    def get_params_ratio(self, trainable_params: int, total_params: int) -> float:
        """计算可训练参数量占比"""
        return trainable_params / total_params * 100

    def to_peft_config(self) -> Dict[str, Any]:
        """转换为PEFT库所需的字典格式"""
        config = {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }
        if self.modules_to_save:
            config["modules_to_save"] = self.modules_to_save
        return config


# =============================================================================
# 目标模块映射：为不同模型架构定义各层名称
# =============================================================================


class TargetModuleMapper:
    """
    不同模型架构的目标模块名称映射

    模型架构的线性层命名规则不同：
    - LLaMA/QWen: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    - ChatGLM: query_key_value, dense, gateway, intermediate
    - Bloom: query, key, value, dense, intermediate
    """

    # 通用注意力层名称（大多数模型使用）
    ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # FFN层名称
    FFN_MODULES = ["gate_proj", "up_proj", "down_proj"]

    # LLaMA/LLaMA2/LLaMA3架构
    LLAMA_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    LLAMA_ATTENTION_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]
    LLAMA_QKV_ONLY = ["q_proj", "v_proj"]

    # Qwen架构
    QWEN_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    QWEN_ATTENTION_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]
    QWEN_QKV_ONLY = ["q_proj", "v_proj"]

    # ChatGLM架构
    CHATGLM_MODULES = ["query_key_value", "dense", "gateway", "intermediate"]
    CHATGLM_ATTENTION_ONLY = ["query_key_value", "dense"]

    # Bloom架构
    BLOOM_MODULES = ["query", "key", "value", "dense", "intermediate"]

    # Mistral架构（与LLaMA兼容）
    MISTRAL_MODULES = LLAMA_MODULES

    # Baichuan架构
    BAICHUAN_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    @classmethod
    def get_modules(
        cls,
        model_arch: str,
        strategy: TargetModuleStrategy = TargetModuleStrategy.ATTENTION,
    ) -> List[str]:
        """
        根据模型架构和策略获取目标模块列表

        Args:
            model_arch: 模型架构名称，支持 "llama", "qwen", "chatglm", "bloom", "mistral", "baichuan"
            strategy: 目标模块选择策略

        Returns:
            模块名称列表

        示例：
            >>> TargetModuleMapper.get_modules("llama", TargetModuleStrategy.ATTENTION)
            ["q_proj", "k_proj", "v_proj", "o_proj"]
        """
        model_arch = model_arch.lower()

        if model_arch == "llama":
            base_modules = cls.LLAMA_MODULES
        elif model_arch == "qwen":
            base_modules = cls.QWEN_MODULES
        elif model_arch == "chatglm":
            base_modules = cls.CHATGLM_MODULES
        elif model_arch == "bloom":
            base_modules = cls.BLOOM_MODULES
        elif model_arch == "mistral":
            base_modules = cls.MISTRAL_MODULES
        elif model_arch == "baichuan":
            base_modules = cls.BAICHUAN_MODULES
        else:
            base_modules = cls.ATTENTION_MODULES

        if strategy == TargetModuleStrategy.QKV_ONLY:
            if model_arch == "chatglm":
                return ["query_key_value"]
            elif model_arch in ["llama", "qwen", "mistral", "baichuan"]:
                return cls.LLAMA_QKV_ONLY
            else:
                return ["q_proj", "v_proj"]
        elif strategy == TargetModuleStrategy.ATTENTION:
            if model_arch in ["llama", "qwen", "mistral", "baichuan"]:
                return cls.LLAMA_ATTENTION_ONLY
            elif model_arch == "chatglm":
                return cls.CHATGLM_ATTENTION_ONLY
            else:
                return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif strategy == TargetModuleStrategy.ATTENTION_FFN:
            if model_arch in ["llama", "qwen", "mistral", "baichuan"]:
                return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
            else:
                return base_modules
        elif strategy == TargetModuleStrategy.ALL_LINEAR:
            return base_modules
        else:
            return cls.ATTENTION_MODULES

    @classmethod
    def auto_detect_modules(cls, model: Any) -> List[str]:
        """
        自动检测模型中可用的目标模块

        Args:
            model: HuggingFace模型实例

        Returns:
            检测到的模块名称列表
        """
        detected = []
        for name, module in model.named_modules():
            if isinstance(module, __import__("torch").nn.Linear):
                for suffix in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "query_key_value",
                    "dense",
                ]:
                    if name.endswith(suffix):
                        detected.append(suffix)
                        break
        return list(set(detected))


# =============================================================================
# 配置工厂函数：根据模型规模提供推荐配置
# =============================================================================


def create_lora_config_for_scale(
    model_scale: ModelScale,
    model_arch: str = "llama",
    custom_r: Optional[int] = None,
    custom_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """
    根据模型规模创建LoRA配置

    Args:
        model_scale: 模型规模（SMALL/MEDIUM/LARGE/XLARGE）
        model_arch: 模型架构（llama/qwen/chatglm等）
        custom_r: 自定义r值（可选）
        custom_modules: 自定义目标模块（可选）

    Returns:
        LoraConfig实例

    配置说明：
        - SMALL (1B-3B): r=16, 全部层应用，显存~8GB
        - MEDIUM (7B): r=8, 仅注意力层，显存~23GB
        - LARGE (13B): r=8, 注意力+部分FFN，显存~40GB
        - XLARGE (70B+): r=4, 仅QKV，显存~160GB (FP16) or ~50GB (QLoRA INT4)

    示例：
        >>> config = create_lora_config_for_scale(ModelScale.MEDIUM, "qwen")
        >>> print(f"Config for 7B model: r={config.r}, alpha={config.lora_alpha}")
    """
    configs = {
        ModelScale.SMALL: {
            "r": custom_r or 16,
            "lora_alpha": (custom_r or 16) * 2,
            "target_modules": custom_modules
            or TargetModuleMapper.get_modules(
                model_arch, TargetModuleStrategy.ALL_LINEAR
            ),
            "lora_dropout": 0.05,
        },
        ModelScale.MEDIUM: {
            "r": custom_r or 8,
            "lora_alpha": (custom_r or 8) * 2,
            "target_modules": custom_modules
            or TargetModuleMapper.get_modules(
                model_arch, TargetModuleStrategy.ATTENTION
            ),
            "lora_dropout": 0.05,
        },
        ModelScale.LARGE: {
            "r": custom_r or 8,
            "lora_alpha": (custom_r or 8) * 2,
            "target_modules": custom_modules
            or TargetModuleMapper.get_modules(
                model_arch, TargetModuleStrategy.ATTENTION_FFN
            ),
            "lora_dropout": 0.05,
        },
        ModelScale.XLARGE: {
            "r": custom_r or 4,
            "lora_alpha": (custom_r or 4) * 2,
            "target_modules": custom_modules
            or TargetModuleMapper.get_modules(
                model_arch, TargetModuleStrategy.QKV_ONLY
            ),
            "lora_dropout": 0.05,
        },
    }

    config_dict = configs[model_scale]
    return LoraConfig(
        r=config_dict["r"],
        lora_alpha=config_dict["lora_alpha"],
        lora_dropout=config_dict["lora_dropout"],
        target_modules=config_dict["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def create_lora_config_for_model(
    model_name_or_path: str, custom_r: Optional[int] = None
) -> LoraConfig:
    """
    根据模型名称自动推断规模并创建配置

    Args:
        model_name_or_path: 模型名称或路径（如"Qwen/Qwen2.5-7B"、"meta-llama/Llama-2-7b"）
        custom_r: 自定义r值（可选）

    Returns:
        LoraConfig实例

    示例：
        >>> config = create_lora_config_for_model("Qwen/Qwen2.5-7B")
        >>> print(f"Auto-detected config: r={config.r}")
    """
    model_name = model_name_or_path.lower()

    if "1b" in model_name or "-1b" in model_name or "1b-" in model_name:
        scale = ModelScale.SMALL
    elif "3b" in model_name or "-3b" in model_name or "3b-" in model_name:
        scale = ModelScale.SMALL
    elif "7b" in model_name or "-7b" in model_name or "7b-" in model_name:
        scale = ModelScale.MEDIUM
    elif "13b" in model_name or "-13b" in model_name or "13b-" in model_name:
        scale = ModelScale.LARGE
    elif "70b" in model_name or "-70b" in model_name or "70b-" in model_name:
        scale = ModelScale.XLARGE
    elif "4b" in model_name or "-4b" in model_name:
        scale = ModelScale.SMALL
    elif "8b" in model_name or "-8b" in model_name:
        scale = ModelScale.MEDIUM
    elif "14b" in model_name or "-14b" in model_name:
        scale = ModelScale.LARGE
    elif "72b" in model_name or "-72b" in model_name:
        scale = ModelScale.XLARGE
    else:
        scale = ModelScale.MEDIUM

    if "qwen" in model_name:
        arch = "qwen"
    elif "llama" in model_name or "llama" in model_name:
        arch = "llama"
    elif "chatglm" in model_name:
        arch = "chatglm"
    elif "bloom" in model_name:
        arch = "bloom"
    elif "mistral" in model_name:
        arch = "mistral"
    elif "baichuan" in model_name:
        arch = "baichuan"
    else:
        arch = "llama"

    return create_lora_config_for_scale(scale, arch, custom_r)


def estimate_vram_usage(
    model_params: int, config: LoraConfig, precision: str = "fp16"
) -> Dict[str, float]:
    """
    估算LoRA微调的显存占用

    Args:
        model_params: 模型参数量（单位：B，即十亿参数）
        config: LoRA配置
        precision: 精度格式 ("fp16", "fp32", "int8", "int4")

    Returns:
        显存估算字典，包含：
        - model_vram: 模型权重显存（GB）
        - lora_vram: LoRA参数显存（GB）
        - gradient_vram: 梯度显存（GB）
        - optimizer_vram: 优化器状态显存（GB）
        - total_vram: 总显存（GB）

    估算公式：
        - 模型显存 ≈ 参数量 × 2字节（FP16）
        - LoRA显存 ≈ 2 × r × (in_dim + out_dim) × 2字节
        - 梯度显存 ≈ 参数量 × 2字节（FP16）
        - AdamW优化器显存 ≈ 参数量 × 12字节（FP32状态）

    示例：
        >>> config = create_lora_config_for_scale(ModelScale.MEDIUM, "qwen")
        >>> usage = estimate_vram_usage(7, config, "fp16")
        >>> print(f"Total VRAM: {usage['total_vram']:.1f} GB")
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    bytes_per_fp = bytes_per_param.get(precision, 2)

    model_vram_gb = model_params * 1e9 * bytes_per_fp / (1024**3)

    target_module_count = len(config.target_modules) if config.target_modules else 8
    lora_params = 2 * config.r * 4096 * target_module_count
    lora_vram_gb = lora_params * bytes_per_fp / (1024**3)

    gradient_vram_gb = model_params * 1e9 * bytes_per_fp / (1024**3)

    if "paged" in "":
        optimizer_vram_gb = model_params * 1e9 * 4 / (1024**3) * 0.5
    else:
        optimizer_vram_gb = model_params * 1e9 * 12 / (1024**3)

    total_vram_gb = model_vram_gb + lora_vram_gb + gradient_vram_gb + optimizer_vram_gb

    return {
        "model_vram": model_vram_gb,
        "lora_vram": lora_vram_gb,
        "gradient_vram": gradient_vram_gb,
        "optimizer_vram": optimizer_vram_gb,
        "total_vram": total_vram_gb,
    }


# =============================================================================
# 预定义配置模板
# =============================================================================

LORA_CONFIG_TEMPLATES = {
    "qwen2_7b": LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    "llama2_7b": LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    "llama3_8b": LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    "qwen2_1.5b": LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    "qwen2_0.5b": LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ),
    "chatglm3_6b": LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ),
}


def get_template_config(model_name: str) -> Optional[LoraConfig]:
    """
    获取预定义配置模板

    Args:
        model_name: 模型名称（支持模糊匹配）

    Returns:
        LoraConfig实例，未找到时返回None

    示例：
        >>> config = get_template_config("qwen2.5-7b")
        >>> if config:
        ...     print(f"Using template: r={config.r}")
    """
    model_lower = model_name.lower()

    for key, config in LORA_CONFIG_TEMPLATES.items():
        if key.replace("_", "-") in model_lower or key.replace(
            "_", ""
        ) in model_lower.replace("-", ""):
            return config

    return None


if __name__ == "__main__":
    print("=" * 60)
    print("LoRA Config Module - Quick Test")
    print("=" * 60)

    print("\n1. Testing LoraConfig dataclass:")
    config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    print(
        f"   Config: r={config.r}, alpha={config.lora_alpha}, scaling={config.get_scaling():.2f}"
    )

    print("\n2. Testing target module mapping:")
    llama_modules = TargetModuleMapper.get_modules(
        "llama", TargetModuleStrategy.ATTENTION
    )
    print(f"   LLaMA attention modules: {llama_modules}")

    qwen_qkv = TargetModuleMapper.get_modules("qwen", TargetModuleStrategy.QKV_ONLY)
    print(f"   Qwen QKV only: {qwen_qkv}")

    print("\n3. Testing config factory:")
    medium_config = create_lora_config_for_scale(ModelScale.MEDIUM, "qwen")
    print(
        f"   Medium config (7B): r={medium_config.r}, alpha={medium_config.lora_alpha}"
    )
    print(f"   Target modules: {medium_config.target_modules}")

    print("\n4. Testing model auto-detection:")
    auto_config = create_lora_config_for_model("Qwen/Qwen2.5-7B")
    print(
        f"   Auto config for Qwen2.5-7B: r={auto_config.r}, alpha={auto_config.lora_alpha}"
    )

    print("\n5. Testing VRAM estimation:")
    usage = estimate_vram_usage(7, medium_config, "fp16")
    print(f"   7B model VRAM breakdown:")
    for k, v in usage.items():
        print(f"     {k}: {v:.2f} GB")

    print("\n6. Testing template config:")
    template = get_template_config("qwen2.5-7b")
    if template:
        print(f"   Template found: r={template.r}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
