"""
LoRA PyTorch实现 - 从底层理解LoRA原理
=====================================

本模块从零实现LoRA（Low-Rank Adaptation）核心逻辑，不依赖PEFT库。
通过亲手实现，能够：
1. 理解LoRA的数学原理：y = Wx + (scaling * BA)x
2. 掌握权重合并的原理：W_merged = W + scaling * (B @ A)
3. 了解训练稳定性的关键：零初始化设计
4. 分析可训练参数与全量微调的差异

核心公式：
- LoRA-A: 随机初始化，除以sqrt(r)的Xavier变体
- LoRA-B: 零初始化，确保训练初期BA=0
- 缩放因子: scaling = alpha / r

参考论文：LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """LoRA配置参数"""

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    merge_weights: bool = False


class LoRALinear(nn.Module):
    """
    LoRA线性层的PyTorch实现

    核心公式: y = Wx + (scaling * BA)x
    其中 BA = lora_B @ lora_A

    设计要点：
    1. 冻结原始层权重，仅训练A、B矩阵
    2. 矩阵A随机初始化（除以sqrt(r)），矩阵B零初始化
    3. scaling因子控制LoRA分支的贡献度
    4. 支持dropout增强泛化能力
    5. 支持权重合并用于推理加速

    参数说明：
    - original_layer: 原始预训练层，权重被冻结
    - r: LoRA秩，决定低秩矩阵的维度
    - lora_alpha: 缩放因子，用于调整BA的贡献度
    - lora_dropout: LoRA分支的dropout概率
    - merge_weights: 是否在训练前合并权重
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        """
        初始化LoRA线性层

        Args:
            original_layer: 原始预训练层，权重被冻结
            r: LoRA秩，决定低秩矩阵的维度
            lora_alpha: 缩放因子，用于调整BA的贡献度
            lora_dropout: LoRA分支的dropout概率
            merge_weights: 是否在训练前合并权重
        """
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights

        self._has_merged = False

        if r <= 0:
            raise ValueError(f"r must be positive, got {r}")

        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.in_features = in_features
        self.out_features = out_features

        self.scaling = self.lora_alpha / self.r

        self.lora_A = nn.Parameter(torch.randn(r, in_features) / math.sqrt(r))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        公式: y = Wx + scaling * (lora_B @ lora_A @ x)

        训练阶段：返回原始输出 + LoRA分支输出
        合并后：直接返回合并权重计算的结果
        """
        if self._has_merged:
            return self.original_layer(x)

        original_output = self.original_layer(x)

        if self.r == 0:
            return original_output

        x_dropped = self.lora_dropout(x)

        lora_output = x_dropped @ self.lora_A.T @ self.lora_B.T

        return original_output + self.scaling * lora_output

    def merge(self) -> None:
        """
        将LoRA权重合并到原始层中，用于推理加速

        计算: W_merged = W + scaling * (B @ A)
        合并后无需额外计算LoRA分支，前向传播更快

        注意：合并操作是不可逆的，合并前应保存原始权重
        """
        if self.r == 0 or self._has_merged:
            return

        delta_w = self.lora_B @ self.lora_A
        merged_weight = self.original_layer.weight + self.scaling * delta_w

        self.original_layer.weight = nn.Parameter(merged_weight)

        self._has_merged = True

        if self.lora_A is not None:
            self.lora_A.requires_grad = False
        if self.lora_B is not None:
            self.lora_B.requires_grad = False

    def unmerge(self) -> None:
        """
        从合并状态分离（如果需要恢复原始模型）

        注意：仅当之前调用过merge()且保留了原始权重时有效
        """
        if not self._has_merged:
            return

        self._has_merged = False

        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def get_lora_params(self) -> Tuple[nn.Parameter, nn.Parameter]:
        """获取LoRA参数A和B"""
        return self.lora_A, self.lora_B

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"r={self.r}, alpha={self.lora_alpha}, "
            f"dropout={self.lora_dropout}, "
            f"merged={self._has_merged}"
        )


class LoRALayerWrapper(nn.Module):
    """
    LoRA层包装器 - 支持替换模型中任意线性层

    用于将预训练模型中的指定层替换为LoRA实现
    """

    def __init__(
        self,
        model: nn.Module,
        target_names: list[str],
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        """
        初始化LoRA层包装器

        Args:
            model: 预训练模型
            target_names: 目标层名称列表，如["q_proj", "v_proj"]
            r: LoRA秩
            lora_alpha: 缩放因子
            lora_dropout: dropout概率
        """
        super().__init__()
        self.model = model
        self.target_names = target_names
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self._replaced_modules: Dict[str, Tuple[nn.Module, LoRALinear]] = {}

        self._inject_lora()

    def _inject_lora(self) -> None:
        """注入LoRA层到目标位置"""
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_names):
                if isinstance(module, nn.Linear):
                    self._replace_with_lora(name, module)

    def _replace_with_lora(self, name: str, original_layer: nn.Linear) -> None:
        """替换原始层为LoRA实现"""
        lora_layer = LoRALinear(
            original_layer,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )

        self._replaced_modules[name] = (original_layer, lora_layer)

        parts = name.split(".")
        current = self.model
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], lora_layer)

    def merge(self) -> None:
        """合并所有LoRA权重"""
        for name, (_, lora_layer) in self._replaced_modules.items():
            lora_layer.merge()

    def get_trainable_params(self) -> int:
        """获取可训练参数数量"""
        total = 0
        for _, lora_layer in self._replaced_modules.values():
            total += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
        return total

    def restore(self) -> None:
        """恢复原始模型（如果未合并）"""
        for name, (original_layer, _) in self._replaced_modules.items():
            parts = name.split(".")
            current = self.model
            for part in parts[:-1]:
                current = getattr(current, part)
            setattr(current, parts[-1], original_layer)


def count_trainable_params(model: nn.Module) -> Tuple[int, int, float]:
    """
    统计模型中可训练参数数量

    Args:
        model: PyTorch模型

    Returns:
        (trainable_params, total_params, trainable_ratio)
    """
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    ratio = trainable_params / total_params * 100 if total_params > 0 else 0
    return trainable_params, total_params, ratio


def compare_full_vs_lora(
    model: nn.Module, target_names: list[str], r: int = 8
) -> Dict[str, Any]:
    """
    对比全量微调与LoRA的可训练参数量

    Args:
        model: 预训练模型
        target_names: 目标层名称列表
        r: LoRA秩

    Returns:
        包含详细对比信息的字典
    """
    trainable_full, total, _ = count_trainable_params(model)

    wrapper = LoRALayerWrapper(model, target_names, r=r)
    trainable_lora = wrapper.get_trainable_params()

    reduction = (1 - trainable_lora / trainable_full) * 100
    ratio_lora = trainable_lora / total * 100

    return {
        "full_finetune_trainable": trainable_full,
        "lora_trainable": trainable_lora,
        "reduction_percent": reduction,
        "lora_ratio_percent": ratio_lora,
        "total_params": total,
        "rank": r,
        "target_layers": target_names,
    }


def create_lora_training_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, device: str = "cuda"
) -> callable:
    """
    创建LoRA训练步骤函数

    Args:
        model: 包含LoRA层的模型
        optimizer: 优化器（仅优化LoRA参数）
        device: 计算设备

    Returns:
        训练步骤函数
    """
    model.train()

    def training_step(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        执行单个训练步骤

        Args:
            batch: 包含"input_ids", "attention_mask", "labels"等键的字典

        Returns:
            损失值
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss

    return training_step


class LoRAModelWrapper(nn.Module):
    """
    完整LoRA模型包装器

    简化版实现，用于包装AutoModelForCausalLM等HuggingFace模型
    """

    def __init__(
        self,
        base_model: nn.Module,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ):
        """
        初始化LoRA模型包装器

        Args:
            base_model: 基础预训练模型
            r: LoRA秩
            lora_alpha: 缩放因子
            lora_dropout: dropout概率
            target_modules: 目标模块名称列表
        """
        super().__init__()
        self.base_model = base_model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]

        self._inject_lora_to_model()

        self._register_base_model_hooks()

    def _inject_lora_to_model(self) -> None:
        """将LoRA注入到模型的所有目标层"""
        self._lora_layers: Dict[str, LoRALinear] = {}

        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALinear(
                        module,
                        r=self.r,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                    )
                    self._lora_layers[name] = lora_layer

                    parts = name.split(".")
                    current = self.base_model
                    for part in parts[:-1]:
                        current = getattr(current, part)
                    setattr(current, parts[-1], lora_layer)

    def _register_base_model_hooks(self) -> None:
        """注册基础模型相关钩子"""
        pass

    def forward(self, *args, **kwargs):
        """前向传播（透传给基础模型）"""
        return self.base_model(*args, **kwargs)

    def merge_and_unload(self) -> nn.Module:
        """
        合并所有LoRA权重到基础模型

        Returns:
            合并后的基础模型
        """
        for name, lora_layer in self._lora_layers.items():
            lora_layer.merge()

        return self.base_model

    def get_trainable_params_count(self) -> Tuple[int, int, float]:
        """获取可训练参数统计"""
        trainable, total, ratio = count_trainable_params(self)
        lora_only = sum(
            layer.lora_A.numel() + layer.lora_B.numel()
            for layer in self._lora_layers.values()
        )
        return lora_only, total, lora_only / total * 100

    def print_trainable_parameters(self) -> None:
        """打印可训练参数信息"""
        trainable, total, ratio = self.get_trainable_params_count()
        print(
            f"trainable params: {trainable:,} || "
            f"all params: {total:,} || "
            f"{ratio:.3f}% trainable"
        )


def demo_lora_math():
    """
    演示LoRA的数学原理

    展示：
    1. 矩阵A和B的形状
    2. scaling因子的作用
    3. 权重合并的计算过程
    """
    print("=" * 60)
    print("LoRA数学原理演示")
    print("=" * 60)

    r = 8
    alpha = 16
    in_features = 512
    out_features = 512

    scaling = alpha / r
    print(f"\n1. 配置参数:")
    print(f"   r (rank) = {r}")
    print(f"   alpha = {alpha}")
    print(f"   scaling = alpha/r = {scaling}")

    print(f"\n2. 矩阵形状:")
    print(f"   LoRA-A: ({r}, {in_features})")
    print(f"   LoRA-B: ({out_features}, {r})")
    print(f"   BA: ({out_features}, {in_features})")

    print(f"\n3. 参数量计算:")
    a_params = r * in_features
    b_params = out_features * r
    total_lora = a_params + b_params
    print(f"   A参数: {r} × {in_features} = {a_params:,}")
    print(f"   B参数: {out_features} × {r} = {b_params:,}")
    print(f"   LoRA总参数: {total_lora:,}")

    print(f"\n4. 相比全量权重压缩比:")
    full_params = out_features * in_features
    compression = full_params / total_lora
    print(f"   全量权重: {out_features} × {in_features} = {full_params:,}")
    print(f"   压缩比: {compression:.1f}x")

    print(f"\n5. 权重合并演示:")
    W = torch.randn(out_features, in_features)
    A = torch.randn(r, in_features) / math.sqrt(r)
    B = torch.zeros(out_features, r)

    delta_W = B @ A
    print(f"   delta_W形状: {delta_W.shape}")
    print(f"   delta_W (B@A) 全零验证: {delta_W.abs().max().item():.6f}")

    print(f"\n6. 合并后权重:")
    W_merged = W + scaling * delta_W
    print(f"   W_merged形状: {W_merged.shape}")
    print(f"   合并后与原始权重差异: {(W_merged - W).abs().max().item():.6f}")
    print(f"   (零初始化确保训练初期合并权重等于原始权重)")

    print("\n" + "=" * 60)


def demo_lora_training_loop():
    """
    演示LoRA训练循环

    展示如何在纯PyTorch中实现LoRA训练
    """
    print("\n" + "=" * 60)
    print("LoRA训练循环演示")
    print("=" * 60)

    torch.manual_seed(42)

    print("\n1. 创建模拟模型和LoRA配置:")
    vocab_size = 1000
    embed_dim = 256
    hidden_dim = 512

    class MockLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=4,
                        dim_feedforward=hidden_dim,
                        batch_first=True,
                    )
                ]
            )
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        def forward(self, input_ids, attention_mask=None, labels=None):
            x = self.embed(input_ids)
            for layer in self.layers:
                x = layer(x)
            logits = self.lm_head(x)

            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )

            return type("Output", (), {"loss": loss, "logits": logits})()

    model = MockLM()
    print(f"   模型创建完成")

    target_modules = ["q_proj", "v_proj"]
    r = 4
    lora_alpha = 8

    print(f"\n2. 注入LoRA层:")
    print(f"   目标模块: {target_modules}")
    print(f"   r = {r}, alpha = {lora_alpha}")

    wrapper = LoRAModelWrapper(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
    )

    print(f"   LoRA层注入完成")
    wrapper.print_trainable_parameters()

    print(f"\n3. 对比全量微调 vs LoRA:")
    comparison = compare_full_vs_lora(model, target_modules, r)
    print(f"   全量微调可训练参数: {comparison['full_finetune_trainable']:,}")
    print(f"   LoRA可训练参数: {comparison['lora_trainable']:,}")
    print(f"   参数减少比例: {comparison['reduction_percent']:.2f}%")

    print(f"\n4. 模拟训练步骤:")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, wrapper.parameters()), lr=1e-4
    )

    dummy_batch = {
        "input_ids": torch.randint(0, vocab_size, (2, 10)),
        "attention_mask": torch.ones(2, 10),
        "labels": torch.randint(0, vocab_size, (2, 10)),
    }

    train_step = create_lora_training_step(wrapper, optimizer)

    initial_loss = train_step(dummy_batch).item()
    print(f"   初始损失: {initial_loss:.4f}")

    for step in range(3):
        loss = train_step(dummy_batch)
        print(f"   Step {step + 1} 损失: {loss:.4f}")

    print(f"\n5. 权重合并演示:")
    print(f"   合并前 - 检查原始层权重是否冻结: ", end="")
    original_weight = wrapper.base_model.layers[0].self_attn.q_proj.weight
    print(f"requires_grad={original_weight.requires_grad}")

    merged_model = wrapper.merge_and_unload()
    print(
        f"   合并后 - 原始层权重requires_grad: {merged_model.layers[0].self_attn.q_proj.weight.requires_grad}"
    )

    print("\n" + "=" * 60)
    print("LoRA演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_lora_math()
    demo_lora_training_loop()
