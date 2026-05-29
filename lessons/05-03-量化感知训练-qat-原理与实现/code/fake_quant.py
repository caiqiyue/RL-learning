"""
Fake Quantization (伪量化) 实现

展示Fake量化的前向传播（含真实量化操作）和反向传播（使用STE绕过取整）。
Fake量化让模型在FP32训练时"感知"到量化误差，同时保持梯度可正常传递。

STE (Straight-Through Estimator):
    Forward:  y = round(x)           (不可导)
    Backward: ∂L/∂x = ∂L/∂y * 1      (梯度直接穿透，忽略round)
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class FakeQuantize(Function):
    """
    自定义Fake量化算子，实现STE反向传播。

    前向传播: FP32 → 量化(INT8) → 反量化(FP32)
    反向传播: 梯度绕过取整操作，直接传递（STE）
    """

    @staticmethod
    def forward(ctx, x, scale, qmin=-128, qmax=127):
        """
        Args:
            x: 输入FP32张量
            scale: 量化缩放因子（标量或与x形状匹配的张量）
            qmin: 量化最小值（INT8有符号为-128）
            qmax: 量化最大值（INT8有符号为127）
        Returns:
            经过Fake量化（再反量化）的FP32张量
        """
        # 计算缩放因子（处理shape广播）
        if scale.dim() == 0:
            scale_val = scale.item()
            scale_tensor = scale
        else:
            scale_tensor = scale

        # 量化: round(x / scale) → INT8范围
        x_quant = torch.round(x / scale_tensor)
        x_quant = x_quant.clamp(qmin, qmax)

        # 反量化: INT8 → FP32（恢复浮点精度用于后续计算）
        x_fake = x_quant * scale_tensor

        # 保存quantized值供调试/分析（不在backward中使用）
        ctx.save_for_backward(x_quant)
        return x_fake

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：STE实现。

        梯度绕过round操作，直接传递。
        ∂L/∂x = ∂L/∂x_fake * 1 ≈ ∂L/∂y
        （假设 round 的梯度为1，即恒等函数）
        """
        (x_quant,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None, None  # x, scale, qmin, qmax


def fake_quantize(x, scale, qmin=-128, qmax=127):
    """
    便捷封装：对输入张量执行Fake量化。

    Args:
        x: 输入FP32张量
        scale: 缩放因子
        qmin/qmax: 量化范围
    Returns:
        Fake量化后的FP32张量
    """
    return FakeQuantize.apply(x, scale, qmin, qmax)


class FakeQuantizedLinear(nn.Module):
    """
    包装标准nn.Linear，在其权重上应用Fake量化。
    前向传播时权重经过 FakeQuantize → dequantize，
    输出仍是FP32，可正常用于后续计算和梯度回传。
    """

    def __init__(self, in_features, out_features, bias=True, qbit=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qbit = qbit
        self.qmax = 2 ** (qbit - 1) - 1
        self.qmin = -self.qmax - 1 if qbit > 1 else 0

        # 可学习scale（per-output-channel）
        self.scale = nn.Parameter(torch.ones(out_features))

        # 标准FP32权重和偏置
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

        # 初始化权重（Kaiming初始化，适合ReLU类网络）
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        前向传播:
            weight (FP32) → FakeQuantize → weight_fake (FP32) → matmul
        """
        # 对权重进行per-channel fake量化
        w_scale = self.scale.unsqueeze(1)  # [out_features, 1]
        w_fake = fake_quantize(self.weight, w_scale, self.qmin, self.qmax)
        return nn.functional.linear(x, w_fake, self.bias)


def demo_fake_quant():
    """
    演示Fake量化基本行为：
    1. 前向：展示量化→反量化后的值与原始FP32的差异（量化误差）
    2. 反向：验证STE让梯度正常传递（不会因round而消失）
    """
    print("=" * 60)
    print("Fake Quantization Demo")
    print("=" * 60)

    torch.manual_seed(42)

    # 创建一个简单张量
    x = torch.randn(4, 8, requires_grad=True)
    scale = torch.tensor(0.05)

    print(f"\n[1] 原始FP32张量:")
    print(f"    x = {x[0, :5].tolist()} ...")

    # Fake量化
    x_fake = fake_quantize(x, scale)
    print(f"\n[2] Fake量化后的张量（实际仍是FP32但含量化误差）:")
    print(f"    x_fake = {x_fake[0, :5].tolist()} ...")

    # 量化误差
    error = (x_fake - x).abs()
    print(f"\n[3] 量化误差（|x_fake - x|）:")
    print(f"    误差范围: [{error.min().item():.4f}, {error.max().item():.4f}]")
    print(f"    平均误差: {error.mean().item():.4f}")

    # 反向传播测试
    loss = x_fake.sum()  # 简单loss
    loss.backward()

    print(f"\n[4] 反向传播（STE）:")
    print(f"    x.grad (from FakeQuantize backward) = {x.grad[0, :5].tolist()} ...")
    print(
        f"    梯度是否正常传递（非零）: {x.grad is not None and x.grad.abs().sum().item() > 0}"
    )

    # 对比：如果梯度真的流经round操作，大多数位置梯度应为0
    # （因为round是分段常数，小扰动不会改变量化值）
    print(f"\n[5] 梯度统计:")
    print(f"    所有元素x.grad != 0: {(x.grad != 0).all().item()}")
    print(f"    这正是STE的效果——梯度绕过round直接传递")

    print("\n" + "=" * 60)


def demo_qat_vs_ptq():
    """
    演示QAT(STE)和naive PTQ梯度消失的对比。
    """
    print("\n" + "=" * 60)
    print("QAT(STE) vs Naive PTQ 梯度对比")
    print("=" * 60)

    torch.manual_seed(123)

    # 模拟一个权重
    w = torch.tensor([[1.53, 2.47, 3.21]], requires_grad=True)
    scale = torch.tensor(0.05)
    qmin, qmax = -127, 127

    # ===== 方法1: Naive（没有STE）的反向传播梯度 =====
    # 如果严格按round的梯度，几乎所有位置梯度为0
    w_naive = w.clone().detach().requires_grad_(True)

    def naive_quant_backward(w):
        """模拟naive量化——梯度直接为0或被截断"""
        w_quant = torch.round(w / scale).clamp(qmin, qmax)
        # Naive反向: ∂L/∂w = ∂L/∂w_quant * ∂w_quant/∂w ≈ 0
        # 这里用0来模拟不可微round的梯度
        grad = torch.zeros_like(w)
        return grad

    # ===== 方法2: QAT(STE)的反向传播梯度 =====
    w_qat = w.clone().detach().requires_grad_(True)
    x_fake_qat = fake_quantize(w_qat, scale, qmin, qmax)
    loss_qat = x_fake_qat.sum()
    loss_qat.backward()

    print(f"\n[1] 原始权重: {w.tolist()}")
    print(f"    scale = {scale.item()}")

    print(f"\n[2] 量化结果 (round(w/scale)):")
    w_quant = torch.round(w / scale).clamp(qmin, qmax)
    print(f"    w_quant (INT8 representation) = {w_quant.tolist()}")

    print(f"\n[3] 反向传播梯度对比:")
    print(f"    Naive量化梯度: {naive_quant_backward(w_naive).tolist()}")
    print(f"    QAT(STE)梯度: {w_qat.grad.tolist()}")

    print(f"\n[4] 结论:")
    print(f"    Naive量化：round操作导致梯度处处为0，无法训练")
    print(f"    QAT(STE)：梯度绕过round正常传递，模型可训练")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_fake_quant()
    demo_qat_vs_ptq()
