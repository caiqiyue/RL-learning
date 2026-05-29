"""
QAT（量化感知训练）完整示例

展示如何用PyTorch实现一个支持QAT的简单MLP模型，
并在Fake量化环境下进行训练。

包含三个实验对比：
1. FP32普通训练（baseline）
2. QAT训练（使用Fake量化+STE）
3. PTQ后量化（训练后量化，对比精度损失）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fake_quant import FakeQuantizedLinear, fake_quantize


class QATMLP(nn.Module):
    """
    支持量化感知训练的简单MLP。

    每一层都使用 FakeQuantizedLinear（权重经过Fake量化），
    前向传播模拟量化行为，梯度通过STE正常回传。
    """

    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10, qbit=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.qbit = qbit

        # 构建层：逐层增加宽度 → 缩减宽度
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(FakeQuantizedLinear(prev_dim, h_dim, bias=True, qbit=qbit))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim

        layers.append(FakeQuantizedLinear(prev_dim, output_dim, bias=True, qbit=qbit))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.network(x)


class FP32MLP(nn.Module):
    """
    标准FP32 MLP（用于对比baseline）。
    与QATMLP结构相同，但使用标准nn.Linear。
    """

    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim, bias=True))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim, bias=True))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


def create_fake_data(batch_size=32, input_dim=784, num_classes=10):
    """
    创建随机假数据用于演示训练流程。
    真实场景应使用实际数据集（如MNIST、CIFAR-10）。
    """
    x = torch.randn(batch_size, 1, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    return x, y


def train_one_epoch(model, optimizer, batch_size=32, input_dim=784, num_classes=10):
    """
    训练一个epoch。
    对于QAT模型，前向/反向会自动经过Fake量化 + STE。
    对于FP32模型，就是标准训练。
    """
    model.train()
    x, y = create_fake_data(batch_size, input_dim, num_classes)

    optimizer.zero_grad()
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()

    # 梯度裁剪，稳定训练
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    preds = output.argmax(dim=1)
    acc = (preds == y).float().mean().item()
    return loss.item(), acc


def evaluate(model, num_batches=20, batch_size=32, input_dim=784, num_classes=10):
    """
    在随机数据上评估模型。
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = create_fake_data(batch_size, input_dim, num_classes)
            output = model(x)
            loss = F.cross_entropy(output, y)
            preds = output.argmax(dim=1)
            acc = (preds == y).float().mean().item()

            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            num_samples += batch_size

    return total_loss / num_samples, total_acc / num_batches


def ptq_quantize(model, qbit=8):
    """
    简单的PTQ后量化实现：
    将FP32模型权重量化为INT8（存储形式），但推理时仍用FP32。
    这里通过将权重round到量化网格点来模拟PTQ效果。
    """
    qmax = 2 ** (qbit - 1) - 1
    qmin = -qmax - 1

    model_ptq = FP32MLP()
    model_ptq.load_state_dict(model.state_dict())

    with torch.no_grad():
        for name, param in model_ptq.named_parameters():
            if "weight" in name:
                scale = torch.max(torch.abs(param)) / qmax
                param.copy_(torch.round(param / scale).clamp(qmin, qmax) * scale)

    return model_ptq


def demo_qat_training():
    """
    演示QAT训练流程：FP32 baseline vs QAT vs PTQ精度对比。
    """
    print("=" * 60)
    print("QAT Training Demo")
    print("=" * 60)

    batch_size = 64
    input_dim = 784
    hidden_dims = [256, 128]
    output_dim = 10
    qbit = 8
    lr = 1e-3
    epochs = 20

    print(f"\n[Config]")
    print(f"    batch_size = {batch_size}")
    print(f"    hidden_dims = {hidden_dims}")
    print(f"    quantization = INT{qbit}")
    print(f"    epochs = {epochs}")

    # ========== 1. FP32 Baseline ==========
    print(f"\n{'=' * 60}")
    print("Training FP32 Baseline...")
    model_fp32 = FP32MLP(input_dim, hidden_dims, output_dim)
    optimizer_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=lr)

    for epoch in range(epochs):
        loss, acc = train_one_epoch(
            model_fp32, optimizer_fp32, batch_size, input_dim, output_dim
        )
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch + 1:2d}: loss={loss:.4f}, acc={acc:.4f}")

    loss_fp32, acc_fp32 = evaluate(
        model_fp32,
        num_batches=20,
        batch_size=batch_size,
        input_dim=input_dim,
        num_classes=output_dim,
    )
    print(f"    Final FP32 Eval: loss={loss_fp32:.4f}, acc={acc_fp32:.4f}")

    # ========== 2. QAT Training ==========
    print(f"\n{'=' * 60}")
    print("Training QAT (Fake Quantization + STE)...")
    model_qat = QATMLP(input_dim, hidden_dims, output_dim, qbit=qbit)
    optimizer_qat = torch.optim.Adam(model_qat.parameters(), lr=lr)

    for epoch in range(epochs):
        loss, acc = train_one_epoch(
            model_qat, optimizer_qat, batch_size, input_dim, output_dim
        )
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch + 1:2d}: loss={loss:.4f}, acc={acc:.4f}")

    loss_qat, acc_qat = evaluate(
        model_qat,
        num_batches=20,
        batch_size=batch_size,
        input_dim=input_dim,
        num_classes=output_dim,
    )
    print(f"    Final QAT Eval:  loss={loss_qat:.4f}, acc={acc_qat:.4f}")

    # ========== 3. PTQ (FP32模型训练后量化) ==========
    print(f"\n{'=' * 60}")
    print("PTQ: Quantizing trained FP32 model...")
    model_ptq = ptq_quantize(model_fp32, qbit=qbit)

    loss_ptq, acc_ptq = evaluate(
        model_ptq,
        num_batches=20,
        batch_size=batch_size,
        input_dim=input_dim,
        num_classes=output_dim,
    )
    print(f"    Final PTQ Eval:  loss={loss_ptq:.4f}, acc={acc_ptq:.4f}")

    # ========== 对比总结 ==========
    print(f"\n{'=' * 60}")
    print("Comparison Summary")
    print(f"{'=' * 60}")
    print(f"    FP32 Baseline:  acc={acc_fp32:.4f}")
    print(
        f"    QAT (INT{qbit}):      acc={acc_qat:.4f}  (Δ={acc_qat - acc_fp32:+.4f} vs baseline)"
    )
    print(
        f"    PTQ (INT{qbit}):      acc={acc_ptq:.4f}  (Δ={acc_ptq - acc_fp32:+.4f} vs baseline)"
    )
    print(f"\n    QAT vs PTQ improvement: {acc_qat - acc_ptq:+.4f}")

    if acc_qat > acc_ptq:
        print(f"\n    ✓ QAT outperforms PTQ by {acc_qat - acc_ptq:.4f}")
        print(f"      (QAT trained with quantization in the loop)")
        print(f"      (PTQ quantized after training without adaptation)")
    else:
        print(f"\n    Note: QAT and PTQ are close (task is simple enough)")

    print("\n" + "=" * 60)


def demo_qat_weights_inspection():
    """
    检查QAT训练过程中权重的变化，
    展示Fake量化如何影响权重分布。
    """
    print("\n" + "=" * 60)
    print("QAT Weight Distribution Inspection")
    print("=" * 60)

    torch.manual_seed(99)

    model_qat = QATMLP(input_dim=128, hidden_dims=[64, 32], output_dim=10, qbit=8)
    optimizer = torch.optim.Adam(model_qat.parameters(), lr=1e-2)

    print(f"\n[1] 初始权重分布 (Before Training):")
    inspect_weights(model_qat)

    print(f"\n[2] 训练后权重分布 (After Training):")
    for epoch in range(10):
        train_one_epoch(
            model_qat, optimizer, batch_size=32, input_dim=128, num_classes=10
        )

    inspect_weights(model_qat)

    # 获取scale参数
    scales = []
    for name, param in model_qat.named_parameters():
        if "scale" in name:
            scales.append(param.data)
    print(f"\n[3] 学习到的Scale参数 (可学习量化参数):")
    for i, s in enumerate(scales[:3]):
        print(f"    Layer {i}: scale values = {s[:5].tolist()} ...")

    print("\n" + "=" * 60)


def inspect_weights(model):
    """打印模型各层权重的统计信息"""
    for name, param in model.named_parameters():
        if "weight" in name:
            w = param.data
            print(
                f"    {name}: shape={w.shape}, "
                f"min={w.min().item():.4f}, max={w.max().item():.4f}, "
                f"mean={w.mean().item():.4f}, std={w.std().item():.4f}"
            )


if __name__ == "__main__":
    demo_qat_training()
    demo_qat_weights_inspection()
