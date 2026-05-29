"""
SmoothQuant 实现原理与代码框架

SmoothQuant 解决的核心问题：
    激活值(activation)量化困难——Transformer中的激活值经常存在异常值(outlier)，
    导致INT8量化时精度严重下降。

核心思想：
    将激活值中的异常值"迁移"到权重一侧，通过引入逐通道平滑因子s，
    使得变换后：y = (W diag(s)^{-1}) (diag(s) x)
    新的权重和激活值分布都更加均衡，更适合INT8量化。

公式推导：
    原始：y_c = sum_c(W_{c,k} * x_k)
    平滑：y_c = sum_c((W_{c,k} / s_c) * (s_c * x_k))
         = sum_c(W'_c,k * x'_k)

    其中 s_c 是通道c的平滑因子。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothQuantLinear(nn.Module):
    """
    应用SmoothQuant平滑的Linear层。

    通过在forward前应用逐通道平滑因子，对权重和激活值重新分配，
    使得两者都更容易被INT8量化。

    使用方法：
        1. 先正常做一次前向传播（计算激活值统计量）
        2. 调用 compute_smooth_factors() 计算平滑因子
        3. 后续前向传播自动使用平滑后的权重和激活值
    """

    def __init__(self, in_features, out_features, bias=True, alpha=0.5):
        """
        Args:
            in_features: 输入维度
            out_features: 输出维度
            bias: 是否使用偏置
            alpha: 平滑因子计算的超参数，控制迁移程度
                   alpha=0: 不迁移，激活值异常度不变
                   alpha=1: 完全迁移，激活值和权重同等难度
                   实践中通常 alpha=0.5
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

        # 平滑因子 s（逐输出通道，per-token）
        self.s = nn.Parameter(torch.ones(out_features, 1))

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def compute_smooth_factors(self, x, x_stats_collected=False):
        """
        计算SmoothQuant的平滑因子 s。

        目标：使得 |W_c / s_c|_max 与 |x_c|_max 尽可能均衡。

        计算公式（按通道c）:
            s_c = (|W_c|_max)^{alpha} / (|x_c|_max)^{alpha}

        Args:
            x: 输入张量 [batch, seq_len, in_features] 或 [batch, in_features]
            x_stats_collected: 是否已经收集了激活值统计量
                               如果为True，直接用x计算；否则需要先正常forward收集
        """
        if x.dim() == 3:
            # [batch, seq, in_feat] -> 取最后一个维度统计
            x_max = (
                torch.abs(x).transpose(0, 1).reshape(x.size(1), -1).max(dim=1).values
            )
        elif x.dim() == 2:
            # [batch, in_feat]
            x_max = torch.abs(x).max(dim=0).values
        else:
            raise ValueError(f"Unexpected input dim: {x.dim()}")

        # 权重按输出通道的最大绝对值
        W_max = torch.max(torch.abs(self.weight), dim=1).values

        # 计算平滑因子
        # s_c = (W_max_c)^{alpha} / (x_max_c)^{alpha}
        # 为避免除零，加一个小的epsilon
        eps = 1e-8
        s_raw = (W_max**self.alpha) / ((x_max + eps) ** self.alpha)

        # 更新可学习参数 s
        self.s.data = s_raw.unsqueeze(1)

        return self.s

    def forward(self, x):
        """
        前向传播，使用SmoothQuant平滑。

        平滑后：y = (W @ diag(s)^{-1}) @ (diag(s) @ x) + b
              = W' @ x' + b

        其中：
            W' = W @ diag(s)^{-1}  （权重向量化）
            x' = diag(s) @ x       （激活值向量化）
        """
        s_inv = 1.0 / (self.s + 1e-8)  # [out_features, 1]

        # 平滑后的权重：W' = W * s_inv (broadcast)
        W_smooth = self.weight * s_inv  # [out_features, in_features]

        # 平滑后的激活值：x' = x * s (broadcast on in_features dimension)
        # 注意：s 是按输出通道的，但激活值x是按输入通道的
        # SmoothQuant 中 s 的维度对应 weight 的输出通道（即激活值的输入维度）
        x_smooth = x * s_inv.transpose(0, 1)  # 需要正确reshape

        return F.linear(x_smooth, W_smooth, self.bias)


class SmoothQuantWrapper(nn.Module):
    """
    SmoothQuant层的包装器，在训练过程中同时更新平滑因子。

    典型用法：
        1. 正常前向传播收集激活值统计量
        2. 调用 update_smooth_factors() 更新平滑因子
        3. 继续训练
    """

    def __init__(self, layer, alpha=0.5):
        super().__init__()
        self.layer = layer
        self.alpha = alpha

        if hasattr(layer, "weight"):
            self.in_features = layer.weight.shape[1]
            self.out_features = layer.weight.shape[0]
        else:
            raise ValueError("Wrapped layer must have weight attribute")

        self.s = nn.Parameter(torch.ones(self.out_features, 1))

    def update_smooth_factors(self, x):
        """
        基于当前输入x更新平滑因子s。

        使用动态收集的激活值统计量（online更新）。
        实践中常用 running max 来稳定估计。
        """
        if not hasattr(self, "running_x_max"):
            self.register_buffer("running_x_max", torch.zeros(self.in_features))

        x_max = torch.max(torch.abs(x), dim=0).values
        self.running_x_max = torch.max(self.running_x_max, x_max)

        W_max = torch.max(torch.abs(self.layer.weight), dim=1).values

        s_raw = (W_max**self.alpha) / ((self.running_x_max + 1e-8) ** self.alpha)
        self.s.data = s_raw.unsqueeze(1)

    def forward(self, x):
        s_inv = 1.0 / (self.s + 1e-8)
        W_smooth = self.layer.weight * s_inv
        x_smooth = x * s_inv.transpose(0, 1) if x.dim() > 2 else x * s_inv.t()
        return F.linear(x_smooth, W_smooth, self.layer.bias)


def demo_smoothquant():
    """
    演示SmoothQuant的基本效果：
    展示平滑前后权重和激活值的数值分布变化。
    """
    print("=" * 60)
    print("SmoothQuant Demo")
    print("=" * 60)

    torch.manual_seed(42)

    # 创建一个模拟Transformer层中的异常激活值
    # 假设激活值x有少数通道值很大（异常值）
    batch, seq_len, in_feat = 4, 8, 64
    out_feat = 32

    x = torch.randn(batch, seq_len, in_feat) * 0.1
    # 注入异常通道：第5个token的某些维度有大值
    x[:, 5, :8] = torch.randn(batch, 8) * 10  # outlier channels

    W = torch.randn(out_feat, in_feat) * 0.05
    # 注入权重异常值
    W[:4, :4] = torch.randn(4, 4) * 5

    print(f"\n[1] 激活值 x 统计 (Input activation):")
    print(f"    shape: {x.shape}")
    x_max_per_channel = torch.max(torch.abs(x), dim=(0, 1)).values
    print(f"    max per channel (first 16): {x_max_per_channel[:16].tolist()}")
    print(f"    overall max: {x_max_per_channel.max().item():.4f}")
    print(f"    overall min: {x_max_per_channel.min().item():.4f}")

    print(f"\n[2] 权重 W 统计 (Weight):")
    print(f"    shape: {W.shape}")
    W_max_per_channel = torch.max(torch.abs(W), dim=1)
    print(f"    max per channel (first 16): {W_max_per_channel.values[:16].tolist()}")
    print(f"    overall max: {W_max_per_channel.values.max().item():.4f}")

    # 计算平滑因子
    alpha = 0.5
    s_raw = (W_max_per_channel.values**alpha) / ((x_max_per_channel + 1e-8) ** alpha)
    s = s_raw.unsqueeze(1)
    s_inv = 1.0 / s

    print(f"\n[3] 平滑因子 s (Smooth factors):")
    print(f"    s (first 16): {s[:16, 0].tolist()}")

    # 应用平滑
    W_smooth = W * s_inv
    x_smooth = x * s_inv.transpose(0, 1).unsqueeze(0)

    print(f"\n[4] 平滑后激活值 x' 统计:")
    x_smooth_max = torch.max(torch.abs(x_smooth), dim=(0, 1)).values
    print(f"    max per channel (first 16): {x_smooth_max[:16].tolist()}")
    print(f"    overall max: {x_smooth_max.max().item():.4f}")
    print(f"    overall min: {x_smooth_max.min().item():.4f}")
    print(
        f"    max/min 比值 (平滑前): {x_max_per_channel.max().item() / x_max_per_channel.min().item():.1f}x"
    )
    print(
        f"    max/min 比值 (平滑后): {x_smooth_max.max().item() / x_smooth_max.min().item():.1f}x"
    )

    print(f"\n[5] 平滑后权重 W' 统计:")
    W_smooth_max = torch.max(torch.abs(W_smooth), dim=1).values
    print(f"    max per channel (first 16): {W_smooth_max[:16].tolist()}")
    print(f"    overall max: {W_smooth_max.max().item():.4f}")

    print(f"\n[6] 量化难度平衡效果:")
    print(f"    权重难度(W/max): {W_max_per_channel.values.max().item():.4f}")
    print(f"    激活值难度(x/max): {x_max_per_channel.max().item():.4f}")
    print(
        f"    差异比例: {x_max_per_channel.max().item() / W_max_per_channel.values.max().item():.1f}x"
    )
    print(f"    SmoothQuant迁移后差异比例接近1:1，量化更均衡")

    print("\n" + "=" * 60)


def demo_smoothquant_training():
    """
    演示SmoothQuant wrapper在训练中的使用。
    """
    print("\n" + "=" * 60)
    print("SmoothQuant Training Wrapper Demo")
    print("=" * 60)

    torch.manual_seed(777)

    batch, seq, in_feat = 8, 16, 64
    out_feat = 32

    base_layer = nn.Linear(in_feat, out_feat)
    sq_layer = SmoothQuantWrapper(base_layer, alpha=0.5)

    optimizer = torch.optim.Adam(sq_layer.parameters(), lr=1e-3)

    print(f"\n[1] 训练前初始状态:")
    x_dummy = torch.randn(batch, seq, in_feat)
    sq_layer.update_smooth_factors(x_dummy)
    print(f"    s factors: {sq_layer.s[:8, 0].tolist()} ...")

    print(f"\n[2] 执行几步训练:")
    for step in range(5):
        x = torch.randn(batch, seq, in_feat)
        # 模拟激活值异常
        x[:, 5:, :8] = torch.randn(batch, 11, 8) * 8

        optimizer.zero_grad()
        out = sq_layer(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        if step == 0 or step == 4:
            sq_layer.update_smooth_factors(x)
            print(
                f"    Step {step + 1}: loss={loss.item():.4f}, "
                f"s factors mean={sq_layer.s.mean().item():.4f}"
            )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_smoothquant()
    demo_smoothquant_training()
