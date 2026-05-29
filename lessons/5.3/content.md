# 5.3 量化感知训练(QAT)原理与实现

## 课程概述

量化感知训练（Quantization-Aware Training，QAT）是一种在训练过程中模拟量化效应的技术，使模型能够适应量化带来的精度损失，从而在后续量化部署时获得更高的精度保持率。与训练后量化（PTQ）相比，QAT通过在训练图中插入伪量化操作，让模型"提前适应"量化环境。

本节课首先从PTQ的精度损失问题出发，阐明QAT的核心思想——Fake量化与直通估计器（STE），随后深入解析前向传播中的伪量化流程与反向传播中梯度如何穿越不可导的取整操作。接着介绍带可学习缩放因子与零点偏移的QAT变体，以及完整的QAT训练循环结构。最后通过PyTorch实现示例、SmoothQuant原理讲解，以及PTQ与QAT的全面对比，帮助读者掌握何时及如何选择QAT。

## 学习目标

- 理解QAT与PTQ的本质差异，以及QAT为何能获得更好的精度保持
- 掌握Fake量化的前向/反向数据流，以及直通估计器（STE）的原理与作用
- 理解可学习scale和zero_point在QAT变体中的作用
- 能够基于PyTorch实现一个简单的QAT训练循环
- 了解SmoothQuant解决激活值量化难题的核心思路
- 能够根据精度需求和资源限制做出PTQ/QAT的选型决策

## 前置知识

- 线性量化的基本原理：量化公式 `q = round(x/scale + zero_point)`、scale和zero_point的物理含义
- 神经网络训练的基本流程：前向传播、损失计算、反向传播、梯度更新
- PyTorch张量操作与自动求导机制
- 深度学习模型压缩的基本概念（可选）

---

## 1. QAT vs PTQ：为什么需要量化感知训练

训练后量化（Post-Training Quantization，PTQ）流程简单——先训练再量化——但存在一个根本问题：**模型在训练时使用的是高精度FP32权重，而部署时的量化操作对模型是完全陌生的**。PTQ让模型"突然"面对一个从未见过的低精度世界，精度下降不可避免。

量化的本质是一个**非连续、非可微**的操作：

```
x (FP32) → round(x/scale + zero_point) → q (INT8) → (q - zero_point) * scale → x' (FP32近似)
```

在前向传播中，取整操作（round）将连续值映射到离散整数，使得模型参数必须"跳"到量化网格点上。这个过程在反向传播中几乎不传递梯度——权重的一个小扰动可能恰好落在同一个量化桶中，导致梯度为0。

**QAT的核心思想**：与其让模型在训练完成后"突然"面对量化，不如在训练过程中就插入伪量化操作（Fake Quantization），让模型在整个训练阶段都"感受"到量化误差并主动调整参数去适应它。这就好像运动员在比赛前就穿着比赛用鞋训练，而不是比赛当天才第一次穿。

| 对比维度 | PTQ（训练后量化） | QAT（量化感知训练） |
|---------|----------------|-------------------|
| 训练流程 | 训练FP32模型 → 量化 → 部署 | 在训练流程中插入伪量化 |
| 量化时机 | 训练完成后 | 训练过程中 |
| 精度保持 | 较差，INT8通常有1-3%下降 | 更好，接近FP32 |
| 计算成本 | 极低（仅一次性量化） | 高（需完整再训练） |
| 实现复杂度 | 低 | 中高 |
| 适用场景 | 快速部署、资源极度受限 | 精度敏感、延迟敏感 |

---

## 2. Fake量化：模拟量化效应而不真正使用低精度

Fake量化是QAT的核心机制。它的思想是：在前向传播中，权重先经过量化再反量化（quantize → dequantize），使用含有量化误差的"伪量化"值进行计算；但在实际存储和梯度更新时，依然使用FP32全精度权重。这样模型能够感知量化误差，同时训练梯度仍然可以正常反向传播。

### 2.1 Fake量化的前向传播

假设一个FP32权重张量 `w`，Fake量化过程如下：

```
w_fp32 → quantize(w_fp32) → w_int → dequantize(w_int) → w_fake_fp32 → 用于实际计算
```

具体实现（以对称量化为例）：
```python
# 计算缩放因子
scale = torch.max(torch.abs(w)) / 127.0

# 量化：FP32 → INT8
w_int = torch.round(w / scale).clamp(-128, 127)

# 反量化：INT8 → FP32（但实际计算使用的是这个"伪量化"值）
w_fake = w_int * scale
```

此时 `w_fake` 含有量化误差（因为经过了round操作），但类型仍是FP32，可以正常参与梯度计算。

### 2.2 量化误差的来源

Fake量化引入的误差主要有两类：

**截断误差（Clipping Error）**：当权重超出量化范围 `[-128*scale, 127*scale]` 时，超出部分被钳位到边界值。例如 `w=200*scale` 的权重会被量化为127，再反量化后变为 `127*scale < 200*scale`，丢失了精度。

**舍入误差（Rounding Error）**：即便是落在范围内的值，经过round操作后通常也无法精确还原。以 `scale=0.0236` 为例，原始值 `0.77` 量化为 `round(0.77/0.0236) = 33`，再反量化得到 `33*0.0236 = 0.7788`，与原始值存在 `0.0088` 的误差。

---

## 3. Straight-Through Estimator（STE）：梯度如何穿过取整操作

### 3.1 问题：取整操作不可导

量化操作中的取整（round）是**非连续且不可微**的。从数学上看，`d(round(x))/dx` 在几乎所有点上都是0（因为round是分段常数函数），只有，在跨过整数边界时才有冲激函数形式的"梯度"。这意味着如果严格按照导数定义，量化操作的梯度几乎处处为0，反向传播无法有效更新权重。

### 3.2 STE的核心思想

Straight-Through Estimator（直通估计器）的核心思想是：**在前向传播中执行实际的取整操作（因为这代表了真实的量化行为），但在反向传播中，假装这个取整操作不存在，梯度直接穿透过去**。

形象地说：
- **前向传播（现实）**：货物必须装进集装箱（取整到离散值），你无法运送3.7个单位的货物
- **反向传播（规划）**：但在规划如何调整货物量时，假设可以自由调整小数（梯度绕过取整）

Stevens和Bengio在2013年的论文中首次系统性地提出将STE用于二元梯度（binary networks），此后推广到一般的量化场景。

### 3.3 数学表达

标准STE的定义：

```python
# 前向传播：实际执行量化
def forward(x, scale):
    x_quant = torch.round(x / scale).clamp(-128, 127)  # 取整，不可导
    x_fake = x_quant * scale                           # 反量化回FP32
    return x_fake

# 反向传播：梯度绕过取整操作
# ∂L/∂x ≈ ∂L/∂x_fake（直接将下游梯度传递上去，忽略round）
# 即 Ste(x) = ∂x_fake/∂x 对于round操作，Ste ≈ 1（identity）
```

从数学上更精确地描述STE：

```
Forward:  q = round(x)          （不可微）
Backward: ∂L/∂x = ∂L/∂q * 1      （STE将round的梯度设为1，梯度直接传递）
```

### 3.4 STE的形象类比：邮递员送包裹

想象你是一名邮递员，需要将包裹送到街道上的特定门牌号：

**街道设置**：房屋的门牌号只能是整数（10号、11号、12号……），不存在10.5号或11.7号。

**前向传播（实际投递）**：你必须将包裹送到实际存在的门牌号。如果地址是11.3号，你会送到11号（向下取整）；如果地址是11.7号，你会送到12号（向上取整）。

**反向传播（规划路线）的问题**：如果严格按实际规则，地址从11.3改为11.6仍然送到11号，梯度为零；只有跨越11.5的边界才会改变投递结果。这使得训练几乎无法进行。

**使用STE后**：在规划路线时，假装每个微小变化都有效果——虽然实际上只能送到整数门牌号，但方向指导基于"连续地址空间"的梯度进行更新。这就好像在规划时把楼梯看成斜坡，虽然实际移动仍是台阶式的，但方向基于斜坡决定。

### 3.5 PyTorch中的STE实现

在PyTorch中，Fake量化通常通过自定义autograd函数实现：

```python
import torch
from torch.autograd import Function

class FakeQuantize(Function):
    @staticmethod
    def forward(ctx, x, scale):
        # 前向传播：实际执行量化
        x_quant = torch.round(x / scale)
        x_quant = x_quant.clamp(-128, 127)
        return x_quant * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：STE，直接传递梯度（忽略round）
        return grad_output, None  # scale不需要梯度
```

其中 `backward` 方法返回的 `grad_output` 直接就是下游传来的梯度——这正是STE的核心：绕过round操作的不可微性。

---

## 4. QAT训练循环：完整的数据流

一个完整的QAT训练循环与标准FP32训练的主要区别在于每个训练step中插入了Fake量化操作：

```
标准FP32训练:
  input → 前向计算 → loss → 反向传播 → 更新FP32权重

QAT训练:
  input → 权重经过Fake量化 → 前向计算 → loss → 反向传播 → STE梯度 → 更新FP32权重
```

具体步骤：

**Step 1 — 前向传播（含Fake量化）**：
- 权重 `w` 经过 FakeQuantize 得到 `w_fake = FakeQuantize(w, scale)`
- 使用 `w_fake` 进行矩阵乘法等计算
- 所有层的输入输出仍为FP32张量，仅权重"看起来"是量化后的值

**Step 2 — 损失计算**：
- 与标准训练相同，使用FP32 loss

**Step 3 — 反向传播（含STE）**：
- 梯度从loss传来，经过所有层反向传播
- 到达FakeQuantize算子时，STE让梯度直接传递（绕过round）
- 最终得到 `∂L/∂w` 全精度梯度

**Step 4 — 权重更新**：
- 以标准优化器（如Adam、SGD）更新原始FP32权重 `w`
- 更新后的 `w` 在下一个step中再次经过Fake量化

这意味着**训练始终在FP32精度下进行**（优化器状态、权重更新都是FP32），但前向传播模拟了量化行为，使模型学会在量化条件下仍能正确工作。

---

## 5. 可学习的Scale和Zero Point

在一些QAT变体中，量化参数 `scale` 和 `zero_point` 并非在量化前固定计算得到，而是作为**可学习参数**在训练过程中自动优化。这进一步提升了QAT的精度。

### 5.1 为什么需要可学习的量化参数？

固定scale的问题在于：随着训练的进行，权重分布会发生变化。训练初期确定的scale在训练后期可能已经不再是最优的——它无法捕捉权重分布的动态变化。

可学习量化（Learned Quantization）的基本形式是将scale/zero_point作为独立的学习参数：

```python
# 初始化可学习参数
log_scale = nn.Parameter(torch.zeros(num_groups))  # 存储log(scale)避免下溢
zero_point = nn.Parameter(torch.zeros(num_groups))  # 可学习零点
```

### 5.2 可学习量化参数的更新

可学习参数通过标准反向传播自动更新：

```
loss → backward → ∂L/∂w → 更新w
                 ↘ ∂L/∂log_scale → 更新log_scale
                 ↘ ∂L/∂zero_point → 更新zero_point
```

训练过程中，模型会逐渐找到一个"量化友好"的权重分布，使得即使用学习到的scale/zero_point进行量化后，精度损失也最小。

### 5.3 优化技巧

实践中使用可学习scale时通常有几点注意事项：

- **用log形式存储**：scale > 0，存储 `log_scale = log(scale)` 可以避免下溢和梯度爆炸问题，同时保证scale始终为正
- **梯度裁剪**：对scale/zero_point的梯度适当裁剪，防止更新步长过大导致不稳定
- **初始化**：scale的初始值可以使用输入数据的实际标准差来设置一个合理的起点

---

## 6. PyTorch QAT实现

PyTorch提供了两套QAT实现路径：动态量化感知的 `torch.quantization.quantize_dynamic`（主要用于推断）和FX图模式量化（支持训练后转换）。但对于QAT训练场景，最核心的是手工实现Fake量化 + 自定义autograd。

### 6.1 动态量化（用于对比参考）

动态量化是最简单的量化方式，只量化权重，激活值在推理时动态量化：

```python
import torch.quantization

# 加载FP32模型
model_fp32 = MyModel()

# 应用动态量化（仅权重量化为INT8）
model_dynamic = torch.quantization.quantize_dynamic(
    model_fp32, 
    {torch.nn.Linear},  # 指定要量化的层类型
    dtype=torch.qint8
)
```

### 6.2 QAT训练框架

对于需要QAT的场景，需要自定义Fake量化模块并嵌入模型中：

```python
class QATLinear(torch.nn.Module):
    """带量化感知训练的Linear层"""
    def __init__(self, in_features, out_features, bias=True, qbit=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qbit = qbit
        self.qmax = 2 ** (qbit - 1) - 1  # INT8: 127
        
        # 原始FP32权重
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 可学习scale（per-channel，按输出通道）
        self.scale = torch.nn.Parameter(torch.ones(out_features))
    
    def forward(self, x):
        # Fake量化：权重量化后反量化
        w_fake = fake_quantize(self.weight, self.scale, self.qmax)
        return torch.nn.functional.linear(x, w_fake, self.bias)

def fake_quantize(w, scale, qmax):
    """Fake量化操作（STE反向）"""
    w_quant = torch.round(w / scale.unsqueeze(1))
    w_quant = w_quant.clamp(-qmax, qmax)
    return w_quant * scale.unsqueeze(1)
```

### 6.3 完整的QAT训练示例

以一个简单MLP在MNIST上的分类任务为例，对比FP32普通训练、QAT训练和PTQ后量化的精度差异：

```python
# 伪代码框架，完整代码见 lessons/5.3/code/qat_example.py
def qat_train(model, train_loader, epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in epochs:
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            
            # 前向传播（含Fake量化）
            output = model(batch_data)
            loss = torch.nn.functional.cross_entropy(output, batch_labels)
            
            # 反向传播（STE自动处理）
            loss.backward()
            
            # 梯度裁剪（稳定训练）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
```

---

## 7. SmoothQuant：解决激活值量化难题

PTQ和基础QAT的一个核心挑战是**激活值（activation）量化困难**。权重可以提前校准，但激活值随输入变化，且不同输入样本的激活值分布差异很大。特别是Transformer中的激活值经常存在**异常值（outlier）**——少数通道的激活值范围远大于其他通道，这会导致：

- 如果按全局范围设置scale，大部分通道的量化精度严重受损
- 如果按各通道分别设置scale，开销增大且实现复杂

### 7.1 SmoothQuant的核心思想

SmoothQuant观察到：**对于一个线性层 `y = Wx + b`，如果激活值中存在异常值，我们可以将这个"难度"转移到权重一侧**。具体做法是引入一个逐通道的平滑因子 `s`，将权重和激活值重新分配：

```
原始：y = Wx
平滑后：y = (W diag(s)^{-1}) (diag(s) x)
```

其中 `s` 是按通道计算的一个缩放因子，它使得**变换后的激活值和权重都更容易量化**。

### 7.2 平滑因子如何计算

平滑因子 `s` 的设计原则是：让权重通道的"难度"（数值范围）和激活值通道的"难度"平衡。

SmoothQuant定义了一个"难度"指标：

```python
# 对于每通道c，衡量其量化的难度
# 激活值异常度：激活值通道c的最大绝对值
# 权重异常度：权重行c的最大绝对值

# 平滑因子使得：|s_c * a_c|_max / α = |W_c / s_c|_max
# α是一个超参数，控制迁移程度（α越大，越多"难度"迁移到权重）
```

实践中 `s_c = (|W_c|_max)^{α} / (|x_c|_max)^{α}`，其中 `α ∈ [0,1]` 控制迁移程度。

### 7.3 SmoothQuant的效果

经过SmoothQuant处理后，权重和激活值都呈现更加均衡的数值分布：

- 激活值中的异常值被"吸收"到权重中
- 变换后的激活值分布更适合INT8量化
- 权重中的异常值被"吸收"到激活值中（而激活值可以使用更高精度如FP16处理，或通过平滑转移到权重）

这使得**INT8量化可以同时应用于权重和激活值**，而不像基础PTQ那样通常只量化权重、激活值保留FP16。

---

## 8. PTQ与QAT实现对比总结

| 特性 | PTQ | QAT |
|------|-----|-----|
| **实现难度** | 低（仅一次性量化） | 中高（需改造训练流程） |
| **训练时间** | 无额外训练 | 接近完整训练时间 |
| **精度** | INT8通常下降1-3% | 接近FP32 |
| **内存占用** | 无额外开销 | 需存储FP32主权重（量化模型本身更小） |
| **适用场景** | 快速迭代、资源受限 | 精度敏感、长期部署 |
| **PyTorch支持** | `quantize_dynamic` / FX图模式 | 自定义Fake量化模块 |

PTQ更快但精度有损，适合"先跑起来"的场景。QAT代价更高但精度更好，适合对精度要求苛刻的生产环境。

---

## 9. 何时选择QAT

QAT并非所有场景的最优选择。以下情况建议优先考虑QAT：

1. **PTQ精度不可接受**：当PTQ量化后模型精度下降超过可接受阈值（如 >2%）时，QAT是必然选择
2. **极限压缩比**：当需要INT4甚至更激进的量化精度时，QAT几乎是唯一可行方案
3. **高精度要求的垂直领域**：医疗影像、金融风控等对精度要求极高的场景
4. **特定任务微调**：在特定领域数据上做QAT微调，往往比PTQ + 领域微调效果更好
5. **模型结构特殊**：某些模型结构（如含动态路由的模块）对量化特别敏感，PTQ效果差

反之，如果：
- 资源极度受限需要快速部署
- 对精度要求中等（下降1-2%可接受）
- 模型结构标准（纯Transformer、LSTM等）

则优先使用PTQ，先验证可行性，再考虑QAT作为精度优化手段。

---

## 总结

本节课围绕量化感知训练（QAT）展开，主要内容：

1. **PTQ vs QAT**：PTQ在训练后量化，模型未经历量化环境；QAT在训练中模拟量化，使模型主动适应
2. **Fake量化**：通过 `quantize → dequantize` 在FP32精度下模拟量化效果，为模型提供量化误差的感知信号
3. **STE**：通过将前向传播的取整操作在反向传播中绕过（梯度设为1），解决不可导问题，实现梯度流传递
4. **QAT训练循环**：前向含Fake量化、反向通过STE传递梯度、更新仍是FP32权重——训练仍是FP32精度，只是前向"看到"了量化
5. **可学习scale/zero_point**：作为可学习参数在训练中优化，进一步提升量化精度
6. **PyTorch实现**：通过自定义autograd函数实现Fake量化 + STE
7. **SmoothQuant**：通过逐通道平滑因子将激活值异常值迁移到权重，实现权重+激活值INT8量化
8. **选型决策**：PTQ快但精度有损，QAT慢但精度更好，根据资源与精度要求权衡

---

## 扩展阅读

- Bengio, Y., Léonard, N., & Courville, A. (2013). *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation* — 最早系统提出STE的论文之一
- Zhou, S., et al. (2016). *Dorefa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients* — 探讨梯度量化的重要工作
- Xiao, H., et al. (2022). *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* — SmoothQuant原始论文，INT8权重+激活量化的关键工作
- PyTorch QAT文档：https://pytorch.org/docs/stable/quantization.html — PyTorch官方量化支持文档
- GPTQ: https://arxiv.org/abs/2210.17328 — 经典PTQ方法，适合对照学习

---

## 复习题

1. **PTQ vs QAT的核心区别**：解释为什么QAT通常能获得比PTQ更好的量化精度。从训练流程和模型对量化误差的"感知"角度说明。

2. **Fake量化的前向/反向数据流**：描述Fake量化在前向传播和反向传播中的数据流分别是什么样的。哪些量是FP32，哪些量是INT8？梯度如何传递？

3. **STE的物理意义**：用"邮递员送包裹"的类比，解释STE在前向传播和反向传播中分别扮演什么角色。为什么STE能让训练收敛？

4. **可学习scale的优势**：解释为什么将scale作为可学习参数（而非固定校准值）可以提升QAT精度？训练过程中scale会如何变化？

5. **SmoothQuant的动机与核心思想**：解释为什么激活值量化通常比权重量化更困难。SmoothQuant如何解决这个问题？平滑因子s的物理含义是什么？