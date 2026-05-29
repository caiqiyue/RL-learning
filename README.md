# 大模型微调与强化学习

## 需求解析

| 参数 | 值 |
|------|-----|
| 主题 | 大模型微调 + 强化学习 |
| 受众 | AI从业者、算法工程师、研究者 |
| 级别 | 企业级快速入门 |
| 时长 | 60小时 |
| 技术栈 | Python, PyTorch, Transformers, DeepSpeed, TRL |
| 源材料 | 8份Markdown文档 + 1份PDF论文 |

---

## 原始资料整理

### 核心概念

| 概念 | 定义 |
|------|------|
| 预训练大模型 | 基于海量通用数据训练，具备通用知识和强泛化能力的大规模AI模型 |
| 全量微调 | 更新模型所有参数，显存占用大，适合大规模标注数据 |
| LoRA | Low-Rank Adaptation，通过低秩矩阵分解减少可训练参数 |
| QLoRA | 量化版LoRA，4-bit NF4量化基础模型 + LoRA适配器 |
| PPO | Proximal Policy Optimization，近端策略优化算法 |
| GRPO | Group Relative Policy Optimization，DeepSeek改进的无Value网络RL算法 |
| RLHF | Reinforcement Learning from Human Feedback，基于人类反馈的强化学习 |
| DPO | Direct Preference Optimization，直接偏好优化 |

### 关键技术

- LoRA/QLoRA 低秩适配器微调
- PPO/GRPO 策略优化算法
- RLHF/DPO 对齐技术
- NF4/INT8/INT4 量化技术
- LLaMA-Factory/TRL/DeepSpeed 工具链

### 参考资料

- QLoRA 论文: arXiv:2305.14314
- LLaMA 论文: arXiv:2302.13971
- PPO 论文: arXiv:1707.06347
- RLHF 在 InstructGPT 上的应用

---

## 知识框架 v1.1

### 框架概述

本课程分为4大部分，共16个模块，47节课时，覆盖微调基础理论、数据工程体系、微调实战和企业项目实战。

### 第一部分：微调基础理论

#### 模块1：微调范式与硬件基础
- 1.1 大模型概述与分类
- 1.2 微调策略选择：全量/LoRA/QLoRA/适配器
- 1.3 硬件选择与显存估算

#### 模块2：LoRA/QLoRA核心方法
- 2.1 LoRA原理：ΔW=BA低秩分解
- 2.2 LoRA代码实现与参数配置
- 2.3 QLoRA原理：NF4量化与双量化
- 2.4 QLoRA实战：单卡微调65B模型

#### 模块3：强化学习框架
- 3.1 强化学习基础：MDP/折扣因子/价值函数
- 3.2 策略梯度与Actor-Critic架构
- 3.3 PPO算法详解：裁剪机制与KL散度
- 3.4 PPO代码实现：CartPole/Atari游戏
- 3.5 GRPO算法：组计算机制与双层KL控制
- 3.6 GRPO vs PPO：架构对比与优势分析

#### 模块4：强化学习微调与对齐
- 4.1 RLHF三阶段流程详解
- 4.2 奖励模型训练：排序数据与人类反馈
- 4.3 DPO算法：奖励-策略数学映射与实现
- 4.4 DPO实战：偏好数据准备与训练
- 4.5 KTO与IPO：其他偏好优化方法

#### 模块5：模型轻量化与加速
- 5.1 量化技术基础：精度体系与线性量化
- 5.2 量化因子详解：scale/zero_point/校准
- 5.3 量化感知训练(QAT)原理与实现
- 5.4 INT8/INT4部署：TensorRT与vLLM量化推理
- 5.5 知识蒸馏：模型压缩与部署实战

### 第二部分：数据工程体系

#### 模块6：数据工程总览
- 6.1 微调数据体系概览与构建流程
- 6.2 数据质量评估与清洗标准

#### 模块7：SFT数据完整链路
- 7.1 SFT数据格式与构建方法
- 7.2 指令数据合成与多样性增强
- 7.3 高质量数据筛选与质量控制

#### 模块8：偏好数据完整链路
- 8.1 偏好数据构建：人类反馈与排序标注
- 8.2 AI反馈(RLAIF)与弱监督方法
- 8.3 DPO偏好数据准备与训练

#### 模块9：CoT数据完整链路
- 9.1 思维链Prompt设计原则
- 9.2 长推理数据合成与验证

### 第三部分：微调实战

#### 模块10：工具链
- 10.1 LLaMA-Factory：多模型配置与训练
- 10.2 TRL库：SFT/PPO/DPO Trainer完整用法
- 10.3 DeepSpeed ZeRO优化与分布式训练
- 10.4 Unsloth加速：消费级GPU实战

#### 模块11：强化学习微调实战
- 11.1 RLHF完整Pipeline实现
- 11.2 GRPO实战：DeepSeek-R1风格训练

#### 模块12：高效微调技术
- 12.1 QLoRA实战：多模型单卡微调
- 12.2 长上下文扩展：RoPE插值与位置编码

#### 模块13：多模型架构适配
- 13.1 LLaMA/Qwen/ChatGLM/DeepSeek架构差异
- 13.2 跨模型适配器与权重转换

#### 模块14：模型部署与应用
- 14.1 vLLM批量推理与量化部署
- 14.2 Ollama本地部署与API调用
- 14.3 模型评估：自动基准与质量评估体系

### 第四部分：企业项目实战

#### 模块15：行业项目实战
- 15.1 垂直领域SFT：医疗/法律场景案例
- 15.2 对话系统优化：RLHF完整Pipeline
- 15.3 模型压缩上线：INT8量化+加速+部署闭环

---

## 内容结构设计 v1.1

### 课程列表

| 课ID | 标题 | 类型 | 课时(分钟) | 配套文件 |
|------|------|------|-----------|---------|
| 1.1 | 大模型概述与分类 | theory | 45 | content.md |
| 1.2 | 微调策略选择：全量/LoRA/QLoRA/适配器 | theory | 60 | content.md |
| 1.3 | 硬件选择与显存估算 | theory | 45 | content.md |
| 2.1 | LoRA原理：ΔW=BA低秩分解 | theory | 75 | content.md |
| 2.2 | LoRA代码实现与参数配置 | code | 90 | content.md, code/ |
| 2.3 | QLoRA原理：NF4量化与双量化 | theory | 60 | content.md |
| 2.4 | QLoRA实战：单卡微调65B模型 | code | 120 | content.md, code/ |
| 3.1 | 强化学习基础：MDP/折扣因子/价值函数 | theory | 90 | content.md |
| 3.2 | 策略梯度与Actor-Critic架构 | theory | 75 | content.md |
| 3.3 | PPO算法详解：裁剪机制与KL散度 | theory | 90 | content.md |
| 3.4 | PPO代码实现：CartPole/Atari游戏 | code | 120 | content.md, code/ |
| 3.5 | GRPO算法：组计算机制与双层KL控制 | theory | 90 | content.md |
| 3.6 | GRPO vs PPO：架构对比与优势分析 | comparison | 45 | content.md |
| 4.1 | RLHF三阶段流程详解 | theory | 90 | content.md |
| 4.2 | 奖励模型训练：排序数据与人类反馈 | theory | 75 | content.md |
| 4.3 | DPO算法：奖励-策略数学映射与实现 | theory | 90 | content.md |
| 4.4 | DPO实战：偏好数据准备与训练 | code | 120 | content.md, code/ |
| 4.5 | KTO与IPO：其他偏好优化方法 | theory | 45 | content.md |
| 5.1 | 量化技术基础：精度体系与线性量化 | theory | 60 | content.md |
| 5.2 | 量化因子详解：scale/zero_point/校准 | theory | 75 | content.md |
| 5.3 | 量化感知训练(QAT)原理与实现 | code | 90 | content.md, code/ |
| 5.4 | INT8/INT4部署：TensorRT与vLLM量化推理 | code | 90 | content.md, code/ |
| 5.5 | 知识蒸馏：模型压缩与部署实战 | code | 90 | content.md, code/ |
| 6.1 | 微调数据体系概览与构建流程 | theory | 45 | content.md |
| 6.2 | 数据质量评估与清洗标准 | theory | 45 | content.md |
| 7.1 | SFT数据格式与构建方法 | theory | 60 | content.md |
| 7.2 | 指令数据合成与多样性增强 | code | 90 | content.md, code/ |
| 7.3 | 高质量数据筛选与质量控制 | theory | 45 | content.md |
| 8.1 | 偏好数据构建：人类反馈与排序标注 | theory | 60 | content.md |
| 8.2 | AI反馈(RLAIF)与弱监督方法 | theory | 45 | content.md |
| 8.3 | DPO偏好数据准备与训练 | code | 90 | content.md, code/ |
| 9.1 | 思维链Prompt设计原则 | theory | 45 | content.md |
| 9.2 | 长推理数据合成与验证 | code | 90 | content.md, code/ |
| 10.1 | LLaMA-Factory：多模型配置与训练 | code | 120 | content.md, code/ |
| 10.2 | TRL库：SFT/PPO/DPO Trainer完整用法 | code | 120 | content.md, code/ |
| 10.3 | DeepSpeed ZeRO优化与分布式训练 | code | 120 | content.md, code/ |
| 10.4 | Unsloth加速：消费级GPU实战 | code | 90 | content.md, code/ |
| 11.1 | RLHF完整Pipeline实现 | code | 180 | content.md, code/ |
| 11.2 | GRPO实战：DeepSeek-R1风格训练 | code | 150 | content.md, code/ |
| 12.1 | QLoRA实战：多模型单卡微调 | code | 150 | content.md, code/ |
| 12.2 | 长上下文扩展：RoPE插值与位置编码 | code | 90 | content.md, code/ |
| 13.1 | LLaMA/Qwen/ChatGLM/DeepSeek架构差异 | theory | 60 | content.md |
| 13.2 | 跨模型适配器与权重转换 | code | 90 | content.md, code/ |
| 14.1 | vLLM批量推理与量化部署 | code | 120 | content.md, code/ |
| 14.2 | Ollama本地部署与API调用 | code | 90 | content.md, code/ |
| 14.3 | 模型评估：自动基准与质量评估体系 | code | 90 | content.md, code/ |
| 15.1 | 垂直领域SFT：医疗/法律场景案例 | project | 180 | content.md, code/ |
| 15.2 | 对话系统优化：RLHF完整Pipeline | project | 180 | content.md, code/ |
| 15.3 | 模型压缩上线：INT8量化+加速+部署闭环 | project | 180 | content.md, code/ |

### lesson content 类型说明

| 类型 | 说明 |
|------|------|
| theory | 理论课程，以知识点讲解为主 |
| code | 代码课程，包含可运行的代码示例 |
| comparison | 对比课程，分析不同技术的差异 |
| project | 项目课程，端到端的实战项目 |

---

## 构建状态

- [x] 阶段1: 需求解析
- [x] 阶段2: 资料收集
- [x] 阶段3: 知识框架设计
- [x] 阶段4: 内容结构规划
- [x] 阶段5: 框架审核
- [x] 阶段6: 任务拆分
- [x] 阶段7: 并行构建
- [x] 阶段8: 最终审核
- [x] 阶段9: 项目结构

---

## 项目结构

```
output/大模型微调与强化学习/
├── README.md                    # 本文件：知识框架 + 内容结构总览
├── COURSE_META.json             # 课程元信息
├── KNOWLEDGE_FRAMEWORK.json     # 知识框架JSON
├── content_structure.json        # 内容结构JSON (v1.1)
├── build_tasks.json             # 构建任务清单
├── .gitignore                   # Git忽略文件
├── raw-material/                # 原始学习资料
│   ├── 1. 模型微调概览与硬件选取.md
│   ├── 2.Lora微调原理详解.md
│   ├── 3.QLoRA微调原理详解.md
│   ├── 4.强化学习PPO详解.md
│   ├── 5.RLHF详解.md
│   ├── 6. 强化学习微调DPO详解.md
│   ├── 7.GRPO详解.md
│   └── 8.大模型量化与蒸馏技术详解.md
├── lessons/                     # 课程内容
│   ├── 1.1/
│   │   └── content.md
│   ├── 1.2/
│   │   └── content.md
│   ├── 1.3/
│   │   └── content.md
│   ├── 2.1/
│   │   └── content.md
│   ├── 2.2/
│   │   ├── content.md
│   │   └── code/
│   │       ├── lora_config.py
│   │       ├── lora_from_scratch.py
│   │       └── train_lora.py
│   ... (共47节课时)
│   └── 15.3/
│       ├── content.md
│       └── code/
└── assets/
    └── diagrams/                # 架构图/流程图
```

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2026-05-29 | 初始框架，基于8份原始资料整合 |
| v1.1 | 2026-05-29 | 审核后优化：增加RLHF深度、重构Module4-5、课时增至47节 |