# 11.2 GRPO实战：DeepSeek-R1风格训练

## 课程概述

本节课介绍如何使用 GRPO 实现 DeepSeek-R1 风格的强化学习训练。DeepSeek-R1 是首个在推理任务上展现出超长思维链（Chain-of-Thought）能力的大语言模型，其核心训练方法正是基于 GRPO 强化学习框架。我们将深入解析 GRPO 在数学、代码、逻辑推理任务中的应用，掌握从数据准备到训练配置再到评估验证的完整流程。

**学习目标**
- 理解 DeepSeek-R1 的核心技术特点与设计理念
- 掌握 GRPO 在推理任务中的完整实现流程
- 学会为不同推理任务设计奖励函数
- 掌握 GRPO 训练的关键超参数配置
- 理解蒸馏训练与从头训练的适用场景

**前置知识**
- GRPO 算法原理（详见第11.1节）
- Python 编程基础
- PyTorch 基础
- 大语言模型推理机制

---

## 1. DeepSeek-R1 深度解析

### 1.1 什么是 DeepSeek-R1

DeepSeek-R1 是深度求索（DeepSeek）团队发布的大语言模型，其最显著特点是**超长思维链推理能力**。与传统的 CoT 提示不同，DeepSeek-R1 通过强化学习训练，能够自发地产生中间推理步骤，并在推理过程中进行自我反思和纠错。

```
传统模型回答：
"根据勾股定理，a² + b² = c²，所以答案是 5"

DeepSeek-R1 风格回答：
"让我仔细分析这个问题...
首先，我观察到这是一个直角三角形
根据题目给出的条件，边长分别为 3 和 4
根据勾股定理：a² + b² = c²
3² + 4² = 9 + 16 = 25
c = √25 = 5
验证：3-4-5 是经典的勾股数组合，答案正确"
```

### 1.2 是什么让 DeepSeek-R1 与众不同

| 特性 | 传统推理模型 | DeepSeek-R1 |
|------|-------------|-------------|
| **思维链长度** | 短（few-shot 示例） | 长（可生成数千 token） |
| **训练方式** | SFT + CoT Prompting | GRPO 强化学习 |
| **自我反思** | 无 | 内置反思机制 |
| **奖励信号** | 依赖 Reward Model | 可验证奖励（数学/代码） |
| **涌现能力** | 需要人工设计 | 自发涌现 |

### 1.3 核心技术洞察

DeepSeek-R1 的成功可以归结为三个关键因素：

1. **长思维链的涌现**：当模型被允许生成足够长的推理序列时，复杂的推理能力会自发涌现。这不是因为设计者想到了这一点，而是因为强化学习优化过程自然地发现了这一点。

2. **可验证奖励的优势**：与通用对话任务不同，数学和代码任务具有明确的正确答案。DeepSeek-R1 主要在这类任务上训练，因为奖励信号是明确且可验证的——数学题有唯一答案，代码可以运行测试。

3. **GRPO 的稳定性**：GRPO 消除了 PPO 中需要 Value 网络的问题，在长序列生成任务中显著降低了训练方差，使得 1000+ token 的超长推理链训练成为可能。

---

## 2. GRPO 在推理任务中的应用

### 2.1 为什么推理任务适合 GRPO

推理任务具有一个独特的优势：**可验证的奖励信号**。这使得 GRPO 可以在没有 Reward Model 的情况下工作。

```
GRPO 在推理任务中的优势：

┌────────────────────────────────────────────┐
│         传统 RLHF (需要 Reward Model)       │
├────────────────────────────────────────────┤
│  Prompt → Policy → Response               │
│         ↓                                  │
│    Reward Model                            │
│    (训练复杂，可能有reward hacking)         │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│       GRPO + 可验证奖励 (无需 Reward Model)  │
├────────────────────────────────────────────┤
│  Prompt → Policy → Response               │
│         ↓                                  │
│    外部验证器 (Math Checker / Test Runner)  │
│    (直接验证答案是否正确，简单可靠)            │
└────────────────────────────────────────────┘
```

### 2.2 组采样与优势计算

GRPO 的核心是**分组采样**机制：对于每个输入提示，生成 G 个不同的响应，然后通过组内比较计算优势。

```python
# GRPO 优势计算示意
def compute_advantages(rewards: list[float], group_size: int) -> list[float]:
    """
    对一组内的奖励进行标准化，计算相对优势
    
    Args:
        rewards: 一个组内 G 个响应的奖励列表
        group_size: 组大小 G
    Returns:
        advantages: 标准化后的优势值
    """
    rewards_tensor = torch.tensor(rewards)
    mean_reward = rewards_tensor.mean()
    std_reward = rewards_tensor.std()
    
    # 组内标准化
    normalized = (rewards_tensor - mean_reward) / (std_reward + 1e-8)
    
    return normalized.tolist()
```

**工作流程示例**：

```
输入问题: "求解方程 2x + 5 = 13"

生成 G=4 个响应:
├── 响应1: "2x = 13 - 5 = 8, x = 4"        → 奖励: 1.0 (正确)
├── 响应2: "2x = 13 + 5 = 18, x = 9"       → 奖励: 0.0 (错误)
├── 响应3: "2x = 13 - 5 = 8, x = 4"        → 奖励: 1.0 (正确)
└── 响应4: "x = 13 / 2 = 6.5"              → 奖励: 0.0 (错误)

标准化优势:
mean = 0.5, std ≈ 0.5

响应1: (1.0 - 0.5) / 0.5 = 1.0
响应2: (0.0 - 0.5) / 0.5 = -1.0
响应3: (1.0 - 0.5) / 0.5 = 1.0
响应4: (0.0 - 0.5) / 0.5 = -1.0

策略更新: 强化响应1和3，抑制响应2和4
```

### 2.3 KL 约束的重要性

GRPO 使用 KL 散度约束来防止策略过度偏离参考模型。这在推理任务中尤为重要，因为：

1. **保持基础能力**：确保模型不会因为过度优化特定任务而丧失通用能力
2. **训练稳定性**：防止策略在单次更新中变化过大
3. **控制生成长度**：KL 约束间接有助于控制推理链的长度

```python
# KL 散度计算（无偏估计器）
def compute_kl_loss(policy_logp, reference_logp, beta=0.04):
    """
    计算 GRPO 中使用的 KL 散度损失
    
    使用无偏 KL 估计器：
    D_KL(ref||policy) = exp(log_ref/log_policy) - log(exp(log_ref/log_policy)) - 1
    
    相当于: ratio - log(ratio) - 1
    """
    log_ratio = reference_logp - policy_logp
    ratio = torch.exp(log_ratio)
    
    # 无偏 KL 估计
    kl = ratio - log_ratio - 1
    
    return beta * kl.mean()
```

---

## 3. 训练数据准备

### 3.1 数学推理数据

#### GSM8K 数据集

GSM8K（Grade School Math 8K）包含 8,500 道小学数学应用题，涵盖加法、减法、乘法、除法等基础运算。

```python
# GSM8K 数据示例
{
    "problem": "James has 5 baseball cards. He buys 3 more packs with 7 cards each. How many cards does he have now?",
    "solution": "He buys 3 packs with 7 cards each, so he gets 3 * 7 = 21 cards. Before he had 5 cards. So he now has 5 + 21 = 26 cards."
}
```

#### MATH 数据集

MATH 数据集包含 12,500 道来自竞赛的数学题，难度从初级到高级分为 5 个级别。

```python
# MATH 数据级别
LEVEL_1 = "Training data: Level 1 (AMC 8)"      # 基础竞赛题
LEVEL_2 = "Training data: Level 2 (AMC 10)"      # 中等难度
LEVEL_3 = "Training data: Level 3 (AMC 12/AIME)" # 较高难度
LEVEL_4 = "Training data: Level 4 (AIME)"        # 高难度
LEVEL_5 = "Training data: Level 5 (IMO)"         # 竞赛级难度
```

### 3.2 代码生成数据

#### HumanEval 数据集

HumanEval 包含 164 个编程问题，每个问题包含函数签名、文档字符串和测试用例。

```python
# HumanEval 示例
{
    "task_id": "HumanEval/1",
    "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if any two numbers in the list are close to each other.\n    Args:\n        numbers: List of numbers\n        threshold: The threshold below which a number is considered close to another\n    Returns:\n        bool: True if any two numbers are close, False otherwise.\n    \"\"\"",
    "canonical_solution": "...",
    "test_cases": ["assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False", ...]
}
```

#### MBPP 数据集

MBPP（Mostly Basic Programming Problems）包含 974 个编程问题，更侧重于基础编程能力。

### 3.3 逻辑推理数据

#### ARC-Challenge 数据集

ARC-Challenge 包含需要复杂推理的视觉逻辑题，对于多步推理能力的提升很有帮助。

```python
# ARC-Challenge 示例任务类型
TASK_TYPES = [
    "spatial_reasoning",    # 空间推理
    "logical_deduction",    # 逻辑推演
    "pattern_recognition",  # 模式识别
    "abstract_reasoning"    # 抽象推理
]
```

### 3.4 数据集配置示例

```python
# 数据集配置
REASONING_DATASETS = {
    "math": {
        "gsm8k": {
            "train_path": "datasets/gsm8k/train.jsonl",
            "test_path": "datasets/gsm8k/test.jsonl",
            "num_examples": 7473,
            "difficulty": "easy"
        },
        "math_level5": {
            "train_path": "datasets/math/train_level5.jsonl",
            "test_path": "datasets/math/test.jsonl",
            "num_examples": 2729,
            "difficulty": "hard"
        }
    },
    "code": {
        "humaneval": {
            "train_path": "datasets/humaneval/train.jsonl",
            "test_path": "datasets/humaneval/test.jsonl",
            "num_examples": 164,
            "metric": "pass@1"
        },
        "mbpp": {
            "train_path": "datasets/mbpp/train.jsonl",
            "test_path": "datasets/mbpp/test.jsonl",
            "num_examples": 974,
            "metric": "pass@1"
        }
    },
    "logic": {
        "arc_challenge": {
            "train_path": "datasets/arc/train.jsonl",
            "test_path": "datasets/arc/test.jsonl",
            "num_examples": 1200,
            "task_types": ["spatial", "logical", "pattern"]
        }
    }
}
```

---

## 4. 奖励函数设计

### 4.1 数学奖励函数

数学任务的奖励设计相对直接：**答案正确为1，错误为0**。

```python
def math_reward_function(response: str, ground_truth: str) -> float:
    """
    数学任务奖励函数
    
    检查模型输出中最终答案是否与标准答案一致
    """
    # 提取答案（支持多种格式）
    extracted_answer = extract_final_answer(response)
    
    # 标准化后比较
    pred = normalize_answer(extracted_answer)
    gt = normalize_answer(ground_truth)
    
    # 支持带单位的答案
    if extract_unit(pred) != extract_unit(gt):
        return 0.0
    
    # 数值比较（允许小的浮点误差）
    try:
        if abs(float(pred) - float(gt)) < 1e-6:
            return 1.0
    except ValueError:
        # 无法解析为数值，进行字符串匹配
        if pred.strip() == gt.strip():
            return 1.0
    
    return 0.0


def extract_final_answer(text: str) -> str:
    """
    从思维链中提取最终答案
    
    常见模式:
    - "Therefore, the answer is 42"
    - "#### 42" (MATH 数据集格式)
    - "x = 42"
    """
    # 尝试匹配 boxed 格式（LaTeX）
    import re
    
    boxed = re.search(r'\\boxed\{([^{}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()
    
    # 尝试匹配 #### 格式
    final_line = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if final_line:
        return final_line.group(1).strip()
    
    # 尝试匹配 "the answer is" 模式
    answer = re.search(r'(?:the answer is|answer:|therefore|thus)\s*[:=]?\s*(.+?)(?:\n|$)', 
                      text, re.IGNORECASE)
    if answer:
        return answer.group(1).strip()
    
    # 返回最后一行作为答案
    lines = text.strip().split('\n')
    return lines[-1] if lines else ""
```

### 4.2 代码奖励函数

代码任务的奖励通过运行测试用例获取：

```python
import subprocess
import tempfile
import json

def code_reward_function(
    response: str, 
    test_cases: list[str],
    timeout: int = 10
) -> tuple[float, dict]:
    """
    代码任务奖励函数
    
    Args:
        response: 模型生成的代码
        test_cases: 测试用例列表
        timeout: 执行超时时间（秒）
    
    Returns:
        (reward, details): 奖励值和详细信息
    """
    # 提取代码
    code = extract_code(response)
    
    if not code:
        return 0.0, {"error": "no_code_extracted"}
    
    # 创建临时文件执行
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = f"{tmpdir}/solution.py"
        with open(code_path, 'w') as f:
            f.write(code)
        
        passed = 0
        failed_cases = []
        
        for i, test_case in enumerate(test_cases):
            try:
                # 执行单个测试用例
                result = subprocess.run(
                    ['python', code_path],
                    input=test_case,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode == 0:
                    passed += 1
                else:
                    failed_cases.append({"case": i, "error": result.stderr})
                    
            except subprocess.TimeoutExpired:
                return 0.0, {"error": "timeout", "failed_at": i}
            except Exception as e:
                return 0.0, {"error": str(e), "failed_at": i}
        
        reward = passed / len(test_cases) if test_cases else 0.0
        return reward, {"passed": passed, "total": len(test_cases)}


def extract_code(text: str) -> str:
    """
    从模型输出中提取代码块
    """
    import re
    
    # 尝试匹配 python 代码块
    code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # 尝试匹配普通代码块
    code_blocks = re.findall(r'```\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # 尝试匹配 def 语句开始的内容
    def_match = re.search(r'(def\s+\w+.*?)(?=\n\n|\Z)', text, re.DOTALL)
    if def_match:
        return def_match.group(1)
    
    return text.strip()
```

### 4.3 过程监督奖励（高级）

对于更精细的训练，可以对推理过程中的每个步骤进行奖励：

```python
def process_reward_function(
    response: str, 
    ground_truth: str,
    step_rewards: list[float] = None
) -> float:
    """
    过程监督奖励函数
    
    与结果监督不同，过程监督对每个推理步骤给予奖励
    这需要提前定义每个步骤的期望行为
    """
    steps = extract_reasoning_steps(response)
    
    if step_rewards is None:
        # 简化版本：只有最终答案正确才给奖励
        return math_reward_function(response, ground_truth)
    
    total_reward = 0.0
    for step, reward in zip(steps, step_rewards):
        total_reward += reward * len(step)  # 按步骤长度加权
    
    # 加上最终答案的奖励
    final_reward = math_reward_function(response, ground_truth)
    total_reward += final_reward * 0.5  # 最终答案占 50% 权重
    
    return total_reward
```

---

## 5. 训练配置与实现

### 5.1 GRPO 关键超参数

```python
# grpo_config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GRPOConfig:
    """GRPO 训练配置"""
    
    # 组采样配置
    group_size: int = 16
    """每个问题生成的响应数量，G 值"""
    
    # KL 约束配置
    kl_coef: float = 0.04
    """KL 惩罚系数，通常 0.01-0.1"""
    
    gamma: float = 1.0
    """折扣因子，用于过程监督"""
    
    # PPO 相关
    clip_range: float = 0.2
    """PPO 裁剪范围"""
    
    clip_range_ratio: float = 2.0
    """新旧策略比率裁剪上限"""
    
    # 优化器配置
    learning_rate: float = 1e-6
    """学习率，通常比 SFT 小 10-100 倍"""
    
    lr_scheduler_name: str = "cosine"
    
    warmup_steps: int = 100
    """预热步数"""
    
    # 训练稳定性
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    """梯度裁剪阈值"""
    
    beta: float = 0.004
    """替代 kl_coef 的另一命名"""
    
    # 模型配置
    ref_model: Optional[str] = None
    """参考模型路径或名称"""
    
    # 生成配置
    max_prompt_length: int = 512
    max_response_length: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    
    # 训练控制
    num_episodes: int = 100
    """训练轮数"""
    episodes_per_batch: int = 1
    """每批次训练的episode数量"""
    
    def __post_init__(self):
        """验证配置"""
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO to work")
        if self.kl_coef <= 0:
            raise ValueError("kl_coef must be positive")
```

### 5.2 训练流程实现

```python
# train_grpo.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


classGRPOTrainer:
    """GRPO 训练器"""
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: GRPOConfig,
        reward_fn: Callable,
        tokenizer,
        data_collator=None
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.config = config
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.data_collator = data_collator or self._default_collator
        
        # 设备配置
        self.device = next(policy_model.parameters()).device
        
        # 冻结参考模型
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def step(self, batch: dict) -> dict:
        """
        执行一次 GRPO 训练步骤
        """
        prompts = batch["prompt"]
        ground_truths = batch.get("ground_truth", [None] * len(prompts))
        
        # 1. 分组采样：为每个 prompt 生成 G 个响应
        responses, log_probs = self._group_sampling(prompts)
        
        # 2. 计算奖励
        rewards = self._compute_rewards(responses, ground_truths)
        
        # 3. 计算优势（组内标准化）
        advantages = self._compute_advantages(rewards)
        
        # 4. 计算策略损失
        loss, info = self._compute_policy_loss(
            prompts, responses, log_probs, advantages
        )
        
        # 5. 反向传播
        loss.backward()
        
        # 梯度裁剪
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # 记录统计信息
        info["reward_mean"] = torch.tensor(rewards).mean().item()
        info["reward_std"] = torch.tensor(rewards).std().item()
        info["advantage_mean"] = torch.tensor(advantages).mean().item()
        
        return info
    
    def _group_sampling(self, prompts: list[str]) -> tuple[list[list[str]], list[torch.Tensor]]:
        """
        分组采样：为每个 prompt 生成 group_size 个响应
        """
        all_responses = []
        all_log_probs = []
        
        for prompt in prompts:
            prompt_responses = []
            prompt_log_probs = []
            
            # 使用相同 prompt 生成 G 个不同的响应
            for _ in range(self.config.group_size):
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_prompt_length
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.policy_model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_response_length,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                response_ids = outputs[0][inputs["input_ids"].shape[1]:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                # 计算 log prob（用于策略更新）
                response_inputs = self.tokenizer(
                    response,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_response_length
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.policy_model(
                        input_ids=response_inputs["input_ids"],
                        attention_mask=response_inputs["attention_mask"]
                    )
                    log_probs = torch.log_softmax(outputs.logits, dim=-1)
                
                # 获取响应部分的 log prob
                response_logp = log_probs[:-1].gather(
                    2, response_inputs["input_ids"][1:].unsqueeze(-1)
                ).squeeze(-1).mean()
                
                prompt_responses.append(response)
                prompt_log_probs.append(response_logp)
            
            all_responses.append(prompt_responses)
            all_log_probs.append(torch.stack(prompt_log_probs))
        
        return all_responses, all_log_probs
    
    def _compute_rewards(
        self, 
        responses: list[list[str]], 
        ground_truths: list[str]
    ) -> list[list[float]]:
        """
        计算奖励
        """
        all_rewards = []
        
        for prompt_responses, gt in zip(responses, ground_truths):
            prompt_rewards = []
            for response in prompt_responses:
                reward = self.reward_fn(response, gt)
                prompt_rewards.append(reward)
            all_rewards.append(prompt_rewards)
        
        return all_rewards
    
    def _compute_advantages(self, rewards: list[list[float]]) -> list[torch.Tensor]:
        """
        组内标准化计算优势
        """
        advantages = []
        
        for group_rewards in rewards:
            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            
            mean = rewards_tensor.mean()
            std = rewards_tensor.std()
            
            # 标准化
            normalized = (rewards_tensor - mean) / (std + 1e-8)
            advantages.append(normalized)
        
        return advantages
    
    def _compute_policy_loss(
        self,
        prompts: list[str],
        responses: list[list[str]],
        log_probs: list[torch.Tensor],
        advantages: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        """
        计算 GRPO 策略损失
        """
        policy_losses = []
        kl_losses = []
        clip_fractions = []
        
        for i, prompt in enumerate(prompts):
            group_size = len(responses[i])
            g = group_size
            
            for j in range(g):
                response = responses[i][j]
                log_prob = log_probs[i][j]
                advantage = advantages[i][j]
                
                # 获取参考模型的 log prob
                ref_inputs = self.tokenizer(
                    prompt + response,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_prompt_length + self.config.max_response_length
                ).to(self.device)
                
                with torch.no_grad():
                    ref_outputs = self.ref_model(**ref_inputs)
                    ref_log_probs = torch.log_softmax(ref_outputs.logits, dim=-1)
                
                # 计算 ratio
                ratio = torch.exp(log_prob - ref_log_probs.mean())
                
                # PPO clip
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip_range,
                    1 + self.config.clip_range
                )
                
                # 策略损失
                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                )
                policy_losses.append(policy_loss)
                
                # KL 损失（无偏估计器）
                kl = ratio - torch.log(ratio + 1e-8) - 1
                kl_losses.append(kl)
                
                # clip 比例
                clip_fraction = (torch.abs(ratio - 1) > self.config.clip_range).float()
                clip_fractions.append(clip_fraction)
        
        total_policy_loss = torch.stack(policy_losses).mean()
        total_kl_loss = self.config.kl_coef * torch.stack(kl_losses).mean()
        total_clip_fraction = torch.stack(clip_fractions).mean()
        
        loss = total_policy_loss + total_kl_loss
        
        info = {
            "policy_loss": total_policy_loss.item(),
            "kl_loss": total_kl_loss.item(),
            "clip_fraction": total_clip_fraction.item()
        }
        
        return loss, info
    
    def _default_collator(self, batch):
        """默认数据整理器"""
        return batch


def train_grpo(
    policy_model: nn.Module,
    ref_model: nn.Module,
    train_loader: DataLoader,
    reward_fn: Callable,
    tokenizer,
    config: GRPOConfig,
    output_dir: str = "./output"
) -> dict:
    """
    GRPO 训练主函数
    """
    trainer =GRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate),
        config=config,
        reward_fn=reward_fn,
        tokenizer=tokenizer
    )
    
    global_step = 0
    history = {"loss": [], "reward": [], "kl": [], "clip": []}
    
    for episode in range(config.num_episodes):
        for batch in train_loader:
            info = trainer.step(batch)
            
            global_step += 1
            
            history["loss"].append(info.get("policy_loss", 0))
            history["reward"].append(info.get("reward_mean", 0))
            history["kl"].append(info.get("kl_loss", 0))
            history["clip"].append(info.get("clip_fraction", 0))
            
            if global_step % 10 == 0:
                logger.info(
                    f"Step {global_step} | "
                    f"Loss: {info.get('policy_loss', 0):.4f} | "
                    f"Reward: {info.get('reward_mean', 0):.4f} | "
                    f"KL: {info.get('kl_loss', 0):.4f} | "
                    f"Clip: {info.get('clip_fraction', 0):.2%}"
                )
        
        # 保存检查点
        if (episode + 1) % 10 == 0:
            torch.save({
                "model_state_dict": policy_model.state_dict(),
                "config": config,
                "episode": episode
            }, f"{output_dir}/checkpoint_{episode}.pt")
    
    return history
```

---

## 6. 蒸馏训练 vs 从头训练

### 6.1 蒸馏训练（Distillation）

蒸馏是从更大的推理模型向小模型传递能力的有效方法：

```
蒸馏训练流程：

┌─────────────────────────────────────────────────┐
│              教师模型 (如 DeepSeek-R1 70B)        │
│                     ↓                           │
│        生成大量高质量 reasoning 数据              │
│                     ↓                           │
│        使用 SFT 训练小模型 (如 1B-7B)             │
│                     ↓                           │
│        使用 GRPO 进一步优化                       │
└─────────────────────────────────────────────────┘
```

**特点**：
- 训练速度快，因为有良好的初始化
- 需要高质量的教师模型
- 适合资源受限的场景
- 小模型可能无法完全学到教师的能力

### 6.2 从头训练（Learning from Scratch）

从头训练是完全使用 GRPO 在推理任务上训练：

**特点**：
- 训练时间长，需要更多计算资源
- 可能学到教师没有的推理模式
- 风险是可能学到错误的推理模式
- 需要更仔细的奖励设计

### 6.3 推荐策略

| 场景 | 推荐策略 | 说明 |
|------|---------|------|
| 资源有限 | 蒸馏 + GRPO | 先 SFT 蒸馏，再用 GRPO 优化 |
| 追求最佳效果 | 从头 GRPO | 如果有足够计算资源 |
| 快速验证 | 小模型蒸馏 | 1B-3B 模型先验证流程 |
| 生产部署 | 蒸馏 + 量化 | 小模型 + INT4 量化 |

---

## 7. 结合 SFT 训练

### 7.1 为什么需要 SFT 热启动

GRPO 在推理任务上表现良好，但模型需要有一定的基础能力才能开始学习推理。纯粹的 RL 训练可能导致模型"忘记"如何进行基础对话。

### 7.2 两阶段训练流程

```
阶段 1: SFT 热启动
├── 目标：让模型学会遵循指令和基础推理格式
├── 数据：通用指令数据 + 少量 CoT 示例
├── 训练：标准 SFT，1-2 epoch
└── 产出：具备基础能力的检查点

阶段 2: GRPO 推理优化
├── 目标：强化数学/代码/逻辑推理能力
├── 数据：高质量推理任务数据
├── 训练：GRPO，100+ episodes
└── 产出：具备强推理能力的最终模型
```

### 7.3 SFT 配置示例

```python
# SFT 热启动配置
SFT_CONFIG = {
    # 模型配置
    "model_name": "Qwen2.5-1.5B",
    "model_type": "qwen2",
    
    # 数据配置
    "dataset": "sft_reasoning_data",
    "dataset_split": "train",
    
    # 训练配置
    "learning_rate": 5e-6,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    
    # 输出配置
    "output_dir": "./output/sft_warmup",
    "logging_steps": 10,
    "save_steps": 500,
}
```

### 7.4 GRPO 训练配置示例

```python
# GRPO 训练配置
GRPO_CONFIG = {
    # 模型配置
    "model_name": "./output/sft_warmup/checkpoint",
    "ref_model": "./output/sft_warmup/checkpoint",  # 参考模型 = SFT 检查点
    
    # GRPO 核心配置
    "group_size": 16,              # 建议 8-64
    "kl_coef": 0.04,              # KL 惩罚系数
    "clip_range": 0.2,            # PPO clip 范围
    
    # 优化器配置
    "learning_rate": 1e-6,         # GRPO 学习率通常较小
    "lr_scheduler_name": "cosine",
    "warmup_steps": 100,
    
    # 生成配置
    "max_prompt_length": 512,
    "max_response_length": 2048,  # 长响应支持长推理链
    "temperature": 0.9,           # 适度随机性
    
    # 训练控制
    "num_episodes": 100,
    "max_grad_norm": 1.0,
    
    # 输出配置
    "output_dir": "./output/grpo_final",
}
```

---

## 总结

本节课我们系统学习了 DeepSeek-R1 风格的 GRPO 推理训练：

1. **DeepSeek-R1 核心特点**：超长思维链、可验证奖励、GRPO 强化学习

2. **GRPO 推理应用**：分组采样、组内标准化优势计算、KL 约束

3. **奖励函数设计**：
   - 数学任务：答案正确性检查
   - 代码任务：测试用例执行
   - 过程监督：每个推理步骤评分

4. **训练数据**：GSM8K、MATH、HumanEval、MBPP、ARC-Challenge

5. **训练配置**：group_size、G=16-64，KL 系数，PPO clip

6. **蒸馏 vs 从头训练**：蒸馏效率高，从头训练潜力大

7. **SFT + GRPO 两阶段**：SFT 热启动 + GRPO 优化

---

## 扩展阅读

- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12599) - DeepSeek-R1 官方技术报告
- [GRPO 论文](https://arxiv.org/abs/2402.03300) - GRPO 原始论文
- [Group Relative Policy Optimization](https://github.com/) - OpenGRPO 开源实现
- [VERIFIED Math Problems](https://arxiv.org/abs/2312.08923) - Math-Shepherd 过程监督
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) - DeepSeek 数学推理模型

---

## 复习题

1. **DeepSeek-R1 与传统推理模型最核心的区别是什么？为什么 GRPO 特别适合训练推理能力？**

2. **假设你正在训练模型解决数学问题，请设计一个完整的奖励函数，包括答案提取和验证逻辑。**

3. **为什么 GRPO 需要 `group_size >= 2`？如果 group_size 过小（如 2）会有什么问题？**

4. **在代码生成任务中，使用 GRPO 训练和传统的 Reward Model + PPO 训练相比，有什么优势和劣势？**

5. **解释为什么需要 SFT 热启动 + GRPO 优化的两阶段训练流程，而不是直接用 GRPO 从头训练。**