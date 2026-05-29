# 9.2 长推理数据合成 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：torch, transformers, sympy, numpy, anthropic |
| `prm_trainer.py` | 过程奖励模型训练：ProcessRewardModel、PRM损失计算 |
| `reasoning_generator.py` | 推理链生成器：Best-of-N采样、PRM引导过滤、格式化 |
| `verify_math.py` | 数学推理验证：方程提取、符号求解、批量验证 |

## 运行方式

```bash
pip install -r requirements.txt
python prm_trainer.py --data ./math_reasoning.json
python reasoning_generator.py --prompt "求积分..." --n 16
```

## 核心概念

- PRM (Process Reward Model): 对每步推理打分而非仅对结果打分
- Best-of-N: 生成N条推理链，用PRM选择最优
- MCTS: 蒙特卡洛树搜索探索推理空间