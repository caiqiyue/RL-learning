# 11.2 GRPO实战 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：torch, transformers, datasets |
| `grpo_config.py` | GRPO配置：GRPOTrainer、GroupConfig、超参数 |
| `train_grpo.py` | GRPO训练实现：_group_sampling、_compute_rewards、_compute_advantages |
| `generate_reasoning.py` | 推理生成：批量生成、评分、过滤 |
| `evaluate_reasoning.py` | 数学/代码评估：正确答案率、推理长度分布 |

## 运行方式

```bash
pip install -r requirements.txt
python train_grpo.py --data ./math_data.json --group_size 16
python evaluate_reasoning.py --model ./grpo_model --dataset math
```

## 核心概念

- GRPO: 无需Value网络，组内标准化优势估计
- Group Size G: 16-64，越大优势估计越稳定但计算成本增加
- 双重KL: 参考模型KL（防遗忘）+ 旧策略KL（稳定性）