# 10.2 TRL库 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：trl, peft, transformers, datasets, torch |
| `train_sft.py` | SFTTrainer示例：LoRA配置、数据packing、梯度检查点 |
| `train_dpo.py` | DPOTrainer示例：偏好数据、beta配置 |
| `train_ppo.py` | PPOTrainer示例：MockRewardModel（需替换为真实奖励模型） |

## 运行方式

```bash
pip install -r requirements.txt
python train_sft.py --dataset ./instruction_data.json
python train_dpo.py --dataset ./preference_data.json --beta 0.1
```

## 核心概念

- SFTTrainer: 有监督微调，支持多模型LoRA/QLoRA
- DPOTrainer: 直接偏好优化，无需奖励模型
- PPOTrainer: 策略优化，需要奖励模型、参考模型、策略模型