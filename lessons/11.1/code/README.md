# 11.1 RLHF完整Pipeline - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：torch, transformers, trl, peft, datasets, accelerate |
| `train_reward_model.py` | 奖励模型训练：RewardModel类、Bradley-Terry损失、偏好对训练 |
| `train_ppo.py` | PPO训练：PPOTrainer、Experience收集、KL惩罚 |
| `rlhf_pipeline.py` | 端到端pipeline：RLHFPipeline类管理三阶段 |
| `evaluate.py` | 评估脚本：Win rate、 Reward曲线、KL监控 |

## 运行方式

```bash
pip install -r requirements.txt
python rlhf_pipeline.py --stage sft --model llama-2-7b
python rlhf_pipeline.py --stage reward --model ./sft_model
python rlhf_pipeline.py --stage rlhf --model ./reward_model
```

## 核心概念

- 三阶段: SFT → Reward Model → PPO/GRPO
- 四模型: Actor、Critic、Reference、Reward
- 内存优化: 梯度检查点、混合精度、CPU卸载