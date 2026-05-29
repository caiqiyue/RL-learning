# 11.1 RLHF完整Pipeline实现

本节实现RLHF的端到端pipeline：SFT→奖励模型→PPO优化，
包含三阶段的数据处理、模型训练和评估方法。