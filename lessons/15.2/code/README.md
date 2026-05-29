# 15.2 对话系统优化 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：torch, transformers, trl, peft |
| `dialogue_sft.py` | 对话SFT：多轮格式、system prompt、角色标注 |
| `dialogue_reward.py` | 对话奖励模型：HHH评估、偏好数据训练 |
| `dialogue_rlhf.py` | 对话RLHF：多目标奖励、安全约束、PPO训练 |
| `deploy_dialogue.py` | 部署：ConversationManager、SafetyGuardrails、A/B测试 |

## 运行方式

```bash
pip install -r requirements.txt
python dialogue_sft.py --data ./dialogues.json
python dialogue_rlhf.py --data ./preference_dialogues.json --reward_config ./rewards.yaml
python deploy_dialogue.py --model ./dialogue_rlhf --port 8080
```

## 核心概念

- HHH评估: 有帮助(Helpful)、无害(Harmless)、诚实(Honest)
- 多目标奖励: 质量+安全+流畅度的加权组合
- 安全Guardrails: 内容过滤、偏好拒绝、幻觉检测