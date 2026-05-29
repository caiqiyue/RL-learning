# 3.4 PPO代码实现 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：gymnasium, torch, numpy |
| `ppo.py` | 完整PPO实现：Actor/Critic网络、GAE计算、PPO裁剪损失、训练循环 |
| `train_cartpole.py` | CartPole训练脚本：配置、训练曲线可视化、模型评估 |

## 运行方式

```bash
pip install -r requirements.txt
python train_cartpole.py --episodes 500
```

## 核心概念

- GAE (Generalized Advantage Estimation): 平衡偏差与方差的优势估计方法
- 裁剪机制：防止策略更新过大导致性能崩溃
- Actor-Critic: 策略网络(Actor) + 价值网络(Critic)双结构