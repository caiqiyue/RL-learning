# 4.4 DPO实战 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：trl, peft, transformers, datasets, torch |
| `prepare_preference_data.py` | 合成偏好数据生成：三种策略(length/judge/SPIN) |
| `train_dpo.py` | DPOTrainer完整训练脚本：偏好数据加载、beta参数配置 |
| `generate_synthetic_prefs.py` | 真实偏好对生成：使用LLM生成response对并标注偏好 |

## 运行方式

```bash
pip install -r requirements.txt
python prepare_preference_data.py --output ./prefs.json
python train_dpo.py --data ./prefs.json --beta 0.1
```

## 核心概念

- DPOTrainer: TRL库的直接偏好优化训练器
- beta: KL惩罚系数，控制策略偏离参考模型的程度
- 偏好对格式: {prompt, chosen, rejected}