# 8.3 DPO偏好数据 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：trl, peft, transformers, datasets, torch |
| `prepare_dpo_data.py` | DPO数据格式转换：JSON→DPO格式、长度平衡、质量过滤 |
| `train_dpo.py` | DPOTrainer完整训练：偏好对加载、beta配置、训练监控 |
| `generate_synthetic_prefs.py` | 合成偏好生成：SPIN风格、配对生成、RLAIF标注 |

## 运行方式

```bash
pip install -r requirements.txt
python prepare_dpo_data.py --input ./hh_rlhf.json --output ./dpo_data.json
python train_dpo.py --data ./dpo_data.json --beta 0.2
```

## 核心概念

- DPO格式: {prompt, chosen, rejected} 或 {prompt, chosen_response, rejected_response}
- beta: KL惩罚系数，越大越保守
- 偏好数据来源: 人类标注、AI反馈(SPIN/RLAIF)