# 15.1 垂直领域SFT - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：torch, transformers, peft, datasets, medical-legal-specific |
| `data_pipeline.py` | 数据处理：PHI脱敏、文本清洗、医疗/法律prompt模板 |
| `sft_train.py` | 领域微调：LoRA/QLoRA配置、领域适配器训练 |
| `domain_eval.py` | 领域评估：医学USMLE/MedQA、法律Bar测试题 |

## 运行方式

```bash
pip install -r requirements.txt
python data_pipeline.py --input ./medical_notes.json --output ./clean_data.json
python sft_train.py --data ./clean_data.json --domain medical --r 16
python domain_eval.py --model ./medical_model --benchmark usmle
```

## 核心概念

- PHI脱敏: 移除患者隐私信息，HIPAA合规
- 领域prompt: 医疗/法律专业术语和格式
- 评估基准: 医学考试题、法律案例分析