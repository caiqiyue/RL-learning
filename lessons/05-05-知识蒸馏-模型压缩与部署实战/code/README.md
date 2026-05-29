# 5.5 知识蒸馏 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：torch, transformers, sentence-transformers, scikit-learn, accelerate |
| `distillation.py` | 蒸馏核心模块：DistillationConfig、温度softmax、KL损失、特征蒸馏 |
| `train_student.py` | 学生模型训练：BERT→DistilBERT完整蒸馏流程 |

## 运行方式

```bash
pip install -r requirements.txt
python train_student.py --teacher bert-base --student distilbert-base --temperature 2.0
```

## 核心概念

- 温度softmax: T>1时软化概率分布，传递更多信息
- KL损失: 最小化教师与学生输出分布的KL散度
- 特征蒸馏: 匹配中间层表示而非仅输出logits