# 7.2 指令数据合成 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：openai或anthropic, tqdm, datasets |
| `self_instruct.py` | Self-Instruct实现：种子池管理、多轮生成-过滤-评分循环 |
| `augmentation.py` | 数据增强：ParaphraseAugmenter、TaskDecomposer、NegativeSampler |

## 运行方式

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python self_instruct.py --seed 100 --output ./synthetic_data.json
```

## 核心概念

- Self-Instruct: 用LLM生成指令-响应对
- 多样性分析: 指令类型分布、主题覆盖率
- 质量评分: 多轮过滤淘汰低质量样本