# 14.3 模型评估 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：lm-eval, vllm, transformers, openai, anthropic |
| `eval_benchmarks.py` | 标准基准测试：MMLU、HumanEval、GSM8K、TruthfulQA、HellaSwag |
| `llm_judge.py` | LLM-as-Judge评估：单响应打分、成对比较、位置偏置缓解 |
| `eval_pipeline.py` | 评估流水线：基准集成、Judge评估、指标聚合、报告生成 |

## 运行方式

```bash
pip install -r requirements.txt
python eval_benchmarks.py --model llama-2-7b --tasks mmlu,humaneval
python llm_judge.py --judge gpt-4 --model llama-2-7b --mode pairwise
```

## 核心概念

- lm-evaluation-harness: 统一评估框架，支持50+基准
- LLM-as-Judge: 用强模型评价弱模型，需注意位置偏置
- 指标聚合: 几何平均、任务加权、综合评分