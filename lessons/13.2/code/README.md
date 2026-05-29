# 13.2 跨模型适配器 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：peft, transformers, accelerate |
| `convert_adapter.py` | 适配器转换：LoRA权重提取、投影矩阵构建、LLaMA↔Qwen转换 |
| `merge_adapters.py` | 合并策略：SimpleAverage、TaskVector、TIES、DARE、WARM |
| `test_conversion.py` | 转换测试：维度验证、权重相似度、投影稳定性分析 |

## 运行方式

```bash
pip install -r requirements.txt
python convert_adapter.py --source ./lora_llama --target qwen --output ./lora_qwen
python merge_adapters.py --adapters ./adapters --strategy ties --output ./merged
```

## 核心概念

- 架构差异: 层维度不同、attention模块命名差异
- 投影矩阵: 对齐不同架构的权重空间
- 合并策略: TIES通过sign voting解决冲突，WARM用Fisher信息加权