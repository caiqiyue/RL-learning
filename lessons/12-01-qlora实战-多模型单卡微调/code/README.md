# 12.1 QLoRA多模型微调 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：bitsandbytes, peft, transformers, datasets |
| `qlora_multi.py` | 多模型训练：MultiModelQLoraTrainer、顺序加载、适配器保存 |
| `adapter_registry.py` | 适配器注册表：AdapterRegistry、批量操作、兼容性检查 |
| `merge_and_eval.py` | 合并与评估：多种合并策略（平均/加权/Task Vector） |

## 运行方式

```bash
pip install -r requirements.txt
python qlora_multi.py --models llama-2-7b qwen-7b chatglm-6b --sequential
python merge_and_eval.py --adapters ./adapters --strategy task_vector
```

## 核心概念

- 单卡多模型: 顺序训练-保存-卸载-加载，节省显存
- 适配器注册: 跟踪多个适配器，支持跨模型对比
- 合并策略: Task Vector合并保留多任务能力