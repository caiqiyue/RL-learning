# 10.1 LLaMA-Factory - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：llamafactory, torch, transformers |
| `llamafactory_train.sh` | LLaMA-Factory训练启动脚本 |
| `dataset_example.json` | 示例数据集：Alpaca格式，含8条样本 |
| `train_config.yaml` | LoRA训练配置：模型、数据、训练参数完整配置 |
| `export_and_inference.py` | 导出与推理：合并LoRA权重、批量推理脚本 |

## 运行方式

```bash
pip install -r requirements.txt
bash llamafactory_train.sh
python export_and_inference.py --adapter ./lora_adapter
```

## 核心概念

- LLaMA-Factory: 支持30+模型的一站式训练平台
- YAML配置: 数据格式、训练模式(LoRA/QLoRA/全量)、超参数
- 模型导出: merge_and_unload生成可推理的完整模型