# 2.2 LoRA代码实现 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：torch, transformers, peft, datasets, accelerate, bitsandbytes |
| `lora_config.py` | LoRA配置模块：LoraConfig类、TargetModuleMapper、不同模型的target_modules映射 |
| `lora_from_scratch.py` | 纯PyTorch实现的LoRA：LoRALinear类、merge/unmerge功能、参数对比工具 |
| `train_lora.py` | 使用HuggingFace PEFT的完整训练脚本：模型加载、数据处理、Trainer训练 |

## 运行方式

```bash
pip install -r requirements.txt
python train_lora.py --model tinyllama --r 8 --alpha 16
```

## 核心概念

- `r`: LoRA秩，决定低秩矩阵的维度，常用值8/16/32
- `alpha`: 缩放因子，通常设为 `2 * r`
- `target_modules`: 要应用LoRA的层，默认`q_proj`, `v_proj`