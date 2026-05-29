# 2.4 QLoRA代码实现 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：bitsandbytes, peft, transformers, accelerate, datasets, scikit-learn, tensorboard |
| `qlora_config.py` | QLoRA配置模块：QLoRAConfig类、MemoryOptimizer、内存估算工具 |
| `train_qlora.py` | 完整QLoRA训练脚本：4位量化、LoRA适配器、训练循环、模型导出 |

## 运行方式

```bash
pip install -r requirements.txt
# 测试小模型（8GB显存可用）
python train_qlora.py --model_name PY007/TinyLlama-1.1B-step-50K-103k
# 7B模型（需要~12GB显存）
python train_qlora.py --model_name meta-llama/Llama-2-7b --lora_r 64
```

## 核心概念

- **NF4量化**：NormalFloat4，优化的4位量化格式，适合神经网络权重
- **双量化**：对量化缩放参数再次量化，节省约1GB显存
- **Paged Optimizer**：将优化器状态分页到CPU，减少GPU显存占用
- **梯度检查点**：用30%额外计算换取40%显存节省