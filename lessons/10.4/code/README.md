# 10.4 Unsloth加速 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：unsloth, torch, transformers, peft, datasets |
| `train_unsloth.py` | Unsloth训练脚本：FastLanguageModel、LoRA配置、训练循环 |
| `benchmark.py` | 性能对比：Unsloth vs 标准PEFT的速度/内存对比 |
| `export_hf.py` | 导出HuggingFace格式：LoRA权重合并、推送Hub |

## 运行方式

```bash
pip install -r requirements.txt
python train_unsloth.py --model tinyllama --r 16
python benchmark.py --compare peft
```

## 核心概念

- Unsloth: 优化CUDA核，2倍加速、50%显存节省
- FastLanguageModel: Unsloth模型加载API
- 兼容性: 可导出为标准HF格式，无 vendor lock-in