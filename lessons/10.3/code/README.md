# 10.3 DeepSpeed ZeRO优化 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：deepspeed, transformers, accelerate |
| `ds_config.json` | DeepSpeed ZeRO-3配置模板 |
| `train_deepspeed.py` | 分布式训练脚本：多GPU启动、ZeRO配置、梯度检查点 |
| `launch_multi_gpu.sh` | 多GPU启动脚本 |

## 运行方式

```bash
pip install -r requirements.txt
deepspeed train_deepspeed.py --num_gpus 4 --ds_config ds_config.json
```

## 核心概念

- ZeRO-1/2/3: 优化器状态/梯度/参数分片，内存逐步降低
- ZeRO-Offload: 将部分数据卸载到CPU， 支持更大模型
- 梯度累积: 通过累计步替代大批量，减少显存峰值