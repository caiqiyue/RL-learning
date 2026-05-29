# 12.2 长上下文扩展 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：torch, transformers, accelerate, sentencepiece, scipy |
| `rope_interpolation.py` | RoPE插值实现：线性插值、NTK-aware、YaRN |
| `extend_context.py` | 上下文扩展：模型加载、KV缓存估算、长prompt生成 |
| `eval_context.py` | 评估：NeedleInHaystack、多跳推理、MemoryProfiler |

## 运行方式

```bash
pip install -r requirements.txt
python extend_context.py --model llama-2-7b --context_len 32768
python eval_context.py --task needle --model llama-2-7b-32k
```

## 核心概念

- RoPE: 旋转位置编码，通过复数乘法编码相对位置
- 线性插值: 将位置压缩到训练范围
- NTK-aware: 非线性缩放，保留高频信息