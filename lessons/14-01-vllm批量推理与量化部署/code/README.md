# 14.1 vLLM批量推理 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：vllm, openai, tqdm, numpy, psutil |
| `start_vllm.py` | vLLM服务器启动：AWQ/GPTQ/FP8量化配置 |
| `batch_inference.py` | 批量推理：异步并发、流式处理、批量请求 |
| `benchmark_vllm.py` | 性能测试：延迟/吞吐量/并发/长上下文基准 |

## 运行方式

```bash
pip install -r requirements.txt
python start_vllm.py --model meta-llama/Llama-2-7b --quantization awq
python benchmark_vllm.py --model Llama-2-7b --precision int8 --batch_size 16
```

## 核心概念

- PagedAttention: 分页管理KV缓存，减少内存碎片
- 连续批处理: 动态batch，最大化GPU利用率
- 量化推理: AWQ权重、FP8激活，低显存高吞吐