# 5.4 INT8/INT4部署 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：torch, transformers, vllm, tensorrt |
| `deploy_vllm.py` | vLLM量化推理服务器：AWQ/GPTQ/FP8加载、批量请求、流式输出 |
| `benchmark.py` | 性能基准测试：延迟/吞吐量/并发扩展性对比 |
| `deploy_trt_guide.py` | TensorRT详细指南：ONNX导出、INT8校准、引擎构建 |

## 运行方式

```bash
pip install -r requirements.txt
python deploy_vllm.py --model meta-llama/Llama-2-7b --quantization awq
python benchmark.py --model Llama-2-7b --precision int8
```

## 核心概念

- vLLM: PagedAttention连续批处理，高吞吐量
- TensorRT: 低延迟推理引擎，INT8量化加速
- AWQ vs GPTQ: 权重激活感知量化 vs 贪婪量化