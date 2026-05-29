# 15.3 模型压缩上线 - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：vllm, transformers, awq, gptq, tensorrt, fastapi, prometheus-client |
| `quantize.py` | 量化流程：GPTQ/AWQ/BBQ、校准数据选择、质量验证 |
| `optimize.py` | 优化加速：TensorRT/ONNX Runtime、KV缓存、投机解码 |
| `deploy_vllm.py` | vLLM部署：异步API、Prometheus监控、自动扩缩容 |
| `monitor.py` | 线上监控：延迟/吞吐/错误率追踪、漂移检测、告警 |

## 运行方式

```bash
pip install -r requirements.txt
python quantize.py --model ./sft_model --method awq --output ./quantized_model
python optimize.py --model ./quantized_model --backend tensorrt --output ./optimized_engine
python deploy_vllm.py --model ./optimized_engine --port 8080
python monitor.py --endpoint http://localhost:8080
```

## 核心概念

- 量化精度: INT8/FP8比FP16更省显存但可能有精度损失
- 部署闭环: 微调→量化→优化→部署→监控→迭代
- 监控指标: P50/P95/P99延迟、吞吐量、GPU利用率、错误率