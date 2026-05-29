# 5.3 量化感知训练(QAT) - 代码目录

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements.txt` | 依赖包：torch, numpy |
| `fake_quant.py` | 伪量化实现：FakeQuantize算子、直通估计器(STE)、FakeQuantizedLinear |
| `qat_example.py` | QAT训练示例：QAT vs FP32 vs PTQ的对比实验 |
| `smoothquant.py` | SmoothQuant实现：通道级平滑、将异常值从激活移到权重 |

## 运行方式

```bash
pip install -r requirements.txt
python qat_example.py --mode qat
```

## 核心概念

- 伪量化：前向模拟量化、后向直通估计器(STE)
- QAT vs PTQ: 量化感知训练精度更高但训练更慢
- SmoothQuant: 避免INT8量化时的通道间差异问题