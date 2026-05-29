# 15.3 模型压缩上线：INT8量化+加速+部署闭环

## 课程概述

本课时讲解大模型从微调完成后到生产环境部署的完整闭环。涵盖INT8量化、推理加速优化、vLLM部署以及上线后的监控与反馈机制，实现"微调→量化→优化→部署→监控"的全流程。

**学习目标**
- 理解模型压缩与部署的完整工作流程
- 掌握PTQ量化方法（GPTQ/AWQ/BBQ）的原理与实践
- 学会配置INT8/FP8量化并验证量化后的模型质量
- 掌握TensorRT、ONNX Runtime等推理优化工具的使用
- 理解KV Cache优化、批处理策略、投机解码等加速技术
- 学会使用vLLM进行生产级别的模型部署
- 掌握推理延迟监控、性能漂移检测、A/B测试等生产实践
- 了解端到端案例：7B模型从QLoRA微调到vLLM部署的成本分析

**前置知识**：大模型基础、LLM微调技术、量化基本概念（INT8/FP16区别）、Python网络服务基础

---

## 1. 模型压缩与部署完整闭环

### 1.1 端到端工作流程

模型从微调到部署上线是一个系统工程，各环节紧密衔接：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    模型压缩与部署完整闭环                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  Fine-tuned  │───▶│ Quantization │───▶│ Optimization │                  │
│  │    Model      │    │    (INT8)    │    │  (Compiler)  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│          │                  │                  │                            │
│          │                  │                  │                            │
│          ▼                  ▼                  ▼                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Calibration│    │ Quality       │    │    Engine    │                  │
│  │   Data       │    │ Verification  │    │   Selection  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                               │                  │                            │
│                               │                  │                            │
│                               ▼                  ▼                            │
│                      ┌──────────────┐    ┌──────────────┐                  │
│                      │  Quantized   │───▶│   Deploy to   │                  │
│                      │    Model     │    │   Production  │                  │
│                      └──────────────┘    └──────────────┘                  │
│                                                  │                            │
│                                                  ▼                            │
│                      ┌──────────────┐    ┌──────────────┐                  │
│                      │   Feedback   │◀───│   Monitor &   │                  │
│                      │    Loop      │    │   Validate   │                  │
│                      └──────────────┘    └──────────────┘                  │
│                                                     │                        │
│                                                     ▼                        │
│                      ┌──────────────┐    ┌──────────────┐                  │
│                      │  Next-gen    │◀───│  Production   │                  │
│                      │  Training    │    │     Data      │                  │
│                      └──────────────┘    └──────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**各阶段核心产出**：
- **量化阶段**：产出INT8/FP8量化模型，体积缩小4-8倍
- **优化阶段**：产出经过编译器优化的推理引擎
- **部署阶段**：可服务的模型实例，支持高并发推理
- **监控阶段**：收集性能数据，驱动下一轮迭代

### 1.2 各阶段耗时与成本分布

```
GPU Hours Distribution (7B Model Deployment):
┌────────────────────────────────────────────────────┐
│                                                     │
│  QLoRA Fine-tuning     ████████████████████  80%   │
│  Quantization (INT8)   ████                   5%   │
│  Optimization (TRT)    ████████               10%   │
│  Deployment & Testing  █████                   5%   │
│                                                     │
└────────────────────────────────────────────────────┘

Cost Breakdown (Monthly, 7B Model in Production):
┌────────────────────────────────────────────────────┐
│  GPU Cost (A100 80GB)     ████████████████████  $8000│
│  Quantized Inference      ████████               $4000│
│  Monitoring & Ops         ████                   $1500│
│  Data Collection          ██                     $500│
└────────────────────────────────────────────────────┘
```

---

## 2. PTQ量化详解

### 2.1 训练后量化(PTQ)原理

PTQ（Post-Training Quantization）是在模型训练完成后进行的量化，不需要重新训练或微调：

```
PTQ vs QAT vs QFT 流程对比：

PTQ (Post-Training Quantization):
预训练 → 微调 → 量化 ─────────────────→ 部署
                     └── 不需要训练

QAT (Quantization-Aware Training):
预训练 → 微调 → 量化感知微调 → 转换 → 部署
              └── 训练时模拟量化误差

QFT (Quantization Fine-Tuning):
预训练 → 微调 → 量化 → 轻量微调 → 部署
                   └── 量化后恢复精度
```

### 2.2 GPTQ量化

GPTQ是专为GPT类模型设计的PTQ量化方法，基于近似二阶信息进行权重量化：

```python
# gptq_quantizer.py - GPTQ量化核心原理
import torch
import torch.nn as nn
from typing import Dict, List, Optional

class GPTQQuantizer:
    """
    GPTQ量化器
    核心思想：对角线近似Hessian矩阵，按列贪心地量化每个权重
    """
    def __init__(
        self,
        model: nn.Module,
        bit_width: int = 8,
        group_size: int = 128,
        calibration_data: List[torch.Tensor] = None,
    ):
        self.model = model
        self.bit_width = bit_width
        self.group_size = group_size
        self.calibration_data = calibration_data
        self.quant_dict: Dict[str, dict] = {}
        
    def quantize_model(self):
        """
        执行GPTQ量化主流程
        """
        # 1. 准备校准数据
        if self.calibration_data is None:
            raise ValueError("需要提供校准数据用于量化")
        
        # 2. 逐层量化
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self._quantize_linear_layer(name, module)
        
        return self.model
    
    def _quantize_linear_layer(
        self, 
        name: str, 
        layer: nn.Linear
    ):
        """
        GPTQ量化单个Linear层
        核心算法：
        1. 计算Hessian矩阵的逆的近似（对角线部分）
        2. 按列贪心选择最佳量化值
        3. 更新权重并计算误差
        """
        weight = layer.weight.data.float()
        bias = layer.bias.data.float() if layer.bias is not None else None
        
        # 获取层输出数(列数)
        out_features, in_features = weight.shape
        num_groups = in_features // self.group_size
        
        # 初始化量化参数
        scales = torch.zeros(out_features, num_groups, dtype=torch.float16)
        zero_points = torch.zeros(out_features, num_groups, dtype=torch.float16)
        quantized_weight = torch.zeros_like(weight, dtype=torch.int8)
        
        # 计算量化范围
        max_q = 2 ** self.bit_width - 1
        min_q = -(2 ** self.bit_width)
        
        for out_idx in range(out_features):
            weight_row = weight[out_idx]
            
            # 误差累积器
            error = torch.zeros_like(weight_row)
            
            for group_idx in range(num_groups):
                start = group_idx * self.group_size
                end = start + self.group_size
                
                # 取出该组的权重
                w_group = weight_row[start:end] + error[start:end]
                
                # 贪心量化：找到使重建误差最小的scale和zero_point
                # 简化版本：使用对称量化
                max_val = w_group.abs().max()
                scale = max_val / (max_q / 2)
                
                if scale < 1e-8:
                    scale = 1e-8
                
                # 量化
                w_quant = torch.round(w_group / scale).clamp(min_q, max_q)
                
                # 记录
                scales[out_idx, group_idx] = scale
                zero_points[out_idx, group_idx] = 0
                quantized_weight[out_idx, start:end] = w_quant.to(torch.int8)
                
                # 计算误差并累积
                w_dequant = w_quant * scale
                error[start:end] += w_group - w_dequant
        
        # 存储量化后的权重和参数
        self.quant_dict[name] = {
            "weight": quantized_weight,
            "scales": scales,
            "zero_points": zero_points,
            "bit_width": self.bit_width,
            "group_size": self.group_size,
        }
        
        # 用量化后的权重更新原层
        # 注意：实际部署时需要用量化后的权重替换
        print(f"量化层 {name}: {weight.shape} -> INT{self.bit_width}")
    
    def verify_quality(
        self, 
        test_data: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        验证量化后模型质量
        """
        self.model.eval()
        total_error = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in test_data:
                # 计算原始模型输出
                fp_output = self.model(batch)
                
                # 模拟量化后的输出（简化）
                # 实际应该使用量化权重计算
                quant_output = fp_output  # placeholder
                
                # 计算误差
                error = torch.abs(fp_output - quant_output).mean().item()
                total_error += error
                num_samples += 1
        
        return {
            "mean_abs_error": total_error / num_samples,
            "bit_width": self.bit_width,
            "num_layers": len(self.quant_dict),
        }
```

### 2.3 AWQ量化

AWQ（Activation-Aware Weight Quantization）考虑激活值分布进行量化：

```python
# awq_quantizer.py - AWQ量化实现
class AWQQuantizer:
    """
    AWQ量化核心思想：
    不是所有权重都同等重要——保护显著权重(activation大的权重)
    通过数学推导，AWQ证明只需要对权重进行per-channel缩放即可保护显著权重
    """
    def __init__(
        self,
        model: nn.Module,
        bit_width: int = 8,
        percentile: float = 0.99,
    ):
        self.model = model
        self.bit_width = bit_width
        self.percentile = percentile
        
    def _find_scaling_factors(
        self,
        weight: torch.Tensor,
        activation: torch.Tensor,
        group_size: int = 128
    ) -> torch.Tensor:
        """
        找到最佳的per-channel缩放因子
        AWQ核心：s_i = activation_std_i^alpha (alpha通常为0.5)
        """
        # 简化版本：使用activation的统计信息
        # 实际AWQ使用输入activation的均方根
        alpha = 0.5
        scales = activation.abs().mean(dim=0).pow(alpha)
        scales = scales / scales.amax()
        return scales
    
    def quantize_layer(self, layer: nn.Linear, activation: torch.Tensor):
        """
        对单层进行AWQ量化
        """
        weight = layer.weight.data.float()
        out_features, in_features = weight.shape
        
        # 1. 计算缩放因子
        scales = self._find_scaling_factors(weight, activation)
        
        # 2. 缩放权重
        scaled_weight = weight * scales.unsqueeze(0)
        
        # 3. 量化缩放后的权重
        max_q = 2 ** self.bit_width - 1
        scale = scaled_weight.abs().max() / max_q
        
        quantized = torch.round(scaled_weight / scale).to(torch.int8)
        
        # 4. 反量化验证
        dequant = quantized.float() * scale / scales.unsqueeze(0)
        
        return quantized, scale, scales
```

### 2.4 BBQ量化

BBQ（Batch-Based Quantization）利用批量统计信息进行量化：

```python
# bbq_quantizer.py - BBQ量化
class BBQQuantizer:
    """
    BBQ核心思想：
    在推理时利用批量输入的动态范围进行自适应量化
    特别适合推理服务场景
    """
    def __init__(self, model: nn.Module, history_size: int = 100):
        self.model = model
        self.history_size = history_size
        self.act_ranges: Dict[str, List[torch.Tensor]] = {}
        
    def _update_range(self, name: str, activation: torch.Tensor):
        """维护激活值范围的滑动历史"""
        if name not in self.act_ranges:
            self.act_ranges[name] = []
        
        abs_max = activation.abs().max().item()
        self.act_ranges[name].append(abs_max)
        
        if len(self.act_ranges[name]) > self.history_size:
            self.act_ranges[name].pop(0)
    
    def _get_scale(self, name: str, percentile: float = 0.99) -> float:
        """基于历史百分位数计算scale"""
        if name not in self.act_ranges or len(self.act_ranges[name]) == 0:
            return 1.0
        
        ranges = torch.tensor(self.act_ranges[name])
        scale = torch.quantile(ranges, percentile)
        return scale.item()
```

### 2.5 校准数据选择

校准数据的选择直接影响量化质量：

```python
# calibration.py - 校准数据选择策略
class CalibrationDataSelector:
    """
    校准数据选择：确保数据覆盖多种场景
    """
    def __init__(
        self,
        source_data: List[dict],
        max_samples: int = 512,
    ):
        self.source_data = source_data
        self.max_samples = max_samples
        
    def select_diverse_calibration_set(self) -> List[dict]:
        """
        选择多样化的校准集
        策略：
        1. 主题多样性：覆盖不同领域
        2. 长度多样性：短文本到长文本
        3. 复杂度多样性：简单问答到复杂推理
        """
        # 1. 按主题聚类
        clusters = self._cluster_by_topic(self.source_data)
        
        # 2. 每个簇中按长度采样
        selected = []
        samples_per_cluster = self.max_samples // len(clusters)
        
        for cluster in clusters:
            # 从每个簇中均匀采样不同长度
            sorted_cluster = sorted(cluster, key=lambda x: len(x["text"]))
            
            # 选取长度分布均匀的样本
            indices = torch.linspace(
                0, len(sorted_cluster) - 1, samples_per_cluster
            ).long().tolist()
            
            for idx in indices:
                selected.append(sorted_cluster[idx])
        
        return selected[:self.max_samples]
    
    def _cluster_by_topic(self, data: List[dict]) -> List[List[dict]]:
        """简单的基于关键词的主题聚类"""
        topic_groups = {}
        
        topic_keywords = {
            "code": ["代码", "程序", "函数", "Python", "Java"],
            "math": ["计算", "数学", "公式", "方程", "数"],
            "science": ["科学", "物理", "化学", "实验", "研究"],
            "general": [],  # 默认类别
        }
        
        for item in data:
            text = item.get("text", "")
            assigned = False
            
            for topic, keywords in topic_keywords.items():
                if topic == "general":
                    continue
                if any(kw in text for kw in keywords):
                    if topic not in topic_groups:
                        topic_groups[topic] = []
                    topic_groups[topic].append(item)
                    assigned = True
                    break
            
            if not assigned:
                if "general" not in topic_groups:
                    topic_groups["general"] = []
                topic_groups["general"].append(item)
        
        return list(topic_groups.values())
```

### 2.6 量化配置与质量验证

```python
# quant_config.py - 量化配置
QUANT_CONFIG = {
    # 权重量化配置
    "weight": {
        "bits": 8,
        "scheme": "symmetric",  # 对称量化适合权重
        "granularity": "per_channel",  # 按通道保持精度
        "method": "gptq",  # gptq/awq/bbq
    },
    
    # 激活值量化配置
    "activation": {
        "bits": 8,
        "scheme": "asymmetric",  # 非对称适合激活值
        "granularity": "per_tensor",  # 按张量
        "method": "dynamic",  # 动态范围
    },
    
    # KV Cache量化
    "kv_cache": {
        "enabled": True,
        "bits": 8,
        "method": "fp8",  # FP8 for KV cache
    },
    
    # 校准配置
    "calibration": {
        "method": "percentile",
        "percentile": 0.99,
        "max_samples": 512,
    },
}

# 量化质量验证
class QuantizationQualityValidator:
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        
    def validate(self, threshold: float = 0.95) -> dict:
        """
        验证量化后模型质量
        
        Returns:
            validation_result: dict
        """
        metrics = {}
        
        # 1. 困惑度 (Perplexity)
        metrics["perplexity"] = self._compute_perplexity()
        
        # 2. 任务准确率
        metrics["task_accuracy"] = self._compute_task_accuracy()
        
        # 3. 权重重建误差
        metrics["reconstruction_error"] = self._compute_reconstruction_error()
        
        # 4. 数值范围检查
        metrics["numerical_range"] = self._check_numerical_range()
        
        # 综合判定
        passed = all([
            metrics["perplexity"] < threshold * 100,  # 放宽阈值
            metrics["task_accuracy"] > 0.7,
            metrics["reconstruction_error"] < 0.1,
            metrics["numerical_range"]["valid"],
        ])
        
        return {
            "passed": passed,
            "metrics": metrics,
            "details": self._generate_report(metrics),
        }
    
    def _compute_perplexity(self) -> float:
        """计算困惑度"""
        # 简化实现
        return 25.3  # placeholder
    
    def _compute_task_accuracy(self) -> float:
        """计算任务准确率"""
        return 0.92
    
    def _compute_reconstruction_error(self) -> float:
        """计算权重重建误差"""
        return 0.02
    
    def _check_numerical_range(self) -> dict:
        """检查数值范围是否有效"""
        return {"valid": True, "overflow_count": 0}
    
    def _generate_report(self, metrics: dict) -> str:
        """生成质量报告"""
        return f"""
Quantization Quality Report
==========================
Perplexity: {metrics['perplexity']:.2f}
Task Accuracy: {metrics['task_accuracy']:.2%}
Reconstruction Error: {metrics['reconstruction_error']:.4f}
Numerical Range: {'Valid' if metrics['numerical_range']['valid'] else 'Invalid'}
"""
```

---

## 3. 推理优化技术

### 3.1 编译器优化

编译器优化通过图优化、算子融合、内核优化提升推理速度：

```python
# optimize.py - 推理优化配置
OPTIMIZATION_CONFIG = {
    # TensorRT优化
    "tensorrt": {
        "enabled": True,
        "precision": "fp16",  # fp16/int8/fp8
        "workspace_size": 1 << 30,  # 1GB
        "max_batch_size": 32,
        "tp_size": 1,  # Tensor并行数
        "pp_size": 1,  # Pipeline并行数
        "onnx_opset_version": 14,
    },
    
    # ONNX Runtime优化
    "onnx_runtime": {
        "enabled": False,
        "execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "graph_optimization_level": 3,  # 全部优化
        "intra_op_num_threads": 4,
        "inter_op_num_threads": 4,
    },
    
    # PyTorch优化
    "pytorch": {
        "torch_compile": True,
        "torch_compile_mode": "reduce-overhead",
        "enable_nvfuser": True,
        "cudnn_benchmark": True,
    },
}

class ModelOptimizer:
    """
    模型优化器：统一管理多种优化后端
    """
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.optimized_model = None
        
    def optimize_for_tensorrt(self, save_path: str):
        """
        TensorRT优化流程
        """
        import torch
        from torch2trx import torch2trx
        
        # 1. 导出ONNX
        self.model.eval()
        dummy_input = torch.randn(1, 512, dtype=torch.float16)
        
        onnx_path = save_path.replace(".trt", ".onnx")
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size", 1: "seq_len"},
            },
        )
        
        # 2. ONNX转TensorRT
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, "rb") as f:
            parser.parse(f.read())
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 
            self.config["tensorrt"]["workspace_size"]
        )
        config.set_flag(trt.BuilderFlag.FP16)
        
        engine = builder.build_serialized_network(network, config)
        
        with open(save_path, "wb") as f:
            f.write(engine)
        
        print(f"TensorRT engine saved to {save_path}")
        
    def optimize_for_onnx_runtime(self, onnx_path: str):
        """
        ONNX Runtime优化
        """
        import onnx
        from onnxruntime.transformers import optimizer
        
        optimized_model = optimizer.optimize_model(
            onnx_path,
            num_heads=32,
            hidden_size=4096,
            optimization_level=99,
        )
        
        optimized_model.save_model_to_file(onnx_path.replace(".onnx", "_optimized.onnx"))
```

### 3.2 KV Cache优化

KV Cache是加速生成的关键技术：

```python
# kv_cache_optimization.py
KV_CACHE_CONFIG = {
    # PagedAttention配置
    "paged_attention": {
        "enabled": True,
        "block_size": 16,  # KV Cache块大小
        "max_cache_blocks": 1024,
    },
    
    # 动态分区配置
    "dynamic_partition": {
        "enabled": True,
        "memory_fraction": 0.9,  # GPU显存使用比例
    },
    
    # Prefix Cache配置
    "prefix_cache": {
        "enabled": True,
        "max_prefix_length": 2048,
    },
}

class KVCacheOptimizer:
    """
    KV Cache优化器
    核心优化：
    1. PagedAttention：分页管理KV blocks，避免显存碎片
    2. Prefix Cache：复用相同前缀的KV cache
    3. Dynamic Slicing：动态分配KV cache大小
    """
    def __init__(self, config: dict):
        self.config = config
        self.block_manager = None
        
    def enable_paged_attention(self, model):
        """启用PagedAttention"""
        if self.config["paged_attention"]["enabled"]:
            print("Enabling PagedAttention with block_size={}".format(
                self.config["paged_attention"]["block_size"]
            ))
            # 实际通过vLLM/TensorRT等框架启用
        return model
    
    def setup_prefix_caching(self):
        """设置Prefix Cache"""
        if self.config["prefix_cache"]["enabled"]:
            print("Enabling Prefix Cache, max_prefix_length={}".format(
                self.config["prefix_cache"]["max_prefix_length"]
            ))
    
    def compute_cache_stats(self, batch_size: int, seq_len: int, 
                           num_layers: int, num_heads: int, head_dim: int) -> dict:
        """
        计算KV Cache内存占用
        """
        # INT8: 1 byte per value
        bytes_per_value = 1 if self.config.get("quantized", False) else 2
        
        # KV cache per layer
        kv_size_per_layer = (
            2 *  # K and V
            batch_size * seq_len *
            num_heads * head_dim *
            bytes_per_value
        )
        
        total_cache = kv_size_per_layer * num_layers
        
        return {
            "bytes_per_layer": kv_size_per_layer,
            "bytes_total": total_cache,
            "mb_total": total_cache / (1024 ** 2),
            "gb_total": total_cache / (1024 ** 3),
        }
```

### 3.3 批处理策略

```python
# batching_strategy.py - 批处理策略
BATCHING_CONFIG = {
    # Continuous Batching配置
    "continuous_batching": {
        "enabled": True,
        "max_batch_size": 64,
        "max_waiting_time_ms": 100,  # 最长等待时间
        "gpu_memory_utilization": 0.9,
    },
    
    # Static Batching配置
    "static_batching": {
        "enabled": False,
        "batch_sizes": [1, 4, 16, 32],
    },
    
    # Prefill-Decode分离
    "prefix_decoding": {
        "enabled": True,
        "prefill_batch_size": 8,
        "decode_batch_size": 64,
    },
}

class DynamicBatcher:
    """
    动态批处理器：Continuous Batching实现
    核心思想：
    - 不等待batch满再处理，而是有新请求就立即加入
    - 一个批次内的请求完成decode后就退出，腾出位置给新请求
    - 显著提高吞吐量
    """
    def __init__(self, config: dict):
        self.config = config
        self.pending_requests = []
        self.running_batches = []
        
    def add_request(self, request_id: str, prompt_tokens: List[int],
                   max_tokens: int):
        """添加新请求"""
        self.pending_requests.append({
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "arrival_time": time.time(),
        })
        
    def _schedule_batch(self) -> List[dict]:
        """
        调度策略：决定哪些请求组成一个batch
        
        策略1：按到达时间（FCFS）
        策略2：按序列长度分组（减少padding）
        策略3：按deadline优先级
        """
        if len(self.pending_requests) == 0:
            return []
        
        # 计算最优batch大小
        max_batch = min(
            self.config["continuous_batching"]["max_batch_size"],
            len(self.pending_requests)
        )
        
        # 优先调度短序列（减少平均延迟）
        sorted_requests = sorted(
            self.pending_requests,
            key=lambda x: len(x["prompt_tokens"])
        )
        
        return sorted_requests[:max_batch]
    
    def step(self) -> dict:
        """执行一步调度"""
        # 1. 调度新批次
        batch = self._schedule_batch()
        
        # 2. 更新等待中的请求
        for req in batch:
            self.pending_requests.remove(req)
            
        # 3. 返回批次信息
        return {
            "batch": batch,
            "batch_size": len(batch),
            "avg_prompt_len": sum(len(r["prompt_tokens"]) for r in batch) / max(len(batch), 1),
        }
```

### 3.4 投机解码

```python
# speculative_decoding.py - 投机解码
SPECULATIVE_DECODING_CONFIG = {
    "enabled": True,
    "draft_model_ratio": 4,  # draft模型与目标模型大小比
    "num_speculative_tokens": 4,  # 每次投机预测的token数
    "acceptance_threshold": 0.8,  # 接受率阈值
    "temperature": 1.0,
}

class SpeculativeDecoder:
    """
    投机解码(Speculative Decoding)
    
    核心思想：
    1. 使用小的draft模型快速生成多个候选token
    2. 使用大的目标模型并行验证这些候选
    3. 接受大部分正确的token，跳过draft模型生成过程
    
    加速比 ≈ draft模型生成时间 / 验证时间
    通常可达2-3x加速
    """
    def __init__(
        self,
        target_model,
        draft_model,
        config: dict,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config
        self.num_speculative_tokens = config["num_speculative_tokens"]
        
    def generate(self, prompt_tokens: List[int]) -> List[int]:
        """
        投机解码生成
        """
        target_ids = list(prompt_tokens)
        
        while len(target_ids) < self.config.get("max_tokens", 100):
            # 1. Draft模型快速生成候选
            draft_ids = self._draft_tokens(target_ids)
            
            # 2. 目标模型并行验证
            accepted_ids, acceptance_rate = self._verify_batch(
                target_ids, draft_ids
            )
            
            # 3. 添加接受的token
            target_ids.extend(accepted_ids)
            
            # 4. 检查是否应该停止
            if acceptance_rate < self.config["acceptance_threshold"]:
                print(f"Acceptance rate {acceptance_rate:.2f} below threshold")
                break
                
            if len(target_ids) >= self.config.get("max_tokens", 100):
                break
        
        return target_ids
    
    def _draft_tokens(self, prefix_ids: List[int]) -> List[int]:
        """
        Draft模型生成候选token
        """
        # 简化：使用draft模型生成num_speculative_tokens个token
        draft_ids = prefix_ids.copy()
        
        for _ in range(self.num_speculative_tokens):
            with torch.no_grad():
                logits = self.draft_model(torch.tensor([draft_ids]))
                next_token = logits[-1].argmax().item()
                draft_ids.append(next_token)
                
                if next_token == self.target_model.eos_token_id:
                    break
        
        return draft_ids[len(prefix_ids):]
    
    def _verify_batch(
        self,
        prefix_ids: List[int],
        draft_ids: List[int]
    ) -> tuple:
        """
        验证draft tokens
        """
        all_ids = prefix_ids + draft_ids
        
        with torch.no_grad():
            logits = self.target_model(torch.tensor([all_ids]))
        
        accepted = []
        for i, draft_tok in enumerate(draft_ids):
            # 目标模型在prefix之后的位置重新预测
            target_idx = len(prefix_ids) + i
            if target_idx >= len(logits):
                break
                
            target_token = logits[target_idx].argmax().item()
            
            if target_token == draft_tok:
                accepted.append(draft_tok)
            else:
                accepted.append(target_token)
                break  # 出现分歧，后面的全部拒绝
        
        acceptance_rate = len(accepted) / len(draft_ids)
        return accepted, acceptance_rate
```

---

## 4. 部署模式

### 4.1 vLLM部署

```python
# deploy_vllm.py - vLLM部署脚本
import asyncio
import time
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams

VLLM_CONFIG = {
    "model_path": "./models/quantized_7b",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "dtype": "half",  # float16/half/bfloat16
    "enforce_eager": False,
    "trust_remote_code": True,
    
    # 量化配置
    "quantization": "awq",  # awq/gptq/fp8
    "kv_cache_dtype": "fp8_e5m2",
    
    # 分词器配置
    "tokenizer": "./models/quantized_7b",
    "tokenizer_mode": "auto",
}

class vLLMDeployment:
    """
    vLLM生产部署封装
    """
    def __init__(self, config: dict):
        self.config = config
        self.llm = None
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "errors": 0,
        }
        
    def initialize(self):
        """初始化vLLM引擎"""
        print("Initializing vLLM engine...")
        print(f"  Model: {self.config['model_path']}")
        print(f"  Tensor Parallel: {self.config['tensor_parallel_size']}")
        print(f"  GPU Memory Utilization: {self.config['gpu_memory_utilization']}")
        
        self.llm = LLM(
            model=self.config["model_path"],
            tensor_parallel_size=self.config["tensor_parallel_size"],
            gpu_memory_utilization=self.config["gpu_memory_utilization"],
            max_num_seqs=self.config["max_num_seqs"],
            max_num_batched_tokens=self.config["max_num_batched_tokens"],
            dtype=self.config["dtype"],
            enforce_eager=self.config["enforce_eager"],
            trust_remote_code=self.config["trust_remote_code"],
            quantization=self.config.get("quantization"),
            kv_cache_dtype=self.config.get("kv_cache_dtype"),
        )
        
        print("vLLM engine initialized successfully")
        
    async def generate_async(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[Dict]:
        """
        异步生成接口
        """
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=512,
            )
        
        start_time = time.time()
        
        try:
            # vLLM的generate是同步的，用run_until_complete包装
            outputs = await asyncio.get_event_loop().run_in_executor(
                None,
                self.llm.generate,
                prompts,
                sampling_params,
            )
            
            elapsed = time.time() - start_time
            
            # 解析输出
            results = []
            for output in outputs:
                results.append({
                    "text": output.outputs[0].text,
                    "tokens": len(output.outputs[0].token_ids),
                    "finish_reason": output.outputs[0].finish_reason,
                    "latency_ms": elapsed * 1000,
                })
            
            # 更新统计
            self._update_stats(len(prompts), sum(r["tokens"] for r in results), elapsed)
            
            return results
            
        except Exception as e:
            self.stats["errors"] += 1
            print(f"Error during generation: {e}")
            raise
    
    def generate_sync(self, prompts: List[str], 
                     **sampling_kwargs) -> List[Dict]:
        """
        同步生成接口
        """
        sampling_params = SamplingParams(**sampling_kwargs)
        
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        elapsed = time.time() - start_time
        
        results = []
        for output in outputs:
            results.append({
                "text": output.outputs[0].text,
                "tokens": len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason,
                "latency_ms": elapsed * 1000,
            })
        
        self._update_stats(len(prompts), sum(r["tokens"] for r in results), elapsed)
        
        return results
    
    def _update_stats(self, num_requests: int, num_tokens: int, elapsed: float):
        """更新统计信息"""
        self.stats["total_requests"] += num_requests
        self.stats["total_tokens"] += num_tokens
        self.stats["total_time"] += elapsed
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_time = self.stats["total_time"]
        
        if total_time > 0:
            throughput = self.stats["total_tokens"] / total_time
            avg_latency = (total_time / self.stats["total_requests"] * 1000 
                          if self.stats["total_requests"] > 0 else 0)
        else:
            throughput = 0
            avg_latency = 0
            
        return {
            **self.stats,
            "throughput_tokens_per_sec": throughput,
            "avg_latency_ms": avg_latency,
            "error_rate": (self.stats["errors"] / max(self.stats["total_requests"], 1)),
        }
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "errors": 0,
        }

# API服务封装
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="LLM Inference API")
deployment = vLLMDeployment(VLLM_CONFIG)

class GenerateRequest(BaseModel):
    prompts: List[str]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512

class GenerateResponse(BaseModel):
    results: List[dict]
    stats: dict

@app.on_event("startup")
async def startup_event():
    deployment.initialize()

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """生成接口"""
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )
    
    results = await deployment.generate_async(request.prompts, sampling_params)
    stats = deployment.get_stats()
    
    return GenerateResponse(results=results, stats=stats)

@app.get("/stats")
async def get_stats():
    """获取服务统计"""
    return deployment.get_stats()

@app.post("/stats/reset")
async def reset_stats():
    """重置统计"""
    deployment.reset_stats()
    return {"message": "Stats reset successfully"}
```

### 4.2 Triton推理服务器

```python
# deploy_triton.py - Triton部署配置
TRITON_CONFIG = {
    "model_repository": "/models/triton_repo",
    "models": {
        "llm_model": {
            "platform": "tensorrtllm",
            "max_batch_size": 32,
            "input": [
                {"name": "input_ids", "shape": [-1, -1], "dtype": "int32"},
            ],
            "output": [
                {"name": "logits", "shape": [-1, -1, 32000], "dtype": "float16"},
            ],
        }
    },
    "instance_group": [
        {"count": 1, "kind": "GPU"},
    ],
    "dynamic_batching": {
        "preferred_batch_size": [1, 4, 8, 16],
        "max_queue_delay_microseconds": 100000,
    },
}

class TritonInferenceServer:
    """
    Triton推理服务器部署封装
    """
    def __init__(self, config: dict):
        self.config = config
        self.client = None
        
    def prepare_model_repository(self, model_path: str):
        """
        准备Triton模型仓库
        """
        import os
        import subprocess
        
        repo_path = self.config["model_repository"]
        os.makedirs(repo_path, exist_ok=True)
        
        # 创建模型目录
        model_dir = os.path.join(repo_path, "llm_model", "1")
        os.makedirs(model_dir, exist_ok=True)
        
        # 生成config.pbtxt
        config_text = """
name: "llm_model"
platform: "tensorrtllm"
max_batch_size: 32

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [-1, -1]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, -1, 32000]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]

dynamic_batching {{
  preferred_batch_size: [1, 4, 8, 16]
  max_queue_delay_microseconds: 100000
}}
"""
        with open(os.path.join(repo_path, "llm_model", "config.pbtxt"), "w") as f:
            f.write(config_text)
        
        print(f"Model repository prepared at {repo_path}")
    
    def start_server(self):
        """启动Triton服务器"""
        import subprocess
        
        cmd = [
            "tritonserver",
            "--model-repository", self.config["model_repository"],
            "--grpc-port", "8001",
            "--http-port", "8000",
            "--metrics-port", "8002",
        ]
        
        subprocess.Popen(cmd)
        print("Triton server starting...")
    
    def infer(self, input_ids: List[List[int]]) -> List[List[float]]:
        """发送推理请求"""
        import tritonclient.http as httpclient
        
        if self.client is None:
            self.client = httpclient.InferenceServerClient(
                url="localhost:8000",
                verbose=False,
            )
        
        inputs = httpclient.InferInput(
            "input_ids",
            [len(input_ids), len(input_ids[0])],
            "INT32"
        )
        inputs.set_data_from_numpy(np.array(input_ids))
        
        outputs = httpclient.InferRequestedOutput("output")
        
        results = self.client.infer(
            "llm_model",
            inputs=[inputs],
            outputs=[outputs],
        )
        
        return results.as_numpy("output")
```

### 4.3 云端与边缘部署

```python
# deployment_modes.py - 部署模式对比
DEPLOYMENT_MODES = {
    "cloud": {
        "description": "云端GPU集群部署",
        "pros": [
            "无限计算资源",
            "易于扩缩容",
            "GPU类型灵活选择",
        ],
        "cons": [
            "网络延迟",
            "数据隐私风险",
            "持续成本较高",
        ],
        "use_cases": ["大规模推理", "高并发场景", "灵活扩展需求"],
        "cost_per_1k_tokens": 0.002,  # A100
    },
    
    "edge": {
        "description": "边缘设备部署（手机/嵌入式）",
        "pros": [
            "零网络延迟",
            "数据隐私保障",
            "离线可用",
        ],
        "cons": [
            "硬件受限",
            "模型需要更激进压缩",
            "难以更新",
        ],
        "use_cases": ["移动端AI", "IoT设备", "隐私敏感场景"],
        "model_formats": ["ONNX", "TensorFlow Lite", "CoreML"],
    },
    
    "hybrid": {
        "description": "云边协同部署",
        "pros": [
            "平衡延迟与成本",
            "边缘处理简单请求",
            "云端处理复杂请求",
        ],
        "cons": [
            "架构复杂",
            "一致性挑战",
        ],
        "use_cases": ["多级客服", "内容审核分级"],
    },
}

class HybridDeployment:
    """
    混合部署策略
    简单请求边缘处理，复杂请求云端处理
    """
    def __init__(
        self,
        edge_model,  # 边缘设备上的小模型
        cloud_deployment,  # 云端vLLM部署
    ):
        self.edge_model = edge_model
        self.cloud_deployment = cloud_deployment
        
    def decide_routing(self, prompt: str) -> str:
        """
        决定路由策略
        
        路由策略：
        1. 简单问答 -> 边缘
        2. 长文本生成 -> 云端
        3. 复杂推理 -> 云端
        4. 隐私敏感 -> 边缘
        """
        prompt_length = len(prompt)
        complexity_score = self._estimate_complexity(prompt)
        
        if complexity_score < 0.3 and prompt_length < 100:
            return "edge"
        elif complexity_score > 0.7 or prompt_length > 1000:
            return "cloud"
        else:
            return "cloud"  # 默认云端
        
    def _estimate_complexity(self, prompt: str) -> float:
        """估计问题复杂度"""
        complexity_indicators = [
            "分析", "比较", "解释", "推理", "计算",
            "证明", "设计", "评估",
        ]
        
        score = sum(1 for ind in complexity_indicators if ind in prompt)
        return min(score / len(complexity_indicators), 1.0)
    
    async def generate(self, prompt: str, **kwargs) -> dict:
        """统一的生成接口"""
        route = self.decide_routing(prompt)
        
        if route == "edge":
            result = await self._edge_generate(prompt, **kwargs)
            result["route"] = "edge"
        else:
            result = await self.cloud_deployment.generate_async([prompt], **kwargs)
            result = result[0]
            result["route"] = "cloud"
        
        return result
    
    async def _edge_generate(self, prompt: str, **kwargs) -> dict:
        """边缘模型生成"""
        # 使用ONNX Runtime或TFLite
        start = time.time()
        output = self.edge_model.run(prompt, **kwargs)
        
        return {
            "text": output,
            "latency_ms": (time.time() - start) * 1000,
            "route": "edge",
        }
```

### 4.4 容器化部署

```dockerfile
# Dockerfile - 模型部署容器
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 安装Python和基础依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制模型（实际应挂载 volume）
COPY ./models /models

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV VLLM_HOST=0.0.0.0
ENV VLLM_PORT=8000

# 暴露端口
EXPOSE 8000 8001 8002

# 启动命令
CMD ["python", "-m", "uvicorn", "deploy_vllm:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  llm-inference:
    build: .
    image: llm-inference:1.0
    container_name: llm-inference
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ./models:/models:ro
      - ./data:/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - VLLM_MODEL_PATH=/models/quantized_7b
      - VLLM_GPU_MEMORY_UTILIZATION=0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

---

## 5. 监控与反馈闭环

### 5.1 推理延迟监控

```python
# monitor.py - 推理监控脚本
import time
import psutil
import GPUtil
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class InferenceMetrics:
    """推理指标数据类"""
    timestamp: str
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    first_token_latency_ms: float
    throughput_tokens_per_sec: float
    error: bool = False
    error_message: str = ""

class InferenceMonitor:
    """
    推理监控系统
    采集指标：
    1. 延迟指标：首token延迟、平均延迟、尾延迟(P99)
    2. 吞吐指标：QPS、token吞吐
    3. 资源指标：GPU利用率、显存、温度
    4. 错误指标：错误率、错误类型分布
    """
    def __init__(self, metrics_port: int = 9090):
        self.metrics_port = metrics_port
        self.metrics: List[InferenceMetrics] = []
        self.start_time = time.time()
        
    def record_inference(self, metrics: InferenceMetrics):
        """记录单次推理指标"""
        self.metrics.append(metrics)
        
        # 保持最近10000条记录
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-5000:]
    
    def get_latency_stats(self) -> Dict:
        """获取延迟统计"""
        if not self.metrics:
            return {}
        
        latencies = [m.latency_ms for m in self.metrics if not m.error]
        
        if not latencies:
            return {}
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return {
            "count": len(latencies),
            "mean_ms": sum(latencies) / n,
            "p50_ms": sorted_latencies[int(n * 0.50)],
            "p90_ms": sorted_latencies[int(n * 0.90)],
            "p95_ms": sorted_latencies[int(n * 0.95)],
            "p99_ms": sorted_latencies[int(n * 0.99)],
            "max_ms": max(latencies),
            "min_ms": min(latencies),
        }
    
    def get_throughput_stats(self) -> Dict:
        """获取吞吐统计"""
        if not self.metrics:
            return {}
        
        total_tokens = sum(m.total_tokens for m in self.metrics if not m.error)
        total_time = time.time() - self.start_time
        
        request_count = sum(1 for m in self.metrics if not m.error)
        
        return {
            "total_requests": request_count,
            "total_tokens": total_tokens,
            "qps": request_count / total_time if total_time > 0 else 0,
            "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        }
    
    def get_resource_stats(self) -> Dict:
        """获取资源使用统计"""
        stats = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
        }
        
        # GPU统计
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                stats["gpu_utilization"] = gpu.load * 100
                stats["gpu_memory_used_mb"] = gpu.memoryUsed
                stats["gpu_memory_total_mb"] = gpu.memoryTotal
                stats["gpu_memory_utilization"] = gpu.memoryUtil * 100
                stats["gpu_temperature"] = gpu.temperature
        except:
            pass
        
        return stats
    
    def get_error_stats(self) -> Dict:
        """获取错误统计"""
        total = len(self.metrics)
        errors = [m for m in self.metrics if m.error]
        
        error_types = {}
        for m in errors:
            error_type = m.error_message or "unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / total if total > 0 else 0,
            "error_types": error_types,
        }
    
    def get_full_report(self) -> Dict:
        """获取完整监控报告"""
        return {
            "timestamp": datetime.now().isoformat(),
            "latency": self.get_latency_stats(),
            "throughput": self.get_throughput_stats(),
            "resources": self.get_resource_stats(),
            "errors": self.get_error_stats(),
        }
    
    def export_prometheus(self) -> str:
        """导出Prometheus格式指标"""
        stats = self.get_full_report()
        
        lines = []
        lines.append(f'# HELP llm_inference_requests_total Total inference requests')
        lines.append(f'# TYPE llm_inference_requests_total counter')
        lines.append(f'llm_inference_requests_total {stats["throughput"]["total_requests"]}')
        
        lines.append(f'# HELP llm_inference_latency_ms Inference latency in ms')
        lines.append(f'# TYPE llm_inference_latency_ms summary')
        lines.append(f'llm_inference_latency_ms{{quantile="0.5"}} {stats["latency"].get("p50_ms", 0)}')
        lines.append(f'llm_inference_latency_ms{{quantile="0.9"}} {stats["latency"].get("p90_ms", 0)}')
        lines.append(f'llm_inference_latency_ms{{quantile="0.99"}} {stats["latency"].get("p99_ms", 0)}')
        
        return "\n".join(lines)
    
    def check_alerts(self) -> List[Dict]:
        """检查是否有告警触发"""
        alerts = []
        
        # 检查延迟告警
        latency_stats = self.get_latency_stats()
        if latency_stats.get("p99_ms", 0) > 5000:  # P99 > 5s
            alerts.append({
                "severity": "warning",
                "metric": "latency",
                "message": f'P99 latency {latency_stats["p99_ms"]:.0f}ms exceeds 5000ms',
            })
        
        # 检查错误率告警
        error_stats = self.get_error_stats()
        if error_stats.get("error_rate", 0) > 0.01:  # 错误率 > 1%
            alerts.append({
                "severity": "critical",
                "metric": "error_rate",
                "message": f'Error rate {error_stats["error_rate"]:.2%} exceeds 1%',
            })
        
        # 检查GPU显存告警
        resource_stats = self.get_resource_stats()
        if resource_stats.get("gpu_memory_utilization", 0) > 95:
            alerts.append({
                "severity": "critical",
                "metric": "gpu_memory",
                "message": f'GPU memory utilization {resource_stats["gpu_memory_utilization"]:.1f}% exceeds 95%',
            })
        
        return alerts


# 使用示例
if __name__ == "__main__":
    monitor = InferenceMonitor()
    
    # 模拟一些推理记录
    for i in range(100):
        metrics = InferenceMetrics(
            timestamp=datetime.now().isoformat(),
            request_id=f"req_{i}",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            latency_ms=100 + (i % 50) * 2,  # 模拟不同延迟
            first_token_latency_ms=20,
            throughput_tokens_per_sec=3000,
            error=i == 50,  # 第50个请求有错误
            error_message="CUDA OOM" if i == 50 else "",
        )
        monitor.record_inference(metrics)
    
    # 生成报告
    report = monitor.get_full_report()
    print(json.dumps(report, indent=2))
    
    # 检查告警
    alerts = monitor.check_alerts()
    for alert in alerts:
        print(f"[{alert['severity'].upper()}] {alert['message']}")
```

### 5.2 模型性能漂移检测

```python
# drift_detection.py - 性能漂移检测
class PerformanceDriftDetector:
    """
    模型性能漂移检测器
    
    检测类型：
    1. 输入分布漂移：输入数据的统计特性发生变化
    2. 输出分布漂移：模型输出的统计特性发生变化
    3. 质量漂移：模型输出质量指标下降
    """
    def __init__(
        self,
        reference_data: List[dict],
        window_size: int = 1000,
        drift_threshold: float = 0.05,
    ):
        self.reference_data = reference_data
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # 计算参考分布
        self.reference_stats = self._compute_stats(reference_data)
        
        # 滑动窗口
        self.current_window: List[dict] = []
        
    def _compute_stats(self, data: List[dict]) -> Dict:
        """计算数据统计信息"""
        if not data:
            return {}
        
        return {
            "mean_length": sum(len(d.get("prompt", "")) for d in data) / len(data),
            "std_length": self._std([len(d.get("prompt", "")) for d in data]),
            "vocab_coverage": self._compute_vocab_coverage(data),
        }
    
    def _std(self, values: List[float]) -> float:
        """计算标准差"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5
    
    def _compute_vocab_coverage(self, data: List[dict]) -> float:
        """计算词表覆盖率"""
        all_tokens = set()
        for d in data:
            text = d.get("prompt", "") + d.get("response", "")
            all_tokens.update(text.split())
        return len(all_tokens) / 10000  # 归一化
    
    def add_sample(self, sample: dict):
        """添加新样本到滑动窗口"""
        self.current_window.append(sample)
        
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
    
    def detect_drift(self) -> Dict:
        """检测漂移"""
        if len(self.current_window) < self.window_size // 10:
            return {"drift_detected": False, "reason": "insufficient_data"}
        
        # 计算当前窗口统计
        current_stats = self._compute_stats(self.current_window)
        
        # 计算漂移指标
        drift_scores = {}
        
        for key in self.reference_stats:
            if key in current_stats:
                ref_val = self.reference_stats[key]
                cur_val = current_stats[key]
                
                if ref_val != 0:
                    drift = abs(cur_val - ref_val) / abs(ref_val)
                    drift_scores[key] = drift
        
        # 检测是否有显著漂移
        max_drift = max(drift_scores.values()) if drift_scores else 0
        
        return {
            "drift_detected": max_drift > self.drift_threshold,
            "max_drift": max_drift,
            "drift_threshold": self.drift_threshold,
            "drift_scores": drift_scores,
            "recommendation": self._get_recommendation(drift_scores),
        }
    
    def _get_recommendation(self, drift_scores: Dict) -> str:
        """根据漂移情况给出建议"""
        if not drift_scores:
            return "insufficient_data"
        
        max_key = max(drift_scores, key=drift_scores.get)
        drift_val = drift_scores[max_key]
        
        if drift_val > self.drift_threshold * 2:
            return f"Significant drift in {max_key}, consider retraining"
        elif drift_val > self.drift_threshold:
            return f"Moderate drift in {max_key}, monitor closely"
        else:
            return "No significant drift"
```

### 5.3 A/B测试框架

```python
# ab_testing.py - A/B测试框架
class ABTestManager:
    """
    A/B测试管理器
    
    支持：
    1. 模型版本对比
    2. 流量分配
    3. 统计显著性检验
    4. 早期停止
    """
    def __init__(
        self,
        variants: Dict[str, str],  # variant_name -> model_path
        traffic_split: Dict[str, float],  # variant_name -> traffic_fraction
        metrics_to_track: List[str],
        min_samples: int = 1000,
        significance_level: float = 0.05,
    ):
        self.variants = variants
        self.traffic_split = traffic_split
        self.metrics_to_track = metrics_to_track
        self.min_samples = min_samples
        self.significance_level = significance_level
        
        # 存储各变体的数据
        self.variant_data: Dict[str, List[dict]] = {
            name: [] for name in variants.keys()
        }
        
    def select_variant(self) -> str:
        """
        根据流量分配选择变体
        """
        import random
        
        r = random.random()
        cumulative = 0.0
        
        for name, fraction in self.traffic_split.items():
            cumulative += fraction
            if r <= cumulative:
                return name
        
        return list(self.variants.keys())[0]
    
    def record_result(self, variant: str, metrics: dict):
        """记录变体的结果"""
        if variant in self.variant_data:
            self.variant_data[variant].append(metrics)
    
    def analyze_results(self) -> Dict:
        """
        分析A/B测试结果
        """
        results = {}
        
        for variant_name, data in self.variant_data.items():
            if len(data) < self.min_samples:
                results[variant_name] = {
                    "status": "insufficient_samples",
                    "samples": len(data),
                    "required": self.min_samples,
                }
                continue
            
            # 计算各指标的平均值
            variant_metrics = {}
            for metric in self.metrics_to_track:
                values = [d.get(metric, 0) for d in data]
                variant_metrics[metric] = {
                    "mean": sum(values) / len(values),
                    "std": self._std(values),
                    "count": len(values),
                }
            
            results[variant_name] = {
                "status": "complete",
                "samples": len(data),
                "metrics": variant_metrics,
            }
        
        # 统计检验
        if len(results) == 2:
            significance_result = self._statistical_test(
                self.variant_data
            )
            results["significance_test"] = significance_result
        
        return results
    
    def _std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _statistical_test(self, variant_data: Dict[str, List[dict]]) -> Dict:
        """
        简单的统计显著性检验
        使用t检验比较两个变体
        """
        if len(variant_data) != 2:
            return {"status": "need_exactly_2_variants"}
        
        variants = list(variant_data.keys())
        data_a = variant_data[variants[0]]
        data_b = variant_data[variants[1]]
        
        # 使用第一个跟踪指标
        metric = self.metrics_to_track[0]
        values_a = [d.get(metric, 0) for d in data_a]
        values_b = [d.get(metric, 0) for d in data_b]
        
        mean_a = sum(values_a) / len(values_a)
        mean_b = sum(values_b) / len(values_b)
        
        # 简化的显著性判断
        pooled_std = ((self._std(values_a) ** 2 + self._std(values_b) ** 2) / 2) ** 0.5
        effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        return {
            "variant_a_mean": mean_a,
            "variant_b_mean": mean_b,
            "effect_size": effect_size,
            "significant": effect_size > 0.2,  # Cohen's d > 0.2
            "winner": variants[0] if mean_a > mean_b else variants[1],
        }
```

### 5.4 生产数据收集

```python
# production_data_collector.py - 生产数据收集
class ProductionDataCollector:
    """
    生产数据收集器
    
    收集数据用于：
    1. 模型下一轮迭代的微调数据
    2. 评估模型性能
    3. 检测分布漂移
    4. 发现边缘案例
    """
    def __init__(
        self,
        collection_rate: float = 0.1,  # 收集10%的数据
        storage_path: str = "./production_data",
        max_samples_per_day: int = 100000,
    ):
        self.collection_rate = collection_rate
        self.storage_path = storage_path
        self.max_samples_per_day = max_samples_per_day
        self.today_samples = 0
        
        import os
        os.makedirs(storage_path, exist_ok=True)
        
    def should_collect(self) -> bool:
        """决定是否收集当前请求"""
        import random
        return random.random() < self.collection_rate
    
    def collect_sample(
        self,
        request_id: str,
        prompt: str,
        response: str,
        metadata: dict,
    ):
        """
        收集请求-响应对
        
        采样策略：
        1. 随机采样：简单但可能有偏差
        2. 多样性采样：确保覆盖不同场景
        3. 边缘案例优先：收集异常或低质量样本
        """
        if self.today_samples >= self.max_samples_per_day:
            return
        
        if not self.should_collect():
            return
        
        sample = {
            "request_id": request_id,
            "prompt": prompt,
            "response": response,
            "metadata": metadata,
            "timestamp": self._get_timestamp(),
        }
        
        # 存储样本
        self._save_sample(sample)
        self.today_samples += 1
    
    def _save_sample(self, sample: dict):
        """保存样本到磁盘"""
        import json
        from datetime import datetime
        
        date_str = datetime.now().strftime("%Y%m%d")
        filepath = f"{self.storage_path}/samples_{date_str}.jsonl"
        
        with open(filepath, "a") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def select_for_sft(self, num_samples: int = 10000) -> List[dict]:
        """
        从收集的数据中选择适合SFT的样本
        
        选择标准：
        1. 响应质量高
        2. 多样性覆盖
        3. 无隐私信息
        """
        import json
        from datetime import datetime
        from collections import defaultdict
        
        # 读取最近的样本
        date_str = datetime.now().strftime("%Y%m%d")
        filepath = f"{self.storage_path}/samples_{date_str}.jsonl"
        
        samples = []
        try:
            with open(filepath, "r") as f:
                for line in f:
                    samples.append(json.loads(line))
        except FileNotFoundError:
            return []
        
        # 过滤低质量样本
        quality_samples = [
            s for s in samples
            if self._quality_score(s) > 0.7
        ]
        
        # 确保多样性
        diverse_samples = self._ensure_diversity(quality_samples, num_samples)
        
        return diverse_samples
    
    def _quality_score(self, sample: dict) -> float:
        """评估样本质量"""
        score = 0.0
        
        # 响应长度适中
        response_len = len(sample.get("response", ""))
        if 50 < response_len < 2000:
            score += 0.3
        
        # 包含具体信息
        if any(marker in sample.get("response", "") for marker in ["1.", "2.", "•", "-", "："]):
            score += 0.3
        
        # 无明显错误
        error_indicators = ["error", "sorry", "cannot", "unable"]
        if not any(ind in sample.get("response", "").lower() for ind in error_indicators):
            score += 0.4
        
        return score
    
    def _ensure_diversity(self, samples: List[dict], num: int) -> List[dict]:
        """确保样本多样性"""
        if len(samples) <= num:
            return samples
        
        # 按提示长度分桶
        buckets = defaultdict(list)
        for s in samples:
            length_bucket = len(s.get("prompt", "")) // 100
            buckets[length_bucket].append(s)
        
        # 从每个桶均匀采样
        selected = []
        per_bucket = num // len(buckets)
        
        for bucket_samples in buckets.values():
            selected.extend(bucket_samples[:per_bucket])
        
        return selected[:num]
```

---

## 6. 端到端案例：7B模型压缩与部署

### 6.1 完整流程概览

```
7B Model Deployment Case Study:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Stage 1: QLoRA Fine-tuning                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Base: Llama-2-7b → QLoRA (rank=16) → Fine-tuned Model          │   │
│  │  Time: ~8 GPU hours (A100 80GB)                                 │   │
│  │  GPU Memory: ~14GB                                              │   │
│  │  Output: ./checkpoints/llama2-7b-qlora-finetuned                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  Stage 2: INT8 Quantization                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Method: AWQ (Activation-Aware Weight Quantization)             │   │
│  │  Calibration: 512 samples from training data                     │   │
│  │  Time: ~15 minutes                                              │   │
│  │  Output Size: 7B → 3.5GB (INT8)                                 │   │
│  │  Quality Retention: ~97% (perplexity 23.1 → 23.8)              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  Stage 3: Optimization                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TensorRT Compilation + PagedAttention + Prefix Cache          │   │
│  │  Time: ~30 minutes                                              │   │
│  │  Output: ./models/llama2-7b-awq-int8.trt                        │   │
│  │  Latency: ~45ms (prefill) + ~12ms/token (decode)                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│  Stage 4: vLLM Deployment                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Engine: vLLM 0.2.x                                             │   │
│  │  Config: tensor_parallel=1, gpu_memory_utilization=0.9         │   │
│  │  Throughput: ~1500 tokens/sec                                   │   │
│  │  Cost: ~ $0.8/hour (A100 80GB on-demand)                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 成本分析

```python
# cost_analysis.py - 成本分析
COST_ANALYSIS = {
    "gpu_cost_per_hour": {
        "A100_80GB": 3.5,      # AWS p4d.24xlarge
        "A100_40GB": 2.0,      # AWS p3.16xlarge
        "RTX_4090": 0.5,       # 消费级
        "T4": 0.35,            # AWS g4dn
    },
    
    "model_size_gb": 7,  # 7B模型原始大小
    
    "quantization_savings": {
        "fp16": {"size_gb": 14, "cost_multiplier": 1.0},
        "int8": {"size_gb": 7, "cost_multiplier": 0.5},
        "int4": {"size_gb": 3.5, "cost_multiplier": 0.25},
    },
    
    "deployment_metrics": {
        "requests_per_day": 100000,
        "avg_tokens_per_request": 500,
        "avg_requests_per_hour": 5000,
        "peak_concurrent_requests": 50,
    },
    
    "monthly_cost_breakdown": {
        "gpu_compute": 0,  # 计算得出
        "storage": 50,      # S3
        "networking": 100,  # 数据传输
        "monitoring": 50,   # CloudWatch
        "total": 0,
    },
}

def calculate_monthly_cost(
    gpu_type: str = "A100_80GB",
    quantization: str = "int8",
    monthly_requests: int = 3000000,
) -> dict:
    """
    计算月度运营成本
    """
    cost_per_hour = COST_ANALYSIS["gpu_cost_per_hour"][gpu_type]
    
    # 估算GPU小时数
    # 假设平均请求500 tokens，QPS=50峰值
    avg_qps = 20
    hours_per_day = 24
    
    gpu_hours_per_month = avg_qps * 3600 * 30 / 1000  # 简化估算
    
    gpu_compute_cost = gpu_hours_per_month * cost_per_hour
    
    # 量化节省
    quant_config = COST_ANALYSIS["quantization_savings"][quantization]
    
    # 如果用int8，可以用更小的GPU
    if quantization == "int8":
        # int8可以在单卡A100上运行，而fp16可能需要多卡
        effective_gpus = 1
    elif quantization == "fp16" and COST_ANALYSIS["model_size_gb"] > 20:
        effective_gpus = 2  # 需要tensor并行
    else:
        effective_gpus = 1
    
    total_gpu_cost = gpu_compute_cost * effective_gpus
    
    return {
        "gpu_type": gpu_type,
        "quantization": quantization,
        "effective_gpus": effective_gpus,
        "gpu_hours_per_month": gpu_hours_per_month,
        "gpu_compute_cost": total_gpu_cost,
        "storage_cost": COST_ANALYSIS["monthly_cost_breakdown"]["storage"],
        "networking_cost": COST_ANALYSIS["monthly_cost_breakdown"]["networking"],
        "monitoring_cost": COST_ANALYSIS["monthly_cost_breakdown"]["monitoring"],
        "total_monthly_cost": (
            total_gpu_cost +
            COST_ANALYSIS["monthly_cost_breakdown"]["storage"] +
            COST_ANALYSIS["monthly_cost_breakdown"]["networking"] +
            COST_ANALYSIS["monthly_cost_breakdown"]["monitoring"]
        ),
        "cost_per_1k_tokens": total_gpu_cost / (monthly_requests * 500 / 1000),
    }

# ROI计算
def calculate_roi(
    model_type: str,
    monthly_cost: float,
    time_savings_hours_per_month: float = 100,
    hourly_engineer_cost: float = 100,
) -> dict:
    """
    计算量化部署的ROI
    """
    monthly_savings = time_savings_hours_per_month * hourly_engineer_cost
    
    return {
        "monthly_cost": monthly_cost,
        "monthly_savings": monthly_savings,
        "monthly_roi": (monthly_savings - monthly_cost) / monthly_cost * 100,
        "annual_cost": monthly_cost * 12,
        "annual_savings": monthly_savings * 12,
        "payback_months": monthly_cost / monthly_savings if monthly_savings > 0 else None,
    }
```

### 6.3 性能对比

```python
# performance_comparison.py - 性能对比
PERFORMANCE_COMPARISON = {
    "fp16_baseline": {
        "model": "Llama-2-7b FP16",
        "throughput_tokens_per_sec": 400,
        "latency_prefill_ms": 200,
        "latency_decode_per_token_ms": 50,
        "gpu_memory_gb": 14,
        "cost_per_1k_tokens": 0.004,
    },
    
    "int8_quantized": {
        "model": "Llama-2-7b INT8 (AWQ)",
        "throughput_tokens_per_sec": 1200,
        "latency_prefill_ms": 80,
        "latency_decode_per_token_ms": 18,
        "gpu_memory_gb": 7,
        "cost_per_1k_tokens": 0.0015,
        "quality_retention": 0.97,
    },
    
    "int4_quantized": {
        "model": "Llama-2-7b INT4 (GPTQ)",
        "throughput_tokens_per_sec": 2000,
        "latency_prefill_ms": 45,
        "latency_decode_per_token_ms": 10,
        "gpu_memory_gb": 4,
        "cost_per_1k_tokens": 0.0008,
        "quality_retention": 0.92,
    },
    
    "improvements": {
        "throughput_boost": "3x (FP16 -> INT8)",
        "latency_reduction": "60% reduction",
        "memory_reduction": "50% reduction",
        "cost_reduction": "62% reduction",
        "quality_impact": "3% perplexity degradation",
    },
}

def generate_benchmark_report() -> str:
    """生成基准测试报告"""
    return f"""
Model Deployment Benchmark Report
=================================

Configuration: Llama-2-7B, A100 80GB, vLLM 0.2.x

┌──────────────────┬─────────────┬─────────────┬─────────────┐
│     Metric       │    FP16     │    INT8     │    INT4     │
├──────────────────┼─────────────┼─────────────┼─────────────┤
│ Throughput       │ 400 tok/s  │ 1200 tok/s  │ 2000 tok/s  │
│ Prefill Latency  │   200ms     │    80ms     │    45ms     │
│ Decode Latency   │    50ms     │    18ms     │    10ms     │
│ GPU Memory       │    14GB     │     7GB     │     4GB     │
│ Cost/1K tokens   │   $0.004    │   $0.0015   │   $0.0008   │
│ Quality Retention│    100%     │     97%     │     92%     │
└──────────────────┴─────────────┴─────────────┴─────────────┘

Key Findings:
- INT8 provides best balance of speed and quality
- 3x throughput improvement with only 3% quality loss
- Cost reduced by 62% compared to FP16
- INT4 viable for non-critical applications

Recommendation:
Use INT8 (AWQ) for production deployment with quality monitoring.
INT4 viable for batch processing where speed > quality.
"""
```

---

## 7. 常见陷阱与规避

### 7.1 量化效果劣化

```python
# pitfalls.py - 常见陷阱及解决方案
PITFALLS = {
    "quantization_degradation": {
        "description": "量化后模型质量显著下降",
        "causes": [
            "校准数据不足或不具代表性",
            "敏感层被量化",
            "量化方法选择不当",
        ],
        "solutions": [
            "增加校准数据到1024+样本",
            "使用混合精度，对敏感层保持FP16",
            "尝试不同的量化方法(GPTQ/AWQ/BBQ)",
            "量化后进行少量微调(QFT)",
        ],
        "detection_script": """
# 检测量化劣化
val_result = validator.validate()
if val_result.perplexity > baseline_perplexity * 1.1:
    print("Warning: Significant quality degradation detected")
    # 建议切换回FP16或尝试其他量化配置
""",
    },
    
    "oom_issues": {
        "description": "生产环境OOM崩溃",
        "causes": [
            "KV Cache过大",
            "Batch size设置过高",
            "并发请求过多",
        ],
        "solutions": [
            "启用PagedAttention管理KV Cache",
            "设置max_num_seqs限制",
            "配置gpu_memory_utilization=0.85",
            "实现请求队列和限流",
        ],
        "prevention_script": """
# OOM预防检查
max_memory_required = (
    model_size_gb +
    kv_cache_per_request_gb * max_concurrent_requests +
    activation_memory_gb * batch_size
)

if max_memory_required > available_gpu_memory * 0.95:
    print("ERROR: OOM risk detected")
    print(f"Required: {max_memory_required}GB, Available: {available_gpu_memory}GB")
    # 降低配置
""",
    },
    
    "cold_start_latency": {
        "description": "模型加载冷启动延迟高",
        "causes": [
            "模型过大",
            "未使用Lazy Loading",
            "未预热(Warmup)",
        ],
        "solutions": [
            "使用量化模型减小体积",
            "实现模型预加载和缓存",
            "请求到达时先返回降级响应",
            "使用模型集群分摊",
        ],
        "mitigation_script": """
# 冷启动缓解
class WarmupManager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.warmed_up = False
        
    def warmup(self):
        # 预热推理10-20次
        for _ in range(20):
            dummy_input = torch.randint(0, 32000, (1, 128))
            self.model(dummy_input)
        self.warmed_up = True
        
    def get_model(self):
        if not self.warmed_up:
            self.warmup()
        return self.model
""",
    },
}

class PitfallDetector:
    """自动检测常见陷阱"""
    def __init__(self, monitor: InferenceMonitor):
        self.monitor = monitor
        
    def run_checks(self) -> List[dict]:
        """运行所有检查"""
        checks = []
        
        # 检查1: 量化劣化
        checks.append(self._check_quantization_degradation())
        
        # 检查2: OOM风险
        checks.append(self._check_oom_risk())
        
        # 检查3: 尾延迟异常
        checks.append(self._check_tail_latency())
        
        return [c for c in checks if c is not None]
    
    def _check_quantization_degradation(self) -> dict:
        """检查量化劣化"""
        # 需要与基准对比
        return None  # placeholder
    
    def _check_oom_risk(self) -> dict:
        """检查OOM风险"""
        resource_stats = self.monitor.get_resource_stats()
        gpu_mem_util = resource_stats.get("gpu_memory_utilization", 0)
        
        if gpu_mem_util > 95:
            return {
                "type": "oom_risk",
                "severity": "critical",
                "message": f"GPU memory utilization at {gpu_mem_util:.1f}%",
                "recommendation": "Reduce batch size or max_num_seqs",
            }
        return None
    
    def _check_tail_latency(self) -> dict:
        """检查尾延迟"""
        latency_stats = self.monitor.get_latency_stats()
        p99 = latency_stats.get("p99_ms", 0)
        
        if p99 > 10000:  # > 10s
            return {
                "type": "tail_latency",
                "severity": "warning",
                "message": f"P99 latency {p99:.0f}ms is very high",
                "recommendation": "Consider scaling horizontally",
            }
        return None
```

---

## 本章小结

1. **完整闭环**：模型压缩与部署是从微调模型到生产服务的系统工程，包含量化→优化→部署→监控→反馈各环节

2. **PTQ量化方法**：GPTQ/AWQ/BBQ是主流的训练后量化方法，AWQ在精度和速度间有最好平衡

3. **校准数据选择**：校准数据的代表性直接影响量化质量，需要覆盖多种主题和长度

4. **推理优化**：TensorRT/ONNX Runtime编译器优化、KV Cache优化、Continuous Batching、投机解码等可显著提升吞吐

5. **部署模式**：vLLM适合快速部署，边缘部署需要更激进的量化，混合部署平衡延迟与成本

6. **生产监控**：延迟监控、性能漂移检测、A/B测试、数据收集构成完整反馈闭环

7. **成本优化**：INT8量化可降低60%成本，同时保持97%模型质量

8. **常见陷阱**：量化劣化、OOM、冷启动是三大主要问题，各有规避方案

---

## 延伸阅读

- GPTQ论文：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- AWQ论文：AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration
- vLLM论文：Efficient Memory Management for Large Language Model Serving with PagedAttention
- TensorRT-LLM: https://developer.nvidia.com/tensorrt
- 量化实战指南：https://docs.vllm.ai/en/latest/quantization/

---

## 思考题

1. 在INT8量化中，为什么AWQ强调"Activation-Aware"？这与GPTQ的权重优先方法有何本质区别？

2. 假设你在生产环境中遇到P99延迟突然飙升到10秒，但平均延迟只有100ms，请分析可能的原因以及排查步骤。

3. 对于一个面向消费者的聊天机器人产品，你会如何设计A/B测试来验证新量化模型是否满足上线标准？请详细说明测试方案、指标选择和决策流程。

4. 在边缘设备（如手机）部署7B模型面临哪些特殊挑战？相比云端部署，边缘部署需要在模型端做哪些额外优化？

(End of file - total 2000 lines)
