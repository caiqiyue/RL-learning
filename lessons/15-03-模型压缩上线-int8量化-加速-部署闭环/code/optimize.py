"""
Model Optimization using TensorRT and ONNX Runtime

Supports:
- TensorRT engine building
- ONNX export and optimization
- Performance benchmarking
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import numpy as np


@dataclass
class OptimizationConfig:
    """Optimization configuration"""

    engine: str = "tensorrt"  # tensorrt, onnxruntime, pytorch
    precision: str = "fp16"  # fp32, fp16, int8, fp8
    workspace_size_gb: int = 4
    max_batch_size: int = 32
    max_seq_length: int = 4096
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_graph_optimization: bool = True
    enable_profiling: bool = False


class TensorRTOptimizer:
    """
    TensorRT optimization for LLM models.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def export_to_onnx(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, int],
    ) -> str:
        """
        Export model to ONNX format first.
        """
        print(f"Exporting model to ONNX...")

        model.eval()

        dummy_input = torch.randint(
            0, 32000, input_shape, dtype=torch.long, device="cuda"
        )

        onnx_path = os.path.join(output_path, "model.onnx")

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size", 1: "seq_len"},
            },
            verbose=False,
        )

        print(f"ONNX model saved to {onnx_path}")
        return onnx_path

    def build_trt_engine(
        self,
        onnx_path: str,
        output_path: str,
    ) -> str:
        """
        Build TensorRT engine from ONNX model.
        """
        try:
            import tensorrt as trt
        except ImportError:
            print("TensorRT not installed. Using fallback ONNX optimization.")
            return self._optimize_onnx_fallback(onnx_path, output_path)

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
            trt.MemoryPoolType.WORKSPACE, self.config.workspace_size_gb * (1024**3)
        )

        if self.config.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.config.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)

        print(f"Building TensorRT engine...")
        start_time = time.time()

        engine = builder.build_serialized_network(network, config)

        if engine is None:
            raise RuntimeError("TensorRT engine build failed")

        build_time = time.time() - start_time
        print(f"Engine built in {build_time:.1f}s")

        engine_path = os.path.join(output_path, "model.trt")

        with open(engine_path, "wb") as f:
            f.write(engine)

        print(f"TensorRT engine saved to {engine_path}")

        return engine_path

    def _optimize_onnx_fallback(
        self,
        onnx_path: str,
        output_path: str,
    ) -> str:
        """
        Fallback ONNX optimization when TensorRT is not available.
        """
        from onnxruntime.transformers import optimizer

        optimized_path = os.path.join(output_path, "model_optimized.onnx")

        optimized_model = optimizer.optimize_model(
            onnx_path,
            num_heads=32,
            hidden_size=4096,
            optimization_level=99,
        )

        optimized_model.save_model_to_file(optimized_path)

        return optimized_path

    def optimize(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, int] = (1, 512),
    ) -> Dict:
        """
        Full optimization pipeline: ONNX export + TensorRT build.
        """
        print(f"\n{'=' * 60}")
        print(f"Optimization Pipeline: {self.config.engine.upper()}")
        print(f"{'=' * 60}")

        os.makedirs(output_path, exist_ok=True)

        start_time = time.time()

        onnx_path = self.export_to_onnx(model, output_path, input_shape)

        if self.config.engine == "tensorrt":
            engine_path = self.build_trt_engine(onnx_path, output_path)
        else:
            engine_path = self._optimize_onnx_fallback(onnx_path, output_path)

        total_time = time.time() - start_time

        engine_size_gb = os.path.getsize(engine_path) / (1024**3)

        return {
            "engine_type": self.config.engine,
            "precision": self.config.precision,
            "engine_path": engine_path,
            "engine_size_gb": engine_size_gb,
            "optimization_time_minutes": total_time / 60,
        }


class ONNXRuntimeOptimizer:
    """
    ONNX Runtime optimization.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def optimize(
        self,
        onnx_path: str,
        output_path: str,
    ) -> str:
        """
        Optimize ONNX model using ONNX Runtime.
        """
        from onnxruntime.transformers import optimizer

        print(f"Optimizing ONNX model with ONNX Runtime...")

        optimized_path = os.path.join(output_path, "model_ort_optimized.onnx")

        optimized_model = optimizer.optimize_model(
            onnx_path,
            num_heads=32,
            hidden_size=4096,
            optimization_level=7,
            use_mask_index=True,
            use_node_names=True,
            io_binding=True,
        )

        optimized_model.save_model_to_file(optimized_path)

        print(f"Optimized model saved to {optimized_path}")

        return optimized_path

    def create_inference_session(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
    ):
        """
        Create ONNX Runtime inference session.
        """
        import onnxruntime as ort

        if providers is None:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        sess_options = ort.SessionOptions()

        if self.config.enable_graph_optimization:
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )

        return session


class PyTorchOptimizer:
    """
    PyTorch optimization with torch.compile and other techniques.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def optimize(self, model: nn.Module) -> nn.Module:
        """
        Optimize PyTorch model.
        """
        print(f"Optimizing PyTorch model...")

        if self.config.enable_graph_optimization:
            print("  Enabling torch.compile...")
            model = torch.compile(
                model,
                mode="reduce-overhead",
                fullgraph=False,
            )

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(0)

        print("  Model optimization complete")

        return model


class KVCacheOptimizer:
    """
    KV Cache optimization with PagedAttention.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.block_size = 16
        self.max_blocks = 1024

    def get_cache_config(self) -> Dict:
        """
        Get KV cache configuration for vLLM.
        """
        return {
            "enable": True,
            "block_size": self.block_size,
            "max_cache_blocks": self.max_blocks,
            "gpu_memory_utilization": 0.9,
            "num_attention_layers": 32,
        }


class SpeculativeDecodingOptimizer:
    """
    Speculative decoding optimization.
    """

    def __init__(
        self,
        draft_model: Optional[nn.Module] = None,
        num_speculative_tokens: int = 4,
        acceptance_threshold: float = 0.8,
    ):
        self.draft_model = draft_model
        self.num_speculative_tokens = num_speculative_tokens
        self.acceptance_threshold = acceptance_threshold

    def create_config(self) -> Dict:
        """Get speculative decoding configuration."""
        return {
            "enable": True,
            "num_speculative_tokens": self.num_speculative_tokens,
            "acceptance_threshold": self.acceptance_threshold,
            "prompt_lookup_num_tokens": self.num_speculative_tokens,
        }


class BatchOptimizer:
    """
    Batch optimization strategies.
    """

    def __init__(
        self,
        max_batch_size: int = 64,
        max_waiting_time_ms: int = 100,
        enable_continuous_batching: bool = True,
    ):
        self.max_batch_size = max_batch_size
        self.max_waiting_time_ms = max_waiting_time_ms
        self.enable_continuous_batching = enable_continuous_batching

    def get_dynamic_batching_config(self) -> Dict:
        """Get dynamic batching configuration for Triton."""
        return {
            "preferred_batch_sizes": [1, 2, 4, 8, 16, 32],
            "max_queue_delay_microseconds": self.max_waiting_time_ms * 1000,
        }

    def get_vllm_batching_config(self) -> Dict:
        """Get batching configuration for vLLM."""
        return {
            "max_num_seqs": self.max_batch_size,
            "max_num_batched_tokens": self.max_batch_size * 512,
            "enable_continuous_batching": self.enable_continuous_batching,
        }


def benchmark_model(
    model,
    input_ids: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict:
    """
    Benchmark model inference performance.
    """
    print(f"\nBenchmarking model ({num_runs} runs, {warmup_runs} warmup)...")

    model.eval()
    device = next(model.parameters()).device

    input_ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        for _ in range(num_runs):
            _ = model(input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_runs

    tokens_per_sec = input_ids.shape[1] / avg_time

    return {
        "total_time_sec": total_time,
        "avg_time_ms": avg_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "num_runs": num_runs,
        "input_length": input_ids.shape[1],
    }


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Model Optimization Pipeline")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--engine",
        type=str,
        default="tensorrt",
        choices=["tensorrt", "onnxruntime", "pytorch"],
    )
    parser.add_argument(
        "--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"]
    )

    args = parser.parse_args()

    config = OptimizationConfig(
        engine=args.engine,
        precision=args.precision,
    )

    if config.engine == "tensorrt":
        optimizer = TensorRTOptimizer(config)
    elif config.engine == "onnxruntime":
        optimizer = ONNXRuntimeOptimizer(config)
    else:
        optimizer = PyTorchOptimizer(config)

    print("Loading model...")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    result = optimizer.optimize(
        model=model,
        output_path=args.output_path,
        input_shape=(1, 512),
    )

    print(f"\nOptimization complete:")
    print(f"  Engine Type: {result['engine_type']}")
    print(f"  Precision: {result['precision']}")
    print(f"  Engine Path: {result['engine_path']}")
    print(f"  Size: {result['engine_size_gb']:.2f} GB")
    print(f"  Time: {result['optimization_time_minutes']:.1f} minutes")


if __name__ == "__main__":
    main()
