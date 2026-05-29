#!/usr/bin/env python3
"""
Benchmark Tool for Comparing Different Precision Levels

This script benchmarks model inference across different precision levels (FP16, INT8, INT4)
to compare latency, throughput, and memory usage.

Usage:
    python benchmark.py --model <model_path> --precision FP16 INT8

Requirements:
    - vLLM for FP16/INT8/INT4 benchmarking
    - Sufficient GPU memory for the model

Note on TensorRT:
    TensorRT requires complex CUDA setup that cannot be fully automated here.
    For TensorRT benchmarking, you would need to:
    1. Install CUDA Toolkit 11.8+
    2. Install TensorRT 8.6+ from NVIDIA developer site
    3. Export model to ONNX format
    4. Build TensorRT engine with INT8 quantization
    5. Run inference using TensorRT C++ API or Python bindings

    See comments in deploy_trt_guide.md for detailed TensorRT setup.
"""

import argparse
import time
import statistics
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import torch
    import numpy as np
except ImportError:
    print("Error: torch or numpy not installed")
    sys.exit(1)

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, FP16 benchmarks disabled")


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    precision: str
    batch_size: int
    num_tokens: int

    # Latency metrics (in milliseconds)
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float

    # Throughput metrics
    throughput_tokens_per_sec: float

    # Memory metrics (in GB)
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float

    # Quality metrics (if applicable)
    perplexity: Optional[float] = None


class PrecisionBenchmark:
    """
    Benchmark tool for comparing model performance across precision levels.

    Measures:
    - Latency: Time to first token and per-token generation
    - Throughput: Tokens generated per second
    - Memory: GPU memory usage
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.results: List[BenchmarkResult] = []

    def _get_gpu_memory(self) -> Tuple[float, float]:
        """Get current GPU memory usage (allocated, reserved) in GB."""
        if not torch.cuda.is_available():
            return 0.0, 0.0

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved

    def _reset_gpu_memory(self):
        """Reset GPU memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def benchmark_precision(
        self,
        precision: str,
        batch_size: int = 1,
        num_tokens: int = 100,
        num_warmup: int = 3,
        num_runs: int = 10,
        max_model_len: int = 2048,
    ) -> BenchmarkResult:
        """
        Benchmark a specific precision level.

        Args:
            precision: "FP16", "INT8", or "INT4"
            batch_size: Batch size for inference
            num_tokens: Number of tokens to generate per request
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs (after warmup)
            max_model_len: Maximum sequence length

        Returns:
            BenchmarkResult with all metrics
        """
        if precision not in ["FP16", "INT8", "INT4"]:
            raise ValueError(f"Unsupported precision: {precision}")

        # For now, FP16 and INT8 use vLLM (simulated INT8 via AWQ would need actual quantized model)
        # INT4 would require llama.cpp or actual INT4 quantized model
        if precision == "FP16":
            quantization = None
        elif precision == "INT8":
            quantization = "AWQ"  # Requires actual INT8 quantized model
        else:  # INT4
            quantization = "AWQ"  # Requires actual INT4 quantized model

        print(f"\n{'=' * 60}")
        print(f"Benchmarking {precision}")
        print(f"  Batch size: {batch_size}")
        print(f"  Tokens to generate: {num_tokens}")
        print(f"  Runs: {num_warmup} warmup + {num_runs} benchmark")
        print(f"{'=' * 60}")

        if not VLLM_AVAILABLE:
            print("Error: vLLM not available for benchmarking")
            return None

        self._reset_gpu_memory()

        # Initialize model
        print("Loading model...")
        start_load = time.time()

        try:
            llm = LLM(
                model=self.model_path,
                quantization=quantization,
                tensor_parallel_size=1,
                max_model_len=max_model_len,
                dtype="float16",
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Note: INT8/INT4 require actual quantized model weights")
            return None

        load_time = time.time() - start_load
        allocated_before, reserved_before = self._get_gpu_memory()
        print(f"Model loaded in {load_time:.2f}s")
        print(
            f"GPU memory: {allocated_before:.2f}GB allocated, {reserved_before:.2f}GB reserved"
        )

        # Prepare prompts
        prompts = [
            "Explain the theory of relativity in simple terms:",
            "Write a Python function to calculate fibonacci numbers:",
            "What are the main differences between SQL and NoSQL databases?",
        ]
        prompt_texts = prompts * (batch_size // len(prompts) + 1)
        prompt_texts = prompt_texts[:batch_size]

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=num_tokens,
        )

        # Warmup
        print(f"Warming up ({num_warmup} runs)...")
        for _ in range(num_warmup):
            _ = llm.generate(
                prompt_texts[: min(batch_size, len(prompt_texts))], sampling_params
            )

        # Benchmark runs
        print(f"Running benchmark ({num_runs} runs)...")
        latencies = []

        for i in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            outputs = llm.generate(
                prompt_texts[: min(batch_size, len(prompt_texts))], sampling_params
            )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)

            if (i + 1) % 5 == 0:
                print(f"  Run {i + 1}/{num_runs} completed")

        # Calculate metrics
        avg_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p90_latency = sorted(latencies)[int(len(latencies) * 0.9)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        min_latency = min(latencies)
        max_latency = max(latencies)

        # Calculate throughput
        total_tokens = num_runs * batch_size * num_tokens
        total_time_sec = sum(latencies) / 1000
        throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0

        # Memory after
        allocated_after, reserved_after = self._get_gpu_memory()

        result = BenchmarkResult(
            precision=precision,
            batch_size=batch_size,
            num_tokens=num_tokens,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p90_latency_ms=p90_latency,
            p99_latency_ms=p99_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            throughput_tokens_per_sec=throughput,
            gpu_memory_allocated_gb=allocated_after,
            gpu_memory_reserved_gb=reserved_after,
        )

        self.results.append(result)

        # Print summary
        print(f"\nResults for {precision}:")
        print(f"  Latency (avg): {avg_latency:.2f}ms")
        print(f"  Latency (P50): {p50_latency:.2f}ms")
        print(f"  Latency (P90): {p90_latency:.2f}ms")
        print(f"  Latency (P99): {p99_latency:.2f}ms")
        print(f"  Throughput: {throughput:.2f} tokens/sec")
        print(f"  GPU Memory: {allocated_after:.2f}GB allocated")

        return result

    def run_comparison(
        self,
        precisions: List[str] = ["FP16", "INT8"],
        batch_sizes: List[int] = [1, 8, 32],
        num_tokens: int = 100,
        num_runs: int = 10,
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks comparing multiple precision levels and batch sizes.
        """
        all_results = []

        for precision in precisions:
            for batch_size in batch_sizes:
                result = self.benchmark_precision(
                    precision=precision,
                    batch_size=batch_size,
                    num_tokens=num_tokens,
                    num_runs=num_runs,
                )
                if result:
                    all_results.append(result)

        return all_results

    def print_comparison_table(self):
        """Print a formatted comparison table of all results."""
        if not self.results:
            print("No results to display")
            return

        print("\n" + "=" * 100)
        print("BENCHMARK COMPARISON TABLE")
        print("=" * 100)
        print(
            f"{'Precision':<10} {'Batch':<8} {'Avg Lat':<12} {'P50 Lat':<12} {'P90 Lat':<12} "
            f"{'Throughput':<15} {'Memory (GB)':<12}"
        )
        print("-" * 100)

        for r in self.results:
            print(
                f"{r.precision:<10} {r.batch_size:<8} {r.avg_latency_ms:<12.2f} "
                f"{r.p50_latency_ms:<12.2f} {r.p90_latency_ms:<12.2f} "
                f"{r.throughput_tokens_per_sec:<15.2f} {r.gpu_memory_allocated_gb:<12.2f}"
            )

        print("=" * 100)

    def save_results(self, filepath: str):
        """Save results to a CSV file."""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Precision",
                    "Batch Size",
                    "Num Tokens",
                    "Avg Latency (ms)",
                    "P50 Latency (ms)",
                    "P90 Latency (ms)",
                    "P99 Latency (ms)",
                    "Throughput (tokens/s)",
                    "GPU Memory (GB)",
                ]
            )

            for r in self.results:
                writer.writerow(
                    [
                        r.precision,
                        r.batch_size,
                        r.num_tokens,
                        f"{r.avg_latency_ms:.2f}",
                        f"{r.p50_latency_ms:.2f}",
                        f"{r.p90_latency_ms:.2f}",
                        f"{r.p99_latency_ms:.2f}",
                        f"{r.throughput_tokens_per_sec:.2f}",
                        f"{r.gpu_memory_allocated_gb:.2f}",
                    ]
                )

        print(f"Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark models across precision levels"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--precisions",
        type=str,
        nargs="+",
        default=["FP16", "INT8"],
        choices=["FP16", "INT8", "INT4"],
        help="Precision levels to benchmark",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate per request",
    )
    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file for results",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PRECISION BENCHMARK TOOL")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Precision levels: {args.precisions}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Tokens per request: {args.num_tokens}")
    print(f"Benchmark runs: {args.num_runs}")

    benchmark = PrecisionBenchmark(model_path=args.model)

    all_results = benchmark.run_comparison(
        precisions=args.precisions,
        batch_sizes=args.batch_sizes,
        num_tokens=args.num_tokens,
        num_runs=args.num_runs,
    )

    benchmark.print_comparison_table()
    benchmark.save_results(args.output)

    print("\nNote on TensorRT benchmarking:")
    print("  TensorRT requires complex setup (CUDA Toolkit, TensorRT installation,")
    print("  ONNX export, engine building). For full TensorRT benchmarks, see")
    print("  the detailed setup guide in deploy_trt_guide.md")


if __name__ == "__main__":
    main()
