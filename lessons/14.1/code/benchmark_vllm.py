#!/usr/bin/env python3
"""
vLLM Benchmark Script

测试vLLM的吞吐量和延迟性能
"""

import asyncio
import time
import statistics
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass
from openai import AsyncOpenAI
import psutil


@dataclass
class BenchmarkResult:
    metric: str
    value: float
    unit: str


class vLLMBenchmark:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def benchmark_latency(
        self, num_runs: int = 20, warmup_runs: int = 5
    ) -> List[BenchmarkResult]:
        """Benchmark single request latency"""
        prompts = [
            "解释机器学习的基本概念",
            "介绍深度学习的发展历史",
            "什么是Transformer架构",
            "解释注意力机制的工作原理",
            "介绍大语言模型的应用场景",
        ]

        print(f"Running {warmup_runs} warmup requests...")
        for i in range(warmup_runs):
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompts[i % len(prompts)]}],
                max_tokens=128,
            )

        print(f"Running {num_runs} latency test requests...")
        latencies = []
        for i in range(num_runs):
            start_time = time.time()
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompts[i % len(prompts)]}],
                max_tokens=256,
            )
            latency = time.time() - start_time
            latencies.append(latency)
            print(f"  Run {i + 1}/{num_runs}: {latency:.3f}s")

        return [
            BenchmarkResult("mean_latency", statistics.mean(latencies), "s"),
            BenchmarkResult("median_latency", statistics.median(latencies), "s"),
            BenchmarkResult("min_latency", min(latencies), "s"),
            BenchmarkResult("max_latency", max(latencies), "s"),
            BenchmarkResult("p50_latency", sorted(latencies)[len(latencies) // 2], "s"),
            BenchmarkResult(
                "p95_latency", sorted(latencies)[int(len(latencies) * 0.95)], "s"
            ),
            BenchmarkResult(
                "p99_latency", sorted(latencies)[int(len(latencies) * 0.99)], "s"
            ),
            BenchmarkResult(
                "std_latency",
                statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "s",
            ),
        ]

    async def benchmark_throughput(
        self,
        duration_seconds: int = 60,
        max_concurrent: int = 10,
        prompt: str = "解释量子计算的基本原理和应用前景",
    ) -> List[BenchmarkResult]:
        """Benchmark throughput under sustained load"""
        print(f"Running throughput benchmark for {duration_seconds}s...")
        print(f"Max concurrent requests: {max_concurrent}")

        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        failed = 0
        latencies = []
        start_time = time.time()

        async def single_request():
            nonlocal completed, failed
            req_start = time.time()
            try:
                await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=128,
                )
                completed += 1
                latencies.append(time.time() - req_start)
            except Exception as e:
                failed += 1

        async def worker():
            while time.time() - start_time < duration_seconds:
                async with semaphore:
                    await single_request()

        workers = [asyncio.create_task(worker()) for _ in range(max_concurrent)]
        await asyncio.gather(*workers, return_exceptions=True)

        total_time = time.time() - start_time
        throughput = completed / total_time

        return [
            BenchmarkResult("completed_requests", completed, "count"),
            BenchmarkResult("failed_requests", failed, "count"),
            BenchmarkResult("total_time", total_time, "s"),
            BenchmarkResult("throughput", throughput, "req/s"),
            BenchmarkResult(
                "avg_latency", statistics.mean(latencies) if latencies else 0, "s"
            ),
        ]

    async def benchmark_concurrent_scaling(
        self,
        concurrency_levels: List[int] = [1, 2, 5, 10, 20],
        requests_per_level: int = 50,
    ) -> Dict[int, List[BenchmarkResult]]:
        """Benchmark how throughput scales with concurrency"""
        results = {}
        prompt = "解释人工智能在现代社会中的应用"

        for conc in concurrency_levels:
            print(f"\nTesting concurrency level: {conc}")
            semaphore = asyncio.Semaphore(conc)
            completed = 0
            latencies = []
            start_time = time.time()

            async def single_request(idx: int):
                nonlocal completed
                req_start = time.time()
                try:
                    await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": f"{prompt} (请求{idx})"}],
                        max_tokens=128,
                    )
                    completed += 1
                    latencies.append(time.time() - req_start)
                except Exception:
                    pass

            tasks = [single_request(i) for i in range(requests_per_level * conc)]
            await asyncio.gather(*tasks, return_exceptions=True)

            total_time = time.time() - start_time
            results[conc] = [
                BenchmarkResult("completed", completed, "count"),
                BenchmarkResult("throughput", completed / total_time, "req/s"),
                BenchmarkResult(
                    "avg_latency", statistics.mean(latencies) if latencies else 0, "s"
                ),
            ]
            print(
                f"  Completed: {completed}, Throughput: {completed / total_time:.2f} req/s"
            )

        return results

    async def benchmark_long_context(
        self, context_lengths: List[int] = [512, 1024, 2048, 4096, 8192]
    ) -> List[BenchmarkResult]:
        """Benchmark performance with varying context lengths"""
        results = []
        prompt_template = "请阅读以下文本然后回答问题。{}"

        for ctx_len in context_lengths:
            context = "这是一段很长的文本。" * (ctx_len // 10)
            prompt = prompt_template.format(context[:ctx_len])

            print(f"Testing context length: {ctx_len}")
            start_time = time.time()

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=64,
                )
                latency = time.time() - start_time

                results.append(BenchmarkResult(f"context_{ctx_len}", latency, "s"))
                print(f"  Latency: {latency:.3f}s")

            except Exception as e:
                print(f"  Failed: {e}")
                results.append(BenchmarkResult(f"context_{ctx_len}", -1, "s"))

        return results

    def get_system_info(self) -> Dict[str, Any]:
        """Get system resource information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "memory_percent": psutil.virtual_memory().percent,
        }


async def main():
    parser = argparse.ArgumentParser(description="vLLM Benchmark")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["all", "latency", "throughput", "scaling", "context"],
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Throughput test duration in seconds"
    )
    args = parser.parse_args()

    benchmark = vLLMBenchmark(base_url=args.base_url, model=args.model)

    print("=" * 60)
    print("vLLM Benchmark Suite")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print("=" * 60)

    sys_info = benchmark.get_system_info()
    print("\nSystem Info:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    if args.benchmark in ["all", "latency"]:
        print("\n" + "-" * 40)
        print("1. Latency Benchmark")
        print("-" * 40)
        results = await benchmark.benchmark_latency(num_runs=20)
        print("\nLatency Results:")
        for r in results:
            print(f"  {r.metric}: {r.value:.4f} {r.unit}")

    if args.benchmark in ["all", "throughput"]:
        print("\n" + "-" * 40)
        print("2. Throughput Benchmark")
        print("-" * 40)
        results = await benchmark.benchmark_throughput(duration_seconds=args.duration)
        print("\nThroughput Results:")
        for r in results:
            print(f"  {r.metric}: {r.value:.4f} {r.unit}")

    if args.benchmark in ["all", "scaling"]:
        print("\n" + "-" * 40)
        print("3. Concurrent Scaling Benchmark")
        print("-" * 40)
        results = await benchmark.benchmark_concurrent_scaling(
            concurrency_levels=[1, 2, 5, 10], requests_per_level=30
        )
        print("\nScaling Results:")
        for conc, metrics in results.items():
            print(f"  Concurrency {conc}:")
            for r in metrics:
                print(f"    {r.metric}: {r.value:.4f} {r.unit}")

    if args.benchmark in ["all", "context"]:
        print("\n" + "-" * 40)
        print("4. Long Context Benchmark")
        print("-" * 40)
        results = await benchmark.benchmark_long_context(
            context_lengths=[512, 1024, 2048, 4096]
        )
        print("\nContext Results:")
        for r in results:
            print(f"  {r.metric}: {r.value:.4f} {r.unit}")


if __name__ == "__main__":
    asyncio.run(main())
