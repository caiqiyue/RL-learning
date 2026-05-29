#!/usr/bin/env python3
"""
Batch Inference with vLLM

使用vLLM API进行批量推理，支持多种并发策略
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
import argparse


@dataclass
class InferenceRequest:
    prompt: str
    system: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class InferenceResult:
    request_id: int
    prompt: str
    response: str
    latency: float
    success: bool
    error: Optional[str] = None


class BatchInferenceEngine:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_concurrent: int = 10,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def single_inference(
        self,
        request: InferenceRequest,
        request_id: int,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
    ) -> InferenceResult:
        """Execute a single inference request"""
        start_time = time.time()

        async with self.semaphore:
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": request.prompt})

            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                )

                latency = time.time() - start_time
                return InferenceResult(
                    request_id=request_id,
                    prompt=request.prompt,
                    response=response.choices[0].message.content,
                    latency=latency,
                    success=True,
                )

            except Exception as e:
                latency = time.time() - start_time
                return InferenceResult(
                    request_id=request_id,
                    prompt=request.prompt,
                    response="",
                    latency=latency,
                    success=False,
                    error=str(e),
                )

    async def run_batch(
        self,
        requests: List[InferenceRequest],
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        show_progress: bool = True,
    ) -> List[InferenceResult]:
        """Run batch inference with progress tracking"""
        tasks = [
            self.single_inference(req, idx, model) for idx, req in enumerate(requests)
        ]

        if show_progress:
            from tqdm import tqdm

            results = []
            for future in asyncio.as_completed(tasks):
                result = await future
                results.append(result)
                tqdm.write(
                    f"Completed {len(results)}/{len(requests)} - "
                    f"Latency: {result.latency:.2f}s"
                )
            return results

        return await asyncio.gather(*tasks)

    async def run_chat_batch(
        self,
        conversations: List[List[Dict[str, str]]],
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 256,
    ) -> List[InferenceResult]:
        """Run batch chat completions"""
        tasks = []
        for idx, conv in enumerate(conversations):
            task = self._chat_single(conv, idx, model, max_tokens)
            tasks.append(task)
        return await asyncio.gather(*tasks)

    async def _chat_single(
        self,
        messages: List[Dict[str, str]],
        request_id: int,
        model: str,
        max_tokens: int,
    ) -> InferenceResult:
        """Single chat completion"""
        start_time = time.time()

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )

                latency = time.time() - start_time
                return InferenceResult(
                    request_id=request_id,
                    prompt=str(messages),
                    response=response.choices[0].message.content,
                    latency=latency,
                    success=True,
                )

            except Exception as e:
                latency = time.time() - start_time
                return InferenceResult(
                    request_id=request_id,
                    prompt=str(messages),
                    response="",
                    latency=latency,
                    success=False,
                    error=str(e),
                )


def generate_sample_prompts(count: int = 100) -> List[InferenceRequest]:
    """Generate sample prompts for testing"""
    prompts = []
    topics = [
        "解释量子计算的基本原理",
        "写一首关于春天的诗",
        "介绍Python中的装饰器",
        "什么是机器学习",
        "解释区块链的工作原理",
        "介绍一下人工智能的发展历程",
        "如何学习一门新编程语言",
        "什么是深度学习",
        "介绍微服务架构的优缺点",
        "解释云计算的概念",
    ]

    for i in range(count):
        topic = topics[i % len(topics)]
        prompts.append(
            InferenceRequest(
                prompt=f"{topic}（用100字概括）",
                system="你是一个简洁的助手，用简短清晰的语言回答。",
                max_tokens=128,
                temperature=0.7,
            )
        )

    return prompts


async def main():
    parser = argparse.ArgumentParser(description="Batch Inference with vLLM")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Batch Inference")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Number of requests: {args.num_requests}")
    print(f"Max concurrent: {args.max_concurrent}")
    print("=" * 60)

    engine = BatchInferenceEngine(
        base_url=args.base_url, max_concurrent=args.max_concurrent
    )

    prompts = generate_sample_prompts(args.num_requests)

    print(f"\nStarting batch inference with {len(prompts)} requests...")
    start_time = time.time()

    results = await engine.run_batch(prompts, model=args.model)

    total_time = time.time() - start_time

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = [r.latency for r in successful]

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(results) / total_time:.2f} req/s")

    if latencies:
        print(f"\nLatency Stats:")
        print(f"  Mean: {sum(latencies) / len(latencies):.2f}s")
        print(f"  Min: {min(latencies):.2f}s")
        print(f"  Max: {max(latencies):.2f}s")

    if failed:
        print(f"\nFailed requests:")
        for r in failed[:5]:
            print(f"  ID {r.request_id}: {r.error}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total": len(results),
                        "successful": len(successful),
                        "failed": len(failed),
                        "total_time": total_time,
                        "throughput": len(results) / total_time,
                    },
                    "results": [
                        {
                            "request_id": r.request_id,
                            "prompt": r.prompt,
                            "response": r.response,
                            "latency": r.latency,
                            "success": r.success,
                            "error": r.error,
                        }
                        for r in results
                    ],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
