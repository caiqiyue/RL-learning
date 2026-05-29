"""
vLLM Deployment with Monitoring

Complete deployment script for quantized LLM models with vLLM.
"""

import os
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import torch

from vllm import LLM, SamplingParams, EngineArgs
from vllm.outputs import RequestOutput

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from sse_starlette import EventSourceResponse
except ImportError:
    print("FastAPI not installed. API server will not be available.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """vLLM deployment configuration"""

    model_path: str = "./models/quantized_7b"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    max_model_len: int = 4096
    dtype: str = "half"  # half, float16, bfloat16
    quantization: Optional[str] = "awq"  # awq, gptq, fp8, None
    kv_cache_dtype: str = "fp8_e5m2"
    enforce_eager: bool = False
    trust_remote_code: bool = True

    vllm_host: str = "0.0.0.0"
    vllm_port: int = 8000

    enable_monitoring: bool = True
    metrics_port: int = 9090


class InferenceStats:
    """Track inference statistics"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_requests = 0
        self.total_tokens = 0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.total_time = 0.0
        self.errors = 0
        self.first_token_times: List[float] = []
        self.request_latencies: List[float] = []

    def record(
        self,
        num_prompt_tokens: int,
        num_completion_tokens: int,
        latency_ms: float,
        first_token_latency_ms: float,
        error: bool = False,
    ):
        self.total_requests += 1
        self.total_tokens += num_prompt_tokens + num_completion_tokens
        self.total_prefill_tokens += num_prompt_tokens
        self.total_decode_tokens += num_completion_tokens
        self.total_time += latency_ms / 1000
        self.first_token_times.append(first_token_latency_ms)
        self.request_latencies.append(latency_ms)

        if error:
            self.errors += 1

        if len(self.request_latencies) > 10000:
            self.request_latencies = self.request_latencies[-5000:]
            self.first_token_times = self.first_token_times[-5000:]

    def get_summary(self) -> Dict:
        """Get statistics summary"""
        if self.total_requests == 0:
            return self._empty_summary()

        latencies = sorted(self.request_latencies)
        n = len(latencies)

        total_time_sec = self.total_time
        throughput = self.total_tokens / total_time_sec if total_time_sec > 0 else 0

        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "throughput_tokens_per_sec": throughput,
            "latency": {
                "mean_ms": sum(latencies) / n if n > 0 else 0,
                "p50_ms": latencies[int(n * 0.50)] if n > 0 else 0,
                "p90_ms": latencies[int(n * 0.90)] if n > 0 else 0,
                "p95_ms": latencies[int(n * 0.95)] if n > 0 else 0,
                "p99_ms": latencies[int(n * 0.99)] if n > 0 else 0,
                "max_ms": max(latencies) if n > 0 else 0,
                "min_ms": min(latencies) if n > 0 else 0,
            },
            "first_token_latency": {
                "mean_ms": sum(self.first_token_times) / len(self.first_token_times)
                if self.first_token_times
                else 0,
                "p99_ms": sorted(self.first_token_times)[
                    int(len(self.first_token_times) * 0.99)
                ]
                if self.first_token_times
                else 0,
            },
            "errors": self.errors,
            "error_rate": self.errors / self.total_requests,
        }

    def _empty_summary(self) -> Dict:
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "throughput_tokens_per_sec": 0,
            "latency": {},
            "errors": 0,
            "error_rate": 0,
        }


class vLLMDeployment:
    """
    vLLM Deployment Manager
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.llm: Optional[LLM] = None
        self.stats = InferenceStats()
        self.is_initialized = False
        self.start_time = time.time()

    def initialize(self) -> bool:
        """
        Initialize vLLM engine.
        """
        if self.is_initialized:
            logger.warning("vLLM engine already initialized")
            return True

        logger.info(f"Initializing vLLM engine...")
        logger.info(f"  Model: {self.config.model_path}")
        logger.info(f"  Tensor Parallel: {self.config.tensor_parallel_size}")
        logger.info(f"  GPU Memory Utilization: {self.config.gpu_memory_utilization}")
        logger.info(f"  Quantization: {self.config.quantization or 'None'}")
        logger.info(f"  KV Cache Dtype: {self.config.kv_cache_dtype}")

        engine_args = EngineArgs(
            model=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_model_len=self.config.max_model_len,
            dtype=self.config.dtype,
            quantization=self.config.quantization,
            kv_cache_dtype=self.config.kv_cache_dtype,
            enforce_eager=self.config.enforce_eager,
            trust_remote_code=self.config.trust_remote_code,
        )

        try:
            self.llm = LLM.from_engine_args(engine_args)
            self.is_initialized = True
            logger.info("vLLM engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            self.llm = None
            return False

    async def generate_async(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[Dict]:
        """
        Async generation interface.
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize vLLM engine")

        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=512,
                stop=None,
            )

        if isinstance(prompts, str):
            prompts = [prompts]

        start_time = time.time()
        first_token_time = None
        error = False
        error_msg = ""

        try:
            loop = asyncio.get_event_loop()

            outputs = await loop.run_in_executor(
                None,
                self.llm.generate,
                prompts,
                sampling_params,
            )

            prefill_time = time.time() - start_time

            results = []
            for output in outputs:
                generated_text = output.outputs[0].text
                num_completion_tokens = len(output.outputs[0].token_ids)

                result = {
                    "text": generated_text,
                    "completion_tokens": num_completion_tokens,
                    "finish_reason": output.outputs[0].finish_reason,
                    "prefill_time_ms": prefill_time * 1000,
                }
                results.append(result)

            total_latency = time.time() - start_time

            num_prompt_tokens = sum(
                len(p) for p in self.llm.get_tokenizer().encode("\n".join(prompts))
            )

            self.stats.record(
                num_prompt_tokens=num_prompt_tokens,
                num_completion_tokens=sum(r["completion_tokens"] for r in results),
                latency_ms=total_latency * 1000,
                first_token_latency_ms=prefill_time * 1000,
                error=False,
            )

            return results

        except Exception as e:
            error = True
            error_msg = str(e)
            logger.error(f"Generation error: {e}")

            self.stats.record(
                num_prompt_tokens=0,
                num_completion_tokens=0,
                latency_ms=0,
                first_token_latency_ms=0,
                error=True,
            )

            raise

    def generate_sync(
        self,
        prompts: Union[str, List[str]],
        **sampling_kwargs,
    ) -> List[Dict]:
        """
        Sync generation interface.
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Failed to initialize vLLM engine")

        sampling_params = SamplingParams(**sampling_kwargs)

        if isinstance(prompts, str):
            prompts = [prompts]

        start_time = time.time()

        try:
            outputs = self.llm.generate(prompts, sampling_params)

            results = []
            for output in outputs:
                results.append(
                    {
                        "text": output.outputs[0].text,
                        "completion_tokens": len(output.outputs[0].token_ids),
                        "finish_reason": output.outputs[0].finish_reason,
                    }
                )

            total_latency = time.time() - start_time

            self.stats.record(
                num_prompt_tokens=0,
                num_completion_tokens=sum(r["completion_tokens"] for r in results),
                latency_ms=total_latency * 1000,
                first_token_latency_ms=total_latency * 1000,
                error=False,
            )

            return results

        except Exception as e:
            self.stats.record(0, 0, 0, 0, error=True)
            raise

    def get_stats(self) -> Dict:
        """Get inference statistics"""
        summary = self.stats.get_summary()
        summary["uptime_seconds"] = time.time() - self.start_time
        return summary

    def reset_stats(self):
        """Reset statistics"""
        self.stats.reset()
        logger.info("Statistics reset")


deployment = None


app = FastAPI(title="vLLM Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompts: Union[str, List[str]]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512
    min_tokens: int = 1
    stop: Optional[List[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class GenerateResponse(BaseModel):
    results: List[Dict]
    stats: Dict


@app.on_event("startup")
async def startup_event():
    """Initialize deployment on startup"""
    global deployment

    config = DeploymentConfig(
        model_path=os.environ.get("VLLM_MODEL_PATH", "./models/quantized_7b"),
        tensor_parallel_size=int(os.environ.get("VLLM_TENSOR_PARALLEL", "1")),
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY", "0.9")),
    )

    deployment = vLLMDeployment(config)

    if not deployment.initialize():
        logger.error("Failed to initialize deployment")
        raise RuntimeError("Failed to initialize vLLM engine")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate completions"""
    global deployment

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        min_tokens=request.min_tokens,
        stop=request.stop,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
    )

    try:
        results = await deployment.generate_async(request.prompts, sampling_params)
        stats = deployment.get_stats()
        return GenerateResponse(results=results, stats=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_stream")
async def generate_stream(request: GenerateRequest):
    """Streaming generation endpoint"""
    global deployment

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
    )

    async def event_generator():
        prompts = (
            [request.prompts] if isinstance(request.prompts, str) else request.prompts
        )

        for result in await deployment.generate_async(prompts, sampling_params):
            yield {
                "event": "result",
                "data": json.dumps(result),
            }

    return EventSourceResponse(event_generator())


@app.get("/stats")
async def get_stats():
    """Get inference statistics"""
    global deployment

    if deployment is None:
        raise HTTPException(status_code=500, detail="Deployment not initialized")

    return deployment.get_stats()


@app.post("/stats/reset")
async def reset_stats():
    """Reset statistics"""
    global deployment

    if deployment is None:
        raise HTTPException(status_code=500, detail="Deployment not initialized")

    deployment.reset_stats()
    return {"message": "Statistics reset successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global deployment

    if deployment is None or not deployment.is_initialized:
        return {"status": "unhealthy", "initialized": False}

    return {
        "status": "healthy",
        "initialized": deployment.is_initialized,
        "uptime_seconds": time.time() - deployment.start_time,
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus-format metrics"""
    global deployment

    if deployment is None:
        raise HTTPException(status_code=500, detail="Deployment not initialized")

    stats = deployment.get_stats()

    lines = [
        "# HELP vllm_requests_total Total number of inference requests",
        "# TYPE vllm_requests_total counter",
        f"vllm_requests_total {stats['total_requests']}",
        "",
        "# HELP vllm_tokens_total Total number of tokens processed",
        "# TYPE vllm_tokens_total counter",
        f"vllm_tokens_total {stats['total_tokens']}",
        "",
        "# HELP vllm_throughput_tokens_per_sec Token throughput",
        "# TYPE vllm_throughput_tokens_per_sec gauge",
        f"vllm_throughput_tokens_per_sec {stats['throughput_tokens_per_sec']:.2f}",
        "",
        "# HELP vllm_latency_ms Request latency in milliseconds",
        "# TYPE vllm_latency_ms summary",
        f'vllm_latency_ms{{quantile="0.5"}} {stats["latency"].get("p50_ms", 0):.2f}',
        f'vllm_latency_ms{{quantile="0.9"}} {stats["latency"].get("p90_ms", 0):.2f}',
        f'vllm_latency_ms{{quantile="0.99"}} {stats["latency"].get("p99_ms", 0):.2f}',
        "",
        "# HELP vllm_errors_total Total number of errors",
        "# TYPE vllm_errors_total counter",
        f"vllm_errors_total {stats['errors']}",
    ]

    return "\n".join(lines)


def run_server():
    """Run the API server"""
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


def main():
    """CLI interface for deployment"""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Deployment")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--quantization",
        type=str,
        default="awq",
        choices=["awq", "gptq", "fp8", "none"],
    )

    args = parser.parse_args()

    config = DeploymentConfig(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization if args.quantization != "none" else None,
    )

    deployment = vLLMDeployment(config)

    if not deployment.initialize():
        logger.error("Failed to initialize deployment")
        return

    print("\n" + "=" * 60)
    print("vLLM Deployment Ready")
    print("=" * 60)
    print(f"Model: {config.model_path}")
    print(f"Quantization: {config.quantization or 'None'}")
    print(f"API Server: http://{args.host}:{args.port}")
    print("\nEndpoints:")
    print("  POST /generate - Generate completions")
    print("  GET  /stats    - Get inference statistics")
    print("  GET  /health   - Health check")
    print("  GET  /metrics  - Prometheus metrics")
    print("=" * 60 + "\n")

    prompts = ["Hello, how are you?"]

    print("Testing generation...")
    results = deployment.generate_sync(
        prompts,
        temperature=0.7,
        max_tokens=100,
    )

    print(f"Result: {results[0]['text'][:200]}...")
    print(f"Stats: {deployment.get_stats()}")

    run_server()


if __name__ == "__main__":
    main()
