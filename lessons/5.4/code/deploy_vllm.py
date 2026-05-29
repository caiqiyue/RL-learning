#!/usr/bin/env python3
"""
vLLM Quantized Inference Server

This script demonstrates how to serve a quantized model (AWQ/GPTQ) using vLLM.
vLLM provides high-throughput inference through:
- PagedAttention: Virtual memory management for KV cache
- Continuous Batching: Dynamic batch scheduling
- Prefix Caching: Reuse computation for shared prefixes

Usage:
    python deploy_vllm.py --model <model_path> --quantization AWQ

Requirements:
    - vLLM installed: pip install vllm
    - CUDA 11.8+ compatible GPU
    - Quantized model (AWQ or GPTQ format)
"""

import argparse
import sys
from typing import List, Optional

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vLLM not installed. Install with: pip install vllm")
    sys.exit(1)


class QuantizedInferenceServer:
    """
    High-throughput inference server using vLLM for quantized models.

    Supports:
    - AWQ and GPTQ quantized models
    - Continuous batching for high throughput
    - OpenAI-compatible API endpoint
    """

    def __init__(
        self,
        model_path: str,
        quantization: Optional[str] = "AWQ",
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize the vLLM inference server.

        Args:
            model_path: Path to model or HuggingFace model ID
            quantization: Quantization method ("AWQ", "GPTQ", or None for FP16)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            dtype: Data type for activations ("float16", "bfloat16")
            gpu_memory_utilization: Fraction of GPU memory to use
        """
        self.model_path = model_path
        self.quantization = quantization

        print(f"Initializing vLLM server...")
        print(f"  Model: {model_path}")
        print(f"  Quantization: {quantization}")
        print(f"  Tensor Parallel Size: {tensor_parallel_size}")
        print(f"  Max Model Length: {max_model_len}")

        self.llm = LLM(
            model=model_path,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        print("Server initialized successfully.")

    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Generate text from prompts using batched inference.

        vLLM automatically handles batching for maximum throughput.

        Args:
            prompts: List of input prompts
            temperature: Sampling temperature (0 = deterministic)
            top_p: Nucleus sampling threshold
            max_tokens: Maximum tokens to generate per prompt
            stop: Stop sequences

        Returns:
            List of result dictionaries with prompt, generated text, and metadata
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            result = {
                "prompt": output.prompt,
                "generated_text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
                "request_id": output.request_id,
            }
            results.append(result)

        return results

    def generate_streaming(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        """
        Streaming generation for real-time applications.

        Yields tokens as they are generated instead of waiting for complete output.
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        for output in self.llm.generate([prompt], sampling_params):
            for t in output.outputs[0].token_ids:
                yield t

    def batch_generate_from_file(
        self,
        input_file: str,
        output_file: str,
        batch_size: int = 32,
        **generate_kwargs,
    ):
        """
        Process a file of prompts and save results.

        Handles large prompt lists by processing in batches.
        """
        with open(input_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            batch_results = self.generate(batch, **generate_kwargs)
            results.extend(batch_results)
            print(
                f"Processed {min(i + batch_size, len(prompts))}/{len(prompts)} prompts"
            )

        with open(output_file, "w") as f:
            for r in results:
                f.write(f"Prompt: {r['prompt']}\n")
                f.write(f"Generated: {r['generated_text']}\n")
                f.write("-" * 80 + "\n")

        print(f"Results saved to {output_file}")
        return results


def main():
    parser = argparse.ArgumentParser(description="vLLM Quantized Inference Server")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model or HuggingFace model ID"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="AWQ",
        choices=["AWQ", "GPTQ", None],
        help="Quantization method (None for FP16)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096, help="Maximum sequence length"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Activation data type",
    )
    parser.add_argument("--test", action="store_true", help="Run a test generation")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (if using api_server mode)",
    )

    args = parser.parse_args()

    if args.test:
        print("\n" + "=" * 60)
        print("Running test generation...")
        print("=" * 60 + "\n")

        server = QuantizedInferenceServer(
            model_path=args.model,
            quantization=args.quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
        )

        test_prompts = [
            "Explain quantum computing in simple terms:",
            "Write a Python function to compute fibonacci:",
        ]

        results = server.generate(test_prompts, temperature=0.7, max_tokens=128)

        for r in results:
            print(f"Prompt: {r['prompt']}")
            print(f"Generated: {r['generated_text']}")
            print("-" * 40)

    else:
        print("\nTo start the API server, run:")
        print(f"  python -m vllm.entrypoints.openai.api_server \\")
        print(f"      --model {args.model} \\")
        print(f"      --quantization {args.quantization} \\")
        print(f"      --tensor-parallel-size {args.tensor_parallel_size}")
        print("\nOr use this script programmatically:")
        print(
            f"  server = QuantizedInferenceServer('{args.model}', '{args.quantization}')"
        )
        print(f"  results = server.generate(['your prompt here'])")


if __name__ == "__main__":
    main()
