#!/usr/bin/env python3
"""
vLLM Server Startup Script

启动vLLM服务器，支持多种量化方法和硬件配置
"""

import argparse
import subprocess
import sys
import time
import requests
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Start vLLM Server")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", "fp8", "int8", None],
        help="Quantization method",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192, help="Maximum model context length"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0-1)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=32768,
        help="Maximum batched tokens per iteration",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph for better debugging",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote model code"
    )
    parser.add_argument(
        "--download-dir", type=str, default=None, help="Model download directory"
    )
    return parser.parse_args()


def build_command(args):
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--port",
        str(args.port),
        "--host",
        args.host,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
    ]

    if args.quantization:
        cmd.extend(["--quantization", args.quantization])

    if args.tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])

    if args.enforce_eager:
        cmd.append("--enforce-eager")

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    if args.download_dir:
        cmd.extend(["--download-dir", args.download_dir])

    return cmd


def wait_for_server(url: str, timeout: int = 300) -> bool:
    """Wait for server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/v1/models", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def main():
    args = parse_args()

    cmd = build_command(args)
    print("=" * 60)
    print("Starting vLLM Server")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantization or 'None (FP16)'}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    print("\nStarting server process...")
    process = subprocess.Popen(cmd)

    try:
        if wait_for_server(f"http://localhost:{args.port}"):
            print(f"\nServer ready at http://localhost:{args.port}/v1")
            print("API Examples:")
            print(f"  curl http://localhost:{args.port}/v1/models")
            print(f"  curl http://localhost:{args.port}/v1/chat/completions \\")
            print(f"    -H 'Content-Type: application/json' \\")
            print(
                f'    -d \'{{"model": "{args.model}", "messages": [{{"role": "user", "content": "Hello"}}]}}\''
            )
        else:
            print("\nServer failed to start within timeout")
            process.terminate()
            sys.exit(1)

        print("\nPress Ctrl+C to stop server")
        process.wait()

    except KeyboardInterrupt:
        print("\nShutting down server...")
        process.terminate()
        process.wait()


if __name__ == "__main__":
    main()
