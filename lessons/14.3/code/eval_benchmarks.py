#!/usr/bin/env python3
"""
Standard Benchmark Evaluation using lm-evaluation-harness

This script demonstrates how to evaluate language models on standard benchmarks
including MMLU, HumanEval, GSM8K, and other popular evaluation tasks.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

try:
    from lm_eval import Evaluator, simple_evaluate
    from lm_eval.models import get_model
    from lm_eval.tasks import get_task
    from lm_eval.logging_utils import WandbLogger
except ImportError:
    print("Error: lm-evaluation-harness not installed. Run: pip install lm-eval")
    raise


def get_common_args() -> Dict:
    """Return common arguments for lm-evaluation-harness."""
    return {
        "batch_size": "auto",
        "device": "cuda",
        "dtype": "float16",
    }


def run_mmlu(
    model_name: str,
    model_args: Optional[str] = None,
    num_fewshot: int = None,
) -> Dict:
    """
    Run MMLU (Massive Multitask Language Understanding) benchmark.

    MMLU contains 57 tasks spanning various knowledge domains.
    """
    tasks = ["mmlu"]

    extra_args = model_args or ""
    if num_fewshot is not None:
        extra_args += f",num_fewshot={num_fewshot}"

    results = simple_evaluate(
        model="hf",
        model_args=f"{get_common_args()},{extra_args}"
        if extra_args
        else get_common_args(),
        tasks=tasks,
        limit=None,
    )

    return results


def run_humaneval(
    model_name: str,
    model_args: Optional[str] = None,
    num_samples: int = None,
) -> Dict:
    """
    Run HumanEval benchmark for code generation.

    HumanEval contains 164 Python programming problems.
    Evaluates using pass@k metric.
    """
    tasks = ["humaneval"]

    extra_args = model_args or ""

    results = simple_evaluate(
        model="hf",
        model_args=f"{get_common_args()},{extra_args}"
        if extra_args
        else get_common_args(),
        tasks=tasks,
        limit=num_samples,
    )

    return results


def run_gsm8k(
    model_name: str,
    model_args: Optional[str] = None,
    num_fewshot: int = 5,
) -> Dict:
    """
    Run GSM8K (Grade School Math 8K) benchmark.

    Contains 8,500 grade school math word problems requiring multi-step reasoning.
    """
    tasks = ["gsm8k"]

    extra_args = model_args or ""
    extra_args += f",num_fewshot={num_fewshot}"

    results = simple_evaluate(
        model="hf",
        model_args=f"{get_common_args()},{extra_args}"
        if extra_args
        else get_common_args(),
        tasks=tasks,
        limit=None,
    )

    return results


def run_truthfulqa(
    model_name: str,
    model_args: Optional[str] = None,
) -> Dict:
    """
    Run TruthfulQA benchmark for evaluating truthfulness.

    Contains questions where humans might commonly give wrong answers.
    """
    tasks = ["truthfulqa_mc2", "truthfulqa_mc1"]

    extra_args = model_args or ""

    results = simple_evaluate(
        model="hf",
        model_args=f"{get_common_args()},{extra_args}"
        if extra_args
        else get_common_args(),
        tasks=tasks,
        limit=None,
    )

    return results


def run_hellaswag(
    model_name: str,
    model_args: Optional[str] = None,
    num_fewshot: int = 10,
) -> Dict:
    """
    Run HellaSwag benchmark for common sense reasoning.
    """
    tasks = ["hellaswag"]

    extra_args = model_args or ""
    extra_args += f",num_fewshot={num_fewshot}"

    results = simple_evaluate(
        model="hf",
        model_args=f"{get_common_args()},{extra_args}"
        if extra_args
        else get_common_args(),
        tasks=tasks,
        limit=None,
    )

    return results


def run_benchmark(
    benchmark_name: str,
    model_name: str,
    model_args: Optional[str] = None,
    **kwargs,
) -> Dict:
    """
    Run a specified benchmark.

    Args:
        benchmark_name: Name of the benchmark (mmlu, humaneval, gsm8k, etc.)
        model_name: HuggingFace model name or path
        model_args: Additional model arguments
        **kwargs: Additional benchmark-specific arguments

    Returns:
        Dictionary containing benchmark results
    """
    benchmark_map = {
        "mmlu": run_mmlu,
        "humaneval": run_humaneval,
        "gsm8k": run_gsm8k,
        "truthfulqa": run_truthfulqa,
        "hellaswag": run_hellaswag,
    }

    if benchmark_name.lower() not in benchmark_map:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {list(benchmark_map.keys())}"
        )

    run_func = benchmark_map[benchmark_name.lower()]
    results = run_func(model_name, model_args, **kwargs)

    return results


def run_multiple_benchmarks(
    benchmarks: List[str],
    model_name: str,
    model_args: Optional[str] = None,
) -> Dict:
    """
    Run multiple benchmarks in sequence and aggregate results.

    Args:
        benchmarks: List of benchmark names to run
        model_name: HuggingFace model name or path
        model_args: Additional model arguments

    Returns:
        Dictionary containing aggregated results from all benchmarks
    """
    all_results = {
        "model": model_name,
        "benchmarks": {},
    }

    for benchmark in benchmarks:
        print(f"\nRunning {benchmark}...")
        try:
            results = run_benchmark(benchmark, model_name, model_args)
            all_results["benchmarks"][benchmark] = results.get("results", {})
            print(f"  {benchmark} completed")
        except Exception as e:
            print(f"  Error running {benchmark}: {e}")
            all_results["benchmarks"][benchmark] = {"error": str(e)}

    return all_results


def save_results(results: Dict, output_path: str):
    """Save evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")


def print_summary(results: Dict):
    """Print a human-readable summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {results.get('model', 'unknown')}")
    print("-" * 60)

    benchmarks = results.get("benchmarks", {})
    for bench_name, bench_results in benchmarks.items():
        print(f"\n{bench_name.upper()}:")

        if "error" in bench_results:
            print(f"  Error: {bench_results['error']}")
            continue

        if isinstance(bench_results, dict):
            for task_name, metrics in bench_results.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            if abs(value) < 1:
                                print(f"  {task_name}.{metric_name}: {value:.4f}")
                            else:
                                print(f"  {task_name}.{metric_name}: {value}")
                elif isinstance(metrics, (int, float)):
                    print(f"  {task_name}: {metrics}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run standard benchmarks on language models"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["mmlu", "hellaswag", "truthfulqa"],
        help="List of benchmarks to run",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default=None,
        help="Additional model arguments (e.g., 'temperature=0.7,top_p=0.9')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./eval_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    print(f"Evaluating model: {args.model}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")

    results = run_multiple_benchmarks(
        benchmarks=args.benchmarks,
        model_name=args.model,
        model_args=args.model_args,
    )

    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
