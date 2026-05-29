#!/usr/bin/env python3
"""
Complete Evaluation Pipeline with Aggregation

This module integrates benchmark evaluation, LLM-as-Judge,
and comprehensive result reporting into a unified pipeline.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from eval_benchmarks import run_multiple_benchmarks
    from llm_judge import (
        LLMasJudge,
        JudgeConfig,
        JudgeModel,
        EvaluationMode,
        EvalCase,
        BatchEvaluator,
        load_eval_cases_from_jsonl,
    )
except ImportError:
    print("Warning: Could not import from eval_benchmarks or llm_judge")
    print("Some functionality may be limited.")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    benchmarks: List[str]
    model_name: str
    model_args: Optional[str] = None
    batch_size: int = 8
    num_fewshot: Optional[int] = None


@dataclass
class JudgeConfigForPipeline:
    """Configuration for LLM-as-Judge evaluation."""

    judge_model: str = "gpt-4"
    judge_type: str = "openai"
    evaluation_mode: str = "single"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class EvalConfig:
    """Main evaluation configuration."""

    model_name: str
    output_dir: str = "./eval_results"
    seed: int = 42

    benchmark_config: Optional[BenchmarkConfig] = None
    judge_config: Optional[JudgeConfigForPipeline] = None

    run_benchmarks: bool = True
    run_llm_judge: bool = True
    judge_test_file: Optional[str] = None


@dataclass
class EvaluationReport:
    """Container for the complete evaluation report."""

    model_name: str
    timestamp: str
    duration_seconds: float

    benchmark_results: Dict[str, Any]
    llm_judge_results: Dict[str, Any]

    summary: Dict[str, Any]
    raw_results: Dict[str, Any]


class MetricsAggregator:
    """Aggregates metrics from multiple evaluation sources."""

    @staticmethod
    def aggregate_benchmark_results(results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate benchmark results into summary statistics.

        Args:
            results: Raw benchmark results

        Returns:
            Dictionary with aggregated metrics
        """
        aggregated = {
            "benchmarks_run": [],
            "overall_metrics": {},
            "detailed_results": {},
        }

        if "benchmarks" not in results:
            return aggregated

        for bench_name, bench_data in results["benchmarks"].items():
            aggregated["benchmarks_run"].append(bench_name)

            if isinstance(bench_data, dict) and "error" not in bench_data:
                for task_name, metrics in bench_data.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                key = f"{bench_name}_{task_name}_{metric_name}"
                                aggregated["overall_metrics"][key] = value

                aggregated["detailed_results"][bench_name] = bench_data

        if aggregated["overall_metrics"]:
            all_values = list(aggregated["overall_metrics"].values())
            aggregated["summary"] = {
                "mean_score": float(np.mean(all_values)),
                "std_score": float(np.std(all_values)),
                "min_score": float(np.min(all_values)),
                "max_score": float(np.max(all_values)),
                "num_metrics": len(all_values),
            }

        return aggregated

    @staticmethod
    def aggregate_judge_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate LLM-as-Judge evaluation results.

        Args:
            results: List of judge evaluation results

        Returns:
            Dictionary with aggregated statistics
        """
        aggregated = {
            "total_cases": len(results),
            "scores": {},
            "pairwise": {},
        }

        all_scores = []
        for result in results:
            if "score" in result and result["score"] is not None:
                all_scores.append(result["score"])

            for key, value in result.items():
                if key not in aggregated["scores"]:
                    aggregated["scores"][key] = []
                aggregated["scores"][key].append(value)

        if all_scores:
            aggregated["score_stats"] = {
                "mean": float(np.mean(all_scores)),
                "std": float(np.std(all_scores)),
                "min": float(np.min(all_scores)),
                "max": float(np.max(all_scores)),
                "median": float(np.median(all_scores)),
            }

        win_counts = {"A": 0, "B": 0, "tie": 0}
        for result in results:
            if "winner" in result and result["winner"]:
                winner = result["winner"]
                if winner in win_counts:
                    win_counts[winner] += 1

        if sum(win_counts.values()) > 0:
            total = sum(win_counts.values())
            aggregated["pairwise"] = {
                "wins_A": win_counts["A"],
                "wins_B": win_counts["B"],
                "ties": win_counts["tie"],
                "win_rate_A": win_counts["A"] / total if total > 0 else 0,
                "win_rate_B": win_counts["B"] / total if total > 0 else 0,
                "tie_rate": win_counts["tie"] / total if total > 0 else 0,
            }

        return aggregated

    @staticmethod
    def compute_overall_score(
        benchmark_agg: Dict[str, Any],
        judge_agg: Dict[str, Any],
        benchmark_weight: float = 0.5,
        judge_weight: float = 0.5,
    ) -> float:
        """
        Compute a weighted overall score.

        Args:
            benchmark_agg: Aggregated benchmark results
            judge_agg: Aggregated judge results
            benchmark_weight: Weight for benchmark score
            judge_weight: Weight for judge score

        Returns:
            Overall weighted score
        """
        benchmark_score = benchmark_agg.get("summary", {}).get("mean_score", 0)
        judge_score = judge_agg.get("score_stats", {}).get("mean", 0)

        if benchmark_score == 0 and judge_score == 0:
            return 0.0

        total_weight = benchmark_weight + judge_weight
        normalized_benchmark = (
            benchmark_score / 100 if benchmark_score > 1 else benchmark_score
        )
        normalized_judge = judge_score / 5 if judge_score > 1 else judge_score

        overall = (
            (normalized_benchmark * benchmark_weight)
            + (normalized_judge * judge_weight)
        ) / total_weight

        return round(overall, 4)


class EvaluationPipeline:
    """
    Complete evaluation pipeline for language models.

    Integrates:
    - Standard benchmark evaluation
    - LLM-as-Judge evaluation
    - Result aggregation and reporting
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.results = {}
        self.start_time = None
        self.end_time = None

        np.random.seed(config.seed)

    def run_benchmarks(self) -> Dict[str, Any]:
        """Run standard benchmark evaluation."""
        if not self.config.run_benchmarks or not self.config.benchmark_config:
            return {}

        print("\n" + "=" * 60)
        print("RUNNING BENCHMARK EVALUATION")
        print("=" * 60)

        bench_config = self.config.benchmark_config

        try:
            results = run_multiple_benchmarks(
                benchmarks=bench_config.benchmarks,
                model_name=bench_config.model_name,
                model_args=bench_config.model_args,
            )

            print("\nBenchmark Results Summary:")
            for bench in bench_config.benchmarks:
                if bench in results.get("benchmarks", {}):
                    print(f"  - {bench}: completed")
                else:
                    print(f"  - {bench}: failed or not run")

            return results

        except Exception as e:
            print(f"Benchmark evaluation error: {e}")
            return {"error": str(e), "benchmarks": {}}

    def run_llm_judge_eval(self) -> List[Dict[str, Any]]:
        """Run LLM-as-Judge evaluation."""
        if not self.config.run_llm_judge:
            return []

        if not self.config.judge_config:
            print("Warning: judge_config not set")
            return []

        judge_test_file = self.config.judge_test_file
        if not judge_test_file or not os.path.exists(judge_test_file):
            print(f"Warning: Judge test file not found: {judge_test_file}")
            return []

        print("\n" + "=" * 60)
        print("RUNNING LLM-as-Judge EVALUATION")
        print("=" * 60)

        judge_cfg = self.config.judge_config

        judge = LLMasJudge(
            config=JudgeConfig(
                model=JudgeModel(judge_cfg.judge_model),
                mode=EvaluationMode(judge_cfg.evaluation_mode),
                temperature=judge_cfg.temperature,
                max_tokens=judge_cfg.max_tokens,
            ),
            judge_type=judge_cfg.judge_type,
        )

        cases = load_eval_cases_from_jsonl(judge_test_file)
        print(f"Loaded {len(cases)} evaluation cases")

        evaluator = BatchEvaluator(judger=judger, verbose=True)
        results = evaluator.evaluate_dataset(
            cases,
            mode=EvaluationMode(judge_cfg.evaluation_mode),
        )

        judge_results = []
        for case, result in zip(cases, results):
            judge_results.append(
                {
                    "question": case.question,
                    "score": result.score,
                    "winner": result.winner,
                    "reasoning": result.reasoning,
                }
            )

        return judge_results

    def generate_report(
        self,
        benchmark_results: Dict[str, Any],
        judge_results: List[Dict[str, Any]],
    ) -> EvaluationReport:
        """Generate comprehensive evaluation report."""

        benchmark_agg = MetricsAggregator.aggregate_benchmark_results(benchmark_results)
        judge_agg = MetricsAggregator.aggregate_judge_results(judge_results)

        overall_score = MetricsAggregator.compute_overall_score(
            benchmark_agg, judge_agg
        )

        strengths = []
        weaknesses = []

        if benchmark_agg.get("overall_metrics"):
            sorted_metrics = sorted(
                benchmark_agg["overall_metrics"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            strengths = [m[0] for m in sorted_metrics[:3]]
            weaknesses = [m[0] for m in sorted_metrics[-3:] if m[1] < 0.7]

        summary = {
            "overall_score": overall_score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "benchmark_count": len(benchmark_agg.get("benchmarks_run", [])),
            "judge_case_count": len(judge_results),
        }

        return EvaluationReport(
            model_name=self.config.model_name,
            timestamp=datetime.now().isoformat(),
            duration_seconds=self.end_time - self.start_time
            if self.end_time and self.start_time
            else 0,
            benchmark_results=benchmark_agg,
            llm_judge_results=judge_agg,
            summary=summary,
            raw_results={
                "benchmark_raw": benchmark_results,
                "judge_raw": judge_results,
            },
        )

    def save_report(self, report: EvaluationReport, format: str = "json"):
        """Save evaluation report to file."""

        os.makedirs(self.config.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = self.config.model_name.replace("/", "_")

        if format == "json":
            filepath = os.path.join(
                self.config.output_dir, f"eval_report_{model_safe}_{timestamp}.json"
            )
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        elif format == "markdown":
            filepath = os.path.join(
                self.config.output_dir, f"eval_report_{model_safe}_{timestamp}.md"
            )
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self._format_markdown(report))

        print(f"\nReport saved to: {filepath}")
        return filepath

    def _format_markdown(self, report: EvaluationReport) -> str:
        """Format report as markdown."""

        md = f"""# Evaluation Report

## Model: {report.model_name}

**Timestamp:** {report.timestamp}

**Duration:** {report.duration_seconds:.2f} seconds

---

## Summary

- **Overall Score:** {report.summary.get("overall_score", "N/A")}
- **Benchmarks Run:** {report.summary.get("benchmark_count", 0)}
- **Judge Cases:** {report.summary.get("judge_case_count", 0)}

### Strengths
"""

        for s in report.summary.get("strengths", []):
            md += f"- {s}\n"

        md += "\n### Weaknesses\n"
        for w in report.summary.get("weaknesses", []):
            md += f"- {w}\n"

        md += "\n---\n\n## Benchmark Results\n\n"

        if report.benchmark_results.get("benchmarks_run"):
            for bench in report.benchmark_results["benchmarks_run"]:
                md += f"### {bench}\n\n"
                if bench in report.benchmark_results.get("detailed_results", {}):
                    for task, metrics in report.benchmark_results["detailed_results"][
                        bench
                    ].items():
                        md += f"- {task}: {metrics}\n"
                md += "\n"

        md += "\n---\n\n## LLM-as-Judge Results\n\n"

        if report.llm_judge_results.get("score_stats"):
            stats = report.llm_judge_results["score_stats"]
            md += f"""### Score Statistics
- Mean: {stats.get("mean", "N/A"):.4f}
- Std: {stats.get("std", "N/A"):.4f}
- Min: {stats.get("min", "N/A"):.4f}
- Max: {stats.get("max", "N/A"):.4f}

"""

        if report.llm_judge_results.get("pairwise"):
            pw = report.llm_judge_results["pairwise"]
            md += f"""### Pairwise Comparison
- Wins A: {pw.get("wins_A", 0)} ({pw.get("win_rate_A", 0):.2%})
- Wins B: {pw.get("wins_B", 0)} ({pw.get("win_rate_B", 0):.2%})
- Ties: {pw.get("ties", 0)} ({pw.get("tie_rate", 0):.2%})

"""

        return md

    def run(self) -> EvaluationReport:
        """Execute the complete evaluation pipeline."""

        self.start_time = time.time()

        print("\n" + "=" * 60)
        print("EVALUATION PIPELINE STARTED")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Output Directory: {self.config.output_dir}")
        print(f"Run Benchmarks: {self.config.run_benchmarks}")
        print(f"Run LLM Judge: {self.config.run_llm_judge}")

        benchmark_results = {}
        if self.config.run_benchmarks:
            benchmark_results = self.run_benchmarks()

        judge_results = []
        if self.config.run_llm_judge:
            judge_results = self.run_llm_judge_eval()

        self.end_time = time.time()

        report = self.generate_report(benchmark_results, judge_results)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED")
        print("=" * 60)
        print(f"Overall Score: {report.summary.get('overall_score', 'N/A')}")
        print(f"Duration: {report.duration_seconds:.2f}s")

        return report


def create_default_config(
    model_name: str,
    benchmarks: Optional[List[str]] = None,
    judge_test_file: Optional[str] = None,
) -> EvalConfig:
    """Create a default evaluation configuration."""

    if benchmarks is None:
        benchmarks = ["mmlu", "hellaswag", "truthfulqa"]

    bench_config = BenchmarkConfig(
        benchmarks=benchmarks,
        model_name=model_name,
    )

    judge_cfg = JudgeConfigForPipeline(
        judge_model="gpt-4",
        judge_type="openai",
        evaluation_mode="single",
    )

    return EvalConfig(
        model_name=model_name,
        benchmark_config=bench_config,
        judge_config=judge_cfg,
        judge_test_file=judge_test_file,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run complete evaluation pipeline for language models"
    )

    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["mmlu", "hellaswag", "truthfulqa"],
        help="List of benchmarks to run",
    )
    parser.add_argument(
        "--judge-test-file",
        type=str,
        default=None,
        help="Path to JSONL file with judge evaluation cases",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-benchmarks", action="store_true", help="Skip benchmark evaluation"
    )
    parser.add_argument(
        "--skip-judge", action="store_true", help="Skip LLM-as-Judge evaluation"
    )
    parser.add_argument(
        "--report-format",
        type=str,
        choices=["json", "markdown", "both"],
        default="both",
        help="Report output format",
    )

    args = parser.parse_args()

    config = create_default_config(
        model_name=args.model,
        benchmarks=args.benchmarks if not args.skip_benchmarks else [],
        judge_test_file=args.judge_test_file,
    )

    config.output_dir = args.output_dir
    config.run_benchmarks = not args.skip_benchmarks
    config.run_llm_judge = not args.skip_judge

    pipeline = EvaluationPipeline(config)
    report = pipeline.run()

    if "json" in args.report_format or args.report_format == "both":
        pipeline.save_report(report, format="json")

    if "markdown" in args.report_format or args.report_format == "both":
        pipeline.save_report(report, format="markdown")


if __name__ == "__main__":
    main()
