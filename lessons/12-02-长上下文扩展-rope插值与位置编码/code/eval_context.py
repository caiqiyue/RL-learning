"""
Evaluate Long-Context Model Capability

This script provides tools to evaluate a model's ability to handle
long context windows. Includes:
1. Synthetic retrieval tasks (needle-in-haystack)
2. Context understanding benchmarks
3. Memory and performance profiling
"""

import torch
import time
import json
import argparse
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import random


@dataclass
class EvalResult:
    """Container for evaluation results."""

    task_name: str
    context_length: int
    success: bool
    metric: float
    latency_seconds: float
    details: Dict[str, Any]


class NeedleInHaystackTask:
    """
    Needle-in-haystack retrieval task.

    Places a specific piece of information (the "needle") in a large context
    (the "haystack") and tests whether the model can retrieve it.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        needle_template: str = "The secret number is: {secret}",
        haystack_topics: Optional[List[str]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.needle_template = needle_template
        self.haystack_topics = haystack_topics or [
            "science",
            "history",
            "technology",
            "arts",
            "philosophy",
            "geography",
            "biology",
            "physics",
            "chemistry",
            "literature",
        ]

    def generate_context(
        self, num_paragraphs: int, needle_position: Optional[int] = None
    ) -> tuple[str, str, int]:
        """
        Generate a haystack context with embedded needle.

        Args:
            num_paragraphs: Number of paragraphs in the haystack
            needle_position: Position for needle (0-1 ratio). None = random.

        Returns:
            Tuple of (context, needle, needle_position_tokens)
        """
        secret = str(random.randint(10000, 99999))
        needle = self.needle_template.format(secret=secret)

        paragraphs = []
        for i in range(num_paragraphs):
            topic = self.haystack_topics[i % len(self.haystack_topics)]
            sentences = [
                f"This paragraph discusses {topic} in detail.",
                f"Various aspects of {topic} have been thoroughly researched.",
                f"The study of {topic} reveals important insights.",
                f"Experts in {topic} have published significant findings.",
            ]
            paragraphs.append(" ".join(sentences))

        if needle_position is None:
            needle_position = random.uniform(0.1, 0.9)

        insert_idx = int(len(paragraphs) * needle_position)
        paragraphs.insert(insert_idx, needle)

        context = "\n\n".join(paragraphs)
        return context, needle, secret

    def create_prompt(self, context: str, question: str) -> str:
        """Create the evaluation prompt."""
        return f"""Given the following text, answer the question at the end.

Text:
{context}

Question: What is the secret number mentioned in the text?
Answer: The secret number is:"""

    def evaluate(
        self, context_length: int, needle_position_ratio: float = 0.5
    ) -> EvalResult:
        """
        Evaluate model on needle-in-haystack task.

        Args:
            context_length: Target context length in tokens
            needle_position_ratio: Where to place needle (0-1)

        Returns:
            EvalResult with success/failure and details
        """
        start_time = time.time()

        num_paragraphs = max(10, context_length // 50)
        context, needle, secret = self.generate_context(
            num_paragraphs=num_paragraphs, needle_position=needle_position_ratio
        )

        prompt = self.create_prompt(context, "What is the secret number?")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        if input_len > self.model.config.max_position_embeddings:
            return EvalResult(
                task_name="needle_in_haystack",
                context_length=input_len,
                success=False,
                metric=0.0,
                latency_seconds=time.time() - start_time,
                details={"error": "Context exceeds model max position"},
            )

        if "cuda" in str(getattr(self.model, "device", "cpu")):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_secret = None

        for char_count in range(5):
            for start in range(len(generated)):
                candidate = generated[start : start + 5 - char_count]
                if candidate.isdigit() and len(candidate) == 5 - char_count:
                    predicted_secret = candidate
                    break
            if predicted_secret:
                break

        success = predicted_secret == secret
        latency = time.time() - start_time

        return EvalResult(
            task_name="needle_in_haystack",
            context_length=input_len,
            success=success,
            metric=1.0 if success else 0.0,
            latency_seconds=latency,
            details={
                "expected_secret": secret,
                "predicted_secret": predicted_secret,
                "needle_position_ratio": needle_position_ratio,
                "input_tokens": input_len,
            },
        )


class MultiHopReasoningTask:
    """
    Multi-hop reasoning task over long context.

    Tests whether the model can perform multi-step reasoning
    across different parts of a long context.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_multi_hop_context(
        self, num_entities: int = 5
    ) -> tuple[str, List[str], str]:
        """
        Generate context requiring multi-hop reasoning.

        Returns:
            Tuple of (context, question_parts, answer)
        """
        entities = [f"Entity{i}" for i in range(num_entities)]
        relations = []

        for i in range(num_entities - 1):
            rel = f"{entities[i]} works with {entities[i + 1]}."
            relations.append(rel)

        relations_text = " ".join(relations)
        context = (
            f"{relations_text} Finally, {entities[-1]} is located in the city of Paris."
        )

        chain_str = " -> ".join(entities)
        answer = entities[-1]

        return context, [chain_str], answer

    def create_prompt(self, context: str, question: str) -> str:
        return f"""Based on the following information, answer the question.

{context}

Question: Following the chain of connections, which entity is located in Paris?
Answer:"""

    def evaluate(self, context_length: int) -> EvalResult:
        """Evaluate multi-hop reasoning capability."""
        start_time = time.time()

        context, _, answer = self.generate_multi_hop_context()

        padded_context = context
        while len(self.tokenizer(padded_context)["input_ids"]) < context_length:
            padded_context += " " + context

        prompt = self.create_prompt(padded_context, "")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        if "cuda" in str(getattr(self.model, "device", "cpu")):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        success = answer in generated
        latency = time.time() - start_time

        return EvalResult(
            task_name="multi_hop_reasoning",
            context_length=input_len,
            success=success,
            metric=1.0 if success else 0.0,
            latency_seconds=latency,
            details={
                "expected": answer,
                "input_tokens": input_len,
            },
        )


class MemoryProfiler:
    """Profile memory usage during inference."""

    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
        }

    @staticmethod
    def reset_stats():
        """Reset CUDA memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()


class LongContextEvaluator:
    """
    Main evaluator class for long-context capabilities.
    """

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_name_or_path}...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=self.device if self.device == "auto" else None,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(
            f"Model loaded. Max position embeddings: {self.model.config.max_position_embeddings}"
        )

    def run_needle_eval(
        self, context_lengths: List[int], num_trials: int = 3
    ) -> List[EvalResult]:
        """Run needle-in-haystack evaluation across different context lengths."""
        needle_task = NeedleInHaystackTask(self.model, self.tokenizer)
        results = []

        print("\n--- Needle-in-Haystack Evaluation ---")
        for ctx_len in context_lengths:
            for trial in range(num_trials):
                position = (trial + 1) / (num_trials + 1)
                result = needle_task.evaluate(ctx_len, needle_position_ratio=position)
                results.append(result)
                status = "PASS" if result.success else "FAIL"
                print(
                    f"  Context {result.context_length:6d} tokens, "
                    f"Position {position:.2f}: {status} "
                    f"(latency: {result.latency_seconds:.2f}s)"
                )

        return results

    def run_profiling_eval(self, context_lengths: List[int]) -> List[Dict[str, Any]]:
        """Profile memory and latency for different context lengths."""
        MemoryProfiler.reset_stats()
        profiling_results = []

        print("\n--- Memory & Latency Profiling ---")
        for ctx_len in context_lengths:
            dummy_text = "This is a test sentence. " * (ctx_len // 5)
            inputs = self.tokenizer(dummy_text, return_tensors="pt")

            if "cuda" in str(self.device):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latency = time.time() - start_time

            mem_stats = MemoryProfiler.get_memory_stats()

            profiling_results.append(
                {"context_length": ctx_len, "latency_seconds": latency, **mem_stats}
            )

            print(
                f"  Context {ctx_len:6d} tokens: "
                f"latency={latency:.2f}s, "
                f"memory={mem_stats.get('allocated_gb', 'N/A'):.2f}GB"
            )

        return profiling_results

    def run_full_evaluation(
        self, context_lengths: List[int] = None, num_needle_trials: int = 3
    ) -> Dict[str, Any]:
        """
        Run full evaluation suite.

        Args:
            context_lengths: List of context lengths to test
            num_needle_trials: Number of trials per context length

        Returns:
            Dictionary with all evaluation results
        """
        if context_lengths is None:
            context_lengths = [512, 1024, 2048, 4096]

        max_model_ctx = self.model.config.max_position_embeddings
        valid_lengths = [l for l in context_lengths if l <= max_model_ctx]

        if len(valid_lengths) < len(context_lengths):
            print(
                f"Warning: Some lengths exceed model max ({max_model_ctx}), skipping those."
            )

        needle_results = self.run_needle_eval(valid_lengths, num_needle_trials)
        profiling_results = self.run_profiling_eval(valid_lengths)

        needle_success_rate = sum(1 for r in needle_results if r.success) / len(
            needle_results
        )

        return {
            "model_name": self.model.name_or_path,
            "max_position_embeddings": max_model_ctx,
            "needle_in_haystack": {
                "results": [vars(r) for r in needle_results],
                "success_rate": needle_success_rate,
            },
            "profiling": profiling_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate long-context model")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path"
    )
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
        help="Context lengths to test in tokens",
    )
    parser.add_argument(
        "--num_trials", type=int, default=3, help="Number of trials per length"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument(
        "--skip_needle", action="store_true", help="Skip needle evaluation"
    )
    parser.add_argument("--skip_profiling", action="store_true", help="Skip profiling")

    args = parser.parse_args()

    print("=" * 60)
    print("Long-Context Evaluation Suite")
    print("=" * 60)

    evaluator = LongContextEvaluator(args.model)

    results = {}

    if not args.skip_needle:
        needle_results = evaluator.run_needle_eval(
            args.context_lengths, args.num_trials
        )
        results["needle"] = [vars(r) for r in needle_results]
        success_rate = sum(1 for r in needle_results if r.success) / len(needle_results)
        print(f"\nNeedle-in-Haystack Success Rate: {success_rate:.2%}")

    if not args.skip_profiling:
        profiling_results = evaluator.run_profiling_eval(args.context_lengths)
        results["profiling"] = profiling_results

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
