#!/usr/bin/env python3
"""
Synthetic Preference Generation Script

Generate synthetic preference pairs using an LLM.
Supports multiple generation strategies: self-improvement, pairwise comparison, etc.

Usage:
    python generate_synthetic_prefs.py --prompts data/prompts.jsonl --output data/synthetic_prefs.jsonl
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    num_candidates: int = 4
    temperature_range: Tuple[float, float] = (0.5, 1.5)
    improvement_iterations: int = 1
    critique_threshold: float = 0.6


def load_prompts(file_path: str) -> List[Dict]:
    """Load prompts from JSONL or JSON file."""
    path = Path(file_path)

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get("prompts", [])
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_preferences(preferences: List[Dict], file_path: str):
    """Save preference pairs to JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in preferences:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class MockModel:
    """
    Mock model for testing without actual LLM API.
    Replace with actual API client in production.
    """

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """Simulate model generation."""
        self.call_count += 1
        temperature = kwargs.get("temperature", 0.8)
        num_return = kwargs.get("num_return", 1)

        responses = []
        for i in range(num_return):
            responses.append(
                f"[Response {i + 1} from {self.model_name} | temp={temperature:.2f}] "
                f"This is a mock response to: {prompt[:50]}..."
            )

        return responses if num_return > 1 else responses[0]

    def rank(self, prompt: str, candidates: List[str]) -> List[float]:
        """Simulate ranking of candidates."""
        return [np.random.random() for _ in candidates]

    def critique(self, prompt: str, response: str) -> str:
        """Simulate critique generation."""
        self.call_count += 1
        critiques = [
            "The response lacks specific examples.",
            "Good detail level but could be more concise.",
            "Accurate but misses some key points.",
            "Well-structured with clear explanations.",
        ]
        return np.random.choice(critiques)

    def evaluate(self, prompt: str, response_a: str, response_b: str) -> str:
        """Simulate pairwise evaluation."""
        self.call_count += 1
        import random

        choice = random.choice(["A", "B", "tie"])
        if choice == "A":
            return "Response A is better."
        elif choice == "B":
            return "Response B is better."
        else:
            return "Both responses are equally good."


class SyntheticPreferenceGenerator:
    """
    Generate synthetic preference pairs using LLM feedback.
    """

    def __init__(self, model, config: GenerationConfig = GenerationConfig()):
        self.model = model
        self.config = config

    def generate_candidates_temperature(
        self, prompt: str, num_candidates: Optional[int] = None
    ) -> List[str]:
        """
        Generate diverse candidates using temperature sampling.
        """
        if num_candidates is None:
            num_candidates = self.config.num_candidates

        temperatures = np.linspace(
            self.config.temperature_range[0],
            self.config.temperature_range[1],
            num_candidates,
        )

        candidates = []
        for temp in temperatures:
            response = self.model.generate(
                prompt, temperature=float(temp), num_return=1
            )
            candidates.append(response)

        return candidates

    def rank_candidates(
        self, prompt: str, candidates: List[str]
    ) -> List[Tuple[float, str]]:
        """
        Rank candidates using model scoring.
        """
        scores = self.model.rank(prompt, candidates)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0])
        return ranked

    def generate_self_improvement_pair(self, prompt: str) -> Optional[Dict]:
        """
        Generate preference pair via self-improvement (SPIN-style).

        Process:
        1. Generate initial response
        2. Model critiques and improves its own response
        3. Preference pair: (improved, initial)
        """
        initial_response = self.model.generate(prompt, temperature=0.7)

        current_best = initial_response

        for iteration in range(self.config.improvement_iterations):
            critique_prompt = f"""
Original prompt: {prompt}

Current response: {current_best}

Please provide constructive critique identifying weaknesses and areas for improvement.
"""
            critique = self.model.critique(critique_prompt, current_best)

            improvement_prompt = f"""
Original prompt: {prompt}

Current response: {current_best}

Critique: {critique}

Please generate an improved version that addresses the critique while maintaining quality.
"""
            improved_response = self.model.generate(improvement_prompt, temperature=0.7)

            if iteration < self.config.improvement_iterations - 1:
                current_best = improved_response

        if current_best == initial_response:
            return None

        return {
            "prompt": prompt,
            "chosen": current_best,
            "rejected": initial_response,
            "generation_method": "self_improvement",
        }

    def generate_pairwise_preference(
        self, prompt: str, num_candidates: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Generate preference pair via pairwise comparison.

        Process:
        1. Generate multiple candidates with different temperatures
        2. Rank candidates
        3. Preference pair: (best, worst)
        """
        if num_candidates is None:
            num_candidates = self.config.num_candidates

        candidates = self.generate_candidates_temperature(prompt, num_candidates)

        if len(set(candidates)) < 2:
            logger.warning(f"All candidates identical for prompt: {prompt[:50]}")
            return None

        ranked = self.rank_candidates(prompt, candidates)
        worst = ranked[0][1]
        best = ranked[-1][1]

        if best == worst:
            return None

        return {
            "prompt": prompt,
            "chosen": best,
            "rejected": worst,
            "generation_method": "pairwise_ranking",
        }

    def generate_rlaif_preference(
        self, prompt: str, response_a: str, response_b: str
    ) -> Optional[Dict]:
        """
        Generate preference using RLAIF (AI feedback).

        Process:
        1. Evaluate two responses
        2. Return preference based on evaluation
        """
        evaluation = self.model.evaluate(prompt, response_a, response_b)

        if "A is better" in evaluation:
            return {
                "prompt": prompt,
                "chosen": response_a,
                "rejected": response_b,
                "generation_method": "rlaif",
            }
        elif "B is better" in evaluation:
            return {
                "prompt": prompt,
                "chosen": response_b,
                "rejected": response_a,
                "generation_method": "rlaif",
            }
        else:
            return None

    def generate_synthetic_dataset(
        self,
        prompts: List[Dict],
        method: str = "pairwise",
        method_probs: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Generate synthetic preference dataset.

        Args:
            prompts: List of prompt dictionaries with 'text' or 'prompt' field
            method: Generation method ('pairwise', 'self_improvement', 'mixed')
            method_probs: Probability distribution for mixed method
        """
        synthetic_pairs = []
        stats = {"total": 0, "success": 0, "failed": 0, "methods": {}}

        if method_probs is None:
            method_probs = {"pairwise": 0.7, "self_improvement": 0.3}

        for i, item in enumerate(prompts):
            prompt = item.get("text", item.get("prompt", ""))
            if not prompt:
                continue

            stats["total"] += 1

            if method == "pairwise":
                result = self.generate_pairwise_preference(prompt)
            elif method == "self_improvement":
                result = self.generate_self_improvement_pair(prompt)
            elif method == "mixed":
                method_choice = np.random.choice(
                    list(method_probs.keys()), p=list(method_probs.values())
                )

                if method_choice == "pairwise":
                    result = self.generate_pairwise_preference(prompt)
                else:
                    result = self.generate_self_improvement_pair(prompt)

                stats["methods"][method_choice] = (
                    stats["methods"].get(method_choice, 0) + 1
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            if result:
                synthetic_pairs.append(result)
                stats["success"] += 1
                stats["methods"]["success"] = stats["methods"].get("success", 0) + 1
            else:
                stats["failed"] += 1

            if (i + 1) % 100 == 0:
                logger.info(
                    f"Progress: {i + 1}/{len(prompts)} - Success rate: {stats['success'] / stats['total']:.2%}"
                )

        logger.info(f"Generation complete: {stats}")
        return synthetic_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic preference pairs")
    parser.add_argument("--prompts", "-p", required=True, help="Input prompts file")
    parser.add_argument("--output", "-o", required=True, help="Output preferences file")
    parser.add_argument(
        "--method",
        default="pairwise",
        choices=["pairwise", "self_improvement", "mixed"],
        help="Generation method",
    )
    parser.add_argument(
        "--num_candidates", type=int, default=4, help="Number of candidates per prompt"
    )
    parser.add_argument(
        "--temperature_min", type=float, default=0.5, help="Min temperature"
    )
    parser.add_argument(
        "--temperature_max", type=float, default=1.5, help="Max temperature"
    )
    parser.add_argument(
        "--improvement_iterations",
        type=int,
        default=1,
        help="Self-improvement iterations",
    )
    parser.add_argument(
        "--use_mock_model", action="store_true", help="Use mock model for testing"
    )

    args = parser.parse_args()

    logger.info(f"Loading prompts from {args.prompts}")
    prompts = load_prompts(args.prompts)
    logger.info(f"Loaded {len(prompts)} prompts")

    if args.use_mock_model:
        logger.info("Using mock model")
        model = MockModel()
    else:
        logger.warning(
            "No actual LLM API configured. Use --use_mock_model for testing."
        )
        logger.info("Falling back to mock model")
        model = MockModel()

    config = GenerationConfig(
        num_candidates=args.num_candidates,
        temperature_range=(args.temperature_min, args.temperature_max),
        improvement_iterations=args.improvement_iterations,
    )

    generator = SyntheticPreferenceGenerator(model, config)

    logger.info(f"Generating synthetic preferences using method: {args.method}")
    synthetic_prefs = generator.generate_synthetic_dataset(prompts, method=args.method)

    logger.info(f"Generated {len(synthetic_prefs)} preference pairs")

    logger.info(f"Saving to {args.output}")
    save_preferences(synthetic_prefs, args.output)
    logger.info("Done!")


if __name__ == "__main__":
    main()
