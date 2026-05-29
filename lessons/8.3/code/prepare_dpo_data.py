#!/usr/bin/env python3
"""
DPO Preference Data Preparation Script

Prepare and convert preference datasets to DPO format.
Supports format conversion, quality filtering, and length balancing.

Usage:
    python prepare_dpo_data.py --input data/rlhf_raw.jsonl --output data/dpo_format.jsonl
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    error_message: Optional[str] = None


def validate_dpo_format(example: Dict) -> ValidationResult:
    """
    Validate that a preference pair has the required DPO format fields.

    Required fields: prompt, chosen, rejected
    """
    required_keys = {"prompt", "chosen", "rejected"}

    if not required_keys.issubset(example.keys()):
        missing = required_keys - set(example.keys())
        return ValidationResult(False, f"Missing required keys: {missing}")

    if not example["prompt"].strip() if isinstance(example["prompt"], str) else True:
        return ValidationResult(False, "Prompt is empty")

    if not example["chosen"].strip():
        return ValidationResult(False, "Chosen response is empty")

    if not example["rejected"].strip():
        return ValidationResult(False, "Rejected response is empty")

    if example["chosen"] == example["rejected"]:
        return ValidationResult(False, "Chosen and rejected responses are identical")

    return ValidationResult(True)


def convert_to_dpo_format(
    raw_data: List[Dict], input_format: str = "rlhf"
) -> List[Dict]:
    """
    Convert various formats to standard DPO format.

    Supported input formats:
    - "rlhf": Standard {chosen, rejected} format
    - "ranked": {prompt, responses: [r1, r2, ...]} - uses best and worst
    - "score": {prompt, response, score} - pairs with reference
    """
    dpo_data = []

    if input_format == "rlhf":
        for item in raw_data:
            prompt = item.get("prompt", "")
            if isinstance(prompt, list):
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in prompt])

            dpo_data.append(
                {
                    "prompt": prompt,
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                }
            )

    elif input_format == "ranked":
        for item in raw_data:
            responses = item.get("responses", [])
            if len(responses) < 2:
                logger.warning(f"Skipping item with fewer than 2 responses: {item}")
                continue

            ranked = sorted(responses)
            prompt = item.get("prompt", "")

            dpo_data.append(
                {"prompt": prompt, "chosen": ranked[-1], "rejected": ranked[0]}
            )

    elif input_format == "score":
        for item in raw_data:
            dpo_data.append(
                {
                    "prompt": item["prompt"],
                    "chosen": item["response"],
                    "rejected": item.get("reference_response", item["response"]),
                }
            )

    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    return dpo_data


def is_repetitive(text: str, n_gram_threshold: int = 3) -> bool:
    """
    Detect repetitive text using simple n-gram analysis.
    Returns True if text contains excessive repetition.
    """
    words = text.split()
    if len(words) < n_gram_threshold * 2:
        return False

    for n in range(2, n_gram_threshold + 1):
        n_grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if len(n_grams) != len(set(n_grams)):
            return True
    return False


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute simple word overlap similarity between two texts.
    Returns Jaccard similarity coefficient.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    intersection = words1 & words2
    union = words1 | words2

    if not union:
        return 0.0
    return len(intersection) / len(union)


def filter_preference_pairs(
    examples: List[Dict],
    min_length: int = 10,
    max_length_ratio: float = 5.0,
    max_examples: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Filter preference pairs based on quality criteria.

    Filters:
    1. Response length must exceed minimum word count
    2. Length ratio between chosen and rejected must be reasonable
    3. Text should not be overly repetitive
    """
    filtered = []
    stats = {
        "total_input": len(examples),
        "filtered_empty": 0,
        "filtered_short": 0,
        "filtered_length_ratio": 0,
        "filtered_repetitive": 0,
        "final_count": 0,
    }

    for example in examples:
        chosen = example["chosen"]
        rejected = example["rejected"]

        if not chosen.strip() or not rejected.strip():
            stats["filtered_empty"] += 1
            continue

        chosen_words = len(chosen.split())
        rejected_words = len(rejected.split())

        if chosen_words < min_length or rejected_words < min_length:
            stats["filtered_short"] += 1
            continue

        len_ratio = max(chosen_words, rejected_words) / (
            min(chosen_words, rejected_words) + 1
        )
        if len_ratio > max_length_ratio:
            stats["filtered_length_ratio"] += 1
            continue

        if is_repetitive(chosen) or is_repetitive(rejected):
            stats["filtered_repetitive"] += 1
            continue

        filtered.append(example)

    if max_examples and len(filtered) > max_examples:
        filtered = filtered[:max_examples]

    stats["final_count"] = len(filtered)
    return filtered, stats


def balance_by_length(examples: List[Dict], max_ratio: float = 3.0) -> List[Dict]:
    """
    Balance preference pairs by response length to reduce length bias.

    Problem: If chosen responses are always longer than rejected,
    DPO may learn "longer = better" instead of "better quality".
    """
    balanced = []

    for example in examples:
        chosen_len = len(example["chosen"].split())
        rejected_len = len(example["rejected"].split())

        ratio = max(chosen_len, rejected_len) / (min(chosen_len, rejected_len) + 1)
        if ratio <= max_ratio:
            balanced.append(example)

    return balanced


def handle_tied_preferences(
    examples: List[Dict], similarity_threshold: float = 0.95
) -> Tuple[List[Dict], List[Dict]]:
    """
    Handle ambiguous or tied preferences.

    Returns:
        - filtered: Examples with clear preference
        - uncertain: Examples where responses are too similar
    """
    filtered = []
    uncertain = []

    for example in examples:
        chosen = example["chosen"]
        rejected = example["rejected"]

        similarity = compute_text_similarity(chosen, rejected)

        if similarity > similarity_threshold:
            uncertain.append({**example, "uncertainty_score": similarity})
        else:
            filtered.append(example)

    return filtered, uncertain


def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSONL or JSON file."""
    path = Path(file_path)

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_dataset(data: List[Dict], file_path: str):
    """Save dataset to JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO preference data")
    parser.add_argument("--input", "-i", required=True, help="Input data file")
    parser.add_argument("--output", "-o", required=True, help="Output DPO format file")
    parser.add_argument(
        "--input-format",
        default="rlhf",
        choices=["rlhf", "ranked", "score"],
        help="Input data format",
    )
    parser.add_argument(
        "--min-length", type=int, default=10, help="Minimum response word count"
    )
    parser.add_argument(
        "--max-length-ratio", type=float, default=5.0, help="Maximum length ratio"
    )
    parser.add_argument(
        "--balance-length", action="store_true", help="Apply length balancing"
    )
    parser.add_argument(
        "--max-examples", type=int, default=None, help="Maximum number of examples"
    )

    args = parser.parse_args()

    logger.info(f"Loading data from {args.input}")
    raw_data = load_dataset(args.input)
    logger.info(f"Loaded {len(raw_data)} raw examples")

    logger.info(f"Converting to DPO format (input format: {args.input_format})")
    dpo_data = convert_to_dpo_format(raw_data, args.input_format)
    logger.info(f"Converted {len(dpo_data)} examples to DPO format")

    logger.info("Filtering low-quality preference pairs")
    filtered_data, stats = filter_preference_pairs(
        dpo_data, min_length=args.min_length, max_length_ratio=args.max_length_ratio
    )
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    if args.balance_length:
        logger.info("Applying length balancing")
        balanced_data = balance_by_length(filtered_data)
        logger.info(f"  Balance filtered: {len(filtered_data)} -> {len(balanced_data)}")
        filtered_data = balanced_data

    filtered_data, _ = handle_tied_preferences(filtered_data)
    logger.info(f"Final dataset size: {len(filtered_data)}")

    logger.info(f"Saving to {args.output}")
    save_dataset(filtered_data, args.output)
    logger.info("Done!")


if __name__ == "__main__":
    main()
