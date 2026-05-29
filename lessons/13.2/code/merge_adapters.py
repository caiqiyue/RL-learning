"""
Multiple Adapter Merging Strategies

This module implements various adapter merging strategies:
1. Simple Average: Equal weight averaging
2. Task Vector: Direction-based merging
3. TIES-Merging: Conflict resolution via majority voting
4. DARE: Drop and Rescale for sparse merging
5. WARM: Fisher-weighted averaging

Reference papers:
- Task Vectors: Efficient Steering of LLMs (2024)
- TIES-Merging: Task Vector Ensemble with Intelligent Sign Resolution (2023)
- WARM: Weight Averaging meets Fisher (2023)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class MergeConfig:
    """Configuration for merging strategy."""

    method: str = "average"
    density: float = 1.0
    weight_scale: float = 1.0
    temperature: float = 1.0
    drop_prob: float = 0.5
    seed: Optional[int] = None


class SimpleAverageMerger:
    """Simple averaging of multiple task vectors."""

    def __init__(self, weights: Optional[List[float]] = None):
        """
        Args:
            weights: Optional per-model weights. If None, equal weights are used.
        """
        self.weights = weights

    def merge(
        self, task_vectors: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Merge task vectors using weighted average.

        Args:
            task_vectors: List of dictionaries containing weight updates

        Returns:
            Merged weights dictionary
        """
        if not task_vectors:
            return {}

        if self.weights is None:
            w = [1.0 / len(task_vectors)] * len(task_vectors)
        else:
            w = [wt / sum(self.weights) for wt in self.weights]

        merged = {}
        for key in task_vectors[0].keys():
            stacked = torch.stack([tv[key] for tv in task_vectors])
            weighted = sum(w[i] * stacked[i] for i in range(len(task_vectors)))
            merged[key] = weighted

        return merged


class TaskVectorMerger:
    """Task vector based merging using direction and magnitude."""

    def __init__(self, base_weights: Optional[Dict[str, torch.Tensor]] = None):
        """
        Args:
            base_weights: Base model weights (for computing task vectors).
                        If None, assumes inputs are already task vectors.
        """
        self.base_weights = base_weights

    def _compute_task_vector(
        self,
        model_weights: Dict[str, torch.Tensor],
        base_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute task vector (model - base)."""
        if base_weights is None:
            base_weights = self.base_weights

        if base_weights is None:
            return model_weights

        return {k: model_weights[k] - base_weights[k] for k in model_weights.keys()}

    def merge(
        self,
        model_weights_list: List[Dict[str, torch.Tensor]],
        base_weights: Optional[Dict[str, torch.Tensor]] = None,
        strategy: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """Merge model weights using task vector approach.

        Args:
            model_weights_list: List of fine-tuned model weights
            base_weights: Base (pre-trained) weights
            strategy: One of 'mean', 'mean_magnitude', 'direction'

        Returns:
            Merged weights
        """
        if not model_weights_list:
            return {}

        task_vectors = [
            self._compute_task_vector(w, base_weights) for w in model_weights_list
        ]

        if strategy == "mean":
            merger = SimpleAverageMerger()
            merged_tv = merger.merge(task_vectors)
        elif strategy == "mean_magnitude":
            merged_tv = self._mean_magnitude_merge(task_vectors)
        elif strategy == "direction":
            merged_tv = self._direction_merge(task_vectors)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if base_weights is not None:
            return {k: base_weights[k] + merged_tv[k] for k in base_weights.keys()}
        else:
            return merged_tv

    def _mean_magnitude_merge(
        self, task_vectors: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Merge by averaging magnitudes while keeping sign."""
        merged = {}
        for key in task_vectors[0].keys():
            stacked = torch.stack([tv[key] for tv in task_vectors])
            signs = torch.sign(stacked.sum(dim=0))
            magnitudes = stacked.abs().mean(dim=0)
            merged[key] = signs * magnitudes
        return merged

    def _direction_merge(
        self, task_vectors: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Merge based on dominant direction."""
        merged = {}
        for key in task_vectors[0].keys():
            stacked = torch.stack([tv[key] for tv in task_vectors])
            mean_direction = stacked.mean(dim=0)
            dominant_sign = torch.sign(mean_direction)
            merged[key] = dominant_sign * stacked.abs().mean(dim=0)
        return merged


class TIESMerger:
    """TIES-Merging: Task Vector Ensemble with Intelligent Sign Resolution.

    Three-step process:
    1. Select reference vector (mean direction)
    2. Sign resolution (majority vote)
    3. Magnitude resolution (keep most salient)
    """

    def __init__(self):
        pass

    def merge(
        self,
        task_vectors: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Merge task vectors using TIES-Merging strategy.

        Args:
            task_vectors: List of task vectors (weight updates)
            weights: Optional weights for each task vector

        Returns:
            Merged weights dictionary
        """
        if not task_vectors:
            return {}

        if weights is None:
            weights = [1.0] * len(task_vectors)

        merged = {}
        for key in task_vectors[0].keys():
            stacked = torch.stack([tv[key] * w for tv, w in zip(task_vectors, weights)])

            mean_direction = stacked.sum(dim=0)

            sign_votes = torch.sign(mean_direction)
            sign_votes = torch.where(
                sign_votes == 0, torch.ones_like(sign_votes), sign_votes
            )

            masked = stacked * sign_votes.unsqueeze(0)
            magnitudes = masked.abs()

            merged[key] = magnitudes.max(dim=0)[0] * sign_votes

        return merged


class DAREMerger:
    """DARE: Drop And Rescale for sparse merging.

    Strategy:
    1. Randomly drop weights with probability p
    2. Rescale remaining weights by factor 1/(1-p)
    3. Average the result
    """

    def __init__(self, drop_prob: float = 0.5, seed: Optional[int] = None):
        """
        Args:
            drop_prob: Probability of dropping each weight
            seed: Random seed for reproducibility
        """
        self.drop_prob = drop_prob
        self.seed = seed

    def merge(
        self,
        task_vectors: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Merge task vectors using DARE strategy.

        Args:
            task_vectors: List of task vectors
            weights: Optional weights for each task vector

        Returns:
            Merged weights dictionary
        """
        if not task_vectors:
            return {}

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        scale_factor = 1.0 / (1.0 - self.drop_prob)

        if weights is None:
            weights = [1.0] * len(task_vectors)
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]

        merged = {}
        for key in task_vectors[0].keys():
            result = torch.zeros_like(task_vectors[0][key])

            for tv, w in zip(task_vectors, normalized_weights):
                weight = tv[key]
                mask = torch.rand_like(weight) > self.drop_prob
                scaled = weight * mask.float() * scale_factor
                result += w * scaled

            merged[key] = result

        return merged


class WARMerger:
    """WARM: Fisher-weighted Averaging using Fisher Information.

    Uses Fisher information matrix to weight parameters based on their
    importance for the task.

    Note: In practice, Fisher information is often approximated using
    gradient statistics or assumed diagonal.
    """

    def __init__(self, use_diagonal_fisher: bool = True):
        """
        Args:
            use_diagonal_fisher: If True, use diagonal approximation of Fisher
        """
        self.use_diagonal_fisher = use_diagonal_fisher

    def merge(
        self,
        task_vectors: List[Dict[str, torch.Tensor]],
        fisher_weights: Optional[List[torch.Tensor]] = None,
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Merge task vectors using WARM strategy.

        Args:
            task_vectors: List of task vectors
            fisher_weights: Per-task Fisher information (diagonal)
            weights: Fallback weights if Fisher info not available

        Returns:
            Merged weights dictionary
        """
        if not task_vectors:
            return {}

        if fisher_weights is None:
            if weights is None:
                weights = [1.0] * len(task_vectors)
            merger = SimpleAverageMerger(weights)
            return merger.merge(task_vectors)

        merged = {}
        for key in task_vectors[0].keys():
            stacked = torch.stack([tv[key] for tv in task_vectors])

            stacked_fisher = torch.stack([fw for fw in fisher_weights])

            normalized_fisher = stacked_fisher / stacked_fisher.sum(dim=0, keepdim=True)

            weights_expanded = (
                normalized_fisher.unsqueeze(-1)
                if normalized_fisher.dim() > 1
                else normalized_fisher.unsqueeze(-1).unsqueeze(-1)
            )

            if weights_expanded.shape[-1] == 1 and stacked.shape[-1] > 1:
                weights_expanded = weights_expanded.expand_as(stacked)

            merged[key] = (weights_expanded * stacked).sum(dim=0)

        return merged


class AdapterMerger:
    """Unified interface for all merging strategies."""

    STRATEGIES = {
        "average": SimpleAverageMerger,
        "task_vector": TaskVectorMerger,
        "ties": TIESMerger,
        "dare": DAREMerger,
        "warm": WARMerger,
    }

    def __init__(self, strategy: str = "average", **kwargs):
        """
        Args:
            strategy: One of 'average', 'task_vector', 'ties', 'dare', 'warm'
            **kwargs: Additional arguments passed to the specific merger
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")

        merger_class = self.STRATEGIES[strategy]

        if strategy == "dare":
            self.merger = DAREMerger(
                drop_prob=kwargs.get("drop_prob", 0.5), seed=kwargs.get("seed", None)
            )
        elif strategy == "warm":
            self.merger = WARMerger(
                use_diagonal_fisher=kwargs.get("use_diagonal_fisher", True)
            )
        elif strategy == "average":
            self.merger = SimpleAverageMerger(weights=kwargs.get("weights", None))
        elif strategy == "task_vector":
            self.merger = TaskVectorMerger(
                base_weights=kwargs.get("base_weights", None)
            )
        else:
            self.merger = merger_class()

    def merge(
        self, task_vectors: List[Dict[str, torch.Tensor]], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Merge task vectors using the configured strategy.

        Args:
            task_vectors: List of weight dictionaries to merge
            **kwargs: Additional arguments specific to each strategy

        Returns:
            Merged weights dictionary
        """
        return self.merger.merge(task_vectors, **kwargs)


def create_merger(strategy: str = "average", **kwargs) -> AdapterMerger:
    """Factory function to create a merger with specified strategy."""
    return AdapterMerger(strategy=strategy, **kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("Adapter Merging Strategies Module")
    print("=" * 60)
    print("\nAvailable strategies:")
    print("  1. average     - Simple weighted averaging")
    print("  2. task_vector  - Task vector based merging")
    print("  3. ties         - TIES-Merging with sign resolution")
    print("  4. dare         - Drop and Rescale")
    print("  5. warm         - Fisher-weighted averaging")
    print("\nUsage example:")
    print("""
    from merge_adapters import create_merger
    
    task_vectors = [delta_weights_1, delta_weights_2, delta_weights_3]
    
    merger = create_merger('ties')
    merged = merger.merge(task_vectors)
    
    merger = create_merger('dare', drop_prob=0.3, seed=42)
    merged = merger.merge(task_vectors)
    """)
