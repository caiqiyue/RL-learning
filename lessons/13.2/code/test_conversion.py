"""
Test Conversion and Merging Quality

This module provides utilities to test the quality of converted adapters
and merged models against various metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class ConversionMetrics:
    """Metrics for evaluating adapter conversion quality."""

    dimension_match_rate: float
    weight_similarity: float
    layer_coverage: float
    projection_stability: float

    def to_dict(self) -> dict:
        return {
            "dimension_match_rate": self.dimension_match_rate,
            "weight_similarity": self.weight_similarity,
            "layer_coverage": self.layer_coverage,
            "projection_stability": self.projection_stability,
        }


class DimensionValidator:
    """Validate that converted weights have expected dimensions."""

    def __init__(self, expected_dims: Dict[str, Tuple[int, int]]):
        """
        Args:
            expected_dims: Dictionary mapping weight names to expected (in_dim, out_dim)
        """
        self.expected_dims = expected_dims
        self.errors = []
        self.warnings = []

    def validate(self, converted_weights: Dict[str, torch.Tensor]) -> ConversionMetrics:
        """Validate converted weights.

        Returns:
            ConversionMetrics with validation results
        """
        total_weights = len(self.expected_dims)
        matched_weights = 0
        dimension_errors = 0

        for name, (exp_in, exp_out) in self.expected_dims.items():
            if name not in converted_weights:
                self.errors.append(f"Missing weight: {name}")
                continue

            weight = converted_weights[name]
            act_in, act_out = weight.shape

            if act_in == exp_in and act_out == exp_out:
                matched_weights += 1
            else:
                dimension_errors += 1
                self.errors.append(
                    f"Dimension mismatch for {name}: "
                    f"expected ({exp_in}, {exp_out}), got ({act_in}, {act_out})"
                )

        return ConversionMetrics(
            dimension_match_rate=matched_weights / total_weights
            if total_weights > 0
            else 0.0,
            weight_similarity=1.0
            - (dimension_errors / total_weights if total_weights > 0 else 1.0),
            layer_coverage=matched_weights / len(converted_weights)
            if converted_weights
            else 0.0,
            projection_stability=1.0,
        )


class WeightSimilarityAnalyzer:
    """Analyze similarity between source and converted weights."""

    @staticmethod
    def cosine_similarity(w1: torch.Tensor, w2: torch.Tensor) -> float:
        """Compute cosine similarity between two weight matrices."""
        w1_flat = w1.flatten()
        w2_flat = w2.flatten()

        dot_product = torch.dot(w1_flat, w2_flat)
        norm_product = torch.norm(w1_flat) * torch.norm(w2_flat)

        if norm_product == 0:
            return 0.0

        return (dot_product / norm_product).item()

    @staticmethod
    def euclidean_distance(w1: torch.Tensor, w2: torch.Tensor) -> float:
        """Compute normalized Euclidean distance between two weight matrices."""
        diff = w1.flatten() - w2.flatten()
        norm_diff = torch.norm(diff)
        norm_w1 = torch.norm(w1.flatten())

        if norm_w1 == 0:
            return float("inf")

        return (norm_diff / norm_w1).item()

    @staticmethod
    def spectral_norm(w: torch.Tensor) -> float:
        """Compute spectral norm (largest singular value) of a matrix."""
        if w.dim() != 2:
            w = w.reshape(w.shape[0], -1)

        _, s, _ = torch.svd(w)
        return s[0].item() if len(s) > 0 else 0.0

    def analyze(
        self,
        source_weights: Dict[str, torch.Tensor],
        converted_weights: Dict[str, torch.Tensor],
        layer_mapping: Optional[Dict[int, int]] = None,
    ) -> Dict[str, float]:
        """Analyze similarity between source and converted weights.

        Args:
            source_weights: Original source model weights
            converted_weights: Converted weights
            layer_mapping: Optional mapping from source to target layer indices

        Returns:
            Dictionary of similarity metrics
        """
        similarities = []
        distances = []

        for key in source_weights:
            if layer_mapping:
                source_layer = self._extract_layer_idx(key)
                if source_layer is not None and source_layer in layer_mapping:
                    mapped_key = key.replace(
                        f".{source_layer}.", f".{layer_mapping[source_layer]}."
                    )
                else:
                    mapped_key = key
            else:
                mapped_key = key

            if mapped_key not in converted_weights:
                continue

            source_w = source_weights[key]
            target_w = converted_weights[mapped_key]

            if source_w.shape != target_w.shape:
                continue

            sim = self.cosine_similarity(source_w, target_w)
            dist = self.euclidean_distance(source_w, target_w)

            similarities.append(sim)
            distances.append(dist)

        return {
            "mean_cosine_similarity": np.mean(similarities) if similarities else 0.0,
            "min_cosine_similarity": np.min(similarities) if similarities else 0.0,
            "mean_euclidean_distance": np.mean(distances) if distances else 0.0,
            "max_euclidean_distance": np.max(distances) if distances else 0.0,
            "num_compared_layers": len(similarities),
        }

    def _extract_layer_idx(self, key: str) -> Optional[int]:
        """Extract layer index from weight name."""
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None


class ProjectionStabilityAnalyzer:
    """Analyze the stability of projection operations."""

    @staticmethod
    def check_orthogonality(projection_matrix: torch.Tensor, tol: float = 1e-5) -> bool:
        """Check if projection matrix is orthogonal (P @ P^T ≈ I)."""
        if projection_matrix.shape[0] != projection_matrix.shape[1]:
            return False

        pt_p = torch.matmul(projection_matrix.t(), projection_matrix)
        identity = torch.eye(pt_p.shape[0], device=pt_p.device)

        error = torch.norm(pt_p - identity).item()
        return error < tol

    @staticmethod
    def compute_projection_error(projection_matrix: torch.Tensor) -> float:
        """Compute projection error: ||P @ P^T - I|| / ||I||."""
        if projection_matrix.shape[0] != projection_matrix.shape[1]:
            return float("inf")

        pt_p = torch.matmul(projection_matrix.t(), projection_matrix)
        identity = torch.eye(pt_p.shape[0], device=pt_p.device)

        error = torch.norm(pt_p - identity) / torch.norm(identity)
        return error.item()

    def analyze_projection(self, projection_matrix: torch.Tensor) -> Dict[str, float]:
        """Analyze stability of a projection matrix."""
        return {
            "is_orthogonal": self.check_orthogonality(projection_matrix),
            "projection_error": self.compute_projection_error(projection_matrix),
            "spectral_norm": WeightSimilarityAnalyzer.spectral_norm(projection_matrix),
        }


class MergingQualityAnalyzer:
    """Analyze quality of merged adapters."""

    def __init__(self):
        self.metrics_history = []

    def compute_diversity(
        self, task_vectors: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Compute diversity between task vectors."""
        if len(task_vectors) < 2:
            return {"mean_pairwise_distance": 0.0, "variance": 0.0}

        distances = []
        for i in range(len(task_vectors)):
            for j in range(i + 1, len(task_vectors)):
                dist = self._compute_pairwise_distance(task_vectors[i], task_vectors[j])
                distances.append(dist)

        return {
            "mean_pairwise_distance": np.mean(distances) if distances else 0.0,
            "variance": np.var(distances) if distances else 0.0,
            "num_pairs": len(distances),
        }

    def _compute_pairwise_distance(
        self, tv1: Dict[str, torch.Tensor], tv2: Dict[str, torch.Tensor]
    ) -> float:
        """Compute average Euclidean distance between two task vectors."""
        distances = []
        for key in tv1:
            if key in tv2:
                w1 = tv1[key].flatten()
                w2 = tv2[key].flatten()
                dist = torch.norm(w1 - w2) / (torch.norm(w1) + 1e-8)
                distances.append(dist.item())

        return np.mean(distances) if distances else 0.0

    def compute_conflict_metrics(
        self, task_vectors: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Compute metrics about conflicts between task vectors."""
        if not task_vectors:
            return {"conflict_rate": 0.0, "sign_disagreement": 0.0}

        conflict_count = 0
        sign_disagreements = 0
        total_elements = 0

        keys = task_vectors[0].keys()
        for key in keys:
            stacked = torch.stack([tv[key] for tv in task_vectors])
            signs = torch.sign(stacked)

            sign_sum = signs.sum(dim=0)
            total_elements += signs.numel()

            conflict_count += (
                torch.any(sign_sum == 0).item() * signs.shape[1]
                if signs.dim() > 1
                else 0
            )

            disagreements = torch.any(signs != signs[0:1], dim=0)
            sign_disagreements += disagreements.sum().item()

        return {
            "conflict_rate": conflict_count / total_elements
            if total_elements > 0
            else 0.0,
            "sign_disagreement": sign_disagreements / total_elements
            if total_elements > 0
            else 0.0,
        }

    def evaluate_merge_result(
        self,
        merged: Dict[str, torch.Tensor],
        task_vectors: List[Dict[str, torch.Tensor]],
        base_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Comprehensive evaluation of merge result."""
        metrics = {}

        diversity = self.compute_diversity(task_vectors)
        metrics.update({f"diversity_{k}": v for k, v in diversity.items()})

        conflict = self.compute_conflict_metrics(task_vectors)
        metrics.update({f"conflict_{k}": v for k, v in conflict.items()})

        if base_weights is not None:
            magnitude_ratio = self._compute_magnitude_ratio(merged, base_weights)
            metrics["magnitude_ratio"] = magnitude_ratio

        return metrics

    def _compute_magnitude_ratio(
        self, merged: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]
    ) -> float:
        """Compute ratio of merged delta magnitude to base model magnitude."""
        merged_norm = sum(torch.norm(v.flatten()) for v in merged.values())
        base_norm = sum(torch.norm(v.flatten()) for v in base.values())

        if base_norm == 0:
            return float("inf")

        return (merged_norm / base_norm).item()


class AdapterTestSuite:
    """Comprehensive test suite for adapter conversion and merging."""

    def __init__(self):
        self.validator = None
        self.similarity_analyzer = WeightSimilarityAnalyzer()
        self.merging_analyzer = MergingQualityAnalyzer()
        self.results = []

    def test_dimension_conversion(
        self,
        converted_weights: Dict[str, torch.Tensor],
        expected_dims: Dict[str, Tuple[int, int]],
    ) -> Dict:
        """Test that all converted weights have correct dimensions."""
        validator = DimensionValidator(expected_dims)
        metrics = validator.validate(converted_weights)

        return {
            "test_name": "dimension_conversion",
            "passed": metrics.dimension_match_rate == 1.0,
            "metrics": metrics.to_dict(),
            "errors": validator.errors,
        }

    def test_weight_similarity(
        self,
        source_weights: Dict[str, torch.Tensor],
        converted_weights: Dict[str, torch.Tensor],
    ) -> Dict:
        """Test similarity between source and converted weights."""
        similarity_metrics = self.similarity_analyzer.analyze(
            source_weights, converted_weights
        )

        return {
            "test_name": "weight_similarity",
            "passed": similarity_metrics["mean_cosine_similarity"] > 0.8,
            "metrics": similarity_metrics,
        }

    def test_projection_stability(
        self, projection_matrices: List[torch.Tensor]
    ) -> Dict:
        """Test stability of projection matrices."""
        projection_analyzer = ProjectionStabilityAnalyzer()

        results = []
        for i, pm in enumerate(projection_matrices):
            analysis = projection_analyzer.analyze_projection(pm)
            results.append({f"projection_{i}": analysis})

        all_orthogonal = all(
            r[f"projection_{i}"]["is_orthogonal"]
            for i, r in enumerate(results)
            if f"projection_{i}" in r
        )

        return {
            "test_name": "projection_stability",
            "passed": all_orthogonal,
            "metrics": results,
        }

    def test_merging_quality(
        self,
        merged: Dict[str, torch.Tensor],
        task_vectors: List[Dict[str, torch.Tensor]],
        base_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict:
        """Test quality of merged adapters."""
        quality_metrics = self.merging_analyzer.evaluate_merge_result(
            merged, task_vectors, base_weights
        )

        return {
            "test_name": "merging_quality",
            "passed": quality_metrics.get("magnitude_ratio", 0) < 0.1,
            "metrics": quality_metrics,
        }

    def run_all_tests(
        self,
        source_weights: Optional[Dict[str, torch.Tensor]] = None,
        converted_weights: Optional[Dict[str, torch.Tensor]] = None,
        expected_dims: Optional[Dict[str, Tuple[int, int]]] = None,
        task_vectors: Optional[List[Dict[str, torch.Tensor]]] = None,
        merged: Optional[Dict[str, torch.Tensor]] = None,
        base_weights: Optional[Dict[str, torch.Tensor]] = None,
        projection_matrices: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """Run all applicable tests.

        Returns:
            Dictionary with test results
        """
        results = []

        if expected_dims is not None and converted_weights is not None:
            result = self.test_dimension_conversion(converted_weights, expected_dims)
            results.append(result)

        if source_weights is not None and converted_weights is not None:
            result = self.test_weight_similarity(source_weights, converted_weights)
            results.append(result)

        if projection_matrices is not None:
            result = self.test_projection_stability(projection_matrices)
            results.append(result)

        if task_vectors is not None and merged is not None:
            result = self.test_merging_quality(merged, task_vectors, base_weights)
            results.append(result)

        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["passed"])

        return {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            },
            "detailed_results": results,
        }


def run_quick_tests(
    source_weights: Dict[str, torch.Tensor],
    converted_weights: Dict[str, torch.Tensor],
    expected_dims: Dict[str, Tuple[int, int]],
) -> Dict:
    """Run quick validation tests without full test suite."""
    validator = DimensionValidator(expected_dims)
    metrics = validator.validate(converted_weights)

    analyzer = WeightSimilarityAnalyzer()
    similarity = analyzer.analyze(source_weights, converted_weights)

    return {
        "dimension_match_rate": metrics.dimension_match_rate,
        "weight_similarity": similarity["mean_cosine_similarity"],
        "errors": validator.errors,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Adapter Conversion Testing Module")
    print("=" * 60)
    print("\nThis module provides testing utilities for:")
    print("  - Dimension validation")
    print("  - Weight similarity analysis")
    print("  - Projection stability analysis")
    print("  - Merging quality evaluation")
    print("\nUsage example:")
    print("""
    from test_conversion import AdapterTestSuite, run_quick_tests
    
    test_suite = AdapterTestSuite()
    results = test_suite.run_all_tests(
        source_weights=source_weights,
        converted_weights=converted_weights,
        expected_dims=expected_dims
    )
    """)
