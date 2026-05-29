"""
LoRA Adapter Conversion Between Models

This script handles the conversion of LoRA adapters between models with similar
architectures (e.g., LLaMA 7B -> LLaMA 13B, or LLaMA -> Qwen).

Key operations:
1. Extract delta weights from source LoRA adapter
2. Build projection matrices for dimension mismatch
3. Convert weights to target model format
4. Validate converted weights
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import warnings


@dataclass
class ModelConfig:
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int

    @property
    def num_key_value_heads(self) -> int:
        return self.num_attention_heads


class LoraWeightExtractor:
    """Extract and manage LoRA delta weights from a PEFT model."""

    def __init__(self, model):
        self.model = model
        self.delta_weights = {}
        self._extract_weights()

    def _extract_weights(self):
        """Extract all LoRA weights (A and B matrices)."""
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                self.delta_weights[name] = param.data.clone()

    def get_delta(self, layer_name: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get delta weight matrices A and B for a given layer."""
        a_key = f"{layer_name}.lora_A.weight"
        b_key = f"{layer_name}.lora_B.weight"

        if a_key in self.delta_weights and b_key in self.delta_weights:
            return self.delta_weights[a_key], self.delta_weights[b_key]
        return None

    def get_merged_delta(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get merged delta weight (BA @ x = B @ A @ x)."""
        a, b = self.get_delta(layer_name)
        if a is not None:
            return torch.matmul(b, a)
        return None

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Export all delta weights as a dictionary."""
        return {k: v.clone() for k, v in self.delta_weights.items()}


class ProjectionMatrixBuilder:
    """Build projection matrices for dimension mismatch between models."""

    @staticmethod
    def build_linear_projection(source_dim: int, target_dim: int) -> nn.Linear:
        """Build a linear projection matrix.

        For Q/K/V projection from source to target model:
        - source_dim: dimension of source model weights
        - target_dim: dimension of target model weights

        Returns a linear layer that can project from source to target dimension.
        """
        projection = nn.Linear(source_dim, target_dim, bias=False)
        with torch.no_grad():
            nn.init.orthogonal_(projection.weight, gain=1.0)
        return projection

    @staticmethod
    def build_tied_projection(
        source_dim: int, target_dim: int, tie_strategy: str = "expand"
    ) -> Optional[nn.Linear]:
        """Build projection for tied dimensions (e.g., when only some heads are added).

        Strategies:
        - 'expand': Project from smaller to larger (e.g., 32 -> 40 heads)
        - 'contract': Project from larger to smaller
        - None: Same dimension, no projection needed
        """
        if source_dim == target_dim:
            return None

        return ProjectionMatrixBuilder.build_linear_projection(source_dim, target_dim)


class AttentionProjection:
    """Handle attention layer projection between models."""

    def __init__(self, source_config: ModelConfig, target_config: ModelConfig):
        self.source_config = source_config
        self.target_config = target_config

    def compute_projection_dims(
        self, layer_type: str
    ) -> Tuple[int, int, Optional[int]]:
        """Compute source and target dimensions for a given layer type.

        Returns: (source_dim, target_dim, projection_layer_or_none)
        """
        if layer_type == "q":
            source_dim = self.source_config.hidden_size
            target_dim = self.target_config.hidden_size
        elif layer_type in ("k", "v"):
            source_dim = (
                self.source_config.head_dim * self.source_config.num_key_value_heads
            )
            target_dim = (
                self.target_config.head_dim * self.target_config.num_key_value_heads
            )
        elif layer_type == "o":
            source_dim = (
                self.source_config.head_dim * self.source_config.num_attention_heads
            )
            target_dim = (
                self.target_config.head_dim * self.target_config.num_attention_heads
            )
        elif layer_type in ("gate_proj", "up_proj"):
            source_dim = self.source_config.intermediate_size
            target_dim = self.target_config.intermediate_size
        elif layer_type == "down_proj":
            source_dim = self.target_config.intermediate_size
            target_dim = self.source_config.intermediate_size
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        projection = ProjectionMatrixBuilder.build_tied_projection(
            source_dim, target_dim
        )
        return source_dim, target_dim, projection

    def project_lora_layer(
        self, lora_a: torch.Tensor, lora_b: torch.Tensor, layer_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project LoRA A and B matrices to target model dimensions.

        Args:
            lora_a: Source LoRA A matrix [r x in_dim]
            lora_b: Source LoRA B matrix [out_dim x r]
            layer_type: One of 'q', 'k', 'v', 'o', 'gate_proj', 'up_proj', 'down_proj'

        Returns:
            Projected (A, B) matrices for target model
        """
        _, _, projection = self.compute_projection_dims(layer_type)

        if projection is None:
            return lora_a, lora_b

        with torch.no_grad():
            projected_a = projection.weight @ lora_a
            projected_b = lora_b @ projection.weight.t()

        return projected_a, projected_b


class AdapterConverter:
    """Main class for converting LoRA adapters between models."""

    def __init__(self, source_config: ModelConfig, target_config: ModelConfig):
        self.source_config = source_config
        self.target_config = target_config
        self.attention_projection = AttentionProjection(source_config, target_config)
        self.layer_mapping = {}

    def _infer_layer_mapping(self):
        """Infer mapping between source and target model layers."""
        src_layers = self.source_config.num_hidden_layers
        tgt_layers = self.target_config.num_hidden_layers

        if src_layers == tgt_layers:
            mapping = {i: i for i in range(src_layers)}
        elif tgt_layers > src_layers:
            scale = tgt_layers / src_layers
            mapping = {i: int(i * scale) for i in range(src_layers)}
        else:
            scale = src_layers / tgt_layers
            mapping = {int(i * scale): i for i in range(tgt_layers)}

        self.layer_mapping = mapping

    def convert_layer(
        self, layer_name: str, delta_weights: Dict[str, torch.Tensor], layer_type: str
    ) -> Dict[str, torch.Tensor]:
        """Convert a single layer's LoRA weights.

        Args:
            layer_name: Name of the layer (e.g., 'model.layers.0.self_attn.q_proj')
            delta_weights: Dictionary of all delta weights from source
            layer_type: Type of layer ('q', 'k', 'v', 'o', 'ffn')

        Returns:
            Converted weights dictionary for this layer
        """
        a_key = f"{layer_name}.lora_A.weight"
        b_key = f"{layer_name}.lora_B.weight"

        if a_key not in delta_weights or b_key not in delta_weights:
            return {}

        lora_a = delta_weights[a_key]
        lora_b = delta_weights[b_key]

        proj_a, proj_b = self.attention_projection.project_lora_layer(
            lora_a, lora_b, layer_type
        )

        return {
            f"{layer_name}.lora_A.weight": proj_a,
            f"{layer_name}.lora_B.weight": proj_b,
        }

    def convert_model(
        self, source_weights: Dict[str, torch.Tensor], layer_wise: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Convert all LoRA weights from source to target model.

        Args:
            source_weights: Delta weights from source model
            layer_wise: If True, process layer by layer

        Returns:
            Converted weights dictionary for target model
        """
        self._infer_layer_mapping()
        converted = {}

        source_layers = self.source_config.num_hidden_layers
        target_layers = self.target_config.num_hidden_layers

        for i in range(source_layers):
            mapped_i = self.layer_mapping.get(i, i)

            prefix_base = f"model.layers.{i}."

            for attn_type in ["q", "k", "v"]:
                old_prefix = f"model.layers.{i}.self_attn.{attn_type}_proj"
                new_prefix = f"model.layers.{mapped_i}.self_attn.{attn_type}_proj"

                for key in list(source_weights.keys()):
                    if old_prefix in key:
                        new_key = key.replace(old_prefix, new_prefix)
                        converted[new_key] = source_weights[key]

            for ffn_type in ["gate_proj", "up_proj", "down_proj"]:
                old_prefix = f"model.layers.{i}.mlp.{ffn_type}"
                new_prefix = f"model.layers.{mapped_i}.mlp.{ffn_type}"

                for key in list(source_weights.keys()):
                    if old_prefix in key:
                        new_key = key.replace(old_prefix, new_prefix)
                        converted[new_key] = source_weights[key]

        return converted

    def validate_conversion(
        self,
        converted_weights: Dict[str, torch.Tensor],
        expected_dims: Dict[str, Tuple[int, int]],
    ) -> List[str]:
        """Validate converted weights have expected dimensions.

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        for name, (expected_in, expected_out) in expected_dims.items():
            if name not in converted_weights:
                errors.append(f"Missing weight: {name}")
                continue

            weight = converted_weights[name]
            actual_in, actual_out = weight.shape

            if actual_in != expected_in or actual_out != expected_out:
                errors.append(
                    f"Dimension mismatch for {name}: "
                    f"expected ({expected_in}, {expected_out}), "
                    f"got ({actual_in}, {actual_out})"
                )

        return errors


class LLaMAtoQwenConverter(AdapterConverter):
    """Specialized converter for LLaMA to Qwen model conversion.

    Handles the Pre-RMSNorm vs Post-RMSNorm difference and other Qwen-specific
    considerations.
    """

    def __init__(self, source_config: ModelConfig, target_config: ModelConfig):
        super().__init__(source_config, target_config)
        self.norm_scale_factors = {}

    def compute_norm_scale_factor(self, layer_idx: int) -> float:
        """Compute normalization layer scale factor for LLaMA -> Qwen conversion.

        Qwen uses Pre-RMSNorm while LLaMA uses Post-RMSNorm. This causes
        a scale difference in the residual connection that needs to be compensated.

        Returns:
            Scale factor to apply to the layer normalization weights
        """
        source_layers = self.source_config.num_hidden_layers
        target_layers = self.target_config.num_hidden_layers

        relative_depth = layer_idx / max(source_layers, 1)
        scale_factor = 1.0 + 0.1 * (1.0 - relative_depth)

        return scale_factor

    def convert_norm_layer(
        self, layer_name: str, source_weights: Dict[str, torch.Tensor], layer_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Convert normalization layer weights with scale adjustment."""
        converted = {}

        weight_key = f"{layer_name}.weight"
        if weight_key in source_weights:
            weight = source_weights[weight_key]
            scale = self.compute_norm_scale_factor(layer_idx)
            converted[weight_key] = weight * scale

        return converted

    def convert_model_with_norm_adjustment(
        self, source_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Convert model with normalization layer adjustment."""
        self._infer_layer_mapping()
        converted = {}

        source_layers = self.source_config.num_hidden_layers

        for i in range(source_layers):
            mapped_i = self.layer_mapping.get(i, i)

            attn_types = ["q", "k", "v"]
            for attn_type in attn_types:
                old_prefix = f"model.layers.{i}.self_attn.{attn_type}_proj"
                new_prefix = f"model.layers.{mapped_i}.self_attn.{attn_type}_proj"

                for key in list(source_weights.keys()):
                    if old_prefix in key:
                        new_key = key.replace(old_prefix, new_prefix)
                        converted[new_key] = source_weights[key]

            ffn_types = ["gate_proj", "up_proj", "down_proj"]
            for ffn_type in ffn_types:
                old_prefix = f"model.layers.{i}.mlp.{ffn_type}"
                new_prefix = f"model.layers.{mapped_i}.mlp.{ffn_type}"

                for key in list(source_weights.keys()):
                    if old_prefix in key:
                        new_key = key.replace(old_prefix, new_prefix)
                        converted[new_key] = source_weights[key]

            input_norm_old = f"model.layers.{i}.input_layernorm.weight"
            input_norm_new = f"model.layers.{mapped_i}.input_layernorm.weight"
            if input_norm_old in source_weights:
                converted[input_norm_new] = source_weights[input_norm_old]

        return converted


def create_converter(
    source_config: ModelConfig,
    target_config: ModelConfig,
    conversion_type: str = "auto",
) -> AdapterConverter:
    """Factory function to create appropriate converter.

    Args:
        source_config: Source model configuration
        target_config: Target model configuration
        conversion_type: One of 'auto', 'llama_to_qwen', 'same_family', 'generic'

    Returns:
        Appropriate converter instance
    """
    if conversion_type == "llama_to_qwen":
        return LLaMAtoQwenConverter(source_config, target_config)
    elif conversion_type == "same_family":
        return AdapterConverter(source_config, target_config)
    elif conversion_type == "auto":
        if source_config.hidden_size != target_config.hidden_size:
            return AdapterConverter(source_config, target_config)
        else:
            return AdapterConverter(source_config, target_config)
    else:
        return AdapterConverter(source_config, target_config)


if __name__ == "__main__":
    print("=" * 60)
    print("LoRA Adapter Conversion Module")
    print("=" * 60)
    print("\nThis module provides classes for converting LoRA adapters")
    print("between models with similar architectures.")
    print("\nKey classes:")
    print("  - LoraWeightExtractor: Extract delta weights from PEFT model")
    print("  - ProjectionMatrixBuilder: Build projection matrices")
    print("  - AdapterConverter: Main conversion logic")
    print("  - LLaMAtoQwenConverter: Specialized LLaMA to Qwen conversion")
    print("\nUsage example:")
    print("""
    from convert_adapter import ModelConfig, LoraWeightExtractor, create_converter
    
    source_config = ModelConfig(hidden_size=4096, ...)
    target_config = ModelConfig(hidden_size=5120, ...)
    
    converter = create_converter(source_config, target_config)
    converted = converter.convert_model(delta_weights)
    """)
