"""
RoPE Interpolation Implementation

This module demonstrates the core concepts of Rotary Position Embedding (RoPE)
and position interpolation techniques for extending context window.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponential frequencies for RoPE.

    Args:
        dim: Dimension of the embeddings (must be even)
        end: Maximum position to compute frequencies for
        theta: Base frequency parameter

    Returns:
        Complex tensor of shape [end, dim // 2]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor to broadcast with input tensor.

    Args:
        freqs_cis: Complex frequencies [seq_len, dim//2]
        x: Input tensor [batch, seq_len, n_heads, head_dim]

    Returns:
        Reshaped frequencies for broadcasting
    """
    ndim = x.ndim
    shape = [1] * (ndim - 2) + [x.shape[1], x.shape[3] // 2]
    return freqs_cis.view(*shape)


def apply_rotary_pos_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.

    Args:
        xq: Query tensor [batch, seq_len, n_heads, head_dim]
        xk: Key tensor [batch, seq_len, n_heads, head_dim]
        freqs_cis: Precomputed complex frequencies [seq_len, head_dim//2]

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class LinearInterpolation:
    """Linear interpolation for RoPE position scaling."""

    def __init__(self, original_max_pos: int, new_max_pos: int, theta: float = 10000.0):
        """
        Args:
            original_max_pos: Original max position embeddings (e.g., 8192)
            new_max_pos: New max position to extend to (e.g., 32768)
            theta: RoPE base frequency
        """
        self.original_max_pos = original_max_pos
        self.new_max_pos = new_max_pos
        self.theta = theta
        self.scale_factor = original_max_pos / new_max_pos

    def get_scaled_freqs_cis(self, seq_len: int) -> torch.Tensor:
        """
        Get scaled frequency complex numbers for given sequence length.

        Positions are scaled linearly: new_pos = old_pos * scale_factor
        """
        freqs = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.original_max_pos, 2).float()
                / self.original_max_pos
            )
        )
        t = torch.arange(seq_len) * self.scale_factor
        freqs = torch.outer(t, freqs)
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs


class NTKawareScaling:
    """
    NTK-aware scaling for RoPE position interpolation.

    This method applies non-linear scaling based on frequency bands,
    preserving more precision for high-frequency (fine-grained) position info.
    """

    def __init__(
        self,
        original_max_pos: int,
        new_max_pos: int,
        dims_per_head: int = 128,
        theta: float = 10000.0,
    ):
        self.original_max_pos = original_max_pos
        self.new_max_pos = new_max_pos
        self.dims_per_head = dims_per_head
        self.theta = theta
        self.scale_factor = original_max_pos / new_max_pos

    def _compute_scaled_base(self) -> float:
        """
        Compute scaled base frequency for NTK-aware method.

        The idea is to adjust the base frequency so that high-frequency
        components are less affected by the scaling.
        """
        base = self.theta * (self.scale_factor ** (2 / math.log(self.dims_per_head)))
        return base

    def get_ntk_freqs_cis(self, seq_len: int) -> torch.Tensor:
        """
        Get NTK-scaled frequency complex numbers.
        """
        dim = self.dims_per_head
        base = self._compute_scaled_base()

        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len)
        freqs = torch.outer(t, freqs)
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs


class YarnMixin:
    """
    YaRN (Yet another RoPE extensioN) method.

    Combines NTK-aware scaling with attention temperature adjustment
    and optional fine-tuning data mixing.
    """

    def __init__(
        self,
        original_max_pos: int,
        new_max_pos: int,
        dims_per_head: int = 128,
        theta: float = 10000.0,
        alpha: float = 1.0,
    ):
        self.original_max_pos = original_max_pos
        self.new_max_pos = new_max_pos
        self.dims_per_head = dims_per_head
        self.theta = theta
        self.alpha = alpha
        self.scale_factor = original_max_pos / new_max_pos

        self._compute_yarn_params()

    def _compute_yarn_params(self) -> None:
        """Compute YaRN-specific parameters."""
        dim = self.dims_per_head

        # Compute the scaled base for NTK-aware part
        self.scaled_base = self.theta * (self.scale_factor ** (2 / math.log(dim)))

        # Attention temperature adjustment factor
        # YaRN suggests using sqrt(dim) scaling for the temperature
        self.attn_temp_factor = math.sqrt(dim) / (math.sqrt(dim) * self.scale_factor)

    def get_yarn_freqs_cis(self, seq_len: int) -> torch.Tensor:
        """
        Get YaRN-scaled frequency complex numbers.
        """
        dim = self.dims_per_head

        freqs = 1.0 / (self.scaled_base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len)
        freqs = torch.outer(t, freqs)
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def get_adjusted_attention_scale(self, original_scale: float) -> float:
        """
        Get attention scale adjusted for position interpolation.

        Args:
            original_scale: Original attention softmax scale

        Returns:
            Adjusted scale factor
        """
        return original_scale * self.attn_temp_factor


def demo_rope_interpolation():
    """Demonstrate RoPE interpolation with different scaling methods."""
    print("=" * 60)
    print("RoPE Interpolation Demonstration")
    print("=" * 60)

    batch_size = 2
    seq_len_original = 2048
    seq_len_extended = 8192
    n_heads = 8
    head_dim = 128
    original_max_pos = 8192
    new_max_pos = 32768

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create sample query and key tensors
    xq = torch.randn(batch_size, seq_len_original, n_heads, head_dim).to(device)
    xk = torch.randn(batch_size, seq_len_original, n_heads, head_dim).to(device)

    print(f"\nInput shape: {xq.shape}")
    print(f"Original max position: {original_max_pos}")
    print(f"New max position: {new_max_pos}")
    print(f"Scale factor: {original_max_pos / new_max_pos:.4f}")

    # Original RoPE
    print("\n--- Original RoPE ---")
    freqs_cis_original = precompute_freqs_cis(head_dim, original_max_pos, theta=10000.0)
    freqs_cis_original = freqs_cis_original.to(device)
    xq_rot, xk_rot = apply_rotary_pos_emb(xq, xk, freqs_cis_original[:seq_len_original])
    print(f"Original RoPE applied, output shape: {xq_rot.shape}")

    # Linear Interpolation
    print("\n--- Linear Interpolation ---")
    linear_interp = LinearInterpolation(original_max_pos, new_max_pos)
    freqs_cis_linear = linear_interp.get_scaled_freqs_cis(seq_len_extended).to(device)
    xq_lin, xk_lin = apply_rotary_pos_emb(xq, xk, freqs_cis_linear[:seq_len_original])
    print(
        f"Linear interpolation applied, scale factor: {linear_interp.scale_factor:.4f}"
    )

    # NTK-aware Scaling
    print("\n--- NTK-aware Scaling ---")
    ntk = NTKawareScaling(original_max_pos, new_max_pos, dims_per_head=head_dim)
    freqs_cis_ntk = ntk.get_ntk_freqs_cis(seq_len_extended).to(device)
    xq_ntk, xk_ntk = apply_rotary_pos_emb(xq, xk, freqs_cis_ntk[:seq_len_original])
    print(f"NTK-aware scaling applied, scaled base: {ntk.scaled_base:.2f}")

    # YaRN
    print("\n--- YaRN ---")
    yarn = YarnMixin(original_max_pos, new_max_pos, dims_per_head=head_dim)
    freqs_cis_yarn = yarn.get_yarn_freqs_cis(seq_len_extended).to(device)
    xq_yarn, xk_yarn = apply_rotary_pos_emb(xq, xk, freqs_cis_yarn[:seq_len_original])
    print(
        f"YaRN applied, scaled base: {yarn.scaled_base:.2f}, attn temp factor: {yarn.attn_temp_factor:.4f}"
    )

    # Compare how different methods affect high vs low frequency components
    print("\n--- Frequency Analysis ---")
    print("First few frequency values (original):")
    print(f"  {freqs_cis_original[100, :5]}")
    print("First few frequency values (linear interpolation):")
    print(f"  {freqs_cis_linear[100, :5]}")
    print("First few frequency values (NTK-aware):")
    print(f"  {freqs_cis_ntk[100, :5]}")

    return {
        "original": (xq_rot, xk_rot),
        "linear": (xq_lin, xk_lin),
        "ntk": (xq_ntk, xk_ntk),
        "yarn": (xq_yarn, xk_yarn),
    }


if __name__ == "__main__":
    results = demo_rope_interpolation()
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
