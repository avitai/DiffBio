"""Differentiable pileup generation for variant calling.

This module provides a differentiable approximation of pileup generation,
which aggregates aligned reads at each position of a reference sequence.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree


@dataclass
class PileupConfig(OperatorConfig):
    """Configuration for differentiable pileup.

    Attributes:
        window_size: Size of context window around each position.
        min_coverage: Minimum coverage threshold.
        max_coverage: Maximum coverage for normalization.
        use_quality_weights: Whether to weight bases by quality scores.
        reference_length: Length of reference sequence (required for batch processing).
            All reads in a batch must align to the same reference length.
        stochastic: Whether the operator uses randomness (always False).
        stream_name: RNG stream name (not used).
    """

    window_size: int = 21
    min_coverage: int = 1
    max_coverage: int = 100
    use_quality_weights: bool = True
    reference_length: int = 100  # Default reference length for batch processing
    stochastic: bool = False
    stream_name: str | None = None


class DifferentiablePileup(OperatorModule):
    """Differentiable pileup generator.

    Aggregates aligned reads into a position-wise nucleotide distribution
    that can be used for variant calling. Unlike traditional pileup which
    simply counts bases, this implementation uses soft weighting that
    allows gradients to flow through.

    Args:
        config: Pileup configuration.
        rngs: Flax NNX random number generators.
        name: Optional operator name.
    """

    def __init__(
        self,
        config: PileupConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize differentiable pileup.

        Args:
            config: Pileup configuration.
            rngs: Random number generators (optional).
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)
        self.temperature = nnx.Param(jnp.array(1.0))

    def compute_pileup(
        self,
        reads: Float[Array, "num_reads read_length 4"],
        positions: Int[Array, "num_reads"],
        quality: Float[Array, "num_reads read_length"],
        reference_length: int,
    ) -> Float[Array, "reference_length 4"]:
        """Generate pileup from aligned reads.

        Args:
            reads: One-hot encoded reads (num_reads, read_length, 4).
            positions: Starting position of each read (num_reads,).
            quality: Quality scores for each base (num_reads, read_length).
            reference_length: Length of reference sequence.

        Returns:
            Pileup array of shape (reference_length, 4) with nucleotide
            distributions at each position.
        """
        _, read_length, _ = reads.shape

        # Convert quality scores to weights
        if self.config.use_quality_weights:
            # Phred to probability: p_error = 10^(-Q/10)
            # Weight = 1 - p_error
            p_error = jnp.power(10.0, -quality / 10.0)
            weights = 1.0 - p_error
        else:
            weights = jnp.ones_like(quality)

        # Create position indices for all bases in all reads
        # For each read i at position p[i], base j maps to reference position p[i] + j
        read_offsets = jnp.arange(read_length)  # [0, 1, ..., read_length-1]
        # Broadcast to get absolute positions: (num_reads, read_length)
        absolute_positions = positions[:, None] + read_offsets[None, :]

        # Flatten everything for scatter operation
        flat_positions = absolute_positions.reshape(-1)  # (num_reads * read_length,)
        flat_reads = reads.reshape(-1, 4)  # (num_reads * read_length, 4)
        flat_weights = weights.reshape(-1, 1)  # (num_reads * read_length, 1)

        # Mask out-of-bounds positions
        in_bounds = (flat_positions >= 0) & (flat_positions < reference_length)
        flat_weights = flat_weights * in_bounds[:, None].astype(jnp.float32)

        # Weighted reads
        weighted_reads = flat_reads * flat_weights

        # Use segment_sum to aggregate bases at each position
        # First, clip positions to valid range (we've already masked weights for invalid)
        clipped_positions = jnp.clip(flat_positions, 0, reference_length - 1)

        # Aggregate nucleotide counts at each position
        pileup = jax.ops.segment_sum(
            weighted_reads,
            clipped_positions.astype(jnp.int32),
            num_segments=reference_length,
        )

        # Aggregate coverage at each position
        coverage = jax.ops.segment_sum(
            flat_weights,
            clipped_positions.astype(jnp.int32),
            num_segments=reference_length,
        )

        # Normalize by coverage to get nucleotide distribution
        # Add small epsilon to avoid division by zero
        coverage = jnp.maximum(coverage, 1e-8)
        pileup_normalized = pileup / coverage

        # Apply softmax to ensure valid probability distribution
        pileup_normalized = jax.nn.softmax(pileup_normalized / self.temperature[...], axis=-1)

        return pileup_normalized

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply pileup generation to read data.

        This method implements the OperatorModule interface for batch processing.
        It expects data containing reads and their positions, and returns pileup.

        Note: reference_length is taken from config (not data) because it must be
        static for JAX's segment_sum. All reads in a batch must align to the same
        reference. Output preserves input keys for Datarax vmap compatibility.

        Args:
            data: Dictionary containing:
                - "reads": One-hot encoded reads (num_reads, read_length, 4)
                - "positions": Starting position of each read (num_reads,)
                - "quality": Quality scores for each base (num_reads, read_length)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (deterministic operator)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains input data plus pileup array
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        reads = data["reads"]
        positions = data["positions"]
        quality = data["quality"]

        # Use reference_length from config (must be static for segment_sum)
        reference_length = self.config.reference_length

        # Compute pileup
        pileup = self.compute_pileup(reads, positions, quality, reference_length)

        # Build output data - preserve input keys for Datarax vmap compatibility
        transformed_data = {
            "reads": reads,
            "positions": positions,
            "quality": quality,
            "pileup": pileup,
        }

        return transformed_data, state, metadata
