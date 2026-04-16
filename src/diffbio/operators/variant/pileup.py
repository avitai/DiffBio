"""Differentiable pileup generation for variant calling."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.configs import TemperatureConfig
from diffbio.constants import EPSILON
from diffbio.core.base_operators import TemperatureOperator


@dataclass(frozen=True)
class PileupConfig(TemperatureConfig):
    """Configuration for differentiable pileup.

    Inherits from TemperatureConfig to get temperature and learnable_temperature fields.

    Attributes:
        use_quality_weights: Whether to weight bases by quality scores.
        reference_length: Length of reference sequence (required for batch processing).
            All reads in a batch must align to the same reference length.
        return_coverage: Whether to return coverage channel in output.
        return_quality: Whether to return mean quality channel in output.
        apply_softmax: Whether to apply softmax to final pileup (set False to preserve
            raw weighted sums, which is better for variant detection).
    """

    use_quality_weights: bool = True
    reference_length: int = 100
    return_coverage: bool = False
    return_quality: bool = False
    apply_softmax: bool = True

    def __post_init__(self) -> None:
        """Validate the supported pileup configuration surface."""
        super().__post_init__()

        if self.reference_length <= 0:
            raise ValueError(f"reference_length must be positive, got {self.reference_length}")


class DifferentiablePileup(TemperatureOperator):
    """Differentiable pileup generator.

    Aggregates aligned reads into a position-wise nucleotide distribution
    that can be used for variant calling. Unlike traditional pileup which
    simply counts bases, this implementation uses soft weighting that
    allows gradients to flow through.

    Inherits from TemperatureOperator to get:

    - _temperature property for temperature-controlled smoothing
    - soft_max() for logsumexp-based smooth maximum
    - soft_argmax() for soft position selection

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
        # Temperature is now managed by TemperatureOperator via self._temperature

    def compute_pileup(
        self,
        reads: Float[Array, "num_reads read_length 4"],
        positions: Int[Array, "num_reads"],
        quality: Float[Array, "num_reads read_length"],
        reference_length: int,
    ) -> dict[str, Float[Array, "..."]]:
        """Generate pileup from aligned reads.

        Args:
            reads: One-hot encoded reads (num_reads, read_length, 4).
            positions: Starting position of each read (num_reads,).
            quality: Quality scores for each base (num_reads, read_length).
            reference_length: Length of reference sequence.

        Returns:
            Dictionary containing:
            - pileup: (reference_length, 4) nucleotide distributions
            - coverage: (reference_length, 1) read depth at each position (if return_coverage)
            - mean_quality: (reference_length, 1) mean quality at each position (if return_quality)
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
        flat_quality = quality.reshape(-1, 1)  # (num_reads * read_length, 1)

        # Mask out-of-bounds positions
        in_bounds = (flat_positions >= 0) & (flat_positions < reference_length)
        in_bounds_mask = in_bounds[:, None].astype(jnp.float32)
        flat_weights_masked = flat_weights * in_bounds_mask

        # Weighted reads
        weighted_reads = flat_reads * flat_weights_masked

        # Use segment_sum to aggregate bases at each position
        # First, clip positions to valid range (we've already masked weights for invalid)
        clipped_positions = jnp.clip(flat_positions, 0, reference_length - 1)

        # Aggregate nucleotide counts at each position
        pileup = jax.ops.segment_sum(
            weighted_reads,
            clipped_positions.astype(jnp.int32),
            num_segments=reference_length,
        )

        # Aggregate coverage at each position (sum of weights)
        coverage = jax.ops.segment_sum(
            flat_weights_masked,
            clipped_positions.astype(jnp.int32),
            num_segments=reference_length,
        )

        # Normalize by coverage to get nucleotide distribution
        # Add small epsilon to avoid division by zero
        coverage_safe = jnp.maximum(coverage, EPSILON)
        pileup_normalized = pileup / coverage_safe

        # Optionally apply softmax
        # Use inherited _temperature property from TemperatureOperator
        if self.config.apply_softmax:
            pileup_normalized = jax.nn.softmax(pileup_normalized / self._temperature, axis=-1)

        result = {"pileup": pileup_normalized}

        # Add coverage channel if requested
        if self.config.return_coverage:
            result["coverage"] = coverage

        # Add mean quality channel if requested
        if self.config.return_quality:
            # Aggregate quality * weight, then divide by weight sum
            weighted_quality = flat_quality * flat_weights_masked
            quality_sum = jax.ops.segment_sum(
                weighted_quality,
                clipped_positions.astype(jnp.int32),
                num_segments=reference_length,
            )
            mean_quality = quality_sum / coverage_safe
            result["mean_quality"] = mean_quality

        return result

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

        # Compute pileup (returns dict with pileup and optional coverage/quality)
        pileup_result = self.compute_pileup(reads, positions, quality, reference_length)

        # Build output data - preserve input keys for Datarax vmap compatibility
        transformed_data = {
            "reads": reads,
            "positions": positions,
            "quality": quality,
            "pileup": pileup_result["pileup"],
        }

        # Add coverage and mean_quality if present
        if "coverage" in pileup_result:
            transformed_data["coverage"] = pileup_result["coverage"]
        if "mean_quality" in pileup_result:
            transformed_data["mean_quality"] = pileup_result["mean_quality"]

        return transformed_data, state, metadata
