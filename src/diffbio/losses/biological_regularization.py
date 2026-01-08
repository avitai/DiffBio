"""Biological regularization losses for differentiable bioinformatics.

This module provides regularization losses that help prevent adversarial
optimization of differentiable bioinformatics components. These losses
encourage biologically plausible sequences and alignments.

Reference:
    Petti et al. (2023) observed that purely differentiable alignment can
    produce biologically implausible solutions without proper regularization.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


@dataclass
class BiologicalRegularizationConfig:
    """Configuration for biological regularization losses.

    Attributes:
        gc_content_weight: Weight for GC content regularization.
        gap_pattern_weight: Weight for gap pattern regularization.
        complexity_weight: Weight for sequence complexity loss.
        target_gc_content: Target GC content (typically 0.4-0.6).
        target_gc_tolerance: Tolerance around target GC content.
    """

    gc_content_weight: float = 1.0
    gap_pattern_weight: float = 1.0
    complexity_weight: float = 1.0
    target_gc_content: float = 0.5
    target_gc_tolerance: float = 0.2


class GCContentRegularization(nnx.Module):
    """Regularization loss for GC content.

    Penalizes sequences with GC content far from biological norms.
    For most organisms, GC content ranges from 25% to 75%.

    Args:
        target_gc: Target GC content (default 0.5 for balanced).
        tolerance: Tolerance around target before penalizing.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        target_gc: float = 0.5,
        tolerance: float = 0.2,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize GC content regularization.

        Args:
            target_gc: Target GC content.
            tolerance: Tolerance around target.
            rngs: Random number generators (optional).
        """
        super().__init__()
        self.target_gc = nnx.Param(jnp.array(target_gc))
        self.tolerance = nnx.Param(jnp.array(tolerance))

    def __call__(
        self,
        sequence: Float[Array, "length alphabet"],
    ) -> Float[Array, ""]:
        """Compute GC content regularization loss.

        Args:
            sequence: Soft one-hot encoded sequence (length, alphabet_size).
                     Assumes alphabet order: A, C, G, T (indices 0, 1, 2, 3).

        Returns:
            Scalar loss penalizing deviation from target GC content.
        """
        # GC content = sum of C and G probabilities
        # C is index 1, G is index 2
        gc_content = jnp.mean(sequence[:, 1] + sequence[:, 2])

        # Compute deviation from target
        target = self.target_gc[...]
        tolerance = self.tolerance[...]

        # Soft penalty: quadratic beyond tolerance
        deviation = jnp.abs(gc_content - target)
        excess = jnp.maximum(deviation - tolerance, 0.0)

        return excess**2


class GapPatternRegularization(nnx.Module):
    """Regularization loss for gap patterns in alignments.

    Penalizes unrealistic gap patterns such as:
    - Very long consecutive gaps
    - Many scattered small gaps

    Args:
        max_gap_length: Maximum expected gap length before penalizing.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        max_gap_length: int = 10,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize gap pattern regularization.

        Args:
            max_gap_length: Maximum expected gap length.
            rngs: Random number generators (optional).
        """
        super().__init__()
        self.max_gap_length = max_gap_length

    def __call__(
        self,
        alignment_weights: Float[Array, "len1 len2"],
    ) -> Float[Array, ""]:
        """Compute gap pattern regularization loss.

        Args:
            alignment_weights: Soft alignment matrix where entry (i,j)
                             indicates probability of aligning position i to j.

        Returns:
            Scalar loss penalizing unrealistic gap patterns.
        """
        # Compute row-wise and column-wise alignment strengths
        row_aligned = jnp.max(alignment_weights, axis=1)
        col_aligned = jnp.max(alignment_weights, axis=0)

        # Penalize positions with very low alignment probability (gaps)
        # Using smooth measure of "gappiness"
        row_gap_penalty = jnp.mean(1.0 - row_aligned)
        col_gap_penalty = jnp.mean(1.0 - col_aligned)

        # Also penalize non-monotonic alignments (jumps)
        # A good alignment should roughly follow the diagonal
        len1, len2 = alignment_weights.shape
        expected_diag = jnp.linspace(0, len2 - 1, len1)

        # Compute weighted average position for each row
        positions = jnp.arange(len2)
        weighted_pos = jnp.sum(
            alignment_weights * positions[None, :], axis=1
        ) / jnp.maximum(jnp.sum(alignment_weights, axis=1), 1e-8)

        # Penalize deviation from expected diagonal progression
        diag_penalty = jnp.mean((weighted_pos - expected_diag) ** 2) / (len2**2)

        return row_gap_penalty + col_gap_penalty + diag_penalty


class SequenceComplexityLoss(nnx.Module):
    """Regularization loss for sequence complexity.

    Penalizes low-complexity sequences that might arise from adversarial
    optimization (e.g., all-A sequences, repetitive patterns).

    Uses entropy as a measure of complexity.

    Args:
        min_entropy: Minimum expected entropy per position.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        min_entropy: float = 1.0,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize sequence complexity loss.

        Args:
            min_entropy: Minimum expected entropy.
            rngs: Random number generators (optional).
        """
        super().__init__()
        self.min_entropy = nnx.Param(jnp.array(min_entropy))

    def __call__(
        self,
        sequence: Float[Array, "length alphabet"],
    ) -> Float[Array, ""]:
        """Compute sequence complexity loss.

        Args:
            sequence: Soft one-hot encoded sequence (length, alphabet_size).

        Returns:
            Scalar loss penalizing low-complexity sequences.
        """
        # Compute per-position entropy
        # Add small epsilon for numerical stability
        eps = 1e-8
        entropy = -jnp.sum(sequence * jnp.log(sequence + eps), axis=-1)

        # Average entropy across positions
        avg_entropy = jnp.mean(entropy)

        # Penalize if entropy is below minimum
        min_ent = self.min_entropy[...]
        deficit = jnp.maximum(min_ent - avg_entropy, 0.0)

        return deficit**2


class BiologicalPlausibilityLoss(nnx.Module):
    """Combined biological plausibility regularization.

    Combines multiple regularization terms to encourage biologically
    plausible sequences and alignments during differentiable optimization.

    Args:
        config: BiologicalRegularizationConfig with weights and targets.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        config: BiologicalRegularizationConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize combined biological plausibility loss.

        Args:
            config: Configuration with weights and targets.
            rngs: Random number generators (optional).
        """
        super().__init__()
        self.config = config

        # Initialize component losses
        self.gc_loss = GCContentRegularization(
            target_gc=config.target_gc_content,
            tolerance=config.target_gc_tolerance,
            rngs=rngs,
        )
        self.complexity_loss = SequenceComplexityLoss(
            min_entropy=1.0,
            rngs=rngs,
        )

    def __call__(
        self,
        sequence: Float[Array, "length alphabet"],
        alignment_weights: Float[Array, "len1 len2"] | None = None,
    ) -> Float[Array, ""]:
        """Compute combined biological plausibility loss.

        Args:
            sequence: Soft one-hot encoded sequence.
            alignment_weights: Optional soft alignment matrix.

        Returns:
            Scalar combined regularization loss.
        """
        total_loss = jnp.array(0.0)

        # GC content regularization
        if self.config.gc_content_weight > 0:
            gc_loss = self.gc_loss(sequence)
            total_loss = total_loss + self.config.gc_content_weight * gc_loss

        # Sequence complexity regularization
        if self.config.complexity_weight > 0:
            complexity_loss = self.complexity_loss(sequence)
            total_loss = total_loss + self.config.complexity_weight * complexity_loss

        # Gap pattern regularization (if alignment provided)
        if alignment_weights is not None and self.config.gap_pattern_weight > 0:
            gap_loss_fn = GapPatternRegularization(rngs=None)
            gap_loss = gap_loss_fn(alignment_weights)
            total_loss = total_loss + self.config.gap_pattern_weight * gap_loss

        return total_loss
