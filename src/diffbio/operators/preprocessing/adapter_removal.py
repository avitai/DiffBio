"""Differentiable adapter removal operator.

This module provides a soft adapter removal operator that uses differentiable
alignment to find adapter sequences and applies soft trimming to maintain
gradient flow.

Key technique: Use SmoothSmithWaterman for adapter matching, then apply
sigmoid-weighted soft trimming based on the match position.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.sequences.dna import encode_dna_string


@dataclass
class AdapterRemovalConfig(OperatorConfig):
    """Configuration for SoftAdapterRemoval.

    Attributes:
        adapter_sequence: Adapter sequence to remove (default: Illumina universal).
        temperature: Temperature for soft matching and trimming.
            Lower = sharper trimming, Higher = smoother.
        match_threshold: Minimum alignment score ratio to consider a match.
        min_overlap: Minimum overlap length to consider adapter presence.
        stochastic: Whether the operator uses randomness (always False).
        stream_name: RNG stream name (not used).
    """

    adapter_sequence: str = "AGATCGGAAGAG"  # Illumina universal adapter
    temperature: float = 1.0
    match_threshold: float = 0.5
    min_overlap: int = 6
    stochastic: bool = False
    stream_name: str | None = None


class SoftAdapterRemoval(OperatorModule):
    """Differentiable adapter removal for sequencing reads.

    This operator performs soft adapter trimming using a differentiable
    approach. It finds potential adapter matches at the 3' end of reads
    and applies sigmoid-weighted trimming that maintains gradient flow.

    The algorithm:
    1. Compute soft alignment scores between sequence suffix and adapter
    2. Find the soft trim position using weighted position averaging
    3. Apply sigmoid-weighted retention to each position

    Args:
        config: AdapterRemovalConfig with adapter parameters.
        rngs: Flax NNX random number generators (optional).
        name: Optional operator name.

    Example:
        >>> config = AdapterRemovalConfig(adapter_sequence="AGATCGGAAGAG")
        >>> remover = SoftAdapterRemoval(config)
        >>> data = {"sequence": encoded_seq, "quality_scores": quality}
        >>> result, state, meta = remover.apply(data, {}, None)
    """

    def __init__(
        self,
        config: AdapterRemovalConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the soft adapter removal operator.

        Args:
            config: Adapter removal configuration.
            rngs: Random number generators (optional).
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        # Encode adapter sequence as one-hot
        adapter_onehot = encode_dna_string(config.adapter_sequence)
        self.adapter = nnx.Param(adapter_onehot)

        # Learnable parameters
        self.temperature = nnx.Param(jnp.array(config.temperature))
        self.match_threshold = nnx.Param(jnp.array(config.match_threshold))

        # Scoring matrix for DNA alignment (match=2, mismatch=-1)
        scoring = jnp.array([
            [2.0, -1.0, -1.0, -1.0],  # A
            [-1.0, 2.0, -1.0, -1.0],  # C
            [-1.0, -1.0, 2.0, -1.0],  # G
            [-1.0, -1.0, -1.0, 2.0],  # T
        ])
        self.scoring_matrix = nnx.Param(scoring)

        # Store config values
        self.min_overlap = config.min_overlap

    def _compute_suffix_adapter_scores(
        self,
        sequence: Float[Array, "length alphabet"],
    ) -> Float[Array, "length"]:
        """Compute adapter match scores for each suffix position.

        For each position i, compute how well the suffix starting at i
        matches the adapter prefix. This finds where the adapter might start.

        Args:
            sequence: One-hot encoded sequence (length, 4).

        Returns:
            Scores for adapter match starting at each position (length,).
        """
        seq_len = sequence.shape[0]
        adapter = self.adapter[...]
        adapter_len = adapter.shape[0]
        scoring = self.scoring_matrix[...]

        # For each starting position, compute match score with adapter
        def score_at_position(start_pos: int) -> Float[Array, ""]:
            """Compute alignment score for suffix starting at start_pos."""
            # Length of overlap
            overlap_len = jnp.minimum(seq_len - start_pos, adapter_len)

            # Extract overlapping regions
            # Use dynamic slicing with masking for differentiability
            positions = jnp.arange(adapter_len)
            mask = positions < overlap_len

            # Compute scores for each position in the overlap
            def score_position(pos: int) -> Float[Array, ""]:
                seq_pos = start_pos + pos
                # Handle out-of-bounds with zeros
                valid = (seq_pos < seq_len) & (pos < adapter_len)
                seq_base = jnp.where(
                    valid,
                    jax.lax.dynamic_slice(sequence, (seq_pos, 0), (1, 4)).squeeze(0),
                    jnp.zeros(4),
                )
                adapter_base = jnp.where(
                    valid,
                    jax.lax.dynamic_slice(adapter, (pos, 0), (1, 4)).squeeze(0),
                    jnp.zeros(4),
                )
                # Score = seq @ scoring @ adapter.T
                score = jnp.einsum("a,ab,b->", seq_base, scoring, adapter_base)
                return jnp.where(valid, score, 0.0)

            # Sum scores across overlap positions
            scores = jax.vmap(score_position)(jnp.arange(adapter_len))
            total_score = jnp.sum(scores * mask)

            # Normalize by maximum possible score for this overlap length
            max_score = 2.0 * overlap_len  # Perfect match score
            normalized = jnp.where(
                overlap_len >= self.min_overlap,
                total_score / jnp.maximum(max_score, 1.0),
                0.0,
            )
            return normalized

        # Compute scores for all suffix positions
        scores = jax.vmap(score_at_position)(jnp.arange(seq_len))
        return scores

    def _compute_soft_trim_position(
        self,
        adapter_scores: Float[Array, "length"],
    ) -> Float[Array, ""]:
        """Compute soft trim position using weighted average.

        Uses softmax over adapter scores to compute a soft trim position.
        High adapter match scores at position i suggest trimming should
        start there.

        Args:
            adapter_scores: Adapter match scores at each position.

        Returns:
            Soft trim position (continuous value).
        """
        seq_len = adapter_scores.shape[0]
        temp = self.temperature[...]
        threshold = self.match_threshold[...]

        # Apply threshold - only consider positions with good matches
        thresholded_scores = jax.nn.relu(adapter_scores - threshold)

        # Soft position selection using softmax
        # Add small epsilon for numerical stability
        weights = jax.nn.softmax(thresholded_scores / temp + 1e-10)

        # Weighted average of positions
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        soft_position = jnp.sum(weights * positions)

        # If no adapter found (all scores below threshold), return seq_len
        has_adapter = jnp.any(adapter_scores > threshold)
        soft_position = jnp.where(has_adapter, soft_position, float(seq_len))

        return soft_position

    def _apply_soft_trimming(
        self,
        sequence: Float[Array, "length alphabet"],
        quality_scores: Float[Array, "length"],
        soft_trim_pos: Float[Array, ""],
    ) -> tuple[Float[Array, "length alphabet"], Float[Array, "length"]]:
        """Apply soft trimming to sequence and quality scores.

        Uses sigmoid to create smooth retention weights based on position
        relative to the trim point. Positions before trim_pos have high
        retention, positions after have low retention.

        Args:
            sequence: One-hot encoded sequence.
            quality_scores: Quality scores for each position.
            soft_trim_pos: Soft trim position.

        Returns:
            Tuple of (weighted_sequence, weighted_quality).
        """
        seq_len = sequence.shape[0]
        temp = self.temperature[...]

        # Create position-based retention weights
        # retention = sigmoid((trim_pos - position) / temperature)
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        retention_weights = jax.nn.sigmoid((soft_trim_pos - positions) / temp)

        # Apply weights
        weighted_sequence = sequence * retention_weights[:, None]
        weighted_quality = quality_scores * retention_weights

        return weighted_sequence, weighted_quality

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply soft adapter removal to sequence data.

        This method finds potential adapter sequences and applies
        differentiable soft trimming to remove them while maintaining
        gradient flow.

        Args:
            data: Dictionary containing:
                - "sequence": One-hot encoded sequence (length, alphabet_size)
                - "quality_scores": Phred quality scores (length,)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (deterministic operator)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:
                    - "sequence": Soft-trimmed sequence
                    - "quality_scores": Soft-trimmed quality scores
                    - "adapter_score": Maximum adapter match score
                    - "trim_position": Soft trim position
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        sequence = data["sequence"]
        quality_scores = data["quality_scores"]

        # Compute adapter match scores for each suffix position
        adapter_scores = self._compute_suffix_adapter_scores(sequence)

        # Find soft trim position
        soft_trim_pos = self._compute_soft_trim_position(adapter_scores)

        # Apply soft trimming
        trimmed_sequence, trimmed_quality = self._apply_soft_trimming(
            sequence, quality_scores, soft_trim_pos
        )

        # Build output data
        transformed_data = {
            "sequence": trimmed_sequence,
            "quality_scores": trimmed_quality,
            "adapter_score": jnp.max(adapter_scores),
            "trim_position": soft_trim_pos,
        }

        return transformed_data, state, metadata
