"""Differentiable CNV Segmentation operator.

This module provides a differentiable implementation of copy number
variation segmentation using attention-based soft changepoint detection.

Key technique: Attention mechanism identifies segment boundaries softly,
enabling gradient flow through the segmentation process.

Applications: CNV analysis, coverage depth segmentation, breakpoint detection.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree


@dataclass
class CNVSegmentationConfig(OperatorConfig):
    """Configuration for DifferentiableCNVSegmentation.

    Attributes:
        max_segments: Maximum number of segments to detect.
        hidden_dim: Hidden dimension for attention layers.
        attention_heads: Number of attention heads.
        temperature: Temperature for softmax operations.
        stochastic: Whether the operator uses randomness.
        stream_name: RNG stream name.
    """

    max_segments: int = 100
    hidden_dim: int = 64
    attention_heads: int = 4
    temperature: float = 1.0
    stochastic: bool = False
    stream_name: str | None = None


class DifferentiableCNVSegmentation(OperatorModule):
    """Soft CNV segmentation using attention-based changepoint detection.

    This operator identifies segment boundaries in coverage data using
    attention mechanisms, replacing hard Circular Binary Segmentation
    with differentiable soft assignments.

    Algorithm:
    1. Project coverage signal into hidden space
    2. Use self-attention to identify changepoint positions
    3. Compute soft segment assignments via attention
    4. Compute segment means as weighted averages

    Args:
        config: CNVSegmentationConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = CNVSegmentationConfig(max_segments=50)
        >>> segmenter = DifferentiableCNVSegmentation(config, rngs=nnx.Rngs(42))
        >>> data = {"coverage": coverage_signal}  # (n_positions,)
        >>> result, state, meta = segmenter.apply(data, {}, None)
    """

    def __init__(
        self,
        config: CNVSegmentationConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the CNV segmentation operator.

        Args:
            config: Segmentation configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.max_segments = config.max_segments
        self.hidden_dim = config.hidden_dim
        self.attention_heads = config.attention_heads
        self.temperature = config.temperature

        # Input projection: coverage value -> hidden
        self.input_proj = nnx.Linear(1, config.hidden_dim, rngs=rngs)

        # Positional encoding projection
        self.pos_proj = nnx.Linear(1, config.hidden_dim, rngs=rngs)

        # Attention projections (for boundary detection)
        self.query_proj = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.key_proj = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.value_proj = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)

        # Boundary detection head
        self.boundary_head = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

        # Segment centroids (learnable)
        key = rngs.params()
        init_centroids = jax.random.normal(key, (config.max_segments, config.hidden_dim)) * 0.1
        self.segment_centroids = nnx.Param(init_centroids)

    def compute_embeddings(
        self,
        coverage: Float[Array, "n_positions"],
    ) -> Float[Array, "n_positions hidden_dim"]:
        """Compute position embeddings from coverage signal.

        Args:
            coverage: Coverage values at each position.

        Returns:
            Embedded representation of each position.
        """
        n_positions = coverage.shape[0]

        # Project coverage values
        coverage_emb = self.input_proj(coverage[:, None])  # (n_positions, hidden_dim)

        # Add positional encoding
        positions = jnp.arange(n_positions, dtype=jnp.float32) / n_positions
        pos_emb = self.pos_proj(positions[:, None])  # (n_positions, hidden_dim)

        embeddings = coverage_emb + pos_emb

        return embeddings

    def compute_boundary_probs(
        self,
        embeddings: Float[Array, "n_positions hidden_dim"],
    ) -> Float[Array, "n_positions"]:
        """Compute soft boundary probabilities using self-attention.

        Args:
            embeddings: Position embeddings.

        Returns:
            Probability of being a segment boundary at each position.
        """
        n_positions = embeddings.shape[0]

        # Self-attention to detect changepoints
        Q = self.query_proj(embeddings)  # (n_positions, hidden_dim)
        K = self.key_proj(embeddings)  # (n_positions, hidden_dim)
        V = self.value_proj(embeddings)  # (n_positions, hidden_dim)

        # Compute attention scores
        head_dim = self.hidden_dim // self.attention_heads
        scale = jnp.sqrt(head_dim).astype(embeddings.dtype)

        # Reshape for multi-head attention
        Q = Q.reshape(n_positions, self.attention_heads, head_dim)
        K = K.reshape(n_positions, self.attention_heads, head_dim)
        V = V.reshape(n_positions, self.attention_heads, head_dim)

        # Attention: (n_positions, n_heads, n_positions)
        attn_scores = jnp.einsum("nhd,mhd->nhm", Q, K) / scale

        # Soft attention weights
        attn_weights = jax.nn.softmax(attn_scores / self.temperature, axis=-1)

        # Attend to values
        attended = jnp.einsum("nhm,mhd->nhd", attn_weights, V)
        attended = attended.reshape(n_positions, self.hidden_dim)

        # Compute boundary probability from attended features
        # Look at how much attention pattern changes
        boundary_logits = self.boundary_head(attended).squeeze(-1)  # (n_positions,)

        # Boundaries at positions where signal changes
        boundary_probs = jax.nn.sigmoid(boundary_logits)

        return boundary_probs

    def compute_segment_assignments(
        self,
        embeddings: Float[Array, "n_positions hidden_dim"],
    ) -> Float[Array, "n_positions max_segments"]:
        """Compute soft segment assignments via attention to centroids.

        Args:
            embeddings: Position embeddings.

        Returns:
            Soft assignment probability to each segment.
        """
        centroids = self.segment_centroids[...]  # (max_segments, hidden_dim)

        # Compute similarity to segment centroids
        # (n_positions, hidden_dim) x (hidden_dim, max_segments) -> (n_positions, max_segments)
        similarities = jnp.einsum("nh,sh->ns", embeddings, centroids)

        # Soft assignments via softmax
        assignments = jax.nn.softmax(similarities / self.temperature, axis=-1)

        return assignments

    def compute_segment_means(
        self,
        coverage: Float[Array, "n_positions"],
        assignments: Float[Array, "n_positions max_segments"],
    ) -> Float[Array, "max_segments"]:
        """Compute segment mean values.

        Args:
            coverage: Coverage values.
            assignments: Soft segment assignments.

        Returns:
            Mean coverage for each segment.
        """
        # Weighted sum of coverage / sum of weights
        weighted_sum = jnp.einsum("n,ns->s", coverage, assignments)  # (max_segments,)
        weight_sum = jnp.sum(assignments, axis=0) + 1e-10  # (max_segments,)

        segment_means = weighted_sum / weight_sum

        return segment_means

    def compute_smoothed_coverage(
        self,
        coverage: Float[Array, "n_positions"],
        assignments: Float[Array, "n_positions max_segments"],
        segment_means: Float[Array, "max_segments"],
    ) -> Float[Array, "n_positions"]:
        """Compute smoothed coverage from segment assignments.

        Args:
            coverage: Original coverage values.
            assignments: Soft segment assignments.
            segment_means: Mean value for each segment.

        Returns:
            Smoothed coverage (soft segmentation result).
        """
        # Weighted combination of segment means
        smoothed = jnp.einsum("ns,s->n", assignments, segment_means)

        return smoothed

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply CNV segmentation to coverage data.

        Args:
            data: Dictionary containing:
                - "coverage": Coverage signal (n_positions,)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:
                    - "coverage": Original coverage
                    - "boundary_probs": Soft boundary probabilities
                    - "segment_assignments": Soft segment memberships
                    - "segment_means": Mean value per segment
                    - "smoothed_coverage": Segmented/smoothed signal
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        coverage = data["coverage"]

        # Compute embeddings
        embeddings = self.compute_embeddings(coverage)

        # Detect boundaries
        boundary_probs = self.compute_boundary_probs(embeddings)

        # Soft segment assignments
        segment_assignments = self.compute_segment_assignments(embeddings)

        # Segment statistics
        segment_means = self.compute_segment_means(coverage, segment_assignments)

        # Smoothed signal
        smoothed_coverage = self.compute_smoothed_coverage(
            coverage, segment_assignments, segment_means
        )

        # Build output data
        transformed_data = {
            "coverage": coverage,
            "boundary_probs": boundary_probs,
            "segment_assignments": segment_assignments,
            "segment_means": segment_means,
            "smoothed_coverage": smoothed_coverage,
        }

        return transformed_data, state, metadata
