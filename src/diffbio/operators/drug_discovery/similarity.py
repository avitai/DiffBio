"""Differentiable molecular similarity operator.

This module implements differentiable similarity metrics for comparing
molecular fingerprints, enabling gradient-based optimization of similarity.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx


@dataclass
class MolecularSimilarityConfig(OperatorConfig):
    """Configuration for molecular similarity operator.

    Attributes:
        similarity_type: Type of similarity metric ("tanimoto", "cosine", "dice").
        temperature: Temperature for soft similarity (higher = sharper).
        stochastic: Whether operator uses random sampling.
        stream_name: Optional stream name for data routing.
    """

    similarity_type: str = "tanimoto"
    temperature: float = 1.0
    stochastic: bool = False
    stream_name: str | None = None


def tanimoto_similarity(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Compute differentiable Tanimoto similarity.

    For continuous vectors, uses the generalized Tanimoto formula:
        T(a, b) = (a · b) / (|a|² + |b|² - a · b)

    Args:
        a: First fingerprint vector.
        b: Second fingerprint vector.
        eps: Small constant for numerical stability.

    Returns:
        Similarity score in [0, 1].
    """
    dot_product = jnp.sum(a * b)
    norm_a_sq = jnp.sum(a * a)
    norm_b_sq = jnp.sum(b * b)

    # Generalized Tanimoto for continuous vectors
    similarity = dot_product / (norm_a_sq + norm_b_sq - dot_product + eps)

    # Clamp to [0, 1] for numerical stability
    return jnp.clip(similarity, 0.0, 1.0)


def cosine_similarity(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Compute cosine similarity.

    Args:
        a: First vector.
        b: Second vector.
        eps: Small constant for numerical stability.

    Returns:
        Similarity score in [-1, 1].
    """
    dot_product = jnp.sum(a * b)
    norm_a = jnp.linalg.norm(a)
    norm_b = jnp.linalg.norm(b)

    return dot_product / (norm_a * norm_b + eps)


def dice_similarity(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Compute Dice similarity coefficient.

    For continuous vectors:
        Dice(a, b) = 2 * (a · b) / (|a|² + |b|²)

    Args:
        a: First vector.
        b: Second vector.
        eps: Small constant for numerical stability.

    Returns:
        Similarity score in [0, 1].
    """
    dot_product = jnp.sum(a * b)
    norm_a_sq = jnp.sum(a * a)
    norm_b_sq = jnp.sum(b * b)

    similarity = 2 * dot_product / (norm_a_sq + norm_b_sq + eps)

    return jnp.clip(similarity, 0.0, 1.0)


class MolecularSimilarityOperator(OperatorModule):
    """Differentiable molecular similarity operator.

    Computes similarity between molecular fingerprints using various
    differentiable metrics. Supports Tanimoto, cosine, and Dice similarity.

    Example:
        ```python
        config = MolecularSimilarityConfig(similarity_type="tanimoto")
        sim_op = MolecularSimilarityOperator(config, rngs=nnx.Rngs(42))
        data = {"fingerprint_a": fp1, "fingerprint_b": fp2}
        result, _, _ = sim_op.apply(data, {}, None)
        similarity = result["similarity"]  # scalar in [0, 1]
        ```
    """

    def __init__(self, config: MolecularSimilarityConfig, *, rngs: nnx.Rngs | None = None):
        """Initialize similarity operator.

        Args:
            config: Similarity configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)
        self.config: MolecularSimilarityConfig = config

        # Fix: wrap _unique_id as static for jax.grad compatibility
        # (datarax stores it as plain int which causes gradient errors)
        self._unique_id = nnx.static(self._unique_id)

        # Select similarity function
        if config.similarity_type == "tanimoto":
            self._similarity_fn = tanimoto_similarity
        elif config.similarity_type == "cosine":
            self._similarity_fn = cosine_similarity
        elif config.similarity_type == "dice":
            self._similarity_fn = dice_similarity
        else:
            raise ValueError(f"Unknown similarity type: {config.similarity_type}")

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Compute similarity between two fingerprints.

        Args:
            data: Input data containing:
                - fingerprint_a: First fingerprint vector
                - fingerprint_b: Second fingerprint vector
            state: Per-element state (passed through).
            metadata: Optional metadata.
            random_params: Unused random parameters.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of:
                - data with added "similarity" key
                - unchanged state
                - unchanged metadata
        """
        fp_a = data["fingerprint_a"]
        fp_b = data["fingerprint_b"]

        similarity = self._similarity_fn(fp_a, fp_b)

        result = {
            **data,
            "similarity": similarity,
        }

        return result, state, metadata


def create_similarity_operator(
    similarity_type: str = "tanimoto",
    temperature: float = 1.0,
) -> MolecularSimilarityOperator:
    """Create a molecular similarity operator.

    Args:
        similarity_type: Type of similarity ("tanimoto", "cosine", "dice").
        temperature: Temperature parameter.

    Returns:
        Configured MolecularSimilarityOperator.
    """
    config = MolecularSimilarityConfig(
        similarity_type=similarity_type,
        temperature=temperature,
    )
    return MolecularSimilarityOperator(config, rngs=nnx.Rngs(42))
