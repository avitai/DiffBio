"""Differentiable quality filter operator for bioinformatics sequences.

This module provides a soft quality filtering operator that down-weights
low-quality positions in sequences using a differentiable sigmoid function.
"""

from dataclasses import dataclass
from typing import Any

import jax
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import PyTree

from diffbio.configs import DiffBioOperatorConfig
from diffbio.constants import PHRED_QUALITY_THRESHOLD
from diffbio.utils.nn_utils import init_learnable_param


@dataclass
class QualityFilterConfig(DiffBioOperatorConfig):
    """Configuration for DifferentiableQualityFilter.

    Attributes:
        initial_threshold: Initial Phred quality score threshold.
            Positions with quality below this are down-weighted.
            Default is 20.0 (1% error rate).
    """

    initial_threshold: float = PHRED_QUALITY_THRESHOLD


class DifferentiableQualityFilter(OperatorModule):
    """Differentiable quality filter for DNA/RNA sequences.

    This operator applies soft quality filtering using a sigmoid function
    to weight sequence positions by their quality scores. High-quality
    positions (above threshold) pass through with high weight, while
    low-quality positions are down-weighted.

    The threshold is a learnable parameter that can be optimized
    end-to-end with the rest of the pipeline.

    Formula:
        retention_weight = sigmoid(quality_score - threshold)
        filtered_sequence = sequence * retention_weight

    Args:
        config: QualityFilterConfig with initial threshold
        rngs: Flax NNX random number generators

    Example:
        >>> config = QualityFilterConfig(initial_threshold=20.0)
        >>> filter_op = DifferentiableQualityFilter(config, rngs=nnx.Rngs(42))
        >>> data = {"sequence": encoded_seq, "quality_scores": quality}
        >>> filtered_data, state, meta = filter_op.apply(data, {}, None, None)
    """

    def __init__(
        self,
        config: QualityFilterConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the quality filter with learnable threshold.

        Args:
            config: Quality filter configuration
            rngs: Random number generators (optional for deterministic ops)
            name: Optional operator name
        """
        super().__init__(config, rngs=rngs, name=name)

        # Learnable threshold parameter
        self.threshold = init_learnable_param(config.initial_threshold)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply soft quality filtering to sequence data.

        This method applies a differentiable quality filter that weights
        each position by sigmoid(quality - threshold). High quality
        positions retain most of their value, while low quality positions
        are down-weighted.

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
                - transformed_data contains weighted sequence and original quality
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        sequence = data["sequence"]
        quality_scores = data["quality_scores"]

        # Compute retention weights using sigmoid
        # sigmoid(q - threshold): high quality -> weight ~1, low quality -> weight ~0
        retention_weights = jax.nn.sigmoid(quality_scores - self.threshold[...])

        # Apply weights to sequence (broadcast over alphabet dimension)
        weighted_sequence = sequence * retention_weights[:, None]

        # Build output data (preserve quality scores for downstream use)
        transformed_data = {
            "sequence": weighted_sequence,
            "quality_scores": quality_scores,
        }

        return transformed_data, state, metadata
