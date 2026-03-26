"""End-to-end differentiable preprocessing pipeline.

This module provides a complete read preprocessing pipeline that composes:
1. Quality filtering - Filter low-quality bases
2. Adapter removal - Soft trim adapter sequences
3. Duplicate weighting - Assign probabilistic weights based on uniqueness
4. Error correction - Neural network-based base correction

The pipeline is fully differentiable, enabling gradient-based optimization
of all preprocessing components jointly.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array

from diffbio.operators.preprocessing import (
    AdapterRemovalConfig,
    DifferentiableDuplicateWeighting,
    DuplicateWeightingConfig,
    ErrorCorrectionConfig,
    SoftAdapterRemoval,
    SoftErrorCorrection,
)
from diffbio.operators.quality_filter import (
    DifferentiableQualityFilter,
    QualityFilterConfig,
)
from diffbio.utils.quality import apply_quality_filter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingPipelineConfig(OperatorConfig):
    # pylint: disable=too-many-instance-attributes
    """Configuration for the preprocessing pipeline.

    Attributes:
        read_length: Expected read length for initialization.
        adapter_sequence: Adapter sequence to remove (Illumina universal default).
        quality_threshold: Initial quality score threshold for filtering.
        adapter_match_threshold: Threshold for adapter matching.
        adapter_temperature: Temperature for soft adapter trimming.
        duplicate_similarity_threshold: Similarity threshold for duplicate detection.
        error_correction_window: Window size for error correction.
        error_correction_hidden_dim: Hidden dimension for error correction network.
        enable_adapter_removal: Whether to enable adapter removal step.
        enable_duplicate_weighting: Whether to enable duplicate weighting step.
        enable_error_correction: Whether to enable error correction step.
    """

    read_length: int = 150
    adapter_sequence: str = "AGATCGGAAGAG"
    quality_threshold: float = 20.0
    adapter_match_threshold: float = 0.8
    adapter_temperature: float = 1.0
    duplicate_similarity_threshold: float = 0.95
    error_correction_window: int = 11
    error_correction_hidden_dim: int = 64
    enable_adapter_removal: bool = True
    enable_duplicate_weighting: bool = True
    enable_error_correction: bool = True


class PreprocessingPipeline(OperatorModule):
    """End-to-end differentiable preprocessing pipeline.

    This pipeline processes sequencing reads through multiple preprocessing steps:

    Input data structure:
        - reads: Float[Array, "num_reads read_length 4"] - One-hot encoded reads
        - quality: Float[Array, "num_reads read_length"] - Base quality scores

    Output data structure (adds):
        - preprocessed_reads: Float[Array, "num_reads read_length 4"] - Processed reads
        - preprocessed_quality: Float[Array, "num_reads read_length"] - Processed quality
        - read_weights: Float[Array, "num_reads"] - Read uniqueness weights

    The pipeline is fully differentiable, supporting gradient-based training
    to optimize all preprocessing components jointly.

    Example:
        ```python
        config = PreprocessingPipelineConfig(read_length=150)
        pipeline = PreprocessingPipeline(config, rngs=nnx.Rngs(42))
        result, state, meta = pipeline.apply(data, {}, None)
        processed = result["preprocessed_reads"]
        ```
    """

    def __init__(
        self,
        config: PreprocessingPipelineConfig,
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ):
        """Initialize the preprocessing pipeline.

        Args:
            config: Pipeline configuration.
            rngs: Random number generators for parameter initialization.
            name: Optional name for the pipeline.
        """
        super().__init__(config, rngs=rngs, name=name)

        # 1. Quality filter (always enabled)
        self.quality_filter = DifferentiableQualityFilter(
            QualityFilterConfig(initial_threshold=config.quality_threshold),
            rngs=rngs,
        )

        # 2. Adapter removal (optional)
        self.adapter_removal = (
            SoftAdapterRemoval(
                AdapterRemovalConfig(
                    adapter_sequence=config.adapter_sequence,
                    match_threshold=config.adapter_match_threshold,
                    temperature=config.adapter_temperature,
                ),
                rngs=rngs,
            )
            if config.enable_adapter_removal
            else None
        )

        # 3. Duplicate weighting (optional)
        self.duplicate_weighting = (
            DifferentiableDuplicateWeighting(
                DuplicateWeightingConfig(
                    similarity_threshold=config.duplicate_similarity_threshold,
                ),
                rngs=rngs,
            )
            if config.enable_duplicate_weighting
            else None
        )

        # 4. Error correction (optional)
        self.error_correction = (
            SoftErrorCorrection(
                ErrorCorrectionConfig(
                    window_size=config.error_correction_window,
                    hidden_dim=config.error_correction_hidden_dim,
                ),
                rngs=rngs,
            )
            if config.enable_error_correction
            else None
        )

    def apply(
        self,
        data: dict[str, Array],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Any], dict[str, Any] | None]:
        """Apply the full preprocessing pipeline to reads.

        Args:
            data: Input data containing:
                - reads: Float[Array, "num_reads read_length 4"]
                - quality: Float[Array, "num_reads read_length"]
            state: Element state (passed through).
            metadata: Element metadata (passed through).
            random_params: Not used (deterministic pipeline).
            stats: Optional statistics dict.

        Returns:
            Tuple of (output_data, state, metadata) where output_data contains
            all input keys plus preprocessed outputs.
        """
        reads = data["reads"]
        quality = data["quality"]
        num_reads = reads.shape[0]

        # Initialize read weights to 1.0 (all reads equally weighted)
        read_weights = jnp.ones((num_reads,))

        # Step 1: Quality filtering (per-base)
        filtered_reads, filtered_quality = apply_quality_filter(self.quality_filter, reads, quality)

        # Step 2: Adapter removal (optional) - apply per-read using vmap
        if self.adapter_removal is not None:

            def apply_adapter_removal(read, quality):
                adapter_data = {"sequence": read, "quality_scores": quality}
                adapter_result, _, _ = self.adapter_removal.apply(adapter_data, {}, None)
                return adapter_result["sequence"], adapter_result["quality_scores"]

            filtered_reads, filtered_quality = jax.vmap(apply_adapter_removal)(
                filtered_reads, filtered_quality
            )

        # Step 3: Duplicate weighting (optional) - operates on full batch for cross-read comparison
        # Use apply_batch which returns weights for all reads (not just first)
        if self.duplicate_weighting is not None:
            raw_weights, _ = self.duplicate_weighting.apply_batch(filtered_reads, filtered_quality)
            # Normalize weights to [0, 1] range for use as probabilities
            read_weights = raw_weights / jnp.max(raw_weights)

        # Step 4: Error correction (optional) - apply per-read using vmap
        if self.error_correction is not None:

            def apply_error_correction(read, quality):
                ec_data = {"sequence": read, "quality_scores": quality}
                ec_result, _, _ = self.error_correction.apply(ec_data, {}, None)
                return ec_result["sequence"]

            filtered_reads = jax.vmap(apply_error_correction)(filtered_reads, filtered_quality)

        # Build output preserving input keys
        output_data = {
            **data,
            "preprocessed_reads": filtered_reads,
            "preprocessed_quality": filtered_quality,
            "read_weights": read_weights,
        }

        return output_data, state, metadata


def create_preprocessing_pipeline(
    read_length: int = 150,
    quality_threshold: float = 20.0,
    adapter_sequence: str = "AGATCGGAAGAG",
    enable_adapter_removal: bool = True,
    enable_duplicate_weighting: bool = True,
    enable_error_correction: bool = True,
    seed: int = 42,
) -> PreprocessingPipeline:
    """Factory function to create a preprocessing pipeline.

    Args:
        read_length: Expected read length.
        quality_threshold: Quality score threshold.
        adapter_sequence: Adapter sequence to remove.
        enable_adapter_removal: Whether to enable adapter removal.
        enable_duplicate_weighting: Whether to enable duplicate weighting.
        enable_error_correction: Whether to enable error correction.
        seed: Random seed.

    Returns:
        Configured PreprocessingPipeline instance.
    """
    config = PreprocessingPipelineConfig(
        read_length=read_length,
        quality_threshold=quality_threshold,
        adapter_sequence=adapter_sequence,
        enable_adapter_removal=enable_adapter_removal,
        enable_duplicate_weighting=enable_duplicate_weighting,
        enable_error_correction=enable_error_correction,
    )
    rngs = nnx.Rngs(seed)
    return PreprocessingPipeline(config, rngs=rngs)
