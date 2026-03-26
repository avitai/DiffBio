"""Enhanced end-to-end differentiable variant calling pipeline.

This module provides an enhanced variant calling pipeline that composes:
1. Quality filtering (preprocessing) - Filter low-quality bases
2. Pileup generation - Aggregate reads at each position
3. CNN classification - DeepVariant-style variant classification
4. Quality recalibration - VQSR-style variant quality filtering

The pipeline is fully differentiable, enabling gradient-based optimization
of all components jointly.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array

from diffbio.operators.quality_filter import (
    DifferentiableQualityFilter,
    QualityFilterConfig,
)
from diffbio.operators.variant import (
    CNNVariantClassifier,
    CNNVariantClassifierConfig,
    DifferentiablePileup,
    PileupConfig,
    SoftVariantQualityFilter,
    VariantQualityFilterConfig,
)
from diffbio.utils.nn_utils import extract_windows_1d

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnhancedVariantCallingPipelineConfig(OperatorConfig):
    # pylint: disable=too-many-instance-attributes
    """Configuration for the enhanced variant calling pipeline.

    Attributes:
        reference_length: Length of reference sequence.
        num_classes: Number of variant classes (default: 3 for ref/snp/indel).
        quality_threshold: Initial quality score threshold for filtering.
        pileup_window_size: Window size for pileup context.
        cnn_input_height: Height of pileup image for CNN (coverage depth).
        cnn_hidden_channels: Hidden channels for CNN classifier.
        cnn_fc_dims: Fully connected layer dimensions for CNN.
        cnn_dropout_rate: Dropout rate for CNN classifier.
        quality_recal_n_components: Number of GMM components for quality recalibration.
        quality_recal_n_features: Number of features for quality recalibration.
        quality_recal_threshold: Threshold for quality filtering.
        enable_preprocessing: Whether to enable quality filtering preprocessing.
        enable_quality_recalibration: Whether to enable quality recalibration.
    """

    reference_length: int = 1000
    num_classes: int = 3
    quality_threshold: float = 20.0
    pileup_window_size: int = 11
    cnn_input_height: int = 100
    cnn_hidden_channels: tuple[int, ...] = (64, 128, 256)
    cnn_fc_dims: tuple[int, ...] = (256, 128)
    cnn_dropout_rate: float = 0.1
    quality_recal_n_components: int = 3
    quality_recal_n_features: int = 4
    quality_recal_threshold: float = 0.5
    enable_preprocessing: bool = True
    enable_quality_recalibration: bool = True

    def __post_init__(self) -> None:
        """Set non-default stochastic fields."""
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "sample")
        super().__post_init__()


class EnhancedVariantCallingPipeline(OperatorModule):
    """Enhanced end-to-end differentiable variant calling pipeline.

    This pipeline processes sequencing reads to call variants using a
    DeepVariant-style CNN classifier followed by VQSR-style quality
    recalibration:

    Input data structure:
        - reads: Float[Array, "num_reads read_length 4"] - One-hot encoded reads
        - positions: Int[Array, "num_reads"] - Read start positions on reference
        - quality: Float[Array, "num_reads read_length"] - Base quality scores

    Output data structure (adds):
        - pileup: Float[Array, "reference_length 4"] - Aggregated base frequencies
        - logits: Float[Array, "reference_length num_classes"] - Raw predictions
        - probabilities: Float[Array, "reference_length num_classes"] - Class probs
        - quality_scores: Float[Array, "reference_length"] - Recalibrated quality
        - filter_weights: Float[Array, "reference_length"] - Soft filter weights

    The pipeline is fully differentiable, supporting gradient-based training
    to optimize all components jointly.

    Example:
        ```python
        config = EnhancedVariantCallingPipelineConfig(reference_length=1000)
        pipeline = EnhancedVariantCallingPipeline(config, rngs=nnx.Rngs(42))
        result, state, meta = pipeline.apply(data, {}, None)
        probs = result["probabilities"]
        ```
    """

    def __init__(
        self,
        config: EnhancedVariantCallingPipelineConfig,
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ):
        """Initialize the enhanced variant calling pipeline.

        Args:
            config: Pipeline configuration.
            rngs: Random number generators for parameter initialization.
            name: Optional name for the pipeline.
        """
        super().__init__(config, rngs=rngs, name=name)

        # 1. Quality filter for preprocessing (optional)
        self.quality_filter = (
            DifferentiableQualityFilter(
                QualityFilterConfig(initial_threshold=config.quality_threshold),
                rngs=rngs,
            )
            if config.enable_preprocessing
            else None
        )

        # 2. Pileup generation
        self.pileup = DifferentiablePileup(
            PileupConfig(
                reference_length=config.reference_length,
                window_size=config.pileup_window_size,
                use_quality_weights=True,
            ),
            rngs=rngs,
        )

        # 3. CNN classifier (DeepVariant-style)
        # Use 4 channels (A, C, G, T) from the pileup
        self.cnn_classifier = CNNVariantClassifier(
            CNNVariantClassifierConfig(
                num_classes=config.num_classes,
                input_height=config.cnn_input_height,
                input_width=config.pileup_window_size,
                num_channels=4,  # A, C, G, T from pileup
                hidden_channels=config.cnn_hidden_channels,
                fc_dims=config.cnn_fc_dims,
                dropout_rate=config.cnn_dropout_rate,
            ),
            rngs=rngs,
        )

        # 4. Quality recalibration (optional)
        self.quality_recalibration = (
            SoftVariantQualityFilter(
                VariantQualityFilterConfig(
                    n_components=config.quality_recal_n_components,
                    n_features=config.quality_recal_n_features,
                    threshold=config.quality_recal_threshold,
                ),
                rngs=rngs,
            )
            if config.enable_quality_recalibration
            else None
        )

    def apply(
        self,
        data: dict[str, Array],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Array], dict[str, Any], dict[str, Any] | None]:
        """Apply the enhanced variant calling pipeline.

        Args:
            data: Input data containing:
                - reads: Float[Array, "num_reads read_length 4"]
                - positions: Int[Array, "num_reads"]
                - quality: Float[Array, "num_reads read_length"]
            state: Element state (passed through).
            metadata: Element metadata (passed through).
            random_params: Random parameters for stochastic operations.
            stats: Optional statistics dict.

        Returns:
            Tuple of (output_data, state, metadata) where output_data contains
            all input keys plus variant calling outputs.
        """
        reads = data["reads"]
        positions = data["positions"]
        quality = data["quality"]

        # Step 1: Quality filtering (optional)
        if self.quality_filter is not None:
            # Apply quality filter per-base
            num_reads, read_length, _ = reads.shape
            reads_flat = reads.reshape(-1, 4)
            quality_flat = quality.reshape(-1)

            filter_data = {"sequence": reads_flat, "quality_scores": quality_flat}
            filter_result, _, _ = self.quality_filter.apply(filter_data, {}, None)

            reads = filter_result["sequence"].reshape(num_reads, read_length, 4)
            quality = filter_result["quality_scores"].reshape(num_reads, read_length)

        # Step 2: Generate pileup
        pileup_data = {
            "reads": reads,
            "positions": positions,
            "quality": quality,
        }
        pileup_result, _, _ = self.pileup.apply(pileup_data, {}, None)
        pileup = pileup_result["pileup"]  # (reference_length, 4)

        # Step 3: CNN classification
        # Extract windows around each position for CNN input
        # CNN expects (batch, height, width, channels)
        ref_length = self.config.reference_length
        window_size = self.config.pileup_window_size

        # Extract windows: (ref_length, window_size, 4)
        # extract_windows_1d handles padding internally
        windows = extract_windows_1d(pileup, window_size)

        # Add depth dimension for CNN: (ref_length, height, width, channels)
        # Use pileup as a simple "pileup image" - repeat to create height
        cnn_input_height = self.config.cnn_input_height
        # Create a simplified pileup image by repeating the window pattern
        pileup_images = jnp.broadcast_to(
            windows[:, None, :, :], (ref_length, cnn_input_height, window_size, 4)
        )

        # Apply CNN classifier
        cnn_data = {"pileup_image": pileup_images}
        cnn_result, _, _ = self.cnn_classifier.apply(cnn_data, {}, None)

        logits = cnn_result["logits"]  # (ref_length, num_classes)
        probabilities = cnn_result["class_probs"]  # CNN outputs class_probs

        # Build output
        output_data = {
            **data,
            "pileup": pileup,
            "logits": logits,
            "probabilities": probabilities,
        }

        # Step 4: Quality recalibration (optional)
        if self.quality_recalibration is not None:
            # Compute variant features for quality recalibration
            # Features: depth, max_prob, entropy, strand_balance
            depth = pileup.sum(axis=-1)  # Total coverage at each position
            max_prob = probabilities.max(axis=-1)  # Confidence of prediction
            entropy = -jnp.sum(
                probabilities * jnp.log(probabilities + 1e-10), axis=-1
            )  # Prediction entropy
            # Simple strand balance proxy (ratio of first two bases)
            strand_balance = jnp.abs(pileup[:, 0] - pileup[:, 1]) / (depth + 1e-10)

            variant_features = jnp.stack(
                [depth, max_prob, entropy, strand_balance], axis=-1
            )  # (ref_length, 4)

            recal_data = {"variant_features": variant_features}
            recal_result, _, _ = self.quality_recalibration.apply(recal_data, {}, None)

            output_data["quality_scores"] = recal_result["quality_scores"]
            output_data["filter_weights"] = recal_result["filter_weights"]

        return output_data, state, metadata


def create_enhanced_variant_calling_pipeline(
    reference_length: int = 1000,
    num_classes: int = 3,
    pileup_window_size: int = 11,
    cnn_hidden_channels: tuple[int, ...] | None = None,
    cnn_fc_dims: tuple[int, ...] | None = None,
    enable_preprocessing: bool = True,
    enable_quality_recalibration: bool = True,
    seed: int = 42,
) -> EnhancedVariantCallingPipeline:
    """Factory function to create an enhanced variant calling pipeline.

    Args:
        reference_length: Length of reference sequence.
        num_classes: Number of variant classes.
        pileup_window_size: Window size for pileup context.
        cnn_hidden_channels: Hidden channels for CNN classifier.
        cnn_fc_dims: Fully connected dimensions for CNN.
        enable_preprocessing: Whether to enable quality filtering.
        enable_quality_recalibration: Whether to enable quality recalibration.
        seed: Random seed.

    Returns:
        Configured EnhancedVariantCallingPipeline instance.
    """
    if cnn_hidden_channels is None:
        cnn_hidden_channels = (64, 128, 256)
    if cnn_fc_dims is None:
        cnn_fc_dims = (256, 128)

    config = EnhancedVariantCallingPipelineConfig(
        reference_length=reference_length,
        num_classes=num_classes,
        pileup_window_size=pileup_window_size,
        cnn_hidden_channels=cnn_hidden_channels,
        cnn_fc_dims=cnn_fc_dims,
        enable_preprocessing=enable_preprocessing,
        enable_quality_recalibration=enable_quality_recalibration,
    )
    rngs = nnx.Rngs(seed)
    return EnhancedVariantCallingPipeline(config, rngs=rngs)
