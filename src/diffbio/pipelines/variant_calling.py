"""End-to-end differentiable variant calling pipeline.

This module provides a complete variant calling pipeline that composes:
1. Quality filtering - Filter low-quality reads
2. Pileup generation - Aggregate reads at each position
3. Variant classification - Classify each position as variant/reference

The pipeline is fully differentiable, enabling gradient-based optimization
of all components jointly.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float

from diffbio.constants import ClassifierType
from diffbio.utils.nn_utils import extract_windows_1d
from diffbio.utils.quality import apply_quality_filter
from diffbio.operators.quality_filter import (
    DifferentiableQualityFilter,
    QualityFilterConfig,
)
from diffbio.operators.variant import (
    CNNVariantClassifier,
    CNNVariantClassifierConfig,
    DifferentiablePileup,
    PileupConfig,
    VariantClassifier,
    VariantClassifierConfig,
)


@dataclass
class VariantCallingPipelineConfig(OperatorConfig):
    """Configuration for the variant calling pipeline.

    Attributes:
        reference_length: Length of reference sequence
        num_classes: Number of variant classes (default: 3 for ref/snp/indel)
        quality_threshold: Initial quality score threshold for filtering
        pileup_window_size: Window size for pileup context
        classifier_hidden_dim: Hidden dimension for classifier MLP
        use_quality_weights: Whether to weight pileup by quality scores
        classifier_type: Type of classifier (ClassifierType.MLP or ClassifierType.CNN)
        cnn_hidden_channels: Hidden channels for CNN classifier
        cnn_fc_dims: Fully connected layer dimensions for CNN
        apply_pileup_softmax: Whether to apply softmax to pileup output
    """

    reference_length: int = 100
    num_classes: int = 3
    quality_threshold: float = 20.0
    pileup_window_size: int = 11
    classifier_hidden_dim: int = 64
    use_quality_weights: bool = True
    classifier_type: str = ClassifierType.MLP  # ClassifierType.MLP or ClassifierType.CNN
    cnn_hidden_channels: list[int] = field(default_factory=lambda: [32, 64])
    cnn_fc_dims: list[int] = field(default_factory=lambda: [64, 32])
    apply_pileup_softmax: bool = True  # False is better for variant detection
    stochastic: bool = field(default=False, repr=False)


class VariantCallingPipeline(OperatorModule):
    """End-to-end differentiable variant calling pipeline.

    This pipeline processes sequencing reads to call variants:

    Input data structure:
        - reads: Float[Array, "num_reads read_length 4"] - One-hot encoded reads
        - positions: Int[Array, "num_reads"] - Read start positions on reference
        - quality: Float[Array, "num_reads read_length"] - Base quality scores

    Output data structure (adds):
        - pileup: Float[Array, "reference_length 4"] - Aggregated base frequencies
        - logits: Float[Array, "reference_length num_classes"] - Raw predictions
        - probabilities: Float[Array, "reference_length num_classes"] - Class probs

    The pipeline is fully differentiable, supporting gradient-based training
    to optimize quality filtering, pileup aggregation, and classification jointly.

    Example:
        ```python
        config = VariantCallingPipelineConfig(reference_length=100)
        pipeline = VariantCallingPipeline(config, rngs=nnx.Rngs(42))
        pipeline.eval_mode()  # Disable dropout for inference
        # Process a batch of samples
        result_batch = pipeline(input_batch)
        probs = result_batch.data.get_value()["probabilities"]
        ```
    """

    def __init__(
        self,
        config: VariantCallingPipelineConfig,
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ):
        """Initialize the variant calling pipeline.

        Args:
            config: Pipeline configuration
            rngs: Random number generators for parameter initialization
            name: Optional name for the pipeline
        """
        super().__init__(config, rngs=rngs, name=name)

        # Store typed config for attribute access
        self.pipeline_config: VariantCallingPipelineConfig = config

        # Initialize sub-operators
        # 1. Quality filter for preprocessing reads
        self.quality_filter = DifferentiableQualityFilter(
            QualityFilterConfig(initial_threshold=config.quality_threshold),
            rngs=rngs,
        )

        # 2. Pileup generator - always return coverage and quality for CNN
        use_multichannel = config.classifier_type == ClassifierType.CNN
        self.pileup = DifferentiablePileup(
            PileupConfig(
                window_size=config.pileup_window_size,
                use_quality_weights=config.use_quality_weights,
                reference_length=config.reference_length,
                return_coverage=use_multichannel,
                return_quality=use_multichannel,
                apply_softmax=config.apply_pileup_softmax,
            ),
            rngs=rngs,
        )

        # 3. Variant classifier (per-position)
        if config.classifier_type == ClassifierType.CNN:
            # CNN classifier takes pileup images with multiple channels
            # Channels: 4 (base) + 1 (coverage) + 1 (quality) = 6
            self.classifier = CNNVariantClassifier(
                CNNVariantClassifierConfig(
                    num_classes=config.num_classes,
                    input_height=1,  # Single "row" per position
                    input_width=config.pileup_window_size,
                    num_channels=6,  # base(4) + coverage(1) + quality(1)
                    hidden_channels=config.cnn_hidden_channels,
                    fc_dims=config.cnn_fc_dims,
                ),
                rngs=rngs,
            )
            self._use_cnn = True
        else:
            # MLP classifier
            self.classifier = VariantClassifier(
                VariantClassifierConfig(
                    num_classes=config.num_classes,
                    hidden_dim=config.classifier_hidden_dim,
                    input_window=config.pileup_window_size,
                ),
                rngs=rngs,
            )
            self._use_cnn = False

    def set_training(self, training: bool = True) -> None:
        """Set pipeline training mode.

        Args:
            training: If True, enable dropout. If False, disable dropout.
        """
        if training:
            self.classifier.train()
        else:
            self.classifier.eval()

    def train_mode(self) -> None:
        """Set pipeline to training mode (enables dropout)."""
        self.classifier.train()

    def eval_mode(self) -> None:
        """Set pipeline to evaluation mode (disables dropout)."""
        self.classifier.eval()

    def apply(
        self,
        data: dict[str, Array],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Any], dict[str, Any] | None]:
        """Apply the full variant calling pipeline to a single sample.

        Args:
            data: Input data containing:
                - reads: Float[Array, "num_reads read_length 4"]
                - positions: Int[Array, "num_reads"]
                - quality: Float[Array, "num_reads read_length"]
            state: Element state (passed through)
            metadata: Element metadata (passed through)
            random_params: Not used (deterministic pipeline)
            stats: Optional statistics dict

        Returns:
            Tuple of (output_data, state, metadata) where output_data contains
            all input keys plus pileup, logits, and probabilities.
        """
        reads = data["reads"]
        positions = data["positions"]
        quality = data["quality"]

        # Step 1: Quality-weighted filtering
        # Apply quality filter to each read position
        filtered_reads, filtered_quality = apply_quality_filter(self.quality_filter, reads, quality)

        # Step 2: Generate pileup
        pileup_data = {
            "reads": filtered_reads,
            "positions": positions,
            "quality": filtered_quality,
        }
        pileup_result, _, _ = self.pileup.apply(pileup_data, {}, None)
        pileup = pileup_result["pileup"]  # Shape: (reference_length, 4)

        # Extract coverage and quality for CNN (if available)
        coverage = pileup_result.get("coverage")  # (reference_length, 1) or None
        mean_quality = pileup_result.get("mean_quality")  # (reference_length, 1) or None

        # Step 3: Classify each position using sliding window
        logits, probabilities = self._classify_positions(pileup, coverage, mean_quality)

        # Build output preserving input keys
        output_data = {
            **data,
            "filtered_reads": filtered_reads,
            "filtered_quality": filtered_quality,
            "pileup": pileup,
            "logits": logits,
            "probabilities": probabilities,
        }

        # Add coverage and quality if available
        if coverage is not None:
            output_data["coverage"] = coverage
        if mean_quality is not None:
            output_data["mean_quality"] = mean_quality

        return output_data, state, metadata

    def _classify_positions(
        self,
        pileup: Float[Array, "reference_length 4"],
        coverage: Float[Array, "reference_length 1"] | None = None,
        mean_quality: Float[Array, "reference_length 1"] | None = None,
    ) -> tuple[
        Float[Array, "reference_length num_classes"], Float[Array, "reference_length num_classes"]
    ]:
        """Classify each reference position using pileup windows.

        Extracts a window around each position and classifies it.
        Uses padding at boundaries.

        For MLP: Uses batch processing through the classifier's linear layers.
        For CNN: Constructs 6-channel pileup images and uses CNN classifier.
        """
        reference_length = pileup.shape[0]
        window_size = self.pipeline_config.pileup_window_size
        half_window = window_size // 2

        if self._use_cnn:
            return self._classify_positions_cnn(
                pileup, coverage, mean_quality, reference_length, window_size, half_window
            )
        else:
            return self._classify_positions_mlp(pileup, reference_length, window_size, half_window)

    def _classify_positions_mlp(
        self,
        pileup: Float[Array, "reference_length 4"],
        reference_length: int,
        window_size: int,
        half_window: int,
    ) -> tuple[
        Float[Array, "reference_length num_classes"], Float[Array, "reference_length num_classes"]
    ]:
        """Classify positions using MLP classifier."""
        del reference_length, half_window  # Not needed - handled by extract_windows_1d

        # Extract all windows using utility function
        all_windows = extract_windows_1d(
            pileup, window_size=window_size, pad_mode="edge"
        )  # (reference_length, window_size, 4)

        # Classify all windows using batch processing through the classifier
        batch_size = all_windows.shape[0]
        x = all_windows.reshape(batch_size, -1)  # (reference_length, window_size * 4)

        # Forward pass through classifier layers with batch dimension
        x = self.classifier.input_layer(x)
        x = nnx.relu(x)

        for layer, dropout in zip(
            self.classifier.layers, self.classifier.dropout_layers, strict=False
        ):
            x = layer(x)
            x = nnx.relu(x)
            x = dropout(x)

        logits = self.classifier.output_layer(x)  # (reference_length, num_classes)
        probabilities = jax.nn.softmax(logits, axis=-1)

        return logits, probabilities

    def _classify_positions_cnn(
        self,
        pileup: Float[Array, "reference_length 4"],
        coverage: Float[Array, "reference_length 1"] | None,
        mean_quality: Float[Array, "reference_length 1"] | None,
        reference_length: int,
        window_size: int,
        half_window: int,
    ) -> tuple[
        Float[Array, "reference_length num_classes"], Float[Array, "reference_length num_classes"]
    ]:
        """Classify positions using CNN classifier.

        Creates 6-channel pileup images:
        - Channels 0-3: Base distributions (A, C, G, T)
        - Channel 4: Coverage (normalized by max_coverage)
        - Channel 5: Mean quality (normalized to 0-1)
        """
        del half_window  # Not needed - handled by extract_windows_1d
        max_coverage = self.pileup.config.max_coverage

        # Normalize coverage and quality if provided
        if coverage is not None:
            norm_coverage = coverage / max_coverage  # (reference_length, 1)
        else:
            norm_coverage = jnp.zeros((reference_length, 1))

        if mean_quality is not None:
            norm_quality = mean_quality / 40.0  # Normalize to ~0-1 (Phred max ~40)
        else:
            norm_quality = jnp.zeros((reference_length, 1))

        # Concatenate all channels: (reference_length, 6)
        pileup_6ch = jnp.concatenate([pileup, norm_coverage, norm_quality], axis=-1)

        # Extract all windows using utility function
        all_windows = extract_windows_1d(
            pileup_6ch, window_size=window_size, pad_mode="edge"
        )  # (reference_length, window_size, 6)

        # Reshape for CNN: (batch, height, width=window_size, channels=6)
        # We need height >= 2 for CNN's max_pool(2,2) to work
        # Replicate the single row to create a 2D image that the CNN can process
        min_height = 8  # Minimum height for CNN pooling
        pileup_images = all_windows[:, None, :, :]  # (reference_length, 1, window_size, 6)
        pileup_images = jnp.tile(pileup_images, (1, min_height, 1, 1))  # (ref_len, 8, window, 6)

        # Classify using CNN
        logits = self.classifier.classify(pileup_images)  # (reference_length, num_classes)
        probabilities = jax.nn.softmax(logits, axis=-1)

        return logits, probabilities

    def call_variants(
        self,
        batch: Batch,
        threshold: float = 0.5,
    ) -> dict[str, Array]:
        """Convenience method to call variants from a batch.

        Args:
            batch: Input batch with reads, positions, quality
            threshold: Probability threshold for variant calling

        Returns:
            Dict containing:
                - predictions: Int[Array, "batch reference_length"] - Predicted classes
                - probabilities: Float[Array, "batch reference_length num_classes"]
                - variant_positions: List of (batch_idx, position) tuples
        """
        # Process batch
        result_batch = self.apply_batch(batch)
        result_data = result_batch.data.get_value()

        probabilities = result_data["probabilities"]
        predictions = jnp.argmax(probabilities, axis=-1)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "pileup": result_data["pileup"],
        }


def create_variant_calling_pipeline(
    reference_length: int = 100,
    num_classes: int = 3,
    quality_threshold: float = 20.0,
    hidden_dim: int = 64,
    classifier_type: str = ClassifierType.MLP,
    pileup_window_size: int = 11,
    apply_pileup_softmax: bool = True,
    seed: int = 42,
) -> VariantCallingPipeline:
    """Factory function to create a variant calling pipeline.

    Args:
        reference_length: Length of reference sequence
        num_classes: Number of variant classes
        quality_threshold: Quality score threshold
        hidden_dim: Hidden dimension for classifier
        classifier_type: Type of classifier (ClassifierType.MLP or ClassifierType.CNN)
        pileup_window_size: Window size for pileup context
        apply_pileup_softmax: Whether to apply softmax to pileup (False is better
            for variant detection as it preserves raw coverage-weighted signals)
        seed: Random seed

    Returns:
        Configured VariantCallingPipeline instance
    """
    config = VariantCallingPipelineConfig(
        reference_length=reference_length,
        num_classes=num_classes,
        quality_threshold=quality_threshold,
        classifier_hidden_dim=hidden_dim,
        classifier_type=classifier_type,
        pileup_window_size=pileup_window_size,
        apply_pileup_softmax=apply_pileup_softmax,
    )
    rngs = nnx.Rngs(seed)
    pipeline = VariantCallingPipeline(config, rngs=rngs)
    pipeline.eval_mode()
    return pipeline


def create_cnn_variant_pipeline(
    reference_length: int = 100,
    num_classes: int = 3,
    quality_threshold: float = 20.0,
    pileup_window_size: int = 21,
    cnn_hidden_channels: list[int] | None = None,
    cnn_fc_dims: list[int] | None = None,
    seed: int = 42,
) -> VariantCallingPipeline:
    """Factory function to create a CNN-based variant calling pipeline.

    This creates a pipeline using CNN-based classification, which processes
    multi-channel pileup images similar to DeepVariant. The 6 channels are:
    - 4 base distribution channels (A, C, G, T)
    - 1 coverage channel (normalized)
    - 1 quality channel (normalized)

    Args:
        reference_length: Length of reference sequence
        num_classes: Number of variant classes
        quality_threshold: Quality score threshold
        pileup_window_size: Window size for pileup context (recommend 21+ for CNN)
        cnn_hidden_channels: Hidden channels for CNN layers (default: [32, 64])
        cnn_fc_dims: FC layer dimensions (default: [64, 32])
        seed: Random seed

    Returns:
        Configured VariantCallingPipeline instance with CNN classifier
    """
    if cnn_hidden_channels is None:
        cnn_hidden_channels = [32, 64]
    if cnn_fc_dims is None:
        cnn_fc_dims = [64, 32]

    config = VariantCallingPipelineConfig(
        reference_length=reference_length,
        num_classes=num_classes,
        quality_threshold=quality_threshold,
        classifier_type=ClassifierType.CNN,
        pileup_window_size=pileup_window_size,
        cnn_hidden_channels=cnn_hidden_channels,
        cnn_fc_dims=cnn_fc_dims,
        apply_pileup_softmax=False,  # Better for variant detection
    )
    rngs = nnx.Rngs(seed)
    pipeline = VariantCallingPipeline(config, rngs=rngs)
    pipeline.eval_mode()
    return pipeline
