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

from diffbio.operators.quality_filter import (
    DifferentiableQualityFilter,
    QualityFilterConfig,
)
from diffbio.operators.variant import (
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
    """

    reference_length: int = 100
    num_classes: int = 3
    quality_threshold: float = 20.0
    pileup_window_size: int = 11
    classifier_hidden_dim: int = 64
    use_quality_weights: bool = True
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
        >>> config = VariantCallingPipelineConfig(reference_length=100)
        >>> pipeline = VariantCallingPipeline(config, rngs=nnx.Rngs(42))
        >>> pipeline.eval_mode()  # Disable dropout for inference
        >>>
        >>> # Process a batch of samples
        >>> result_batch = pipeline(input_batch)
        >>> probs = result_batch.data.get_value()["probabilities"]
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

        # 2. Pileup generator
        self.pileup = DifferentiablePileup(
            PileupConfig(
                window_size=config.pileup_window_size,
                use_quality_weights=config.use_quality_weights,
                reference_length=config.reference_length,
            ),
            rngs=rngs,
        )

        # 3. Variant classifier (per-position)
        self.classifier = VariantClassifier(
            VariantClassifierConfig(
                num_classes=config.num_classes,
                hidden_dim=config.classifier_hidden_dim,
                input_window=config.pileup_window_size,
            ),
            rngs=rngs,
        )

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
        filtered_reads, filtered_quality = self._apply_quality_filter(reads, quality)

        # Step 2: Generate pileup
        pileup_data = {
            "reads": filtered_reads,
            "positions": positions,
            "quality": filtered_quality,
        }
        pileup_result, _, _ = self.pileup.apply(pileup_data, {}, None)
        pileup = pileup_result["pileup"]  # Shape: (reference_length, 4)

        # Step 3: Classify each position using sliding window
        logits, probabilities = self._classify_positions(pileup)

        # Build output preserving input keys
        output_data = {
            **data,
            "filtered_reads": filtered_reads,
            "filtered_quality": filtered_quality,
            "pileup": pileup,
            "logits": logits,
            "probabilities": probabilities,
        }

        return output_data, state, metadata

    def _apply_quality_filter(
        self,
        reads: Float[Array, "num_reads read_length 4"],
        quality: Float[Array, "num_reads read_length"],
    ) -> tuple[Float[Array, "num_reads read_length 4"], Float[Array, "num_reads read_length"]]:
        """Apply quality filtering to reads.

        Uses the differentiable quality filter to soft-mask low-quality bases.
        """
        num_reads, read_length, _ = reads.shape

        # Flatten for quality filter (treats each base independently)
        reads_flat = reads.reshape(-1, 4)
        quality_flat = quality.reshape(-1)

        # Apply filter
        filter_data = {"sequence": reads_flat, "quality_scores": quality_flat}
        filtered_result, _, _ = self.quality_filter.apply(filter_data, {}, None)

        # Reshape back
        filtered_reads = filtered_result["sequence"].reshape(num_reads, read_length, 4)
        filtered_quality = filtered_result["quality_scores"].reshape(num_reads, read_length)

        return filtered_reads, filtered_quality

    def _classify_positions(
        self,
        pileup: Float[Array, "reference_length 4"],
    ) -> tuple[
        Float[Array, "reference_length num_classes"], Float[Array, "reference_length num_classes"]
    ]:
        """Classify each reference position using pileup windows.

        Extracts a window around each position and classifies it.
        Uses padding at boundaries.
        """
        reference_length = pileup.shape[0]
        window_size = self.pipeline_config.pileup_window_size
        half_window = window_size // 2

        # Pad pileup for boundary positions
        # Use edge padding (repeat boundary values)
        padded_pileup = jnp.pad(
            pileup,
            ((half_window, half_window), (0, 0)),
            mode="edge",
        )

        # Extract windows for all positions using vmap
        def extract_and_classify(pos: int) -> tuple[Array, Array]:
            window = jax.lax.dynamic_slice(
                padded_pileup,
                (pos, 0),
                (window_size, 4),
            )
            classifier_data = {"pileup_window": window}
            result, _, _ = self.classifier.apply(classifier_data, {}, None)
            return result["logits"], result["probabilities"]

        # Vectorize over all positions
        positions = jnp.arange(reference_length)
        logits, probabilities = jax.vmap(extract_and_classify)(positions)

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
    seed: int = 42,
) -> VariantCallingPipeline:
    """Factory function to create a variant calling pipeline.

    Args:
        reference_length: Length of reference sequence
        num_classes: Number of variant classes
        quality_threshold: Quality score threshold
        hidden_dim: Hidden dimension for classifier
        seed: Random seed

    Returns:
        Configured VariantCallingPipeline instance
    """
    config = VariantCallingPipelineConfig(
        reference_length=reference_length,
        num_classes=num_classes,
        quality_threshold=quality_threshold,
        classifier_hidden_dim=hidden_dim,
    )
    rngs = nnx.Rngs(seed)
    pipeline = VariantCallingPipeline(config, rngs=rngs)
    pipeline.eval_mode()
    return pipeline
