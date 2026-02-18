"""Quality filtering utilities for DiffBio pipelines.

This module provides shared quality filtering functions used across
multiple pipeline implementations, avoiding code duplication.
"""

from datarax.core.operator import OperatorModule
from jaxtyping import Array, Float


def apply_quality_filter(
    quality_filter: OperatorModule,
    reads: Float[Array, "num_reads read_length 4"],
    quality: Float[Array, "num_reads read_length"],
) -> tuple[Float[Array, "num_reads read_length 4"], Float[Array, "num_reads read_length"]]:
    """Apply quality filtering to reads using a differentiable quality filter.

    Flattens reads and quality scores for per-base filtering, then reshapes
    back to the original dimensions.

    Args:
        quality_filter: A differentiable quality filter operator
            (e.g., DifferentiableQualityFilter).
        reads: One-hot encoded reads of shape (num_reads, read_length, 4).
        quality: Base quality scores of shape (num_reads, read_length).

    Returns:
        Tuple of (filtered_reads, filtered_quality) with the same shapes
        as the inputs, where low-quality bases have been soft-masked.
    """
    num_reads, read_length, _ = reads.shape

    # Flatten for quality filter (treats each base independently)
    reads_flat = reads.reshape(-1, 4)
    quality_flat = quality.reshape(-1)

    # Apply filter
    filter_data = {"sequence": reads_flat, "quality_scores": quality_flat}
    filtered_result, _, _ = quality_filter.apply(filter_data, {}, None)

    # Reshape back
    filtered_reads = filtered_result["sequence"].reshape(num_reads, read_length, 4)
    filtered_quality = filtered_result["quality_scores"].reshape(num_reads, read_length)

    return filtered_reads, filtered_quality
