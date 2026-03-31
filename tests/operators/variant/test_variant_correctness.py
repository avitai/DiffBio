#!/usr/bin/env python3
"""Variant Calling Benchmark for DiffBio.

This benchmark evaluates DiffBio's variant calling operators for
correctness, differentiability, and performance.

Benchmarks:
- Pileup generation shape and differentiability
- Variant classification output shape and probability normalization
- CNV segmentation boundary detection and segment assignments
- Gradient flow through pileup and classifier
- Throughput for pileup generation

Usage:
    python benchmarks/variant/variant_calling_benchmark.py
    python benchmarks/variant/variant_calling_benchmark.py --quick
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._gradient import check_gradient_flow
from diffbio.operators.variant.classifier import (
    VariantClassifier,
    VariantClassifierConfig,
)
from diffbio.operators.variant.cnv_segmentation import (
    CNVSegmentationConfig,
    DifferentiableCNVSegmentation,
)


# -------------------------------------------------------------------
# Synthetic data generators
# -------------------------------------------------------------------


def generate_pileup_data(
    reference_length: int,
    n_reads: int,
    read_length: int,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic pileup input data.

    Creates one-hot encoded reads, integer start positions,
    and Phred-scale quality scores.

    Args:
        reference_length: Length of the reference sequence.
        n_reads: Number of reads to generate.
        read_length: Length of each read in bases.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with reads, positions, and quality arrays.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # One-hot encoded reads: (n_reads, read_length, 4)
    indices = jax.random.randint(k1, (n_reads, read_length), 0, 4)
    reads = jax.nn.one_hot(indices, 4)

    # Start positions within reference bounds
    max_start = max(reference_length - read_length, 1)
    positions = jax.random.randint(k2, (n_reads,), 0, max_start)

    # Phred quality scores in [10, 40]
    quality = jax.random.uniform(k3, (n_reads, read_length)) * 30.0 + 10.0

    return {"reads": reads, "positions": positions, "quality": quality}


def generate_cnv_coverage(
    n_positions: int,
    seed: int = 42,
) -> jnp.ndarray:
    """Generate synthetic coverage with a copy-number gain region.

    Creates a baseline coverage signal and inserts a region at
    approximately 2x depth to simulate a CNV gain event.

    Args:
        n_positions: Number of genomic positions.
        seed: Random seed for reproducibility.

    Returns:
        1-D coverage array of shape ``(n_positions,)``.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)

    # Baseline Poisson coverage (~30x)
    baseline = jax.random.poisson(k1, 30.0, (n_positions,)).astype(jnp.float32)

    # Insert a 2x gain region in the middle third
    start = n_positions // 3
    end = 2 * n_positions // 3
    gain_noise = jax.random.normal(k2, (end - start,)) * 2.0
    gain = jnp.zeros(n_positions)
    gain = gain.at[start:end].set(30.0 + gain_noise)

    return baseline + gain


# -------------------------------------------------------------------
# Individual operator benchmarks
# -------------------------------------------------------------------


def benchmark_classifier(
    window_size: int,
    num_classes: int,
) -> dict:
    """Benchmark the VariantClassifier operator.

    Checks logit shapes, probability normalization, and gradient flow.

    Args:
        window_size: Pileup window size for classification.
        num_classes: Number of variant classes.

    Returns:
        Dictionary of classifier benchmark metrics.
    """
    print("\n--- Variant Classifier Benchmark ---")

    config = VariantClassifierConfig(
        num_classes=num_classes,
        input_window=window_size,
        hidden_dim=64,
    )
    classifier = VariantClassifier(config, rngs=nnx.Rngs(42))

    # Generate a synthetic pileup window
    key = jax.random.key(43)
    pileup_window = jax.nn.softmax(jax.random.normal(key, (window_size, 4)), axis=-1)
    data = {"pileup_window": pileup_window}

    result, _, _ = classifier.apply(data, {}, None)

    # Shape checks
    logits = result["logits"]
    probs = result["probabilities"]

    logits_shape_ok = logits.shape == (num_classes,)
    prob_sum = float(jnp.sum(probs))
    prob_sum_ok = abs(prob_sum - 1.0) < 1e-4

    print(f"  Logits shape {logits.shape}: {'PASS' if logits_shape_ok else 'FAIL'}")
    print(f"  Probability sum {prob_sum:.6f}: {'PASS' if prob_sum_ok else 'FAIL'}")

    # Gradient flow through classifier
    def classifier_loss(model: VariantClassifier, d: dict) -> jnp.ndarray:
        """Scalar loss for gradient check."""
        out, _, _ = model.apply(d, {}, None)
        return jnp.sum(out["logits"])

    grad_metrics = check_gradient_flow(classifier_loss, classifier, data)
    print(f"  Gradient norm: {grad_metrics.gradient_norm:.6f}")
    print(f"  Gradient non-zero: {grad_metrics.gradient_nonzero}")

    return {
        "classifier_logits_shape_correct": logits_shape_ok,
        "classifier_prob_sum_close_to_one": prob_sum_ok,
        "classifier_prob_sum": prob_sum,
        "classifier_gradient_norm": grad_metrics.gradient_norm,
        "classifier_gradient_nonzero": grad_metrics.gradient_nonzero,
    }


def benchmark_cnv_segmentation(
    n_positions: int,
    max_segments: int,
) -> dict:
    """Benchmark the DifferentiableCNVSegmentation operator.

    Checks boundary probabilities, segment assignments, segment
    means, and smoothed coverage shapes and validity.

    Args:
        n_positions: Number of genomic positions.
        max_segments: Maximum number of segments.

    Returns:
        Dictionary of CNV segmentation benchmark metrics.
    """
    print("\n--- CNV Segmentation Benchmark ---")

    config = CNVSegmentationConfig(
        max_segments=max_segments,
        hidden_dim=64,
        attention_heads=4,
        temperature=1.0,
    )
    segmenter = DifferentiableCNVSegmentation(config, rngs=nnx.Rngs(42))

    coverage = generate_cnv_coverage(n_positions)
    data = {"coverage": coverage}

    result, _, _ = segmenter.apply(data, {}, None)

    boundary_probs = result["boundary_probs"]
    segment_assignments = result["segment_assignments"]
    segment_means = result["segment_means"]
    smoothed_cov = result["smoothed_coverage"]

    # Boundary probs are valid probabilities in [0, 1]
    boundary_valid = bool(jnp.all(boundary_probs >= 0.0) & jnp.all(boundary_probs <= 1.0))

    # Shape checks
    assignments_shape_ok = segment_assignments.shape == (
        n_positions,
        max_segments,
    )
    means_shape_ok = segment_means.shape == (max_segments,)
    smoothed_shape_ok = smoothed_cov.shape == (n_positions,)

    print(f"  Boundary probs in [0,1]: {'PASS' if boundary_valid else 'FAIL'}")
    print(
        f"  Segment assignments shape "
        f"{segment_assignments.shape}: "
        f"{'PASS' if assignments_shape_ok else 'FAIL'}"
    )
    print(f"  Segment means shape {segment_means.shape}: {'PASS' if means_shape_ok else 'FAIL'}")
    print(
        f"  Smoothed coverage shape {smoothed_cov.shape}: {'PASS' if smoothed_shape_ok else 'FAIL'}"
    )

    return {
        "cnv_boundary_probs_valid": boundary_valid,
        "cnv_segment_assignments_shape_correct": assignments_shape_ok,
        "cnv_segment_means_shape_correct": means_shape_ok,
        "cnv_smoothed_coverage_shape_correct": smoothed_shape_ok,
    }
