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

import argparse
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._common import (
    check_gradient_flow,
    measure_throughput,
    save_benchmark_result,
)
from diffbio.operators.variant.classifier import (
    VariantClassifier,
    VariantClassifierConfig,
)
from diffbio.operators.variant.cnv_segmentation import (
    CNVSegmentationConfig,
    DifferentiableCNVSegmentation,
)
from diffbio.operators.variant.pileup import (
    DifferentiablePileup,
    PileupConfig,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class VariantCallingBenchmarkResult:
    """Results from the variant calling benchmark.

    Captures shape correctness, probability normalization, gradient
    flow, and throughput metrics for all three variant operators.
    """

    timestamp: str

    # Pileup metrics
    pileup_shape_correct: bool
    pileup_coverage_shape_correct: bool
    pileup_quality_shape_correct: bool

    # Variant classification metrics
    classifier_logits_shape_correct: bool
    classifier_prob_sum_close_to_one: bool
    classifier_prob_sum: float

    # CNV segmentation metrics
    cnv_boundary_probs_valid: bool
    cnv_segment_assignments_shape_correct: bool
    cnv_segment_means_shape_correct: bool
    cnv_smoothed_coverage_shape_correct: bool

    # Gradient flow
    pileup_gradient_norm: float
    pileup_gradient_nonzero: bool
    classifier_gradient_norm: float
    classifier_gradient_nonzero: bool

    # Throughput
    pileup_per_item_ms: float
    pileup_items_per_sec: float

    # Configuration
    reference_length: int
    n_reads: int
    read_length: int
    window_size: int
    num_classes: int
    max_segments: int
    n_cnv_positions: int

    # Per-operator wall times
    wall_times: dict[str, float] = field(default_factory=dict)


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
    quality = (
        jax.random.uniform(k3, (n_reads, read_length)) * 30.0 + 10.0
    )

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
    baseline = jax.random.poisson(
        k1, 30.0, (n_positions,)
    ).astype(jnp.float32)

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


def benchmark_pileup(
    reference_length: int,
    n_reads: int,
    read_length: int,
) -> dict:
    """Benchmark the DifferentiablePileup operator.

    Checks output shapes, gradient flow, and throughput.

    Args:
        reference_length: Reference sequence length.
        n_reads: Number of reads.
        read_length: Length of each read.

    Returns:
        Dictionary of pileup benchmark metrics.
    """
    print("\n--- Pileup Benchmark ---")

    config = PileupConfig(
        reference_length=reference_length,
        temperature=1.0,
        return_coverage=True,
        return_quality=True,
    )
    pileup_op = DifferentiablePileup(config, rngs=nnx.Rngs(42))

    data = generate_pileup_data(
        reference_length, n_reads, read_length
    )
    result, _, _ = pileup_op.apply(data, {}, None)

    # Shape checks
    pileup_arr = result["pileup"]
    coverage_arr = result["coverage"]
    quality_arr = result["mean_quality"]

    pileup_shape_ok = pileup_arr.shape == (reference_length, 4)
    coverage_shape_ok = coverage_arr.shape == (reference_length, 1)
    quality_shape_ok = quality_arr.shape == (reference_length, 1)

    print(f"  Pileup shape {pileup_arr.shape}: "
          f"{'PASS' if pileup_shape_ok else 'FAIL'}")
    print(f"  Coverage shape {coverage_arr.shape}: "
          f"{'PASS' if coverage_shape_ok else 'FAIL'}")
    print(f"  Mean quality shape {quality_arr.shape}: "
          f"{'PASS' if quality_shape_ok else 'FAIL'}")

    # Gradient flow through pileup
    def pileup_loss(model: DifferentiablePileup, d: dict) -> float:
        """Scalar loss for gradient check."""
        out, _, _ = model.apply(d, {}, None)
        return jnp.sum(out["pileup"])

    grad_metrics = check_gradient_flow(pileup_loss, pileup_op, data)
    print(f"  Gradient norm: {grad_metrics.gradient_norm:.6f}")
    print(
        f"  Gradient non-zero: {grad_metrics.gradient_nonzero}"
    )

    # Throughput
    def run_pileup(d: dict) -> tuple:
        """Single pileup invocation for throughput measurement."""
        return pileup_op.apply(d, {}, None)

    throughput = measure_throughput(
        run_pileup, (data,), n_iterations=50, warmup=5
    )
    print(f"  Throughput: {throughput['per_item_ms']:.2f} ms/item, "
          f"{throughput['items_per_sec']:.1f} items/sec")

    return {
        "pileup_shape_correct": pileup_shape_ok,
        "pileup_coverage_shape_correct": coverage_shape_ok,
        "pileup_quality_shape_correct": quality_shape_ok,
        "pileup_gradient_norm": grad_metrics.gradient_norm,
        "pileup_gradient_nonzero": grad_metrics.gradient_nonzero,
        "pileup_per_item_ms": throughput["per_item_ms"],
        "pileup_items_per_sec": throughput["items_per_sec"],
    }


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
    pileup_window = jax.nn.softmax(
        jax.random.normal(key, (window_size, 4)), axis=-1
    )
    data = {"pileup_window": pileup_window}

    result, _, _ = classifier.apply(data, {}, None)

    # Shape checks
    logits = result["logits"]
    probs = result["probabilities"]

    logits_shape_ok = logits.shape == (num_classes,)
    prob_sum = float(jnp.sum(probs))
    prob_sum_ok = abs(prob_sum - 1.0) < 1e-4

    print(f"  Logits shape {logits.shape}: "
          f"{'PASS' if logits_shape_ok else 'FAIL'}")
    print(f"  Probability sum {prob_sum:.6f}: "
          f"{'PASS' if prob_sum_ok else 'FAIL'}")

    # Gradient flow through classifier
    def classifier_loss(
        model: VariantClassifier, d: dict
    ) -> float:
        """Scalar loss for gradient check."""
        out, _, _ = model.apply(d, {}, None)
        return jnp.sum(out["logits"])

    grad_metrics = check_gradient_flow(
        classifier_loss, classifier, data
    )
    print(f"  Gradient norm: {grad_metrics.gradient_norm:.6f}")
    print(
        f"  Gradient non-zero: {grad_metrics.gradient_nonzero}"
    )

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
    segmenter = DifferentiableCNVSegmentation(
        config, rngs=nnx.Rngs(42)
    )

    coverage = generate_cnv_coverage(n_positions)
    data = {"coverage": coverage}

    result, _, _ = segmenter.apply(data, {}, None)

    boundary_probs = result["boundary_probs"]
    segment_assignments = result["segment_assignments"]
    segment_means = result["segment_means"]
    smoothed_cov = result["smoothed_coverage"]

    # Boundary probs are valid probabilities in [0, 1]
    boundary_valid = bool(
        jnp.all(boundary_probs >= 0.0)
        & jnp.all(boundary_probs <= 1.0)
    )

    # Shape checks
    assignments_shape_ok = segment_assignments.shape == (
        n_positions,
        max_segments,
    )
    means_shape_ok = segment_means.shape == (max_segments,)
    smoothed_shape_ok = smoothed_cov.shape == (n_positions,)

    print(
        f"  Boundary probs in [0,1]: "
        f"{'PASS' if boundary_valid else 'FAIL'}"
    )
    print(
        f"  Segment assignments shape "
        f"{segment_assignments.shape}: "
        f"{'PASS' if assignments_shape_ok else 'FAIL'}"
    )
    print(
        f"  Segment means shape {segment_means.shape}: "
        f"{'PASS' if means_shape_ok else 'FAIL'}"
    )
    print(
        f"  Smoothed coverage shape {smoothed_cov.shape}: "
        f"{'PASS' if smoothed_shape_ok else 'FAIL'}"
    )

    return {
        "cnv_boundary_probs_valid": boundary_valid,
        "cnv_segment_assignments_shape_correct": assignments_shape_ok,
        "cnv_segment_means_shape_correct": means_shape_ok,
        "cnv_smoothed_coverage_shape_correct": smoothed_shape_ok,
    }


# -------------------------------------------------------------------
# Main benchmark orchestrator
# -------------------------------------------------------------------


def run_benchmark(
    quick: bool = False,
) -> VariantCallingBenchmarkResult:
    """Run the complete variant calling benchmark.

    Args:
        quick: If True, use smaller problem sizes for faster runs.

    Returns:
        Aggregated benchmark results.
    """
    print("=" * 60)
    print("DiffBio Variant Calling Benchmark")
    print(f"  Mode: {'quick' if quick else 'full'}")
    print("=" * 60)

    # Problem sizes
    reference_length = 100 if quick else 500
    read_length = 20 if quick else 50
    n_reads = int(reference_length * 20 / read_length)
    window_size = 21
    num_classes = 3
    max_segments = 20 if quick else 50
    n_cnv_positions = 100 if quick else 500

    print(f"\n  Reference length: {reference_length}")
    print(f"  Reads: {n_reads} x {read_length}bp (20x coverage)")
    print(f"  Window size: {window_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Max segments: {max_segments}")
    print(f"  CNV positions: {n_cnv_positions}")

    # Run individual benchmarks
    pileup_metrics = benchmark_pileup(
        reference_length, n_reads, read_length
    )
    classifier_metrics = benchmark_classifier(
        window_size, num_classes
    )
    cnv_metrics = benchmark_cnv_segmentation(
        n_cnv_positions, max_segments
    )

    # Compile results
    result = VariantCallingBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        # Pileup
        pileup_shape_correct=pileup_metrics[
            "pileup_shape_correct"
        ],
        pileup_coverage_shape_correct=pileup_metrics[
            "pileup_coverage_shape_correct"
        ],
        pileup_quality_shape_correct=pileup_metrics[
            "pileup_quality_shape_correct"
        ],
        # Classifier
        classifier_logits_shape_correct=classifier_metrics[
            "classifier_logits_shape_correct"
        ],
        classifier_prob_sum_close_to_one=classifier_metrics[
            "classifier_prob_sum_close_to_one"
        ],
        classifier_prob_sum=classifier_metrics[
            "classifier_prob_sum"
        ],
        # CNV
        cnv_boundary_probs_valid=cnv_metrics[
            "cnv_boundary_probs_valid"
        ],
        cnv_segment_assignments_shape_correct=cnv_metrics[
            "cnv_segment_assignments_shape_correct"
        ],
        cnv_segment_means_shape_correct=cnv_metrics[
            "cnv_segment_means_shape_correct"
        ],
        cnv_smoothed_coverage_shape_correct=cnv_metrics[
            "cnv_smoothed_coverage_shape_correct"
        ],
        # Gradients
        pileup_gradient_norm=pileup_metrics[
            "pileup_gradient_norm"
        ],
        pileup_gradient_nonzero=pileup_metrics[
            "pileup_gradient_nonzero"
        ],
        classifier_gradient_norm=classifier_metrics[
            "classifier_gradient_norm"
        ],
        classifier_gradient_nonzero=classifier_metrics[
            "classifier_gradient_nonzero"
        ],
        # Throughput
        pileup_per_item_ms=pileup_metrics["pileup_per_item_ms"],
        pileup_items_per_sec=pileup_metrics[
            "pileup_items_per_sec"
        ],
        # Configuration
        reference_length=reference_length,
        n_reads=n_reads,
        read_length=read_length,
        window_size=window_size,
        num_classes=num_classes,
        max_segments=max_segments,
        n_cnv_positions=n_cnv_positions,
    )

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    return result


def save_results(
    result: VariantCallingBenchmarkResult,
    output_dir: Path | None = None,
) -> Path:
    """Save benchmark results to JSON via shared utility.

    Args:
        result: Benchmark result dataclass.
        output_dir: Root directory for results. Defaults to
            ``benchmarks/results``.

    Returns:
        Path to the saved JSON file.
    """
    if output_dir is None:
        output_dir = Path("benchmarks/results")

    path = save_benchmark_result(
        asdict(result),
        domain="variant",
        benchmark_name="variant_calling_benchmark",
        output_dir=output_dir,
    )
    print(f"Results saved to: {path}")
    return path


def main() -> None:
    """Main entry point for the variant calling benchmark."""
    parser = argparse.ArgumentParser(
        description="DiffBio Variant Calling Benchmark",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with smaller problem sizes for faster execution.",
    )
    args = parser.parse_args()

    result = run_benchmark(quick=args.quick)
    save_results(result)


if __name__ == "__main__":
    main()
