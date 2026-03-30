#!/usr/bin/env python3
"""CRISPR Guide Scoring Benchmark for DiffBio.

This benchmark evaluates DiffBio's DifferentiableCRISPRScorer operator
for output correctness, differentiability, and throughput on synthetic
one-hot encoded guide RNA sequences.

Benchmarks:
- Efficiency score shape and finite-value validation
- Output value range [0, 1]
- Differentiability verification (gradient flow)
- Performance measurement (guides/second)

Usage:
    python benchmarks/specialized/crispr_benchmark.py
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._gradient import check_gradient_flow
from diffbio.operators.crispr.guide_scoring import (
    CRISPRScorerConfig,
    DifferentiableCRISPRScorer,
)

logger = logging.getLogger(__name__)

# -- Constants ---------------------------------------------------------------

GUIDE_LENGTH = 23
ALPHABET_SIZE = 4


# -- Result dataclass --------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class CRISPRBenchmarkResult:
    """Results from CRISPR guide scoring benchmark.

    Attributes:
        timestamp: ISO-format timestamp of the benchmark run.
        n_guides: Number of guide sequences evaluated.
        score_shape_valid: Whether efficiency_scores has shape
            (n_guides,).
        scores_finite: Whether all efficiency scores are finite.
        scores_in_range: Whether all scores fall within [0, 1].
        features_finite: Whether extracted features are finite.
        gradient_norm: L2 norm of gradients through the operator.
        gradient_nonzero: Whether gradient norm exceeds threshold.
        guides_per_second: Throughput in guides per second.
        wall_time_ms: Time per operator call in milliseconds.
        guide_length: Length of guide sequences used.
    """

    timestamp: str
    n_guides: int
    score_shape_valid: bool
    scores_finite: bool
    scores_in_range: bool
    features_finite: bool
    gradient_norm: float
    gradient_nonzero: bool
    guides_per_second: float
    wall_time_ms: float
    guide_length: int


# -- Synthetic data ----------------------------------------------------------


def generate_synthetic_guides(
    n_guides: int,
    guide_length: int = GUIDE_LENGTH,
    seed: int = 42,
) -> jnp.ndarray:
    """Generate random one-hot encoded guide RNA sequences.

    Args:
        n_guides: Number of guides to generate.
        guide_length: Nucleotide length per guide (default 23).
        seed: Random seed for reproducibility.

    Returns:
        One-hot array of shape (n_guides, guide_length, 4).
    """
    key = jax.random.key(seed)
    indices = jax.random.randint(
        key, (n_guides, guide_length), 0, ALPHABET_SIZE
    )
    return jax.nn.one_hot(indices, ALPHABET_SIZE)


# -- Validation helpers ------------------------------------------------------


def _validate_scores(
    scores: jnp.ndarray,
    n_guides: int,
) -> dict[str, bool]:
    """Validate efficiency score shape and value constraints.

    Args:
        scores: Predicted efficiency scores.
        n_guides: Expected number of guides.

    Returns:
        Dictionary with shape, finiteness, and range flags.
    """
    shape_valid = scores.shape == (n_guides,)
    finite = bool(jnp.all(jnp.isfinite(scores)))
    in_range = bool(
        jnp.all(scores >= -1e-6) and jnp.all(scores <= 1.0 + 1e-6)
    )
    return {
        "score_shape_valid": shape_valid,
        "scores_finite": finite,
        "scores_in_range": in_range,
    }


def _validate_features(features: jnp.ndarray) -> dict[str, bool]:
    """Validate extracted feature vectors.

    Args:
        features: Feature vectors from the CNN encoder.

    Returns:
        Dictionary with finiteness flag.
    """
    return {"features_finite": bool(jnp.all(jnp.isfinite(features)))}


# -- Benchmark runner --------------------------------------------------------


def run_benchmark(*, quick: bool = False) -> CRISPRBenchmarkResult:
    """Run the complete CRISPR guide scoring benchmark.

    Args:
        quick: If True, use fewer guides for faster execution.

    Returns:
        Benchmark results dataclass.
    """
    n_guides = 20 if quick else 50

    print("=" * 60)
    print("DiffBio CRISPR Guide Scoring Benchmark")
    print("=" * 60)

    # -- Synthetic data ------------------------------------------------------
    print("\nGenerating synthetic guide sequences...")
    guides = generate_synthetic_guides(n_guides, guide_length=GUIDE_LENGTH)
    print(f"  Guides: {n_guides}  Shape: {guides.shape}")

    # -- Create operator -----------------------------------------------------
    print("\nCreating DifferentiableCRISPRScorer operator...")
    config = CRISPRScorerConfig(
        guide_length=GUIDE_LENGTH,
        alphabet_size=ALPHABET_SIZE,
    )
    scorer = DifferentiableCRISPRScorer(config, rngs=nnx.Rngs(42))

    # -- Run prediction ------------------------------------------------------
    print("\nRunning guide scoring...")
    data: dict[str, jnp.ndarray] = {"guides": guides}
    result, _, _ = scorer.apply(data, {}, None)

    scores = result["efficiency_scores"]
    features = result["features"]

    # -- Validate outputs ----------------------------------------------------
    print("\nValidating outputs...")
    score_validity = _validate_scores(scores, n_guides)
    feature_validity = _validate_features(features)

    print(f"  Score shape valid: {score_validity['score_shape_valid']}")
    print(f"  Scores finite: {score_validity['scores_finite']}")
    print(f"  Scores in [0, 1]: {score_validity['scores_in_range']}")
    print(f"  Features finite: {feature_validity['features_finite']}")

    # -- Gradient flow -------------------------------------------------------
    print("\nChecking gradient flow...")

    def loss_fn(
        model: DifferentiableCRISPRScorer,
        input_data: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Scalar loss for gradient checking."""
        out, _, _ = model.apply(input_data, {}, None)
        return jnp.sum(out["efficiency_scores"])

    grad_metrics = check_gradient_flow(loss_fn, scorer, data)
    print(f"  Gradient norm: {grad_metrics.gradient_norm:.6f}")
    print(f"  Gradient non-zero: {grad_metrics.gradient_nonzero}")

    # -- Throughput ----------------------------------------------------------
    print("\nMeasuring throughput...")
    n_iters = 20 if quick else 100
    warmup = 3 if quick else 5

    def _run_apply(
        input_data: dict[str, jnp.ndarray],
    ) -> tuple:
        """Single operator call for throughput measurement."""
        return scorer.apply(input_data, {}, None)

    throughput_metrics = measure_throughput(
        _run_apply,
        args=(data,),
        n_iterations=n_iters,
        warmup=warmup,
    )
    guides_per_sec = n_guides * throughput_metrics["items_per_sec"]
    wall_time_ms = throughput_metrics["per_item_ms"]

    print(f"  Guides/sec: {guides_per_sec:.1f}")
    print(f"  Time per call: {wall_time_ms:.2f} ms")

    # -- Compile results -----------------------------------------------------
    benchmark_result = CRISPRBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        n_guides=n_guides,
        **score_validity,
        **feature_validity,
        gradient_norm=grad_metrics.gradient_norm,
        gradient_nonzero=grad_metrics.gradient_nonzero,
        guides_per_second=guides_per_sec,
        wall_time_ms=wall_time_ms,
        guide_length=GUIDE_LENGTH,
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Scores valid: {score_validity['scores_in_range']}")
    print(f"  Gradient flows: {grad_metrics.gradient_nonzero}")
    print(f"  Throughput: {guides_per_sec:.1f} guides/sec")
    print("=" * 60)

    return benchmark_result


# -- Entry point -------------------------------------------------------------


def main() -> None:
    """Run benchmark and save results."""
    result = run_benchmark()
    result_dict = asdict(result)
    output_path = save_benchmark_result(
        result_dict,
        domain="specialized",
        benchmark_name="crispr",
    )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
