#!/usr/bin/env python3
"""Metabolomics Spectral Similarity Benchmark for DiffBio.

This benchmark evaluates DiffBio's DifferentiableSpectralSimilarity
operator for output correctness, differentiability, and throughput on
synthetic binned mass spectra pairs.

Benchmarks:
- Similarity score shape and value range validation
- Embedding shape and finiteness
- Differentiability verification (gradient flow)
- Performance measurement (pairs/second)

Usage:
    python benchmarks/specialized/metabolomics_benchmark.py
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._common import (
    check_gradient_flow,
    measure_throughput,
    save_benchmark_result,
)
from diffbio.operators.metabolomics.spectral_similarity import (
    DifferentiableSpectralSimilarity,
    SpectralSimilarityConfig,
)

logger = logging.getLogger(__name__)

# -- Constants ---------------------------------------------------------------

N_BINS = 500


# -- Result dataclass --------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MetabolomicsBenchmarkResult:
    """Results from metabolomics spectral similarity benchmark.

    Attributes:
        timestamp: ISO-format timestamp of the benchmark run.
        n_pairs: Number of spectra pairs evaluated.
        n_bins: Number of m/z bins per spectrum.
        similarity_shape_valid: Whether similarity_scores has shape
            (n_pairs,).
        similarity_finite: Whether all similarity scores are finite.
        similarity_in_range: Whether scores fall within [-1, 1]
            (cosine similarity range).
        embeddings_shape_valid: Whether embeddings have expected shape.
        embeddings_finite: Whether all embedding values are finite.
        gradient_norm: L2 norm of gradients through the operator.
        gradient_nonzero: Whether gradient norm exceeds threshold.
        pairs_per_second: Throughput in spectrum pairs per second.
        wall_time_ms: Time per operator call in milliseconds.
    """

    timestamp: str
    n_pairs: int
    n_bins: int
    similarity_shape_valid: bool
    similarity_finite: bool
    similarity_in_range: bool
    embeddings_shape_valid: bool
    embeddings_finite: bool
    gradient_norm: float
    gradient_nonzero: bool
    pairs_per_second: float
    wall_time_ms: float


# -- Synthetic data ----------------------------------------------------------


def generate_synthetic_spectra_pairs(
    n_pairs: int,
    n_bins: int = N_BINS,
    seed: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate synthetic binned spectra pairs.

    Creates non-negative spectra normalized so that the maximum
    intensity per spectrum is 1.0, mimicking real mass spectrometry
    binning output.

    Args:
        n_pairs: Number of spectrum pairs.
        n_bins: Number of m/z bins per spectrum.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (spectra_a, spectra_b) each with shape
        (n_pairs, n_bins).
    """
    key = jax.random.key(seed)
    k_a, k_b = jax.random.split(key)

    # Sparse non-negative spectra (most bins near zero)
    raw_a = jax.random.exponential(k_a, (n_pairs, n_bins))
    raw_b = jax.random.exponential(k_b, (n_pairs, n_bins))

    # Normalize each spectrum to max intensity = 1.0
    spectra_a = raw_a / jnp.maximum(
        jnp.max(raw_a, axis=-1, keepdims=True), 1e-8
    )
    spectra_b = raw_b / jnp.maximum(
        jnp.max(raw_b, axis=-1, keepdims=True), 1e-8
    )

    return spectra_a, spectra_b


# -- Validation helpers ------------------------------------------------------


def _validate_similarity(
    scores: jnp.ndarray,
    n_pairs: int,
) -> dict[str, bool]:
    """Validate similarity score shape and value constraints.

    Args:
        scores: Predicted similarity scores.
        n_pairs: Expected number of pairs.

    Returns:
        Dictionary with shape, finiteness, and range flags.
    """
    shape_valid = scores.shape == (n_pairs,)
    finite = bool(jnp.all(jnp.isfinite(scores)))
    # Cosine similarity lies in [-1, 1]
    in_range = bool(
        jnp.all(scores >= -1.0 - 1e-6)
        and jnp.all(scores <= 1.0 + 1e-6)
    )
    return {
        "similarity_shape_valid": shape_valid,
        "similarity_finite": finite,
        "similarity_in_range": in_range,
    }


def _validate_embeddings(
    emb_a: jnp.ndarray,
    emb_b: jnp.ndarray,
    n_pairs: int,
    embedding_dim: int,
) -> dict[str, bool]:
    """Validate embedding shape and finiteness.

    Args:
        emb_a: Embeddings for spectra_a.
        emb_b: Embeddings for spectra_b.
        n_pairs: Expected number of pairs.
        embedding_dim: Expected embedding dimension.

    Returns:
        Dictionary with shape and finiteness flags.
    """
    expected = (n_pairs, embedding_dim)
    shape_valid = emb_a.shape == expected and emb_b.shape == expected
    finite = bool(
        jnp.all(jnp.isfinite(emb_a))
        and jnp.all(jnp.isfinite(emb_b))
    )
    return {
        "embeddings_shape_valid": shape_valid,
        "embeddings_finite": finite,
    }


# -- Benchmark runner --------------------------------------------------------


def run_benchmark(*, quick: bool = False) -> MetabolomicsBenchmarkResult:
    """Run the complete metabolomics spectral similarity benchmark.

    Args:
        quick: If True, use fewer pairs for faster execution.

    Returns:
        Benchmark results dataclass.
    """
    n_pairs = 10 if quick else 30

    print("=" * 60)
    print("DiffBio Metabolomics Spectral Similarity Benchmark")
    print("=" * 60)

    # -- Synthetic data ------------------------------------------------------
    print("\nGenerating synthetic spectra pairs...")
    spectra_a, spectra_b = generate_synthetic_spectra_pairs(
        n_pairs, n_bins=N_BINS
    )
    print(f"  Pairs: {n_pairs}  Bins: {N_BINS}")
    print(f"  spectra_a shape: {spectra_a.shape}")
    print(f"  spectra_b shape: {spectra_b.shape}")

    # -- Create operator -----------------------------------------------------
    print("\nCreating DifferentiableSpectralSimilarity operator...")
    embedding_dim = 200
    config = SpectralSimilarityConfig(
        n_bins=N_BINS,
        embedding_dim=embedding_dim,
    )
    operator = DifferentiableSpectralSimilarity(
        config, rngs=nnx.Rngs(42)
    )

    # -- Run prediction (paired mode) ----------------------------------------
    print("\nRunning spectral similarity computation...")
    data: dict[str, jnp.ndarray] = {
        "spectra_a": spectra_a,
        "spectra_b": spectra_b,
    }
    result, _, _ = operator.apply(data, {}, None)

    scores = result["similarity_scores"]
    emb_a = result["embeddings_a"]
    emb_b = result["embeddings_b"]

    # -- Validate outputs ----------------------------------------------------
    print("\nValidating outputs...")
    sim_validity = _validate_similarity(scores, n_pairs)
    emb_validity = _validate_embeddings(
        emb_a, emb_b, n_pairs, embedding_dim
    )

    print(
        f"  Similarity shape valid: "
        f"{sim_validity['similarity_shape_valid']}"
    )
    print(
        f"  Similarity finite: "
        f"{sim_validity['similarity_finite']}"
    )
    print(
        f"  Similarity in [-1, 1]: "
        f"{sim_validity['similarity_in_range']}"
    )
    print(
        f"  Embeddings shape valid: "
        f"{emb_validity['embeddings_shape_valid']}"
    )
    print(
        f"  Embeddings finite: "
        f"{emb_validity['embeddings_finite']}"
    )

    # -- Gradient flow -------------------------------------------------------
    print("\nChecking gradient flow...")

    def loss_fn(
        model: DifferentiableSpectralSimilarity,
        input_data: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Scalar loss for gradient checking."""
        out, _, _ = model.apply(input_data, {}, None)
        return jnp.sum(out["similarity_scores"])

    grad_metrics = check_gradient_flow(loss_fn, operator, data)
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
        return operator.apply(input_data, {}, None)

    throughput_metrics = measure_throughput(
        _run_apply,
        args=(data,),
        n_iterations=n_iters,
        warmup=warmup,
    )
    pairs_per_sec = n_pairs * throughput_metrics["items_per_sec"]
    wall_time_ms = throughput_metrics["per_item_ms"]

    print(f"  Pairs/sec: {pairs_per_sec:.1f}")
    print(f"  Time per call: {wall_time_ms:.2f} ms")

    # -- Compile results -----------------------------------------------------
    benchmark_result = MetabolomicsBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        n_pairs=n_pairs,
        n_bins=N_BINS,
        **sim_validity,
        **emb_validity,
        gradient_norm=grad_metrics.gradient_norm,
        gradient_nonzero=grad_metrics.gradient_nonzero,
        pairs_per_second=pairs_per_sec,
        wall_time_ms=wall_time_ms,
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        f"  Similarity valid: "
        f"{sim_validity['similarity_in_range']}"
    )
    print(f"  Gradient flows: {grad_metrics.gradient_nonzero}")
    print(f"  Throughput: {pairs_per_sec:.1f} pairs/sec")
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
        benchmark_name="metabolomics_spectral",
    )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
