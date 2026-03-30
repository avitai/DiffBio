#!/usr/bin/env python3
"""Population Ancestry Estimation Benchmark for DiffBio.

This benchmark evaluates DiffBio's DifferentiableAncestryEstimator
operator for output correctness, differentiability, and throughput on
synthetic genotype data with known population structure.

Benchmarks:
- Ancestry proportion shape and sum-to-one validation
- Reconstructed genotype shape and value range
- Differentiability verification (gradient flow)
- Performance measurement (individuals/second)

Usage:
    python benchmarks/specialized/population_benchmark.py
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._gradient import check_gradient_flow
from diffbio.operators.population.ancestry_estimation import (
    AncestryEstimatorConfig,
    DifferentiableAncestryEstimator,
)

logger = logging.getLogger(__name__)

# -- Constants ---------------------------------------------------------------

N_SNPS = 200
N_POPULATIONS = 3


# -- Result dataclass --------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class PopulationBenchmarkResult:
    """Results from population ancestry estimation benchmark.

    Attributes:
        timestamp: ISO-format timestamp of the benchmark run.
        n_individuals: Number of individuals evaluated.
        n_snps: Number of SNP markers used.
        n_populations: Number of reference populations (K).
        ancestry_shape_valid: Whether ancestry_proportions has shape
            (n_individuals, n_populations).
        ancestry_sums_valid: Whether each row sums to approximately 1.
        ancestry_nonnegative: Whether all proportions are non-negative.
        reconstructed_shape_valid: Whether reconstructed genotypes have
            correct shape (n_individuals, n_snps).
        reconstructed_finite: Whether reconstructed values are finite.
        gradient_norm: L2 norm of gradients through the operator.
        gradient_nonzero: Whether gradient norm exceeds threshold.
        individuals_per_second: Throughput in individuals per second.
        wall_time_ms: Time per operator call in milliseconds.
    """

    timestamp: str
    n_individuals: int
    n_snps: int
    n_populations: int
    ancestry_shape_valid: bool
    ancestry_sums_valid: bool
    ancestry_nonnegative: bool
    reconstructed_shape_valid: bool
    reconstructed_finite: bool
    gradient_norm: float
    gradient_nonzero: bool
    individuals_per_second: float
    wall_time_ms: float


# -- Synthetic data ----------------------------------------------------------


def generate_synthetic_genotypes(
    n_individuals: int,
    n_snps: int = N_SNPS,
    n_populations: int = N_POPULATIONS,
    seed: int = 42,
) -> jnp.ndarray:
    """Generate synthetic genotype data with population structure.

    Simulates allele-count genotypes (values in {0, 1, 2}) drawn from
    population-specific allele frequencies. Each individual is randomly
    assigned to one of the reference populations.

    Args:
        n_individuals: Number of individuals.
        n_snps: Number of SNP markers.
        n_populations: Number of source populations.
        seed: Random seed for reproducibility.

    Returns:
        Genotype matrix of shape (n_individuals, n_snps) with float
        values 0.0, 1.0, or 2.0.
    """
    key = jax.random.key(seed)
    k_freq, k_assign, k_allele1, k_allele2 = jax.random.split(key, 4)

    # Population-specific allele frequencies in (0, 1)
    pop_freqs = jax.random.beta(k_freq, 0.5, 0.5, (n_populations, n_snps))

    # Assign each individual to a population
    pop_labels = jax.random.randint(
        k_assign, (n_individuals,), 0, n_populations
    )
    individual_freqs = pop_freqs[pop_labels]  # (n_individuals, n_snps)

    # Draw two independent alleles per locus (diploid)
    allele1 = (
        jax.random.uniform(k_allele1, (n_individuals, n_snps))
        < individual_freqs
    ).astype(jnp.float32)
    allele2 = (
        jax.random.uniform(k_allele2, (n_individuals, n_snps))
        < individual_freqs
    ).astype(jnp.float32)

    return allele1 + allele2


# -- Validation helpers ------------------------------------------------------


def _validate_ancestry(
    ancestry: jnp.ndarray,
    n_individuals: int,
    n_populations: int,
) -> dict[str, bool]:
    """Validate ancestry proportion shape and constraints.

    Args:
        ancestry: Predicted ancestry proportions.
        n_individuals: Expected number of individuals.
        n_populations: Expected number of populations.

    Returns:
        Dictionary with shape, sum-to-one, and non-negativity flags.
    """
    shape_valid = ancestry.shape == (n_individuals, n_populations)
    row_sums = jnp.sum(ancestry, axis=-1)
    sums_valid = bool(jnp.allclose(row_sums, 1.0, atol=1e-4))
    nonneg = bool(jnp.all(ancestry >= -1e-6))
    return {
        "ancestry_shape_valid": shape_valid,
        "ancestry_sums_valid": sums_valid,
        "ancestry_nonnegative": nonneg,
    }


def _validate_reconstructed(
    reconstructed: jnp.ndarray,
    n_individuals: int,
    n_snps: int,
) -> dict[str, bool]:
    """Validate reconstructed genotype matrix.

    Args:
        reconstructed: Reconstructed genotype matrix.
        n_individuals: Expected number of individuals.
        n_snps: Expected number of SNPs.

    Returns:
        Dictionary with shape and finiteness flags.
    """
    shape_valid = reconstructed.shape == (n_individuals, n_snps)
    finite = bool(jnp.all(jnp.isfinite(reconstructed)))
    return {
        "reconstructed_shape_valid": shape_valid,
        "reconstructed_finite": finite,
    }


# -- Benchmark runner --------------------------------------------------------


def run_benchmark(*, quick: bool = False) -> PopulationBenchmarkResult:
    """Run the complete population ancestry estimation benchmark.

    Args:
        quick: If True, use fewer individuals for faster execution.

    Returns:
        Benchmark results dataclass.
    """
    n_individuals = 20 if quick else 50

    print("=" * 60)
    print("DiffBio Population Ancestry Estimation Benchmark")
    print("=" * 60)

    # -- Synthetic data ------------------------------------------------------
    print("\nGenerating synthetic genotype data...")
    genotypes = generate_synthetic_genotypes(
        n_individuals,
        n_snps=N_SNPS,
        n_populations=N_POPULATIONS,
    )
    print(
        f"  Individuals: {n_individuals}  "
        f"SNPs: {N_SNPS}  Populations: {N_POPULATIONS}"
    )
    print(f"  Genotype shape: {genotypes.shape}")

    # -- Create operator -----------------------------------------------------
    print("\nCreating DifferentiableAncestryEstimator operator...")
    config = AncestryEstimatorConfig(
        n_snps=N_SNPS,
        n_populations=N_POPULATIONS,
    )
    estimator = DifferentiableAncestryEstimator(
        config, rngs=nnx.Rngs(42)
    )

    # -- Run prediction ------------------------------------------------------
    print("\nRunning ancestry estimation...")
    data: dict[str, jnp.ndarray] = {"genotypes": genotypes}
    result, _, _ = estimator.apply(data, {}, None)

    ancestry = result["ancestry_proportions"]
    reconstructed = result["reconstructed"]

    # -- Validate outputs ----------------------------------------------------
    print("\nValidating outputs...")
    ancestry_validity = _validate_ancestry(
        ancestry, n_individuals, N_POPULATIONS
    )
    recon_validity = _validate_reconstructed(
        reconstructed, n_individuals, N_SNPS
    )

    print(
        f"  Ancestry shape valid: "
        f"{ancestry_validity['ancestry_shape_valid']}"
    )
    print(
        f"  Ancestry sums to ~1: "
        f"{ancestry_validity['ancestry_sums_valid']}"
    )
    print(
        f"  Ancestry non-negative: "
        f"{ancestry_validity['ancestry_nonnegative']}"
    )
    print(
        f"  Reconstructed shape valid: "
        f"{recon_validity['reconstructed_shape_valid']}"
    )
    print(
        f"  Reconstructed finite: "
        f"{recon_validity['reconstructed_finite']}"
    )

    # -- Gradient flow -------------------------------------------------------
    print("\nChecking gradient flow...")

    def loss_fn(
        model: DifferentiableAncestryEstimator,
        input_data: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Scalar loss for gradient checking."""
        out, _, _ = model.apply(input_data, {}, None)
        return jnp.sum(out["ancestry_proportions"])

    grad_metrics = check_gradient_flow(loss_fn, estimator, data)
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
        return estimator.apply(input_data, {}, None)

    throughput_metrics = measure_throughput(
        _run_apply,
        args=(data,),
        n_iterations=n_iters,
        warmup=warmup,
    )
    individuals_per_sec = (
        n_individuals * throughput_metrics["items_per_sec"]
    )
    wall_time_ms = throughput_metrics["per_item_ms"]

    print(f"  Individuals/sec: {individuals_per_sec:.1f}")
    print(f"  Time per call: {wall_time_ms:.2f} ms")

    # -- Compile results -----------------------------------------------------
    benchmark_result = PopulationBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        n_individuals=n_individuals,
        n_snps=N_SNPS,
        n_populations=N_POPULATIONS,
        **ancestry_validity,
        **recon_validity,
        gradient_norm=grad_metrics.gradient_norm,
        gradient_nonzero=grad_metrics.gradient_nonzero,
        individuals_per_second=individuals_per_sec,
        wall_time_ms=wall_time_ms,
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        f"  Ancestry valid: "
        f"{ancestry_validity['ancestry_sums_valid']}"
    )
    print(f"  Gradient flows: {grad_metrics.gradient_nonzero}")
    print(f"  Throughput: {individuals_per_sec:.1f} individuals/sec")
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
        benchmark_name="population_ancestry",
    )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
