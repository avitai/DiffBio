#!/usr/bin/env python3
"""Population Ancestry Estimation Benchmark for DiffBio.

This benchmark evaluates DiffBio's DifferentiableAncestryEstimator
operator for output correctness, differentiability, and throughput on
synthetic genotype data with known population structure.

Benchmarks:
- Ancestry proportion shape and sum-to-one validation
- Reconstructed genotype shape and value range
- Differentiability verification (gradient flow)

Usage:
    python benchmarks/specialized/population_benchmark.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# -- Constants ---------------------------------------------------------------

N_SNPS = 200
N_POPULATIONS = 3


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
    pop_labels = jax.random.randint(k_assign, (n_individuals,), 0, n_populations)
    individual_freqs = pop_freqs[pop_labels]  # (n_individuals, n_snps)

    # Draw two independent alleles per locus (diploid)
    allele1 = (jax.random.uniform(k_allele1, (n_individuals, n_snps)) < individual_freqs).astype(
        jnp.float32
    )
    allele2 = (jax.random.uniform(k_allele2, (n_individuals, n_snps)) < individual_freqs).astype(
        jnp.float32
    )

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
