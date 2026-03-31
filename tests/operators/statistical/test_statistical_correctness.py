#!/usr/bin/env python3
"""Statistical Model Benchmark for DiffBio.

This benchmark evaluates DiffBio's statistical model operators
for correctness, differentiability, and performance:

- DifferentiableHMM (Hidden Markov Model)
- DifferentiableNBGLM (Negative Binomial GLM)
- DifferentiableEMQuantifier (EM transcript quantification)

Metrics:
- Correctness checks (shapes, value constraints)
- Differentiability verification (gradient flow)
- Throughput measurement (iterations/second)

Usage:
    python benchmarks/statistical/statistical_benchmark.py
    python benchmarks/statistical/statistical_benchmark.py --quick
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------


def generate_hmm_data(
    seq_len: int = 500,
    num_states: int = 3,
    num_emissions: int = 4,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic HMM observation sequence.

    Constructs transition and emission matrices, then samples
    a sequence of observations from the resulting Markov chain.

    Args:
        seq_len: Length of the observation sequence.
        num_states: Number of hidden states.
        num_emissions: Number of distinct emission symbols.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``observations`` key containing an
        integer-encoded array of shape ``(seq_len,)``.
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 4)

    # Build a transition matrix with strong self-loops
    raw_trans = jnp.eye(num_states) * 5.0 + jax.random.uniform(keys[0], (num_states, num_states))
    trans = raw_trans / raw_trans.sum(axis=1, keepdims=True)

    # Build a random emission matrix
    raw_emit = jax.random.uniform(keys[1], (num_states, num_emissions)) + 0.1
    emit = raw_emit / raw_emit.sum(axis=1, keepdims=True)

    # Sample an initial state then roll forward
    initial_state = int(jax.random.categorical(keys[2], jnp.zeros(num_states)))
    state = initial_state
    observations_list: list[int] = []
    step_key = keys[3]
    for _ in range(seq_len):
        step_key, emit_key, trans_key = jax.random.split(step_key, 3)
        obs = int(jax.random.categorical(emit_key, jnp.log(emit[state])))
        observations_list.append(obs)
        state = int(jax.random.categorical(trans_key, jnp.log(trans[state])))

    observations = jnp.array(observations_list, dtype=jnp.int32)
    return {"observations": observations}


def generate_nb_glm_data(
    n_features: int = 50,
    n_covariates: int = 2,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic NB-GLM count data for a single sample.

    Produces a count vector drawn from a negative binomial
    distribution, a design row, and a scalar size factor.

    Args:
        n_features: Number of genes/features.
        n_covariates: Number of covariates in the design row.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``counts``, ``design``, and ``size_factor``.
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 4)

    # Design row: intercept + treatment indicator
    design = jnp.zeros(n_covariates).at[0].set(1.0)
    if n_covariates > 1:
        design = design.at[1].set(1.0)

    # Simulate counts from a Gamma-Poisson mixture
    log_mean = jax.random.normal(keys[0], (n_features,)) * 0.5 + 3.0
    rates = jnp.exp(log_mean)
    dispersion = 5.0
    gamma_samples = jax.random.gamma(keys[1], dispersion, (n_features,))
    scaled_rates = rates * gamma_samples / dispersion
    counts = jax.random.poisson(keys[2], scaled_rates).astype(jnp.float32)

    size_factor = jnp.array(1.0)

    return {
        "counts": counts,
        "design": design,
        "size_factor": size_factor,
    }


def generate_em_data(
    n_reads: int = 200,
    n_transcripts: int = 20,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic EM quantification data.

    Creates a read-transcript compatibility matrix where each read
    is compatible with a small subset of transcripts, plus effective
    transcript lengths.

    Args:
        n_reads: Number of reads.
        n_transcripts: Number of transcripts.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``compatibility`` and ``effective_lengths``.
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 3)

    # Sparse-ish compatibility: each read maps to ~3 transcripts
    raw = jax.random.uniform(keys[0], (n_reads, n_transcripts))
    threshold = jnp.percentile(raw, 85.0)
    compatibility = jnp.where(raw > threshold, raw, 0.0)

    # Ensure every read has at least one compatible transcript
    max_idx = jnp.argmax(raw, axis=1)
    row_indices = jnp.arange(n_reads)
    compatibility = compatibility.at[row_indices, max_idx].set(raw[row_indices, max_idx])

    # Effective transcript lengths (positive)
    effective_lengths = jax.random.uniform(keys[1], (n_transcripts,)) * 1000.0 + 200.0

    return {
        "compatibility": compatibility,
        "effective_lengths": effective_lengths,
    }
