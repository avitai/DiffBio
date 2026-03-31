#!/usr/bin/env python3
"""Metabolomics Spectral Similarity Benchmark for DiffBio.

This benchmark evaluates DiffBio's DifferentiableSpectralSimilarity
operator for output correctness, differentiability, and throughput on
synthetic binned mass spectra pairs.

Benchmarks:
- Similarity score shape and value range validation
- Embedding shape and finiteness
- Differentiability verification (gradient flow)

Usage:
    python benchmarks/specialized/metabolomics_benchmark.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# -- Constants ---------------------------------------------------------------

N_BINS = 500


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
    spectra_a = raw_a / jnp.maximum(jnp.max(raw_a, axis=-1, keepdims=True), 1e-8)
    spectra_b = raw_b / jnp.maximum(jnp.max(raw_b, axis=-1, keepdims=True), 1e-8)

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
    in_range = bool(jnp.all(scores >= -1.0 - 1e-6) and jnp.all(scores <= 1.0 + 1e-6))
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
    finite = bool(jnp.all(jnp.isfinite(emb_a)) and jnp.all(jnp.isfinite(emb_b)))
    return {
        "embeddings_shape_valid": shape_valid,
        "embeddings_finite": finite,
    }
