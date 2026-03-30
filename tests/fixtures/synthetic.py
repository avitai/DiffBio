"""Synthetic data generators for correctness tests.

These generators produce deterministic, reproducible test data for
operator shape/gradient/value-range verification. They are NOT used
in benchmarks (benchmarks use real datasets via datarax DataSources).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def generate_synthetic_expression(
    n_cells: int = 500,
    n_genes: int = 200,
    n_types: int = 3,
    n_batches: int = 2,
    batch_effect_strength: float = 3.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate synthetic single-cell expression data with known structure.

    Creates count data with per-type expression profiles, batch effects,
    and negative binomial sampling for realistic count distributions.

    Args:
        n_cells: Total number of cells.
        n_genes: Number of genes.
        n_types: Number of cell types.
        n_batches: Number of experimental batches.
        batch_effect_strength: Magnitude of additive batch shift (log scale).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys: counts, library_size, batch_labels,
        cell_type_labels, embeddings, n_cells, n_genes, n_batches, n_types.
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 6)

    type_log_means = jax.random.normal(keys[0], (n_types, n_genes)) * 1.5 + 2.0

    cells_per_type = n_cells // n_types
    type_labels_list: list[int] = []
    for t in range(n_types):
        count = (
            cells_per_type if t < n_types - 1
            else n_cells - len(type_labels_list)
        )
        type_labels_list.extend([t] * count)
    cell_type_labels = jnp.array(type_labels_list)

    cells_per_batch = n_cells // n_batches
    batch_labels_list: list[int] = []
    for b in range(n_batches):
        count = (
            cells_per_batch if b < n_batches - 1
            else n_cells - len(batch_labels_list)
        )
        batch_labels_list.extend([b] * count)
    batch_labels = jnp.array(batch_labels_list)

    batch_shifts = (
        jax.random.normal(keys[1], (n_batches, n_genes))
        * batch_effect_strength
    )
    cell_noise = jax.random.normal(keys[2], (n_cells, n_genes)) * 0.3
    cell_log_means = (
        type_log_means[cell_type_labels]
        + batch_shifts[batch_labels]
        + cell_noise
    )

    rates = jnp.exp(cell_log_means)
    dispersion = 5.0
    gamma_samples = jax.random.gamma(keys[3], dispersion, (n_cells, n_genes))
    scaled_rates = rates * gamma_samples / dispersion
    counts = jax.random.poisson(keys[4], scaled_rates).astype(jnp.float32)

    library_size = jnp.sum(counts, axis=-1)

    clean_embeddings = type_log_means[cell_type_labels] + cell_noise
    proj = jax.random.normal(keys[5], (n_genes, min(50, n_genes)))
    embeddings = clean_embeddings @ proj / jnp.sqrt(n_genes)

    return {
        "counts": counts,
        "library_size": library_size,
        "batch_labels": batch_labels,
        "cell_type_labels": cell_type_labels,
        "embeddings": embeddings,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_batches": n_batches,
        "n_types": n_types,
    }


def generate_synthetic_sequences(
    n_seqs: int = 100,
    seq_len: int = 50,
    alphabet_size: int = 4,
    seed: int = 42,
) -> jnp.ndarray:
    """Generate random one-hot encoded sequences.

    Args:
        n_seqs: Number of sequences.
        seq_len: Length of each sequence.
        alphabet_size: Size of the alphabet (4 for DNA/RNA).
        seed: Random seed.

    Returns:
        One-hot encoded array of shape ``(n_seqs, seq_len, alphabet_size)``.
    """
    key = jax.random.key(seed)
    indices = jax.random.randint(key, (n_seqs, seq_len), 0, alphabet_size)
    return jax.nn.one_hot(indices, alphabet_size)


def generate_synthetic_coverage(
    length: int = 10000,
    n_peaks: int = 20,
    peak_width_range: tuple[int, int] = (50, 500),
    background_rate: float = 5.0,
    peak_height_range: tuple[float, float] = (20.0, 100.0),
    seed: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate synthetic coverage signal with known peak positions.

    Args:
        length: Signal length in positions.
        n_peaks: Number of peaks to insert.
        peak_width_range: (min_width, max_width) of each peak.
        background_rate: Mean Poisson rate for background.
        peak_height_range: (min_height, max_height) above background.
        seed: Random seed.

    Returns:
        Tuple of (signal, truth_mask) where truth_mask is 1.0 at peak
        positions and 0.0 elsewhere.
    """
    import numpy as np  # noqa: PLC0415

    key = jax.random.key(seed)
    keys = jax.random.split(key, 4)

    background = jax.random.poisson(
        keys[0], background_rate, (length,)
    ).astype(jnp.float32)

    positions = jax.random.randint(keys[1], (n_peaks,), 0, length)
    min_w, max_w = peak_width_range
    widths = jax.random.randint(
        keys[2], (n_peaks,), min_w, max(min_w + 1, max_w)
    )
    min_h, max_h = peak_height_range
    heights = jax.random.uniform(
        keys[3], (n_peaks,), minval=min_h, maxval=max_h
    )

    signal_np = np.array(background)
    truth_np = np.zeros(length, dtype=np.float32)

    for i in range(n_peaks):
        center = int(positions[i])
        width = int(widths[i])
        height = float(heights[i])
        half_w = width // 2
        start = max(0, center - half_w)
        end = min(length, center + half_w)

        x = np.arange(start, end) - center
        peak = height * np.exp(-0.5 * (x / (width / 6.0)) ** 2)
        signal_np[start:end] += peak
        truth_np[start:end] = 1.0

    return jnp.array(signal_np), jnp.array(truth_np)
