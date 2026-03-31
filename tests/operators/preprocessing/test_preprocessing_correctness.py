#!/usr/bin/env python3
"""Preprocessing Benchmark for DiffBio.

This benchmark evaluates DiffBio's read preprocessing operators:
- SoftAdapterRemoval (soft adapter trimming via differentiable alignment)
- DifferentiableDuplicateWeighting (probabilistic duplicate weighting)
- SoftErrorCorrection (neural network-based error correction)

Metrics:
- Output shape correctness for all operators
- Values are finite and in expected ranges
- Gradient flow for all operators

Usage:
    python benchmarks/preprocessing/preprocessing_benchmark.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from tests.fixtures.synthetic import generate_synthetic_sequences
from diffbio.operators.preprocessing.adapter_removal import (
    AdapterRemovalConfig,
    SoftAdapterRemoval,
)
from diffbio.operators.preprocessing.duplicate_filter import (
    DifferentiableDuplicateWeighting,
    DuplicateWeightingConfig,
)
from diffbio.operators.preprocessing.error_correction import (
    ErrorCorrectionConfig,
    SoftErrorCorrection,
)


# ------------------------------------------------------------------
# Synthetic data generators
# ------------------------------------------------------------------


def _generate_reads_with_adapter(
    n_reads: int,
    read_length: int,
    adapter_sequence: str = "AGATCGGAAGAG",
    seed: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate one-hot reads with an adapter appended at the 3' end.

    Creates random DNA reads and appends the adapter at a random
    position within the last portion of each read so the operator
    has something to detect.

    Args:
        n_reads: Number of reads to generate.
        read_length: Length of each read (including adapter portion).
        adapter_sequence: Adapter sequence string.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sequences, quality_scores) where sequences is
        shape ``(n_reads, read_length, 4)`` and quality_scores
        is shape ``(n_reads, read_length)``.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # Generate random base sequences
    sequences = generate_synthetic_sequences(
        n_seqs=n_reads,
        seq_len=read_length,
        alphabet_size=4,
        seed=seed,
    )

    # Encode adapter bases (A=0, C=1, G=2, T=3)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    adapter_indices = jnp.array([base_to_idx[b] for b in adapter_sequence])
    adapter_onehot = jax.nn.one_hot(adapter_indices, 4)
    adapter_len = len(adapter_sequence)

    # Place adapter at the tail of each read
    # Overwrite the last adapter_len positions with adapter bases
    insert_start = read_length - adapter_len
    insert_end = min(insert_start + adapter_len, read_length)
    actual_len = insert_end - insert_start

    updated = sequences.at[:, insert_start:insert_end, :].set(
        jnp.broadcast_to(
            adapter_onehot[:actual_len][None, :, :],
            (n_reads, actual_len, 4),
        )
    )

    # Generate quality scores (Phred-like, range [0, 40])
    quality_scores = jax.random.uniform(
        k3,
        (n_reads, read_length),
        minval=10.0,
        maxval=40.0,
    )

    return updated, quality_scores


def _generate_duplicate_data(
    n_reads: int,
    read_length: int,
    n_duplicate_groups: int = 5,
    seed: int = 43,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate batched reads with known duplicate groups.

    Creates ``n_duplicate_groups`` unique reads and duplicates each
    several times (with minor noise) to fill ``n_reads`` total.

    Args:
        n_reads: Total number of reads in the batch.
        read_length: Length of each read.
        n_duplicate_groups: Number of distinct sequence groups.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sequences, quality_scores) where sequences is
        shape ``(n_reads, read_length, 4)`` and quality_scores
        is shape ``(n_reads, read_length)``.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # Generate unique template sequences
    templates = generate_synthetic_sequences(
        n_seqs=n_duplicate_groups,
        seq_len=read_length,
        alphabet_size=4,
        seed=seed,
    )

    # Assign each read to a group (round-robin)
    group_indices = jnp.arange(n_reads) % n_duplicate_groups
    sequences = templates[group_indices]  # (n_reads, read_length, 4)

    # Add slight noise so embeddings are not perfectly identical
    noise = jax.random.normal(k2, sequences.shape) * 0.01
    sequences = jax.nn.softmax(
        jnp.log(sequences + 1e-6) + noise,
        axis=-1,
    )

    # Generate quality scores
    quality_scores = jax.random.uniform(
        k3,
        (n_reads, read_length),
        minval=10.0,
        maxval=40.0,
    )

    return sequences, quality_scores


def _generate_error_correction_data(
    n_reads: int,
    read_length: int,
    seed: int = 44,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate reads with quality scores for error correction.

    Creates random one-hot sequences paired with Phred-quality
    scores. Some positions are given low quality to simulate
    sequencing errors.

    Args:
        n_reads: Number of reads.
        read_length: Length of each read.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sequences, quality_scores) where sequences is
        shape ``(n_reads, read_length, 4)`` and quality_scores
        is shape ``(n_reads, read_length)``.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)

    sequences = generate_synthetic_sequences(
        n_seqs=n_reads,
        seq_len=read_length,
        alphabet_size=4,
        seed=seed,
    )

    # Quality scores: most positions high quality, some low
    base_quality = jax.random.uniform(
        k1,
        (n_reads, read_length),
        minval=20.0,
        maxval=40.0,
    )
    # Mark ~10% of positions as low quality
    error_mask = jax.random.bernoulli(k2, 0.1, (n_reads, read_length))
    quality_scores = jnp.where(error_mask, 5.0, base_quality)

    return sequences, quality_scores


# ------------------------------------------------------------------
# Operator tests
# ------------------------------------------------------------------


def _test_adapter_removal(
    sequence: jnp.ndarray,
    quality_scores: jnp.ndarray,
    read_length: int,
) -> tuple[dict[str, bool], SoftAdapterRemoval]:
    """Test SoftAdapterRemoval on a single read.

    Args:
        sequence: One-hot encoded read of shape ``(read_length, 4)``.
        quality_scores: Quality scores of shape ``(read_length,)``.
        read_length: Expected read length.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    config = AdapterRemovalConfig(
        adapter_sequence="AGATCGGAAGAG",
        temperature=1.0,
        match_threshold=0.5,
        min_overlap=6,
    )
    remover = SoftAdapterRemoval(config, rngs=nnx.Rngs(42))

    data = {"sequence": sequence, "quality_scores": quality_scores}
    result, _, _ = remover.apply(data, {}, None)

    # Shape checks
    shape_ok = (
        result["sequence"].shape == (read_length, 4)
        and result["quality_scores"].shape == (read_length,)
        and result["adapter_score"].shape == ()
        and result["trim_position"].shape == ()
    )

    # Finite value checks
    values_finite = bool(
        jnp.all(jnp.isfinite(result["sequence"]))
        and jnp.all(jnp.isfinite(result["quality_scores"]))
        and jnp.isfinite(result["adapter_score"])
        and jnp.isfinite(result["trim_position"])
    )

    # Trim position should be in [0, read_length]
    trim_pos = float(result["trim_position"])
    trim_in_range = 0.0 <= trim_pos <= float(read_length)

    return {
        "shape_ok": shape_ok,
        "values_finite": values_finite,
        "trim_in_range": trim_in_range,
    }, remover


def _test_duplicate_weighting(
    sequences: jnp.ndarray,
    quality_scores: jnp.ndarray,
) -> tuple[dict[str, bool], DifferentiableDuplicateWeighting]:
    """Test DifferentiableDuplicateWeighting on a batch of reads.

    Args:
        sequences: Batched one-hot sequences of shape
            ``(batch, read_length, 4)``.
        quality_scores: Batched quality scores of shape
            ``(batch, read_length)``.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    config = DuplicateWeightingConfig(
        temperature=1.0,
        similarity_threshold=0.9,
        embedding_dim=32,
    )
    weighter = DifferentiableDuplicateWeighting(
        config,
        rngs=nnx.Rngs(42),
    )

    data = {"sequence": sequences, "quality_scores": quality_scores}
    result, _, _ = weighter.apply(data, {}, None)

    batch_size = sequences.shape[0]
    read_length = sequences.shape[1]

    # Shape checks (batched input returns original sequences unchanged,
    # plus scalar weight and embedding for the first element)
    shape_ok = (
        result["sequence"].shape == (batch_size, read_length, 4)
        and result["quality_scores"].shape == (batch_size, read_length)
        and result["uniqueness_weight"].shape == ()
        and result["embedding"].shape == (32,)
    )

    # Finite value checks
    values_finite = bool(
        jnp.all(jnp.isfinite(result["uniqueness_weight"]))
        and jnp.all(jnp.isfinite(result["embedding"]))
    )

    # Weights should be positive
    weights_positive = bool(result["uniqueness_weight"] > 0.0)

    return {
        "shape_ok": shape_ok,
        "values_finite": values_finite,
        "weights_positive": weights_positive,
    }, weighter


def _test_error_correction(
    sequence: jnp.ndarray,
    quality_scores: jnp.ndarray,
    read_length: int,
) -> tuple[dict[str, bool], SoftErrorCorrection]:
    """Test SoftErrorCorrection on a single read.

    Args:
        sequence: One-hot encoded read of shape ``(read_length, 4)``.
        quality_scores: Quality scores of shape ``(read_length,)``.
        read_length: Expected read length.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    config = ErrorCorrectionConfig(
        window_size=11,
        hidden_dim=64,
        num_layers=2,
        use_quality=True,
        temperature=1.0,
    )
    corrector = SoftErrorCorrection(config, rngs=nnx.Rngs(42))

    data = {"sequence": sequence, "quality_scores": quality_scores}
    result, _, _ = corrector.apply(data, {}, None)

    # Shape checks
    shape_ok = (
        result["sequence"].shape == (read_length, 4)
        and result["quality_scores"].shape == (read_length,)
        and result["correction_confidence"].shape == ()
    )

    # Finite value checks
    values_finite = bool(
        jnp.all(jnp.isfinite(result["sequence"]))
        and jnp.all(jnp.isfinite(result["quality_scores"]))
        and jnp.isfinite(result["correction_confidence"])
    )

    # Corrected sequence rows should sum to ~1 (probability dist)
    row_sums = jnp.sum(result["sequence"], axis=-1)
    probs_valid = bool(jnp.allclose(row_sums, 1.0, atol=1e-3))

    return {
        "shape_ok": shape_ok,
        "values_finite": values_finite,
        "probs_valid": probs_valid,
    }, corrector
