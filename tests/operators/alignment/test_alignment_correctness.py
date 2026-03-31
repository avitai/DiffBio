#!/usr/bin/env python3
"""Alignment correctness tests for DiffBio.

Validates DiffBio's SmoothSmithWaterman operator for correctness
and differentiability on synthetic sequence pairs.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from diffbio.operators.alignment import (
    SmoothSmithWaterman,
    SmithWatermanConfig,
    create_dna_scoring_matrix,
)
from diffbio.sequences import encode_dna_string


def create_aligner(
    temperature: float = 1.0,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    match: float = 2.0,
    mismatch: float = -1.0,
) -> SmoothSmithWaterman:
    """Create a configured aligner."""
    scoring_matrix = create_dna_scoring_matrix(match=match, mismatch=mismatch)
    config = SmithWatermanConfig(
        temperature=temperature,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )
    return SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=nnx.Rngs(42))


def test_accuracy(
    aligner: SmoothSmithWaterman,
) -> dict[str, float]:
    """Test alignment accuracy on known cases.

    Returns:
        Dictionary of test case scores.
    """
    results: dict[str, float] = {}

    # Test 1: Perfect match
    seq = "ACGTACGT"
    seq1 = encode_dna_string(seq)
    seq2 = encode_dna_string(seq)
    data = {"seq1": seq1, "seq2": seq2}
    result, _, _ = aligner.apply(data, {}, None)
    results["perfect_match_score"] = float(result["score"])

    # Test 2: Single mismatch
    seq1 = encode_dna_string("ACGTACGT")
    seq2 = encode_dna_string("ACGAACGT")  # T->A at pos 3
    data = {"seq1": seq1, "seq2": seq2}
    result, _, _ = aligner.apply(data, {}, None)
    results["single_mismatch_score"] = float(result["score"])

    # Test 3: Insertion
    seq1 = encode_dna_string("ACGTACGT")
    seq2 = encode_dna_string("ACGTTACGT")  # T inserted
    data = {"seq1": seq1, "seq2": seq2}
    result, _, _ = aligner.apply(data, {}, None)
    results["insertion_score"] = float(result["score"])

    # Test 4: Deletion
    seq1 = encode_dna_string("ACGTACGT")
    seq2 = encode_dna_string("ACGACGT")  # T deleted
    data = {"seq1": seq1, "seq2": seq2}
    result, _, _ = aligner.apply(data, {}, None)
    results["deletion_score"] = float(result["score"])

    return results


def test_differentiability(
    aligner: SmoothSmithWaterman,
) -> dict[str, float | bool]:
    """Test gradient flow through alignment w.r.t. sequence inputs.

    Uses ``jax.grad`` on the input data dictionary (not model params)
    to verify that gradients propagate through the alignment operator.

    Returns:
        Dictionary with gradient metrics.
    """
    seq1 = encode_dna_string("ACGTACGT")
    seq2 = encode_dna_string("ACGTTACGT")
    data = {"seq1": seq1, "seq2": seq2}

    def loss_fn(data: dict[str, jnp.ndarray]) -> jnp.ndarray:
        result, _, _ = aligner.apply(data, {}, None)
        return result["score"]

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(data)

    seq1_grad = grads["seq1"]
    grad_norm = float(jnp.linalg.norm(seq1_grad))
    grad_nonzero = bool(grad_norm > 1e-6)

    return {
        "gradient_norm": grad_norm,
        "gradient_nonzero": grad_nonzero,
    }


def test_temperature_sweep() -> dict[str, float]:
    """Test alignment behavior across temperatures.

    Returns:
        Dictionary mapping temperature to alignment score.
    """
    results: dict[str, float] = {}
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    seq1 = encode_dna_string("ACGTACGT")
    seq2 = encode_dna_string("ACGTTACGT")

    for temp in temperatures:
        aligner = create_aligner(temperature=temp)
        data = {"seq1": seq1, "seq2": seq2}
        result, _, _ = aligner.apply(data, {}, None)
        results[f"temp_{temp}"] = float(result["score"])

    return results


def _generate_sequence_pairs(
    n_pairs: int,
    seq_len: int = 50,
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """Generate random one-hot encoded sequence pairs.

    Args:
        n_pairs: Number of pairs to generate.
        seq_len: Length of each sequence.

    Returns:
        List of (seq1, seq2) one-hot encoded pairs.
    """
    key = jax.random.key(42)
    pairs: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    for _ in range(n_pairs):
        key, k1, k2 = jax.random.split(key, 3)
        idx1 = jax.random.randint(k1, (seq_len,), 0, 4)
        idx2 = jax.random.randint(k2, (seq_len,), 0, 4)
        pairs.append((jax.nn.one_hot(idx1, 4), jax.nn.one_hot(idx2, 4)))
    return pairs
