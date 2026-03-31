#!/usr/bin/env python3
"""RNA Structure Benchmark for DiffBio.

This benchmark evaluates DiffBio's DifferentiableRNAFold operator for
correctness, differentiability, and performance on synthetic RNA sequences
with known secondary structure.

Benchmarks:
- Base pair probability matrix shape and value range
- Symmetry of the bp_probs matrix
- Known base pair detection (hairpin and stem structures)
- Partition function validity
- Differentiability verification (gradient flow)
- Throughput measurement (nucleotides/second)

Usage:
    python benchmarks/rna_structure/rna_structure_benchmark.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._gradient import GradientFlowResult, check_gradient_flow
from diffbio.operators.rna_structure.rna_folding import (
    DifferentiableRNAFold,
    RNAFoldConfig,
)

# RNA nucleotide one-hot indices: A=0, C=1, G=2, U=3
_RNA_CHAR_TO_INDEX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def encode_rna_string(sequence: str) -> jnp.ndarray:
    """Encode an RNA string as a one-hot (length, 4) array.

    Args:
        sequence: RNA string containing only A, C, G, U characters.

    Returns:
        One-hot encoded array of shape ``(len(sequence), 4)``.

    Raises:
        ValueError: If the sequence contains invalid characters.
    """
    indices: list[int] = []
    for char in sequence.upper():
        if char not in _RNA_CHAR_TO_INDEX:
            msg = f"Invalid RNA character '{char}'. Expected one of A, C, G, U."
            raise ValueError(msg)
        indices.append(_RNA_CHAR_TO_INDEX[char])
    return jax.nn.one_hot(jnp.array(indices), num_classes=4)


# ------------------------------------------------------------------
# Synthetic test cases
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RNATestCase:
    """A synthetic RNA test case with known base pairs.

    Attributes:
        name: Human-readable label for the test case.
        sequence: RNA nucleotide string.
        known_pairs: List of (i, j) 0-indexed base pair positions.
    """

    name: str
    sequence: str
    known_pairs: list[tuple[int, int]] = field(default_factory=list)


_HAIRPIN = RNATestCase(
    name="simple_hairpin",
    sequence="GGGAAACCC",
    known_pairs=[(0, 8), (1, 7), (2, 6)],
)

_LONGER_STEM = RNATestCase(
    name="longer_stem",
    sequence="GGGGAAAACCCC",
    known_pairs=[(0, 11), (1, 10), (2, 9), (3, 8)],
)

_SHORT = RNATestCase(
    name="short",
    sequence="GCGAAGC",
    known_pairs=[(0, 6), (1, 5)],
)


# ------------------------------------------------------------------
# Individual benchmark routines
# ------------------------------------------------------------------


def _create_predictor(
    temperature: float = 1.0,
    min_hairpin_loop: int = 3,
) -> DifferentiableRNAFold:
    """Create a configured RNA fold predictor.

    Args:
        temperature: Boltzmann temperature parameter.
        min_hairpin_loop: Minimum unpaired nucleotides in hairpin.

    Returns:
        Initialised ``DifferentiableRNAFold`` instance.
    """
    config = RNAFoldConfig(
        temperature=temperature,
        min_hairpin_loop=min_hairpin_loop,
    )
    return DifferentiableRNAFold(config, rngs=nnx.Rngs(0))


def _check_shape_and_range(
    predictor: DifferentiableRNAFold,
    test_case: RNATestCase,
) -> dict[str, bool]:
    """Verify bp_probs shape, value range, and symmetry.

    Args:
        predictor: RNA fold operator.
        test_case: Test case to evaluate.

    Returns:
        Dictionary with shape_correct, values_in_range, is_symmetric.
    """
    seq = encode_rna_string(test_case.sequence)
    result, _, _ = predictor.apply({"sequence": seq}, {}, None)
    bp_probs = result["bp_probs"]

    length = len(test_case.sequence)
    shape_correct = bp_probs.shape == (length, length)
    values_in_range = bool(jnp.all(bp_probs >= -1e-6) & jnp.all(bp_probs <= 1.0 + 1e-6))
    is_symmetric = bool(jnp.allclose(bp_probs, bp_probs.T, atol=1e-6))

    return {
        "shape_correct": shape_correct,
        "values_in_range": values_in_range,
        "is_symmetric": is_symmetric,
    }


def _check_partition_function(
    predictor: DifferentiableRNAFold,
    test_case: RNATestCase,
) -> dict[str, bool]:
    """Verify partition function is finite and positive.

    Args:
        predictor: RNA fold operator.
        test_case: Test case to evaluate.

    Returns:
        Dictionary with partition_function_finite and
        partition_function_positive.
    """
    seq = encode_rna_string(test_case.sequence)
    result, _, _ = predictor.apply({"sequence": seq}, {}, None)
    log_z = result["partition_function"]

    is_finite = bool(jnp.isfinite(log_z))
    # exp(log_z) > 0 iff log_z is finite (not -inf)
    is_positive = bool(jnp.exp(log_z) > 0.0)

    return {
        "partition_function_finite": is_finite,
        "partition_function_positive": is_positive,
    }


def _check_known_pairs(
    predictor: DifferentiableRNAFold,
    hairpin_case: RNATestCase,
    stem_case: RNATestCase,
) -> dict[str, float | bool]:
    """Check that known base pairs have elevated probabilities.

    Compares the mean bp_prob at known pair positions against
    the mean over all valid (non-diagonal) positions as a random
    baseline.

    Args:
        predictor: RNA fold operator.
        hairpin_case: Hairpin test case with known pairs.
        stem_case: Longer-stem test case with known pairs.

    Returns:
        Dictionary with hairpin_pair_mean_prob, stem_pair_mean_prob,
        random_baseline_mean_prob, and pairs_above_baseline.
    """

    def _mean_at_pairs(
        bp_probs: jnp.ndarray,
        pairs: list[tuple[int, int]],
    ) -> float:
        """Average bp_prob at the given (i, j) positions."""
        values = [float(bp_probs[i, j]) for i, j in pairs]
        return sum(values) / len(values) if values else 0.0

    # Hairpin
    seq_h = encode_rna_string(hairpin_case.sequence)
    res_h, _, _ = predictor.apply({"sequence": seq_h}, {}, None)
    bp_h = res_h["bp_probs"]
    hairpin_mean = _mean_at_pairs(bp_h, hairpin_case.known_pairs)

    # Stem
    seq_s = encode_rna_string(stem_case.sequence)
    res_s, _, _ = predictor.apply({"sequence": seq_s}, {}, None)
    bp_s = res_s["bp_probs"]
    stem_mean = _mean_at_pairs(bp_s, stem_case.known_pairs)

    # Random baseline: mean of all entries (includes zeros on diagonal
    # and positions too close for hairpin)
    baseline = float(bp_h.mean())

    pairs_above = hairpin_mean > baseline and stem_mean > baseline

    return {
        "hairpin_pair_mean_prob": hairpin_mean,
        "stem_pair_mean_prob": stem_mean,
        "random_baseline_mean_prob": baseline,
        "pairs_above_baseline": pairs_above,
    }


def _check_gradient_flow(
    predictor: DifferentiableRNAFold,
    test_case: RNATestCase,
) -> GradientFlowResult:
    """Verify gradients flow through the RNA fold operator.

    Args:
        predictor: RNA fold operator.
        test_case: Test case to use for the forward pass.

    Returns:
        :class:`GradientFlowResult` with gradient_norm and gradient_nonzero.
    """
    seq = encode_rna_string(test_case.sequence)
    data = {"sequence": seq}

    def loss_fn(
        model: DifferentiableRNAFold,
        input_data: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sum of bp_probs as a scalar loss."""
        result, _, _ = model.apply(input_data, {}, None)
        return result["bp_probs"].sum()

    return check_gradient_flow(loss_fn, predictor, data)
