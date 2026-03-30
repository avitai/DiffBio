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

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

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
            msg = (
                f"Invalid RNA character '{char}'. "
                "Expected one of A, C, G, U."
            )
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
# Result dataclass
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RNAStructureBenchmarkResult:
    """Results from the RNA structure benchmark.

    Attributes:
        timestamp: ISO-format timestamp of the run.
        temperature: Temperature parameter used.
        min_hairpin_loop: Minimum hairpin loop size.
        shape_correct: Whether bp_probs had the expected (L, L) shape.
        values_in_range: Whether all bp_probs values fall in [0, 1].
        is_symmetric: Whether bp_probs is symmetric within tolerance.
        partition_function_finite: Whether partition_function is finite.
        partition_function_positive: Whether exp(partition_function) > 0.
        hairpin_pair_mean_prob: Mean bp_prob at known hairpin pairs.
        stem_pair_mean_prob: Mean bp_prob at known stem pairs.
        random_baseline_mean_prob: Mean bp_prob across all valid entries.
        pairs_above_baseline: Whether known pairs exceed the baseline.
        gradient_norm: L2 norm of parameter gradients.
        gradient_nonzero: Whether gradient norm exceeds threshold.
        nucleotides_per_second: Throughput in nucleotides / second.
        per_fold_ms: Wall time per single fold call (ms).
    """

    timestamp: str
    temperature: float
    min_hairpin_loop: int
    # Shape and range
    shape_correct: bool
    values_in_range: bool
    is_symmetric: bool
    # Partition function
    partition_function_finite: bool
    partition_function_positive: bool
    # Known-pair detection
    hairpin_pair_mean_prob: float
    stem_pair_mean_prob: float
    random_baseline_mean_prob: float
    pairs_above_baseline: bool
    # Differentiability
    gradient_norm: float
    gradient_nonzero: bool
    # Performance
    nucleotides_per_second: float
    per_fold_ms: float


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
    values_in_range = bool(
        jnp.all(bp_probs >= -1e-6) & jnp.all(bp_probs <= 1.0 + 1e-6)
    )
    is_symmetric = bool(
        jnp.allclose(bp_probs, bp_probs.T, atol=1e-6)
    )

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


def _measure_throughput(
    predictor: DifferentiableRNAFold,
    seq_length: int,
    n_iterations: int,
    warmup: int,
) -> dict[str, float]:
    """Measure folding throughput in nucleotides per second.

    Args:
        predictor: RNA fold operator.
        seq_length: Length of the random sequence to fold.
        n_iterations: Number of timed iterations.
        warmup: Number of untimed warmup iterations.

    Returns:
        Dictionary with nucleotides_per_second and per_fold_ms.
    """
    key = jax.random.key(99)
    indices = jax.random.randint(key, (seq_length,), 0, 4)
    seq = jax.nn.one_hot(indices, 4)
    data = {"sequence": seq}

    def fold_fn(
        input_data: dict[str, jnp.ndarray],
    ) -> tuple[dict, dict, None]:
        """Wrapper to pass to measure_throughput."""
        return predictor.apply(input_data, {}, None)

    raw = measure_throughput(
        fold_fn,
        args=(data,),
        n_iterations=n_iterations,
        warmup=warmup,
    )

    nucleotides_per_sec = (
        seq_length * raw["items_per_sec"]
    )

    return {
        "nucleotides_per_second": nucleotides_per_sec,
        "per_fold_ms": raw["per_item_ms"],
    }


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def run_benchmark(
    quick: bool = False,
) -> RNAStructureBenchmarkResult:
    """Run the complete RNA structure benchmark.

    Args:
        quick: If True, use shorter sequences and fewer iterations
            for a faster run (useful in CI).

    Returns:
        Frozen dataclass with all benchmark metrics.
    """
    print("=" * 60)
    print("DiffBio RNA Structure Benchmark")
    print("=" * 60)

    temperature = 1.0
    min_hairpin_loop = 3
    predictor = _create_predictor(
        temperature=temperature,
        min_hairpin_loop=min_hairpin_loop,
    )

    # -- Shape, range, symmetry --
    test_case = _SHORT if quick else _HAIRPIN
    print("\nChecking bp_probs shape, range, symmetry...")
    shape_results = _check_shape_and_range(predictor, test_case)
    for key, val in shape_results.items():
        print(f"  {key}: {val}")

    # -- Partition function --
    print("\nChecking partition function...")
    pf_results = _check_partition_function(predictor, test_case)
    for key, val in pf_results.items():
        print(f"  {key}: {val}")

    # -- Known pair detection --
    print("\nChecking known base pair detection...")
    pair_results = _check_known_pairs(
        predictor, _HAIRPIN, _LONGER_STEM
    )
    for key, val in pair_results.items():
        print(f"  {key}: {val}")

    # -- Gradient flow --
    print("\nChecking gradient flow...")
    grad_case = _SHORT if quick else _HAIRPIN
    grad_results = _check_gradient_flow(predictor, grad_case)
    print(f"  gradient_norm: {grad_results.gradient_norm:.6f}")
    print(f"  gradient_nonzero: {grad_results.gradient_nonzero}")

    # -- Throughput --
    seq_length = 20 if quick else 50
    n_iter = 20 if quick else 100
    warmup_iter = 3 if quick else 5
    print(f"\nMeasuring throughput (seq_length={seq_length})...")
    perf_results = _measure_throughput(
        predictor, seq_length, n_iter, warmup_iter
    )
    print(
        f"  nucleotides/sec: "
        f"{perf_results['nucleotides_per_second']:.0f}"
    )
    print(f"  per_fold_ms: {perf_results['per_fold_ms']:.2f}")

    # -- Assemble result --
    result = RNAStructureBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        temperature=temperature,
        min_hairpin_loop=min_hairpin_loop,
        **shape_results,
        **pf_results,
        **pair_results,  # pyright: ignore[reportArgumentType]
        **asdict(grad_results),
        **perf_results,
    )

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    return result


def main() -> None:
    """Entry point for the RNA structure benchmark."""
    result = run_benchmark()
    output_path = save_benchmark_result(
        result=asdict(result),
        domain="rna_structure",
        benchmark_name="rna_structure_benchmark",
        output_dir=Path("benchmarks/results"),
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
