#!/usr/bin/env python3
"""Alignment Benchmark for DiffBio.

This benchmark evaluates DiffBio's SmoothSmithWaterman operator
for correctness, differentiability, and performance.

Benchmarks:
- Alignment accuracy on synthetic sequence pairs
- Differentiability verification (gradient flow)
- Temperature sweep analysis
- Performance measurement (sequences/second)

Usage:
    python benchmarks/alignment/alignment_benchmark.py
    python benchmarks/alignment/alignment_benchmark.py --quick
"""

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from diffbio.operators.alignment import (
    SmoothSmithWaterman,
    SmithWatermanConfig,
    create_dna_scoring_matrix,
)
from diffbio.sequences import encode_dna_string

_DEFAULT_N_PAIRS = 100
_QUICK_N_PAIRS = 20


@dataclass(frozen=True, kw_only=True)
class AlignmentBenchmarkResult:
    """Results from alignment benchmark."""

    timestamp: str
    # Accuracy metrics
    perfect_match_score: float
    single_mismatch_score: float
    insertion_score: float
    deletion_score: float
    # Differentiability
    gradient_norm: float
    gradient_nonzero: bool
    # Temperature analysis
    temperature_scores: dict[str, float]
    # Performance
    align_time_per_pair_ms: float
    sequences_per_second: float
    # Configuration
    temperature: float
    gap_open: float
    gap_extend: float
    match: float
    mismatch: float


def create_aligner(
    temperature: float = 1.0,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    match: float = 2.0,
    mismatch: float = -1.0,
) -> SmoothSmithWaterman:
    """Create a configured aligner."""
    scoring_matrix = create_dna_scoring_matrix(
        match=match, mismatch=mismatch
    )
    config = SmithWatermanConfig(
        temperature=temperature,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )
    return SmoothSmithWaterman(
        config, scoring_matrix=scoring_matrix, rngs=nnx.Rngs(42)
    )


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
        pairs.append(
            (jax.nn.one_hot(idx1, 4), jax.nn.one_hot(idx2, 4))
        )
    return pairs


def test_performance(
    aligner: SmoothSmithWaterman,
    n_pairs: int = _DEFAULT_N_PAIRS,
) -> dict[str, float]:
    """Measure alignment performance using shared infrastructure.

    Args:
        aligner: Configured aligner.
        n_pairs: Number of sequence pairs to align.

    Returns:
        Performance metrics.
    """
    pairs = _generate_sequence_pairs(n_pairs)

    # Build a single callable that aligns one pre-selected pair.
    # ``measure_throughput`` will call it ``n_pairs`` times, giving
    # aggregate timing. We cycle through pairs using a mutable index.
    call_index = [0]

    def align_one() -> tuple[dict, dict, None]:
        idx = call_index[0] % len(pairs)
        call_index[0] += 1
        seq1, seq2 = pairs[idx]
        data = {"seq1": seq1, "seq2": seq2}
        return aligner.apply(data, {}, None)

    throughput = measure_throughput(
        fn=align_one,
        args=(),
        n_iterations=n_pairs,
        warmup=min(5, n_pairs),
    )

    return {
        "align_time_per_pair_ms": throughput["per_item_ms"],
        "sequences_per_second": throughput["items_per_sec"],
    }


def run_benchmark(
    *,
    quick: bool = False,
) -> AlignmentBenchmarkResult:
    """Run the complete alignment benchmark.

    Args:
        quick: If True, use fewer sequence pairs for a faster run.

    Returns:
        Benchmark results.
    """
    from datetime import datetime  # noqa: PLC0415

    n_pairs = _QUICK_N_PAIRS if quick else _DEFAULT_N_PAIRS

    print("=" * 60)
    print("DiffBio Alignment Benchmark")
    if quick:
        print("  (quick mode)")
    print("=" * 60)

    # Configuration
    temperature = 1.0
    gap_open = -2.0
    gap_extend = -0.5
    match = 2.0
    mismatch = -1.0

    # Create aligner
    print("\nCreating aligner...")
    aligner = create_aligner(
        temperature=temperature,
        gap_open=gap_open,
        gap_extend=gap_extend,
        match=match,
        mismatch=mismatch,
    )

    # Test accuracy
    print("\nTesting alignment accuracy...")
    accuracy_results = test_accuracy(aligner)
    for name, score in accuracy_results.items():
        print(f"  {name}: {score:.4f}")

    # Test differentiability
    print("\nTesting differentiability...")
    diff_results = test_differentiability(aligner)
    print(f"  Gradient norm: {diff_results['gradient_norm']:.6f}")
    print(
        f"  Gradient non-zero: {diff_results['gradient_nonzero']}"
    )

    # Temperature sweep
    print("\nTemperature sweep...")
    temp_results = test_temperature_sweep()
    for temp_key, score in temp_results.items():
        print(f"  {temp_key}: {score:.4f}")

    # Performance test
    print(f"\nMeasuring performance ({n_pairs} pairs)...")
    perf_results = test_performance(aligner, n_pairs=n_pairs)
    print(
        f"  Time per pair: "
        f"{perf_results['align_time_per_pair_ms']:.2f} ms"
    )
    print(
        f"  Throughput: "
        f"{perf_results['sequences_per_second']:.1f} pairs/sec"
    )

    # Compile results
    result = AlignmentBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        **accuracy_results,  # pyright: ignore[reportArgumentType]
        **diff_results,
        temperature_scores=temp_results,
        **perf_results,
        temperature=temperature,
        gap_open=gap_open,
        gap_extend=gap_extend,
        match=match,
        mismatch=mismatch,
    )

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    return result


def main() -> None:
    """Main entry point."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="DiffBio Alignment Benchmark",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with fewer iterations for a faster check",
    )
    args = parser.parse_args()

    result = run_benchmark(quick=args.quick)

    output_path = save_benchmark_result(
        result=asdict(result),
        domain="alignment",
        benchmark_name="alignment_benchmark",
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
