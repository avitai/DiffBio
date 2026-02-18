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
    python benchmarks/alignment_benchmark.py
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from diffbio.operators.alignment import (
    SmoothSmithWaterman,
    SmithWatermanConfig,
    create_dna_scoring_matrix,
)
from diffbio.sequences import encode_dna_string


@dataclass
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
    scoring_matrix = create_dna_scoring_matrix(match=match, mismatch=mismatch)
    config = SmithWatermanConfig(
        temperature=temperature,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )
    return SmoothSmithWaterman(config, scoring_matrix=scoring_matrix, rngs=nnx.Rngs(42))


def test_accuracy(aligner: SmoothSmithWaterman) -> dict[str, float]:
    """Test alignment accuracy on known cases.

    Returns:
        Dictionary of test case scores
    """
    results = {}

    # Test 1: Perfect match
    seq = "ACGTACGT"
    seq1 = encode_dna_string(seq)
    seq2 = encode_dna_string(seq)
    data = {"seq1": seq1, "seq2": seq2}
    result, _, _ = aligner.apply(data, {}, None)
    results["perfect_match_score"] = float(result["score"])

    # Test 2: Single mismatch
    seq1 = encode_dna_string("ACGTACGT")
    seq2 = encode_dna_string("ACGAACGT")  # T->A at position 3
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


def test_differentiability(aligner: SmoothSmithWaterman) -> dict[str, float | bool]:
    """Test gradient flow through alignment.

    Returns:
        Dictionary with gradient metrics
    """
    seq1 = encode_dna_string("ACGTACGT")
    seq2 = encode_dna_string("ACGTTACGT")
    data = {"seq1": seq1, "seq2": seq2}

    def loss_fn(data):
        result, _, _ = aligner.apply(data, {}, None)
        return result["score"]

    # Compute gradient with respect to seq1
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
        Dictionary mapping temperature to alignment score
    """
    results = {}
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    seq1 = encode_dna_string("ACGTACGT")
    seq2 = encode_dna_string("ACGTTACGT")

    for temp in temperatures:
        aligner = create_aligner(temperature=temp)
        data = {"seq1": seq1, "seq2": seq2}
        result, _, _ = aligner.apply(data, {}, None)
        results[f"temp_{temp}"] = float(result["score"])

    return results


def test_performance(aligner: SmoothSmithWaterman, n_pairs: int = 100) -> dict[str, float]:
    """Measure alignment performance.

    Args:
        aligner: Configured aligner
        n_pairs: Number of sequence pairs to align

    Returns:
        Performance metrics
    """
    # Generate random sequence pairs
    key = jax.random.key(42)
    seq_len = 50

    # Pre-generate sequences
    sequences = []
    for i in range(n_pairs * 2):
        key, subkey = jax.random.split(key)
        indices = jax.random.randint(subkey, (seq_len,), 0, 4)
        seq = jax.nn.one_hot(indices, 4)
        sequences.append(seq)

    # Time alignments
    start_time = time.time()
    for i in range(n_pairs):
        data = {"seq1": sequences[2 * i], "seq2": sequences[2 * i + 1]}
        result, _, _ = aligner.apply(data, {}, None)
        _ = result["score"].block_until_ready()  # Wait for GPU
    total_time = time.time() - start_time

    align_time_ms = (total_time / n_pairs) * 1000
    seqs_per_second = n_pairs / total_time

    return {
        "align_time_per_pair_ms": align_time_ms,
        "sequences_per_second": seqs_per_second,
    }


def run_benchmark() -> AlignmentBenchmarkResult:
    """Run the complete alignment benchmark.

    Returns:
        Benchmark results
    """
    print("=" * 60)
    print("DiffBio Alignment Benchmark")
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
    print(f"  Gradient non-zero: {diff_results['gradient_nonzero']}")

    # Temperature sweep
    print("\nTemperature sweep...")
    temp_results = test_temperature_sweep()
    for temp_key, score in temp_results.items():
        print(f"  {temp_key}: {score:.4f}")

    # Performance test
    print("\nMeasuring performance...")
    perf_results = test_performance(aligner, n_pairs=100)
    print(f"  Time per pair: {perf_results['align_time_per_pair_ms']:.2f} ms")
    print(f"  Throughput: {perf_results['sequences_per_second']:.1f} pairs/sec")

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


def save_results(result: AlignmentBenchmarkResult, output_dir: Path) -> None:
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"alignment_benchmark_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"Results saved to: {output_file}")


def main():
    """Main entry point."""
    result = run_benchmark()
    save_results(result, Path("benchmarks/results"))


if __name__ == "__main__":
    main()
