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
- Throughput (reads/second)

Usage:
    python benchmarks/preprocessing/preprocessing_benchmark.py
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._common import (
    check_gradient_flow,
    generate_synthetic_sequences,
    measure_throughput,
    save_benchmark_result,
)
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

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class PreprocessingBenchmarkResult:
    """Results from the preprocessing benchmark suite.

    Attributes:
        timestamp: ISO-formatted timestamp of the run.
        n_reads: Number of reads used for testing.
        read_length: Length of each synthetic read.
        adapter_shape_ok: Whether adapter removal outputs have
            correct shapes.
        adapter_values_finite: Whether adapter removal outputs
            contain only finite values.
        adapter_trim_position_in_range: Whether the soft trim
            position falls within [0, read_length].
        duplicate_shape_ok: Whether duplicate weighting outputs
            have correct shapes.
        duplicate_values_finite: Whether duplicate weighting outputs
            contain only finite values.
        duplicate_weights_positive: Whether uniqueness weights
            are positive.
        error_correction_shape_ok: Whether error correction outputs
            have correct shapes.
        error_correction_values_finite: Whether error correction
            outputs contain only finite values.
        error_correction_probs_valid: Whether corrected base
            probabilities sum to approximately 1.
        adapter_gradient: Gradient flow results for adapter removal.
        duplicate_gradient: Gradient flow results for duplicate
            weighting.
        error_correction_gradient: Gradient flow results for error
            correction.
        adapter_throughput: Throughput metrics for adapter removal.
        duplicate_throughput: Throughput metrics for duplicate
            weighting.
        error_correction_throughput: Throughput metrics for error
            correction.
    """

    timestamp: str
    n_reads: int
    read_length: int
    # Adapter removal validation
    adapter_shape_ok: bool
    adapter_values_finite: bool
    adapter_trim_position_in_range: bool
    # Duplicate weighting validation
    duplicate_shape_ok: bool
    duplicate_values_finite: bool
    duplicate_weights_positive: bool
    # Error correction validation
    error_correction_shape_ok: bool
    error_correction_values_finite: bool
    error_correction_probs_valid: bool
    # Gradient flow
    adapter_gradient: dict[str, float | bool] = field(
        default_factory=dict,
    )
    duplicate_gradient: dict[str, float | bool] = field(
        default_factory=dict,
    )
    error_correction_gradient: dict[str, float | bool] = field(
        default_factory=dict,
    )
    # Throughput
    adapter_throughput: dict[str, float] = field(
        default_factory=dict,
    )
    duplicate_throughput: dict[str, float] = field(
        default_factory=dict,
    )
    error_correction_throughput: dict[str, float] = field(
        default_factory=dict,
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
    adapter_indices = jnp.array(
        [base_to_idx[b] for b in adapter_sequence]
    )
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
        k3, (n_reads, read_length), minval=10.0, maxval=40.0,
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
    noise = (
        jax.random.normal(k2, sequences.shape) * 0.01
    )
    sequences = jax.nn.softmax(
        jnp.log(sequences + 1e-6) + noise, axis=-1,
    )

    # Generate quality scores
    quality_scores = jax.random.uniform(
        k3, (n_reads, read_length), minval=10.0, maxval=40.0,
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
        k1, (n_reads, read_length), minval=20.0, maxval=40.0,
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
        config, rngs=nnx.Rngs(42),
    )

    data = {"sequence": sequences, "quality_scores": quality_scores}
    result, _, _ = weighter.apply(data, {}, None)

    batch_size = sequences.shape[0]
    read_length = sequences.shape[1]

    # Shape checks (batched input returns original sequences unchanged,
    # plus scalar weight and embedding for the first element)
    shape_ok = (
        result["sequence"].shape == (batch_size, read_length, 4)
        and result["quality_scores"].shape
        == (batch_size, read_length)
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


# ------------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------------


def run_benchmark(
    *, quick: bool = False,
) -> PreprocessingBenchmarkResult:
    """Run the complete preprocessing benchmark.

    Args:
        quick: If True, use smaller data for faster execution.

    Returns:
        Benchmark results dataclass.
    """
    n_reads = 20 if quick else 50
    read_length = 50 if quick else 100
    n_duplicate_groups = 3 if quick else 5
    n_throughput_iters = 20 if quick else 100

    print("=" * 60)
    print("DiffBio Preprocessing Benchmark")
    print("=" * 60)
    print(f"  Reads       : {n_reads}")
    print(f"  Read length : {read_length}")
    print(f"  Quick mode  : {quick}")

    # ----- Synthetic data -----
    print("\nGenerating synthetic data...")
    adapter_seqs, adapter_quals = _generate_reads_with_adapter(
        n_reads=n_reads, read_length=read_length,
    )
    dup_seqs, dup_quals = _generate_duplicate_data(
        n_reads=n_reads,
        read_length=read_length,
        n_duplicate_groups=n_duplicate_groups,
    )
    ec_seqs, ec_quals = _generate_error_correction_data(
        n_reads=n_reads, read_length=read_length,
    )

    # ----- Adapter Removal -----
    # Test on a single read (adapter removal operates per-read)
    print("\nTesting SoftAdapterRemoval...")
    adapter_metrics, adapter_op = _test_adapter_removal(
        adapter_seqs[0], adapter_quals[0], read_length,
    )
    print(f"  Shape OK              : {adapter_metrics['shape_ok']}")
    print(
        f"  Values finite         : "
        f"{adapter_metrics['values_finite']}"
    )
    print(
        f"  Trim position in range: "
        f"{adapter_metrics['trim_in_range']}"
    )

    # ----- Duplicate Weighting -----
    print("\nTesting DifferentiableDuplicateWeighting...")
    dup_metrics, dup_op = _test_duplicate_weighting(
        dup_seqs, dup_quals,
    )
    print(f"  Shape OK              : {dup_metrics['shape_ok']}")
    print(
        f"  Values finite         : "
        f"{dup_metrics['values_finite']}"
    )
    print(
        f"  Weights positive      : "
        f"{dup_metrics['weights_positive']}"
    )

    # ----- Error Correction -----
    print("\nTesting SoftErrorCorrection...")
    ec_metrics, ec_op = _test_error_correction(
        ec_seqs[0], ec_quals[0], read_length,
    )
    print(f"  Shape OK              : {ec_metrics['shape_ok']}")
    print(
        f"  Values finite         : "
        f"{ec_metrics['values_finite']}"
    )
    print(
        f"  Probs valid           : "
        f"{ec_metrics['probs_valid']}"
    )

    # ----- Gradient flow -----
    print("\nChecking gradient flow...")

    single_adapter_seq = adapter_seqs[0]
    single_adapter_qual = adapter_quals[0]

    def _adapter_loss(model: SoftAdapterRemoval) -> jax.Array:
        """Loss for adapter removal gradient check."""
        out, _, _ = model.apply(
            {
                "sequence": single_adapter_seq,
                "quality_scores": single_adapter_qual,
            },
            {},
            None,
        )
        return jnp.sum(out["sequence"])

    adapter_grad = check_gradient_flow(_adapter_loss, adapter_op)
    print(
        f"  AdapterRemoval  : "
        f"norm={adapter_grad.gradient_norm:.6f}"
        f"  nonzero={adapter_grad.gradient_nonzero}"
    )

    def _dup_loss(
        model: DifferentiableDuplicateWeighting,
    ) -> jax.Array:
        """Loss for duplicate weighting gradient check."""
        out, _, _ = model.apply(
            {"sequence": dup_seqs, "quality_scores": dup_quals},
            {},
            None,
        )
        return jnp.sum(out["embedding"])

    dup_grad = check_gradient_flow(_dup_loss, dup_op)
    print(
        f"  DuplicateWeight : "
        f"norm={dup_grad.gradient_norm:.6f}"
        f"  nonzero={dup_grad.gradient_nonzero}"
    )

    single_ec_seq = ec_seqs[0]
    single_ec_qual = ec_quals[0]

    def _ec_loss(model: SoftErrorCorrection) -> jax.Array:
        """Loss for error correction gradient check."""
        out, _, _ = model.apply(
            {
                "sequence": single_ec_seq,
                "quality_scores": single_ec_qual,
            },
            {},
            None,
        )
        return jnp.sum(out["sequence"])

    ec_grad = check_gradient_flow(_ec_loss, ec_op)
    print(
        f"  ErrorCorrection : "
        f"norm={ec_grad.gradient_norm:.6f}"
        f"  nonzero={ec_grad.gradient_nonzero}"
    )

    # ----- Throughput -----
    print("\nMeasuring throughput...")

    adapter_tp = measure_throughput(
        lambda: adapter_op.apply(
            {
                "sequence": single_adapter_seq,
                "quality_scores": single_adapter_qual,
            },
            {},
            None,
        ),
        args=(),
        n_iterations=n_throughput_iters,
        warmup=3,
    )
    adapter_reads_per_sec = adapter_tp["items_per_sec"]
    print(
        f"  AdapterRemoval  : "
        f"{adapter_reads_per_sec:.0f} reads/s"
        f"  ({adapter_tp['per_item_ms']:.2f} ms/read)"
    )

    dup_tp = measure_throughput(
        lambda: dup_op.apply(
            {"sequence": dup_seqs, "quality_scores": dup_quals},
            {},
            None,
        ),
        args=(),
        n_iterations=n_throughput_iters,
        warmup=3,
    )
    dup_reads_per_sec = n_reads * dup_tp["items_per_sec"]
    print(
        f"  DuplicateWeight : "
        f"{dup_reads_per_sec:.0f} reads/s"
        f"  ({dup_tp['per_item_ms']:.2f} ms/batch)"
    )

    ec_tp = measure_throughput(
        lambda: ec_op.apply(
            {
                "sequence": single_ec_seq,
                "quality_scores": single_ec_qual,
            },
            {},
            None,
        ),
        args=(),
        n_iterations=n_throughput_iters,
        warmup=3,
    )
    ec_reads_per_sec = ec_tp["items_per_sec"]
    print(
        f"  ErrorCorrection : "
        f"{ec_reads_per_sec:.0f} reads/s"
        f"  ({ec_tp['per_item_ms']:.2f} ms/read)"
    )

    # ----- Compile result -----
    result = PreprocessingBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        n_reads=n_reads,
        read_length=read_length,
        adapter_shape_ok=adapter_metrics["shape_ok"],
        adapter_values_finite=adapter_metrics["values_finite"],
        adapter_trim_position_in_range=adapter_metrics[
            "trim_in_range"
        ],
        duplicate_shape_ok=dup_metrics["shape_ok"],
        duplicate_values_finite=dup_metrics["values_finite"],
        duplicate_weights_positive=dup_metrics[
            "weights_positive"
        ],
        error_correction_shape_ok=ec_metrics["shape_ok"],
        error_correction_values_finite=ec_metrics[
            "values_finite"
        ],
        error_correction_probs_valid=ec_metrics["probs_valid"],
        adapter_gradient=adapter_grad,
        duplicate_gradient=dup_grad,
        error_correction_gradient=ec_grad,
        adapter_throughput={
            **adapter_tp,
            "reads_per_sec": adapter_reads_per_sec,
        },
        duplicate_throughput={
            **dup_tp,
            "reads_per_sec": dup_reads_per_sec,
        },
        error_correction_throughput={
            **ec_tp,
            "reads_per_sec": ec_reads_per_sec,
        },
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        f"  Adapter shapes OK     : "
        f"{adapter_metrics['shape_ok']}"
    )
    print(
        f"  Duplicate shapes OK   : "
        f"{dup_metrics['shape_ok']}"
    )
    print(
        f"  ErrorCorr shapes OK   : "
        f"{ec_metrics['shape_ok']}"
    )
    all_grads_ok = (
        adapter_grad.gradient_nonzero
        and dup_grad.gradient_nonzero
        and ec_grad.gradient_nonzero
    )
    print(f"  All gradients nonzero : {all_grads_ok}")
    print("=" * 60)

    return result


def main() -> None:
    """Entry point for the preprocessing benchmark."""
    result = run_benchmark()
    output_path = save_benchmark_result(
        asdict(result),
        domain="preprocessing",
        benchmark_name="preprocessing_benchmark",
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
