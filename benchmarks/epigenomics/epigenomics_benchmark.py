#!/usr/bin/env python3
"""Epigenomics Benchmark for DiffBio.

This benchmark evaluates DiffBio's epigenomics operators:
- DifferentiablePeakCaller (CNN-based peak calling)
- FNOPeakCaller (Fourier Neural Operator peak calling)
- ChromatinStateAnnotator (HMM-based chromatin state annotation)

Metrics:
- Peak caller output shape correctness and probability ranges
- CNN vs FNO peak caller correlation comparison
- Chromatin state posterior normalization
- Gradient flow for all operators
- Throughput (positions/second)

Usage:
    python benchmarks/epigenomics/epigenomics_benchmark.py
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
    generate_synthetic_coverage,
    measure_throughput,
    save_benchmark_result,
)
from diffbio.operators.epigenomics.chromatin_state import (
    ChromatinStateAnnotator,
    ChromatinStateConfig,
)
from diffbio.operators.epigenomics.fno_peak_calling import (
    FNOPeakCaller,
    FNOPeakCallerConfig,
)
from diffbio.operators.epigenomics.peak_calling import (
    DifferentiablePeakCaller,
    PeakCallerConfig,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class EpigenomicsBenchmarkResult:
    """Results from the epigenomics benchmark suite.

    Attributes:
        timestamp: ISO-formatted timestamp of the run.
        signal_length: Length of the synthetic coverage signal.
        n_peaks: Number of peaks in the synthetic signal.
        peak_caller_shape_ok: Whether CNN peak caller outputs
            have correct shapes.
        peak_caller_probs_in_range: Whether CNN peak probabilities
            are in [0, 1].
        fno_shape_ok: Whether FNO peak caller outputs have correct
            shapes.
        fno_probs_in_range: Whether FNO peak probabilities
            are in [0, 1].
        cnn_fno_correlation: Pearson correlation between CNN and FNO
            peak probabilities.
        chromatin_posteriors_sum_ok: Whether chromatin state posteriors
            sum to 1 per position.
        peak_caller_gradient: Gradient flow results for CNN peak
            caller.
        fno_gradient: Gradient flow results for FNO peak caller.
        chromatin_gradient: Gradient flow results for chromatin state
            annotator.
        peak_caller_throughput: Throughput metrics for CNN peak caller.
        fno_throughput: Throughput metrics for FNO peak caller.
        chromatin_throughput: Throughput metrics for chromatin state
            annotator.
    """

    timestamp: str
    signal_length: int
    n_peaks: int
    # Shape correctness
    peak_caller_shape_ok: bool
    peak_caller_probs_in_range: bool
    fno_shape_ok: bool
    fno_probs_in_range: bool
    # CNN vs FNO comparison
    cnn_fno_correlation: float
    # Chromatin state validation
    chromatin_posteriors_sum_ok: bool
    # Gradient flow
    peak_caller_gradient: dict[str, float | bool] = field(
        default_factory=dict,
    )
    fno_gradient: dict[str, float | bool] = field(
        default_factory=dict,
    )
    chromatin_gradient: dict[str, float | bool] = field(
        default_factory=dict,
    )
    # Throughput
    peak_caller_throughput: dict[str, float] = field(
        default_factory=dict,
    )
    fno_throughput: dict[str, float] = field(
        default_factory=dict,
    )
    chromatin_throughput: dict[str, float] = field(
        default_factory=dict,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _generate_histone_marks(
    length: int,
    n_marks: int = 6,
    n_states: int = 4,
    seed: int = 99,
) -> jnp.ndarray:
    """Generate synthetic histone mark data with known state structure.

    Creates a signal where positions are assigned to one of
    ``n_states`` latent states, and each state has a characteristic
    pattern of histone marks.

    Args:
        length: Number of genomic positions.
        n_marks: Number of histone marks (channels).
        n_states: Number of latent chromatin states.
        seed: Random seed for reproducibility.

    Returns:
        Histone mark array of shape ``(length, n_marks)``.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key, 2)

    # State-specific mark profiles (logit scale)
    state_profiles = jax.random.normal(k1, (n_states, n_marks)) * 2.0

    # Assign positions to states in blocks
    block_size = length // n_states
    state_indices = jnp.repeat(
        jnp.arange(n_states), block_size,
    )
    # Pad to exact length if needed
    if state_indices.shape[0] < length:
        pad = jnp.full(
            (length - state_indices.shape[0],), n_states - 1,
        )
        state_indices = jnp.concatenate([state_indices, pad])
    state_indices = state_indices[:length]

    # Build signal from profiles plus noise
    marks = state_profiles[state_indices]
    noise = jax.random.normal(k2, (length, n_marks)) * 0.5
    return marks + noise


# ------------------------------------------------------------------
# Operator tests
# ------------------------------------------------------------------


def _test_peak_caller(
    coverage: jnp.ndarray,
) -> tuple[dict, DifferentiablePeakCaller]:
    """Test DifferentiablePeakCaller on synthetic coverage.

    Args:
        coverage: Coverage signal of shape ``(length,)``.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    config = PeakCallerConfig(
        window_size=200,
        num_filters=16,
        kernel_sizes=(5, 11, 21),
        threshold=0.5,
        temperature=1.0,
    )
    rngs = nnx.Rngs(42)
    peak_caller = DifferentiablePeakCaller(config, rngs=rngs)

    data = {"coverage": coverage}
    result, _, _ = peak_caller.apply(data, {}, None)

    length = coverage.shape[0]
    shape_ok = (
        result["peak_scores"].shape == (length,)
        and result["peak_probabilities"].shape == (length,)
        and result["peak_summits"].shape == (length,)
        and result["peak_starts"].shape == (length,)
        and result["peak_ends"].shape == (length,)
    )
    probs = result["peak_probabilities"]
    probs_in_range = bool(
        jnp.all(probs >= 0.0) and jnp.all(probs <= 1.0)
    )

    return {
        "shape_ok": shape_ok,
        "probs_in_range": probs_in_range,
        "peak_probabilities": probs,
    }, peak_caller


def _test_fno_peak_caller(
    coverage: jnp.ndarray,
) -> tuple[dict, FNOPeakCaller]:
    """Test FNOPeakCaller on synthetic coverage.

    Args:
        coverage: Coverage signal of shape ``(length,)``.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    config = FNOPeakCallerConfig(
        hidden_channels=16,
        modes=8,
        num_layers=2,
        threshold=0.5,
        temperature=1.0,
    )
    rngs = nnx.Rngs(42)
    fno_caller = FNOPeakCaller(config, rngs=rngs, name="fno_peak")

    data = {"coverage": coverage}
    result, _, _ = fno_caller.apply(data, {}, None)

    length = coverage.shape[0]
    shape_ok = (
        result["peak_scores"].shape == (length,)
        and result["peak_probabilities"].shape == (length,)
    )
    probs = result["peak_probabilities"]
    probs_in_range = bool(
        jnp.all(probs >= 0.0) and jnp.all(probs <= 1.0)
    )

    return {
        "shape_ok": shape_ok,
        "probs_in_range": probs_in_range,
        "peak_probabilities": probs,
    }, fno_caller


def _test_chromatin_state(
    histone_marks: jnp.ndarray,
    n_marks: int,
    n_states: int,
) -> tuple[dict, ChromatinStateAnnotator]:
    """Test ChromatinStateAnnotator on synthetic histone data.

    Args:
        histone_marks: Histone mark array of shape
            ``(length, n_marks)``.
        n_marks: Number of histone mark channels.
        n_states: Number of chromatin states.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    config = ChromatinStateConfig(
        num_states=n_states,
        num_marks=n_marks,
        temperature=1.0,
    )
    rngs = nnx.Rngs(42)
    annotator = ChromatinStateAnnotator(config, rngs=rngs)

    data = {"histone_marks": histone_marks}
    result, _, _ = annotator.apply(data, {}, None)

    # Posteriors should sum to 1 along the state axis
    posteriors = result["state_posteriors"]
    sums = jnp.sum(posteriors, axis=-1)
    posteriors_sum_ok = bool(
        jnp.allclose(sums, 1.0, atol=1e-4)
    )

    return {
        "posteriors_sum_ok": posteriors_sum_ok,
    }, annotator


def _compute_pearson_correlation(
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> float:
    """Compute Pearson correlation between two 1-D arrays.

    Args:
        x: First array.
        y: Second array.

    Returns:
        Pearson correlation coefficient.
    """
    x_centered = x - jnp.mean(x)
    y_centered = y - jnp.mean(y)
    numerator = jnp.sum(x_centered * y_centered)
    denominator = jnp.sqrt(
        jnp.sum(x_centered ** 2) * jnp.sum(y_centered ** 2)
    )
    # Guard against zero variance
    safe_denom = jnp.where(denominator > 1e-12, denominator, 1.0)
    return float(numerator / safe_denom)


# ------------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------------


def run_benchmark(
    *, quick: bool = False,
) -> EpigenomicsBenchmarkResult:
    """Run the complete epigenomics benchmark.

    Args:
        quick: If True, use smaller data for faster execution.

    Returns:
        Benchmark results dataclass.
    """
    signal_length = 500 if quick else 5000
    n_peaks = 3 if quick else 10
    n_marks = 6
    n_states = 15
    n_throughput_iters = 20 if quick else 100

    print("=" * 60)
    print("DiffBio Epigenomics Benchmark")
    print("=" * 60)
    print(f"  Signal length : {signal_length}")
    print(f"  Peaks         : {n_peaks}")
    print(f"  Quick mode    : {quick}")

    # ----- Synthetic data -----
    print("\nGenerating synthetic data...")
    coverage, _truth_mask = generate_synthetic_coverage(
        length=signal_length, n_peaks=n_peaks,
    )
    histone_marks = _generate_histone_marks(
        length=signal_length, n_marks=n_marks, n_states=4,
    )

    # ----- CNN Peak Caller -----
    print("\nTesting DifferentiablePeakCaller...")
    pc_metrics, peak_caller = _test_peak_caller(coverage)
    print(f"  Shape OK        : {pc_metrics['shape_ok']}")
    print(f"  Probs in [0,1]  : {pc_metrics['probs_in_range']}")

    # ----- FNO Peak Caller -----
    print("\nTesting FNOPeakCaller...")
    fno_metrics, fno_caller = _test_fno_peak_caller(coverage)
    print(f"  Shape OK        : {fno_metrics['shape_ok']}")
    print(f"  Probs in [0,1]  : {fno_metrics['probs_in_range']}")

    # ----- CNN vs FNO correlation -----
    cnn_probs = pc_metrics["peak_probabilities"]
    fno_probs = fno_metrics["peak_probabilities"]
    correlation = _compute_pearson_correlation(cnn_probs, fno_probs)
    print(f"\nCNN vs FNO correlation: {correlation:.4f}")

    # ----- Chromatin State Annotator -----
    print("\nTesting ChromatinStateAnnotator...")
    cs_metrics, annotator = _test_chromatin_state(
        histone_marks, n_marks=n_marks, n_states=n_states,
    )
    print(
        f"  Posteriors sum to 1: {cs_metrics['posteriors_sum_ok']}"
    )

    # ----- Gradient flow -----
    print("\nChecking gradient flow...")

    def _pc_loss(model: DifferentiablePeakCaller) -> jax.Array:
        """Loss for CNN peak caller gradient check."""
        out, _, _ = model.apply({"coverage": coverage}, {}, None)
        return jnp.sum(out["peak_probabilities"])

    pc_grad = check_gradient_flow(_pc_loss, peak_caller)
    print(
        f"  PeakCaller       : norm={pc_grad.gradient_norm:.6f}"
        f"  nonzero={pc_grad.gradient_nonzero}"
    )

    def _fno_loss(model: FNOPeakCaller) -> jax.Array:
        """Loss for FNO peak caller gradient check."""
        out, _, _ = model.apply({"coverage": coverage}, {}, None)
        return jnp.sum(out["peak_probabilities"])

    fno_grad = check_gradient_flow(_fno_loss, fno_caller)
    print(
        f"  FNOPeakCaller    : norm={fno_grad.gradient_norm:.6f}"
        f"  nonzero={fno_grad.gradient_nonzero}"
    )

    def _cs_loss(model: ChromatinStateAnnotator) -> jax.Array:
        """Loss for chromatin state gradient check."""
        out, _, _ = model.apply(
            {"histone_marks": histone_marks}, {}, None,
        )
        return out["log_likelihood"]

    cs_grad = check_gradient_flow(_cs_loss, annotator)
    print(
        f"  ChromatinState   : norm={cs_grad.gradient_norm:.6f}"
        f"  nonzero={cs_grad.gradient_nonzero}"
    )

    # ----- Throughput -----
    print("\nMeasuring throughput...")

    pc_tp = measure_throughput(
        lambda: peak_caller.apply({"coverage": coverage}, {}, None),
        args=(),
        n_iterations=n_throughput_iters,
        warmup=3,
    )
    pc_positions_per_sec = signal_length * pc_tp["items_per_sec"]
    print(
        f"  PeakCaller       : {pc_positions_per_sec:.0f} pos/s"
        f"  ({pc_tp['per_item_ms']:.2f} ms/call)"
    )

    fno_tp = measure_throughput(
        lambda: fno_caller.apply({"coverage": coverage}, {}, None),
        args=(),
        n_iterations=n_throughput_iters,
        warmup=3,
    )
    fno_positions_per_sec = signal_length * fno_tp["items_per_sec"]
    print(
        f"  FNOPeakCaller    : {fno_positions_per_sec:.0f} pos/s"
        f"  ({fno_tp['per_item_ms']:.2f} ms/call)"
    )

    cs_tp = measure_throughput(
        lambda: annotator.apply(
            {"histone_marks": histone_marks}, {}, None,
        ),
        args=(),
        n_iterations=n_throughput_iters,
        warmup=3,
    )
    cs_positions_per_sec = signal_length * cs_tp["items_per_sec"]
    print(
        f"  ChromatinState   : {cs_positions_per_sec:.0f} pos/s"
        f"  ({cs_tp['per_item_ms']:.2f} ms/call)"
    )

    # ----- Compile result -----
    result = EpigenomicsBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        signal_length=signal_length,
        n_peaks=n_peaks,
        peak_caller_shape_ok=pc_metrics["shape_ok"],
        peak_caller_probs_in_range=pc_metrics["probs_in_range"],
        fno_shape_ok=fno_metrics["shape_ok"],
        fno_probs_in_range=fno_metrics["probs_in_range"],
        cnn_fno_correlation=correlation,
        chromatin_posteriors_sum_ok=cs_metrics["posteriors_sum_ok"],
        peak_caller_gradient=pc_grad,
        fno_gradient=fno_grad,
        chromatin_gradient=cs_grad,
        peak_caller_throughput={
            **pc_tp,
            "positions_per_sec": pc_positions_per_sec,
        },
        fno_throughput={
            **fno_tp,
            "positions_per_sec": fno_positions_per_sec,
        },
        chromatin_throughput={
            **cs_tp,
            "positions_per_sec": cs_positions_per_sec,
        },
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        f"  PeakCaller shapes OK    : {pc_metrics['shape_ok']}"
    )
    print(
        f"  FNO shapes OK           : {fno_metrics['shape_ok']}"
    )
    print(f"  CNN/FNO correlation     : {correlation:.4f}")
    print(
        f"  Posteriors sum to 1     :"
        f" {cs_metrics['posteriors_sum_ok']}"
    )
    all_grads_ok = (
        pc_grad.gradient_nonzero
        and fno_grad.gradient_nonzero
        and cs_grad.gradient_nonzero
    )
    print(f"  All gradients nonzero   : {all_grads_ok}")
    print("=" * 60)

    return result


def main() -> None:
    """Entry point for the epigenomics benchmark."""
    result = run_benchmark()
    output_path = save_benchmark_result(
        asdict(result),
        domain="epigenomics",
        benchmark_name="epigenomics_benchmark",
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
