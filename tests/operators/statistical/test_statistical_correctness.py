#!/usr/bin/env python3
"""Statistical Model Benchmark for DiffBio.

This benchmark evaluates DiffBio's statistical model operators
for correctness, differentiability, and performance:

- DifferentiableHMM (Hidden Markov Model)
- DifferentiableNBGLM (Negative Binomial GLM)
- DifferentiableEMQuantifier (EM transcript quantification)

Metrics:
- Correctness checks (shapes, value constraints)
- Differentiability verification (gradient flow)
- Throughput measurement (iterations/second)

Usage:
    python benchmarks/statistical/statistical_benchmark.py
    python benchmarks/statistical/statistical_benchmark.py --quick
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._gradient import check_gradient_flow
from diffbio.operators.statistical.em_quantification import (
    DifferentiableEMQuantifier,
    EMQuantifierConfig,
)
from diffbio.operators.statistical.hmm import (
    DifferentiableHMM,
    HMMConfig,
)
from diffbio.operators.statistical.nb_glm import (
    DifferentiableNBGLM,
    NBGLMConfig,
)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------


@dataclass(frozen=True)
class HMMMetrics:
    """Metrics for the HMM operator benchmark."""

    log_likelihood_finite: bool
    posteriors_sum_to_one: bool
    posteriors_shape_correct: bool
    gradient_norm: float
    gradient_nonzero: bool
    throughput_items_per_sec: float
    throughput_per_item_ms: float


@dataclass(frozen=True)
class NBGLMMetrics:
    """Metrics for the NB-GLM operator benchmark."""

    predicted_mean_positive: bool
    dispersion_positive: bool
    shape_correct: bool
    gradient_norm: float
    gradient_nonzero: bool
    throughput_items_per_sec: float
    throughput_per_item_ms: float


@dataclass(frozen=True)
class EMMetrics:
    """Metrics for the EM quantifier operator benchmark."""

    abundances_sum_to_one: bool
    tpm_positive: bool
    shape_correct: bool
    gradient_norm: float
    gradient_nonzero: bool
    throughput_items_per_sec: float
    throughput_per_item_ms: float


@dataclass(frozen=True)
class StatisticalBenchmarkResult:
    """Results from the statistical model benchmark."""

    timestamp: str
    quick_mode: bool
    hmm: HMMMetrics
    nb_glm: NBGLMMetrics
    em: EMMetrics
    platform_info: dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------


def generate_hmm_data(
    seq_len: int = 500,
    num_states: int = 3,
    num_emissions: int = 4,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic HMM observation sequence.

    Constructs transition and emission matrices, then samples
    a sequence of observations from the resulting Markov chain.

    Args:
        seq_len: Length of the observation sequence.
        num_states: Number of hidden states.
        num_emissions: Number of distinct emission symbols.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``observations`` key containing an
        integer-encoded array of shape ``(seq_len,)``.
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 4)

    # Build a transition matrix with strong self-loops
    raw_trans = jnp.eye(num_states) * 5.0 + jax.random.uniform(
        keys[0], (num_states, num_states)
    )
    trans = raw_trans / raw_trans.sum(axis=1, keepdims=True)

    # Build a random emission matrix
    raw_emit = jax.random.uniform(
        keys[1], (num_states, num_emissions)
    ) + 0.1
    emit = raw_emit / raw_emit.sum(axis=1, keepdims=True)

    # Sample an initial state then roll forward
    initial_state = int(
        jax.random.categorical(keys[2], jnp.zeros(num_states))
    )
    state = initial_state
    observations_list: list[int] = []
    step_key = keys[3]
    for _ in range(seq_len):
        step_key, emit_key, trans_key = jax.random.split(
            step_key, 3
        )
        obs = int(
            jax.random.categorical(emit_key, jnp.log(emit[state]))
        )
        observations_list.append(obs)
        state = int(
            jax.random.categorical(trans_key, jnp.log(trans[state]))
        )

    observations = jnp.array(observations_list, dtype=jnp.int32)
    return {"observations": observations}


def generate_nb_glm_data(
    n_features: int = 50,
    n_covariates: int = 2,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic NB-GLM count data for a single sample.

    Produces a count vector drawn from a negative binomial
    distribution, a design row, and a scalar size factor.

    Args:
        n_features: Number of genes/features.
        n_covariates: Number of covariates in the design row.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``counts``, ``design``, and ``size_factor``.
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 4)

    # Design row: intercept + treatment indicator
    design = jnp.zeros(n_covariates).at[0].set(1.0)
    if n_covariates > 1:
        design = design.at[1].set(1.0)

    # Simulate counts from a Gamma-Poisson mixture
    log_mean = jax.random.normal(keys[0], (n_features,)) * 0.5 + 3.0
    rates = jnp.exp(log_mean)
    dispersion = 5.0
    gamma_samples = jax.random.gamma(
        keys[1], dispersion, (n_features,)
    )
    scaled_rates = rates * gamma_samples / dispersion
    counts = jax.random.poisson(
        keys[2], scaled_rates
    ).astype(jnp.float32)

    size_factor = jnp.array(1.0)

    return {
        "counts": counts,
        "design": design,
        "size_factor": size_factor,
    }


def generate_em_data(
    n_reads: int = 200,
    n_transcripts: int = 20,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic EM quantification data.

    Creates a read-transcript compatibility matrix where each read
    is compatible with a small subset of transcripts, plus effective
    transcript lengths.

    Args:
        n_reads: Number of reads.
        n_transcripts: Number of transcripts.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``compatibility`` and ``effective_lengths``.
    """
    key = jax.random.key(seed)
    keys = jax.random.split(key, 3)

    # Sparse-ish compatibility: each read maps to ~3 transcripts
    raw = jax.random.uniform(keys[0], (n_reads, n_transcripts))
    threshold = jnp.percentile(raw, 85.0)
    compatibility = jnp.where(raw > threshold, raw, 0.0)

    # Ensure every read has at least one compatible transcript
    max_idx = jnp.argmax(raw, axis=1)
    row_indices = jnp.arange(n_reads)
    compatibility = compatibility.at[row_indices, max_idx].set(
        raw[row_indices, max_idx]
    )

    # Effective transcript lengths (positive)
    effective_lengths = (
        jax.random.uniform(keys[1], (n_transcripts,)) * 1000.0
        + 200.0
    )

    return {
        "compatibility": compatibility,
        "effective_lengths": effective_lengths,
    }


# ------------------------------------------------------------------
# Individual operator benchmarks
# ------------------------------------------------------------------


def benchmark_hmm(quick: bool = False) -> HMMMetrics:
    """Benchmark the DifferentiableHMM operator.

    Args:
        quick: If True, use a shorter observation sequence.

    Returns:
        HMMMetrics with correctness, gradient, and throughput data.
    """
    print("\n--- DifferentiableHMM ---")
    num_states = 3
    num_emissions = 4
    seq_len = 100 if quick else 500

    # Create operator
    config = HMMConfig(
        num_states=num_states,
        num_emissions=num_emissions,
    )
    hmm = DifferentiableHMM(config, rngs=nnx.Rngs(42))

    # Generate data
    data = generate_hmm_data(
        seq_len=seq_len,
        num_states=num_states,
        num_emissions=num_emissions,
    )

    # Run operator
    result, _, _ = hmm.apply(data, {}, None)
    log_likelihood = result["log_likelihood"]
    state_posteriors = result["state_posteriors"]

    # Correctness checks
    ll_finite = bool(jnp.isfinite(log_likelihood))
    posterior_sums = jnp.sum(state_posteriors, axis=-1)
    posteriors_sum_ok = bool(
        jnp.allclose(posterior_sums, 1.0, atol=1e-4)
    )
    shape_ok = state_posteriors.shape == (seq_len, num_states)

    print(f"  Log-likelihood: {float(log_likelihood):.4f}")
    print(f"  Log-likelihood finite: {ll_finite}")
    print(
        f"  Posteriors sum to 1: {posteriors_sum_ok} "
        f"(max dev: {float(jnp.max(jnp.abs(posterior_sums - 1.0))):.2e})"
    )
    print(f"  Shape correct: {shape_ok} {state_posteriors.shape}")

    # Gradient flow
    def hmm_loss(model: DifferentiableHMM, obs: dict) -> jnp.ndarray:
        """Scalar loss for gradient check."""
        out, _, _ = model.apply(obs, {}, None)
        return out["log_likelihood"]

    grad_info = check_gradient_flow(hmm_loss, hmm, data)
    print(f"  Gradient norm: {grad_info.gradient_norm:.6f}")
    print(f"  Gradient nonzero: {grad_info.gradient_nonzero}")

    # Throughput
    n_iters = 20 if quick else 100
    tp = measure_throughput(
        fn=lambda d: hmm.apply(d, {}, None),
        args=(data,),
        n_iterations=n_iters,
        warmup=3,
    )
    print(f"  Throughput: {tp['items_per_sec']:.1f} seqs/s")
    print(f"  Per item: {tp['per_item_ms']:.2f} ms")

    return HMMMetrics(
        log_likelihood_finite=ll_finite,
        posteriors_sum_to_one=posteriors_sum_ok,
        posteriors_shape_correct=shape_ok,
        gradient_norm=grad_info.gradient_norm,
        gradient_nonzero=grad_info.gradient_nonzero,
        throughput_items_per_sec=tp["items_per_sec"],
        throughput_per_item_ms=tp["per_item_ms"],
    )


def benchmark_nb_glm(quick: bool = False) -> NBGLMMetrics:
    """Benchmark the DifferentiableNBGLM operator.

    Args:
        quick: If True, use fewer features.

    Returns:
        NBGLMMetrics with correctness, gradient, and throughput data.
    """
    print("\n--- DifferentiableNBGLM ---")
    n_features = 20 if quick else 50
    n_covariates = 2

    # Create operator
    config = NBGLMConfig(
        n_features=n_features,
        n_covariates=n_covariates,
    )
    glm = DifferentiableNBGLM(config, rngs=nnx.Rngs(42))

    # Generate data
    data = generate_nb_glm_data(
        n_features=n_features,
        n_covariates=n_covariates,
    )

    # Run operator
    result, _, _ = glm.apply(data, {}, None)
    predicted_mean = result["predicted_mean"]
    dispersion = result["dispersion"]

    # Correctness checks
    mean_positive = bool(jnp.all(predicted_mean > 0))
    disp_positive = bool(jnp.all(dispersion > 0))
    shape_ok = (
        predicted_mean.shape == (n_features,)
        and dispersion.shape == (n_features,)
    )

    print(f"  Log-likelihood: {float(result['log_likelihood']):.4f}")
    print(
        f"  Predicted mean positive: {mean_positive} "
        f"(min: {float(jnp.min(predicted_mean)):.4e})"
    )
    print(
        f"  Dispersion positive: {disp_positive} "
        f"(min: {float(jnp.min(dispersion)):.4e})"
    )
    print(f"  Shape correct: {shape_ok}")

    # Gradient flow
    def glm_loss(
        model: DifferentiableNBGLM,
        obs: dict,
    ) -> jnp.ndarray:
        """Scalar loss for gradient check."""
        out, _, _ = model.apply(obs, {}, None)
        return out["log_likelihood"]

    grad_info = check_gradient_flow(glm_loss, glm, data)
    print(f"  Gradient norm: {grad_info.gradient_norm:.6f}")
    print(f"  Gradient nonzero: {grad_info.gradient_nonzero}")

    # Throughput
    n_iters = 50 if quick else 200
    tp = measure_throughput(
        fn=lambda d: glm.apply(d, {}, None),
        args=(data,),
        n_iterations=n_iters,
        warmup=5,
    )
    print(f"  Throughput: {tp['items_per_sec']:.1f} samples/s")
    print(f"  Per item: {tp['per_item_ms']:.2f} ms")

    return NBGLMMetrics(
        predicted_mean_positive=mean_positive,
        dispersion_positive=disp_positive,
        shape_correct=shape_ok,
        gradient_norm=grad_info.gradient_norm,
        gradient_nonzero=grad_info.gradient_nonzero,
        throughput_items_per_sec=tp["items_per_sec"],
        throughput_per_item_ms=tp["per_item_ms"],
    )


def benchmark_em(quick: bool = False) -> EMMetrics:
    """Benchmark the DifferentiableEMQuantifier operator.

    Args:
        quick: If True, use fewer reads and transcripts.

    Returns:
        EMMetrics with correctness, gradient, and throughput data.
    """
    print("\n--- DifferentiableEMQuantifier ---")
    n_reads = 50 if quick else 200
    n_transcripts = 10 if quick else 20

    # Create operator
    config = EMQuantifierConfig(
        n_transcripts=n_transcripts,
        n_iterations=10,
    )
    quantifier = DifferentiableEMQuantifier(
        config, rngs=nnx.Rngs(42)
    )

    # Generate data
    data = generate_em_data(
        n_reads=n_reads,
        n_transcripts=n_transcripts,
    )

    # Run operator
    result, _, _ = quantifier.apply(data, {}, None)
    abundances = result["abundances"]
    tpm = result["tpm"]

    # Correctness checks
    abundance_sum = float(jnp.sum(abundances))
    abundances_sum_ok = bool(jnp.isclose(abundance_sum, 1.0, atol=1e-3))
    tpm_positive = bool(jnp.all(tpm >= 0.0))
    shape_ok = (
        abundances.shape == (n_transcripts,)
        and tpm.shape == (n_transcripts,)
    )

    print(
        f"  Abundances sum: {abundance_sum:.6f} "
        f"(~1: {abundances_sum_ok})"
    )
    print(
        f"  TPM positive: {tpm_positive} "
        f"(min: {float(jnp.min(tpm)):.4e})"
    )
    print(
        f"  TPM sum: {float(jnp.sum(tpm)):.1f} "
        f"(expected ~1e6)"
    )
    print(f"  Shape correct: {shape_ok}")

    # Gradient flow
    def em_loss(
        model: DifferentiableEMQuantifier,
        obs: dict,
    ) -> jnp.ndarray:
        """Scalar loss for gradient check."""
        out, _, _ = model.apply(obs, {}, None)
        return jnp.sum(out["abundances"])

    grad_info = check_gradient_flow(em_loss, quantifier, data)
    print(f"  Gradient norm: {grad_info.gradient_norm:.6f}")
    print(f"  Gradient nonzero: {grad_info.gradient_nonzero}")

    # Throughput
    n_iters = 20 if quick else 100
    tp = measure_throughput(
        fn=lambda d: quantifier.apply(d, {}, None),
        args=(data,),
        n_iterations=n_iters,
        warmup=3,
    )
    print(f"  Throughput: {tp['items_per_sec']:.1f} iters/s")
    print(f"  Per item: {tp['per_item_ms']:.2f} ms")

    return EMMetrics(
        abundances_sum_to_one=abundances_sum_ok,
        tpm_positive=tpm_positive,
        shape_correct=shape_ok,
        gradient_norm=grad_info.gradient_norm,
        gradient_nonzero=grad_info.gradient_nonzero,
        throughput_items_per_sec=tp["items_per_sec"],
        throughput_per_item_ms=tp["per_item_ms"],
    )


# ------------------------------------------------------------------
# Top-level benchmark runner
# ------------------------------------------------------------------


def run_benchmark(
    quick: bool = False,
) -> StatisticalBenchmarkResult:
    """Run the complete statistical model benchmark.

    Args:
        quick: If True, use smaller problem sizes for faster
            execution.

    Returns:
        StatisticalBenchmarkResult with all operator metrics.
    """
    print("=" * 60)
    print("DiffBio Statistical Model Benchmark")
    if quick:
        print("  (quick mode -- reduced problem sizes)")
    print("=" * 60)

    hmm_metrics = benchmark_hmm(quick=quick)
    nb_glm_metrics = benchmark_nb_glm(quick=quick)
    em_metrics = benchmark_em(quick=quick)

    result = StatisticalBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        quick_mode=quick,
        hmm=hmm_metrics,
        nb_glm=nb_glm_metrics,
        em=em_metrics,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  HMM gradient nonzero:    {hmm_metrics.gradient_nonzero}")
    print(f"  NB-GLM gradient nonzero: {nb_glm_metrics.gradient_nonzero}")
    print(f"  EM gradient nonzero:     {em_metrics.gradient_nonzero}")
    print("=" * 60)

    return result


# ------------------------------------------------------------------
# Serialization and entry point
# ------------------------------------------------------------------


def save_results(
    result: StatisticalBenchmarkResult,
    output_dir: Path = Path("benchmarks/results"),
) -> None:
    """Save benchmark results to a timestamped JSON file.

    Args:
        result: The benchmark result to persist.
        output_dir: Root directory for result files.
    """
    stat_dir = output_dir / "statistical"
    stat_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        stat_dir / f"statistical_benchmark_{timestamp}.json"
    )

    with open(output_file, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    print(f"Results saved to: {output_file}")


def main() -> None:
    """Entry point for the statistical benchmark script."""
    quick = "--quick" in sys.argv
    result = run_benchmark(quick=quick)
    save_results(result)


if __name__ == "__main__":
    main()
