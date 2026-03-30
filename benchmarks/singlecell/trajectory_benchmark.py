#!/usr/bin/env python3
"""Trajectory Inference Benchmark for DiffBio.

This benchmark evaluates DiffBio's trajectory inference operators:
- DifferentiablePseudotime (diffusion-map pseudotime ordering)
- DifferentiableFateProbability (absorption-based fate estimation)
- DifferentiableVelocity (RNA velocity via Neural ODEs)
- DifferentiableOTTrajectory (Waddington-OT trajectory inference)

Metrics:
- Pseudotime output shape and range
- Fate probability row-sum validity
- Velocity output shape correctness
- OT transport plan non-negativity and row-sum normalization
- Gradient flow through pseudotime and velocity
- Throughput

Usage:
    python benchmarks/singlecell/trajectory_benchmark.py
    python benchmarks/singlecell/trajectory_benchmark.py --quick
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks._common import (
    check_gradient_flow,
    generate_synthetic_expression,
    measure_throughput,
)
from diffbio.operators.singlecell.ot_trajectory import (
    DifferentiableOTTrajectory,
    OTTrajectoryConfig,
)
from diffbio.operators.singlecell.trajectory import (
    DifferentiableFateProbability,
    DifferentiablePseudotime,
    FateProbabilityConfig,
    PseudotimeConfig,
)
from diffbio.operators.singlecell.velocity import (
    DifferentiableVelocity,
    VelocityConfig,
)


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryBenchmarkResult:
    """Results from trajectory inference benchmark.

    Attributes:
        timestamp: ISO-format timestamp of the benchmark run.
        n_cells: Number of cells in the synthetic dataset.
        n_genes: Number of genes in the synthetic dataset.
        pseudotime_shape_ok: Whether pseudotime output shape matches.
        pseudotime_range_ok: Whether pseudotime values are non-negative.
        pseudotime_root_zero: Whether root cell has pseudotime zero.
        pseudotime_gradient_norm: Gradient norm through pseudotime op.
        pseudotime_gradient_nonzero: Whether gradients are non-zero.
        pseudotime_time_ms: Pseudotime computation time in ms.
        fate_shape_ok: Whether fate probabilities shape matches.
        fate_row_sum_ok: Whether fate probabilities sum to ~1/row.
        fate_time_ms: Fate probability computation time in ms.
        velocity_shape_ok: Whether velocity output shape matches input.
        velocity_latent_time_range_ok: Whether latent_time is in [0,1].
        velocity_gradient_norm: Gradient norm through velocity op.
        velocity_gradient_nonzero: Whether gradients are non-zero.
        velocity_time_ms: Velocity computation time in ms.
        ot_transport_nonneg: Whether transport plan is non-negative.
        ot_row_sum_approx_one: Whether transport rows approx. sum to 1.
        ot_interpolated_shape_ok: Whether interpolated shape matches.
        ot_time_ms: OT trajectory computation time in ms.
        throughput_pseudotime: Pseudotime items/sec.
        throughput_velocity: Velocity items/sec.
    """

    timestamp: str
    n_cells: int
    n_genes: int
    # Pseudotime
    pseudotime_shape_ok: bool
    pseudotime_range_ok: bool
    pseudotime_root_zero: bool
    pseudotime_gradient_norm: float
    pseudotime_gradient_nonzero: bool
    pseudotime_time_ms: float
    # Fate probability
    fate_shape_ok: bool
    fate_row_sum_ok: bool
    fate_time_ms: float
    # Velocity
    velocity_shape_ok: bool
    velocity_latent_time_range_ok: bool
    velocity_gradient_norm: float
    velocity_gradient_nonzero: bool
    velocity_time_ms: float
    # OT trajectory
    ot_transport_nonneg: bool
    ot_row_sum_approx_one: bool
    ot_interpolated_shape_ok: bool
    ot_time_ms: float
    # Throughput
    throughput_pseudotime: float
    throughput_velocity: float


# -------------------------------------------------------------------
# Individual operator benchmarks
# -------------------------------------------------------------------


def test_pseudotime(
    embeddings: jnp.ndarray,
    n_cells: int,
) -> dict[str, Any]:
    """Benchmark DifferentiablePseudotime operator.

    Args:
        embeddings: Cell embeddings of shape ``(n_cells, n_features)``.
        n_cells: Number of cells.

    Returns:
        Dictionary with pseudotime benchmark metrics.
    """
    print("\n  Testing DifferentiablePseudotime...")

    config = PseudotimeConfig(
        n_neighbors=min(15, n_cells - 1),
        n_diffusion_components=min(10, n_cells - 2),
        root_cell_index=0,
    )
    op = DifferentiablePseudotime(config, rngs=nnx.Rngs(42))

    input_data = {"embeddings": embeddings}

    # Forward pass + timing
    start = time.perf_counter()
    result, _, _ = op.apply(input_data, {}, None)
    elapsed_ms = (time.perf_counter() - start) * 1000

    pseudotime = result["pseudotime"]
    dc = result["diffusion_components"]
    transition = result["transition_matrix"]

    # Shape checks
    shape_ok = pseudotime.shape == (n_cells,)
    range_ok = bool(jnp.all(pseudotime >= -1e-6))
    root_zero = bool(jnp.abs(pseudotime[0]) < 1e-6)

    print(f"    pseudotime shape: {pseudotime.shape} (ok={shape_ok})")
    print(f"    range [min, max]: [{float(jnp.min(pseudotime)):.4f},"
          f" {float(jnp.max(pseudotime)):.4f}] (non-neg={range_ok})")
    print(f"    root_cell pseudotime: {float(pseudotime[0]):.6f}"
          f" (zero={root_zero})")
    print(f"    diffusion_components: {dc.shape}")
    print(f"    transition_matrix: {transition.shape}")
    print(f"    Time: {elapsed_ms:.2f}ms")

    # Gradient flow (differentiate through embeddings input)
    def loss_fn(data: dict) -> jnp.ndarray:
        """Loss function summing pseudotime for gradient check."""
        res, _, _ = op.apply(data, {}, None)
        return jnp.sum(res["pseudotime"])

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(input_data)
    grad_norm = float(jnp.linalg.norm(grads["embeddings"]))
    grad_nonzero = grad_norm > 1e-8

    print(f"    Gradient norm: {grad_norm:.6f}"
          f" (nonzero={grad_nonzero})")

    # Throughput
    tp = measure_throughput(
        fn=lambda d: op.apply(d, {}, None),
        args=(input_data,),
        n_iterations=20,
        warmup=3,
    )
    print(f"    Throughput: {tp['items_per_sec']:.1f} calls/sec")

    return {
        "shape_ok": shape_ok,
        "range_ok": range_ok,
        "root_zero": root_zero,
        "gradient_norm": grad_norm,
        "gradient_nonzero": grad_nonzero,
        "time_ms": elapsed_ms,
        "throughput": tp["items_per_sec"],
    }


def test_fate_probability(
    transition_matrix: jnp.ndarray,
    n_cells: int,
    n_terminal: int = 2,
) -> dict[str, Any]:
    """Benchmark DifferentiableFateProbability operator.

    Args:
        transition_matrix: Markov transition matrix from pseudotime.
        n_cells: Number of cells.
        n_terminal: Number of terminal states to designate.

    Returns:
        Dictionary with fate probability benchmark metrics.
    """
    print("\n  Testing DifferentiableFateProbability...")

    config = FateProbabilityConfig(n_macrostates=n_terminal)
    op = DifferentiableFateProbability(config, rngs=nnx.Rngs(42))

    # Designate the last n_terminal cells as terminal states
    terminal_states = jnp.arange(n_cells - n_terminal, n_cells)
    input_data = {
        "transition_matrix": transition_matrix,
        "terminal_states": terminal_states,
    }

    # Forward pass + timing
    start = time.perf_counter()
    result, _, _ = op.apply(input_data, {}, None)
    elapsed_ms = (time.perf_counter() - start) * 1000

    fate = result["fate_probabilities"]
    macrostates = result["macrostates"]

    # Shape check
    shape_ok = fate.shape == (n_cells, n_terminal)

    # Row sums should be close to 1 for each cell
    row_sums = jnp.sum(fate, axis=1)
    row_sum_ok = bool(jnp.allclose(row_sums, 1.0, atol=0.05))

    print(f"    fate_probabilities shape: {fate.shape}"
          f" (ok={shape_ok})")
    print(f"    row sums [min, max]: [{float(jnp.min(row_sums)):.4f},"
          f" {float(jnp.max(row_sums)):.4f}] (approx_1={row_sum_ok})")
    print(f"    macrostates shape: {macrostates.shape}")
    print(f"    Time: {elapsed_ms:.2f}ms")

    return {
        "shape_ok": shape_ok,
        "row_sum_ok": row_sum_ok,
        "time_ms": elapsed_ms,
    }


def test_velocity(
    spliced: jnp.ndarray,
    unspliced: jnp.ndarray,
    n_cells: int,
    n_genes: int,
) -> dict[str, Any]:
    """Benchmark DifferentiableVelocity operator.

    Args:
        spliced: Spliced mRNA counts ``(n_cells, n_genes)``.
        unspliced: Unspliced mRNA counts ``(n_cells, n_genes)``.
        n_cells: Number of cells.
        n_genes: Number of genes.

    Returns:
        Dictionary with velocity benchmark metrics.
    """
    print("\n  Testing DifferentiableVelocity...")

    config = VelocityConfig(n_genes=n_genes, hidden_dim=32)
    op = DifferentiableVelocity(config, rngs=nnx.Rngs(42))

    input_data = {"spliced": spliced, "unspliced": unspliced}

    # Forward pass + timing
    start = time.perf_counter()
    result, _, _ = op.apply(input_data, {}, None)
    elapsed_ms = (time.perf_counter() - start) * 1000

    velocity = result["velocity"]
    latent_time = result["latent_time"]
    alpha = result["alpha"]
    beta = result["beta"]
    gamma = result["gamma"]

    # Shape checks
    shape_ok = velocity.shape == (n_cells, n_genes)
    lt_range_ok = bool(
        jnp.all(latent_time >= 0.0) and jnp.all(latent_time <= 1.0)
    )

    print(f"    velocity shape: {velocity.shape} (ok={shape_ok})")
    print(f"    latent_time range: [{float(jnp.min(latent_time)):.4f},"
          f" {float(jnp.max(latent_time)):.4f}]"
          f" (in_0_1={lt_range_ok})")
    print(f"    alpha shape: {alpha.shape}")
    print(f"    beta shape: {beta.shape}")
    print(f"    gamma shape: {gamma.shape}")
    print(f"    Time: {elapsed_ms:.2f}ms")

    # Gradient flow through model parameters
    def loss_fn(
        model: DifferentiableVelocity,
        data: dict,
    ) -> jnp.ndarray:
        """Loss for gradient check on velocity operator."""
        res, _, _ = model.apply(data, {}, None)
        return jnp.sum(res["velocity"])

    grad_info = check_gradient_flow(loss_fn, op, input_data)
    grad_norm = grad_info.gradient_norm
    grad_nonzero = grad_info.gradient_nonzero

    print(f"    Gradient norm: {grad_norm:.6f}"
          f" (nonzero={grad_nonzero})")

    # Throughput
    tp = measure_throughput(
        fn=lambda d: op.apply(d, {}, None),
        args=(input_data,),
        n_iterations=20,
        warmup=3,
    )
    print(f"    Throughput: {tp['items_per_sec']:.1f} calls/sec")

    return {
        "shape_ok": shape_ok,
        "latent_time_range_ok": lt_range_ok,
        "gradient_norm": float(grad_norm),
        "gradient_nonzero": bool(grad_nonzero),
        "time_ms": elapsed_ms,
        "throughput": tp["items_per_sec"],
    }


def test_ot_trajectory(
    counts_t1: jnp.ndarray,
    counts_t2: jnp.ndarray,
    n1: int,
    n2: int,
    n_genes: int,
) -> dict[str, Any]:
    """Benchmark DifferentiableOTTrajectory operator.

    Args:
        counts_t1: Expression at timepoint 1 ``(n1, n_genes)``.
        counts_t2: Expression at timepoint 2 ``(n2, n_genes)``.
        n1: Number of cells at timepoint 1.
        n2: Number of cells at timepoint 2.
        n_genes: Number of genes.

    Returns:
        Dictionary with OT trajectory benchmark metrics.
    """
    print("\n  Testing DifferentiableOTTrajectory...")

    config = OTTrajectoryConfig(
        n_genes=n_genes,
        sinkhorn_epsilon=0.1,
        sinkhorn_iters=50,
    )
    op = DifferentiableOTTrajectory(config, rngs=nnx.Rngs(42))

    input_data = {"counts_t1": counts_t1, "counts_t2": counts_t2}

    # Forward pass + timing
    start = time.perf_counter()
    result, _, _ = op.apply(input_data, {}, None)
    elapsed_ms = (time.perf_counter() - start) * 1000

    transport = result["transport_plan"]
    growth = result["growth_rates"]
    interpolated = result["interpolated_counts"]

    # Non-negativity of transport plan
    nonneg = bool(jnp.all(transport >= -1e-8))

    # Row sums: Sinkhorn uses uniform marginals so rows should
    # approximately sum to 1/n1 (the marginal for each source cell).
    # Normalising by n1 gives row sums near 1.
    row_sums = jnp.sum(transport, axis=1) * n1
    row_approx_one = bool(jnp.allclose(row_sums, 1.0, atol=0.15))

    # Interpolated shape
    interp_shape_ok = interpolated.shape == (n1, n_genes)

    print(f"    transport_plan shape: {transport.shape}")
    print(f"    non-negative: {nonneg}")
    print(f"    row sums*n1 [min, max]: "
          f"[{float(jnp.min(row_sums)):.4f},"
          f" {float(jnp.max(row_sums)):.4f}]"
          f" (approx_1={row_approx_one})")
    print(f"    growth_rates shape: {growth.shape}")
    print(f"    interpolated_counts shape: {interpolated.shape}"
          f" (ok={interp_shape_ok})")
    print(f"    Time: {elapsed_ms:.2f}ms")

    return {
        "transport_nonneg": nonneg,
        "row_sum_approx_one": row_approx_one,
        "interpolated_shape_ok": interp_shape_ok,
        "time_ms": elapsed_ms,
    }


# -------------------------------------------------------------------
# Main benchmark
# -------------------------------------------------------------------


def run_benchmark(
    quick: bool = False,
) -> TrajectoryBenchmarkResult:
    """Run the complete trajectory inference benchmark.

    Args:
        quick: If True, use smaller dataset for faster execution.

    Returns:
        TrajectoryBenchmarkResult with all metrics.
    """
    print("=" * 60)
    print("DiffBio Trajectory Inference Benchmark")
    print("=" * 60)

    # Dataset sizes
    if quick:
        n_cells, n_genes, n_types = 50, 20, 3
    else:
        n_cells, n_genes, n_types = 200, 50, 3

    # Generate synthetic expression data
    print(f"\nGenerating synthetic data"
          f" (n_cells={n_cells}, n_genes={n_genes})...")
    synth = generate_synthetic_expression(
        n_cells=n_cells,
        n_genes=n_genes,
        n_types=n_types,
    )
    embeddings = synth["embeddings"]
    counts = synth["counts"]
    print(f"  embeddings: {embeddings.shape}")
    print(f"  counts: {counts.shape}")

    # -- Pseudotime --
    pt_metrics = test_pseudotime(embeddings, n_cells)

    # -- Fate probability (chained from pseudotime) --
    # Re-run pseudotime to get transition_matrix for fate probability
    pt_config = PseudotimeConfig(
        n_neighbors=min(15, n_cells - 1),
        n_diffusion_components=min(10, n_cells - 2),
    )
    pt_op = DifferentiablePseudotime(pt_config, rngs=nnx.Rngs(42))
    pt_result, _, _ = pt_op.apply({"embeddings": embeddings}, {}, None)
    transition_matrix = pt_result["transition_matrix"]

    n_terminal = 2
    fate_metrics = test_fate_probability(
        transition_matrix, n_cells, n_terminal=n_terminal
    )

    # -- Velocity --
    # Generate spliced / unspliced from counts with added noise
    key = jax.random.key(99)
    k1, k2 = jax.random.split(key)
    spliced = counts + jax.random.normal(k1, counts.shape) * 0.1
    spliced = jnp.maximum(spliced, 0.0)
    unspliced = counts * 0.3 + jax.random.normal(k2, counts.shape) * 0.05
    unspliced = jnp.maximum(unspliced, 0.0)

    vel_metrics = test_velocity(spliced, unspliced, n_cells, n_genes)

    # -- OT trajectory --
    # Split counts into two timepoints (first half / second half)
    mid = n_cells // 2
    counts_t1 = counts[:mid]
    counts_t2 = counts[mid:]
    n1, n2 = counts_t1.shape[0], counts_t2.shape[0]

    ot_metrics = test_ot_trajectory(
        counts_t1, counts_t2, n1, n2, n_genes
    )

    # -- Compile results --
    result = TrajectoryBenchmarkResult(
        timestamp=datetime.now().isoformat(),
        n_cells=n_cells,
        n_genes=n_genes,
        # Pseudotime
        pseudotime_shape_ok=pt_metrics["shape_ok"],
        pseudotime_range_ok=pt_metrics["range_ok"],
        pseudotime_root_zero=pt_metrics["root_zero"],
        pseudotime_gradient_norm=pt_metrics["gradient_norm"],
        pseudotime_gradient_nonzero=pt_metrics["gradient_nonzero"],
        pseudotime_time_ms=pt_metrics["time_ms"],
        # Fate
        fate_shape_ok=fate_metrics["shape_ok"],
        fate_row_sum_ok=fate_metrics["row_sum_ok"],
        fate_time_ms=fate_metrics["time_ms"],
        # Velocity
        velocity_shape_ok=vel_metrics["shape_ok"],
        velocity_latent_time_range_ok=vel_metrics["latent_time_range_ok"],
        velocity_gradient_norm=vel_metrics["gradient_norm"],
        velocity_gradient_nonzero=vel_metrics["gradient_nonzero"],
        velocity_time_ms=vel_metrics["time_ms"],
        # OT
        ot_transport_nonneg=ot_metrics["transport_nonneg"],
        ot_row_sum_approx_one=ot_metrics["row_sum_approx_one"],
        ot_interpolated_shape_ok=ot_metrics["interpolated_shape_ok"],
        ot_time_ms=ot_metrics["time_ms"],
        # Throughput
        throughput_pseudotime=pt_metrics["throughput"],
        throughput_velocity=vel_metrics["throughput"],
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Pseudotime: shape_ok={pt_metrics['shape_ok']},"
          f" grad_nonzero={pt_metrics['gradient_nonzero']}")
    print(f"  Fate:       shape_ok={fate_metrics['shape_ok']},"
          f" row_sum_ok={fate_metrics['row_sum_ok']}")
    print(f"  Velocity:   shape_ok={vel_metrics['shape_ok']},"
          f" grad_nonzero={vel_metrics['gradient_nonzero']}")
    print(f"  OT:         nonneg={ot_metrics['transport_nonneg']},"
          f" interp_ok={ot_metrics['interpolated_shape_ok']}")
    print("=" * 60)

    return result


def save_results(
    result: TrajectoryBenchmarkResult,
    output_dir: Path,
) -> None:
    """Save benchmark results to JSON.

    Args:
        result: Benchmark result dataclass to serialize.
        output_dir: Directory to write the JSON file into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        output_dir / f"trajectory_benchmark_{timestamp}.json"
    )

    with open(output_file, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"Results saved to: {output_file}")


def main() -> None:
    """Main entry point for the trajectory benchmark."""
    quick = "--quick" in sys.argv
    result = run_benchmark(quick=quick)
    save_results(result, Path("benchmarks/results"))


if __name__ == "__main__":
    main()
