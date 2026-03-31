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

import time
from typing import Any

import jax.numpy as jnp
from flax import nnx

from diffbio.operators.singlecell.ot_trajectory import (
    DifferentiableOTTrajectory,
    OTTrajectoryConfig,
)
from diffbio.operators.singlecell.trajectory import (
    DifferentiableFateProbability,
    FateProbabilityConfig,
)


# -------------------------------------------------------------------
# Individual operator benchmarks
# -------------------------------------------------------------------


def _test_fate_probability(
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

    print(f"    fate_probabilities shape: {fate.shape} (ok={shape_ok})")
    print(
        f"    row sums [min, max]: [{float(jnp.min(row_sums)):.4f},"
        f" {float(jnp.max(row_sums)):.4f}] (approx_1={row_sum_ok})"
    )
    print(f"    macrostates shape: {macrostates.shape}")
    print(f"    Time: {elapsed_ms:.2f}ms")

    return {
        "shape_ok": shape_ok,
        "row_sum_ok": row_sum_ok,
        "time_ms": elapsed_ms,
    }


def _test_ot_trajectory(
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
    print(
        f"    row sums*n1 [min, max]: "
        f"[{float(jnp.min(row_sums)):.4f},"
        f" {float(jnp.max(row_sums)):.4f}]"
        f" (approx_1={row_approx_one})"
    )
    print(f"    growth_rates shape: {growth.shape}")
    print(f"    interpolated_counts shape: {interpolated.shape} (ok={interp_shape_ok})")
    print(f"    Time: {elapsed_ms:.2f}ms")

    return {
        "transport_nonneg": nonneg,
        "row_sum_approx_one": row_approx_one,
        "interpolated_shape_ok": interp_shape_ok,
        "time_ms": elapsed_ms,
    }
