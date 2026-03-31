#!/usr/bin/env python3
"""Trajectory benchmark: Pseudotime + Velocity on pancreas.

Evaluates DiffBio's DifferentiablePseudotime on PCA embeddings and
DifferentiableVelocity on spliced/unspliced counts from the scVelo
pancreas endocrinogenesis dataset (3,696 cells, 27,998 genes).

Usage:
    python benchmarks/singlecell/bench_trajectory.py
    python benchmarks/singlecell/bench_trajectory.py --quick
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult
from calibrax.profiling.timing import TimingCollector
from flax import nnx

from benchmarks._baselines.trajectory import TRAJECTORY_BASELINES
from benchmarks._gradient import check_gradient_flow
from diffbio.operators.singlecell.trajectory import (
    DifferentiablePseudotime,
    PseudotimeConfig,
)
from diffbio.operators.singlecell.velocity import (
    DifferentiableVelocity,
    VelocityConfig,
)
from diffbio.sources.pancreas import PancreasConfig, PancreasSource

logger = logging.getLogger(__name__)

_BASELINES = TRAJECTORY_BASELINES


class TrajectoryBenchmark:
    """Evaluate trajectory inference on the pancreas dataset.

    Args:
        quick: If True, subsample to 500 cells.
        data_dir: Directory containing the scVelo pancreas h5ad.
    """

    def __init__(
        self,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Data/scvelo",
    ) -> None:
        self.quick = quick
        self.data_dir = data_dir

    def run(self) -> BenchmarkResult:
        """Execute the trajectory benchmark."""
        subsample = 500 if self.quick else None
        n_iters = 5 if self.quick else 20

        # 1. Load dataset
        print("Loading pancreas dataset...")
        source = PancreasSource(
            PancreasConfig(
                data_dir=self.data_dir, subsample=subsample
            )
        )
        data = source.load()
        n_cells = data["n_cells"]
        n_genes = data["n_genes"]
        embeddings = data["embeddings"]
        spliced = data["spliced"]
        unspliced = data["unspliced"]

        print(f"  {n_cells} cells, {n_genes} genes")

        rngs = nnx.Rngs(42)
        metrics: dict[str, Metric] = {}

        # 2. Pseudotime inference
        print("Running DifferentiablePseudotime...")
        pt_config = PseudotimeConfig(
            n_neighbors=min(15, n_cells - 1),
            n_diffusion_components=min(10, n_cells - 1),
        )
        pt_op = DifferentiablePseudotime(pt_config, rngs=rngs)
        pt_input = {"embeddings": embeddings}

        start = time.perf_counter()
        pt_result, _, _ = pt_op.apply(pt_input, {}, None)
        pt_time = time.perf_counter() - start

        pseudotime = pt_result["pseudotime"]
        pt_range = float(jnp.max(pseudotime) - jnp.min(pseudotime))
        pt_finite = bool(jnp.all(jnp.isfinite(pseudotime)))

        print(f"  Range: {pt_range:.4f}, Finite: {pt_finite}")
        print(f"  Time: {pt_time:.2f}s")

        metrics["pseudotime_range"] = Metric(value=pt_range)
        metrics["pseudotime_finite"] = Metric(
            value=1.0 if pt_finite else 0.0
        )

        # 3. Velocity inference (on subset of genes for speed)
        print("Running DifferentiableVelocity...")
        n_vel_genes = min(200, n_genes) if self.quick else min(
            2000, n_genes
        )
        spliced_sub = spliced[:, :n_vel_genes]
        unspliced_sub = unspliced[:, :n_vel_genes]

        vel_config = VelocityConfig(
            n_genes=n_vel_genes,
            hidden_dim=32 if self.quick else 64,
        )
        vel_op = DifferentiableVelocity(vel_config, rngs=rngs)
        vel_input = {
            "spliced": spliced_sub,
            "unspliced": unspliced_sub,
        }

        start = time.perf_counter()
        vel_result, _, _ = vel_op.apply(vel_input, {}, None)
        vel_time = time.perf_counter() - start

        velocity = vel_result["velocity"]
        vel_shape_ok = velocity.shape == (n_cells, n_vel_genes)
        vel_finite = bool(jnp.all(jnp.isfinite(velocity)))

        print(f"  Shape correct: {vel_shape_ok}")
        print(f"  Finite: {vel_finite}, Time: {vel_time:.2f}s")

        metrics["velocity_shape_correct"] = Metric(
            value=1.0 if vel_shape_ok else 0.0
        )
        metrics["velocity_finite"] = Metric(
            value=1.0 if vel_finite else 0.0
        )

        # 4. Gradient flow (velocity has learnable params)
        print("Checking gradient flow...")

        def loss_fn(
            model: DifferentiableVelocity, d: dict[str, Any]
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["velocity"])

        grad = check_gradient_flow(loss_fn, vel_op, vel_input)
        print(f"  Gradient norm: {grad.gradient_norm:.4f}")

        metrics["gradient_norm"] = Metric(value=grad.gradient_norm)
        metrics["gradient_nonzero"] = Metric(
            value=1.0 if grad.gradient_nonzero else 0.0
        )

        # 5. Throughput
        print("Measuring throughput...")
        collector = TimingCollector(warmup_iterations=2)
        timing = collector.measure_iteration(
            iterator=iter(range(n_iters)),
            num_batches=n_iters,
            process_fn=lambda _: pt_op.apply(pt_input, {}, None),
            count_fn=lambda _: n_cells,
        )
        cells_per_sec = timing.num_elements / timing.wall_clock_sec
        metrics["cells_per_sec"] = Metric(value=cells_per_sec)
        print(f"  {cells_per_sec:.0f} cells/sec")

        # 6. Comparison
        print("\nComparison (pseudotime):")
        print(f"  DiffBio Pseudotime: range={pt_range:.4f}")
        for name, point in _BASELINES.items():
            sp = point.metrics.get(
                "spearman_pancreas", Metric(value=0)
            ).value
            print(f"  {name}: Spearman={sp}")

        return BenchmarkResult(
            name="singlecell/trajectory",
            domain="diffbio_benchmarks",
            tags={
                "operator": "DifferentiablePseudotime,DifferentiableVelocity",
                "dataset": "pancreas",
                "framework": "diffbio",
            },
            timing=timing,
            metrics=metrics,
            config={
                "n_neighbors": pt_config.n_neighbors,
                "n_vel_genes": n_vel_genes,
                "quick": self.quick,
            },
            metadata={
                "dataset_info": {
                    "n_cells": n_cells,
                    "n_genes": n_genes,
                },
                "baselines": {k: p.to_dict() for k, p in _BASELINES.items()},
            },
        )


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("DiffBio Benchmark: Trajectory Inference")
    print("=" * 60)

    bench = TrajectoryBenchmark(quick=quick)
    result = bench.run()

    from pathlib import Path  # noqa: PLC0415

    out = Path("benchmarks/results/singlecell")
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "trajectory.json")
    print(f"\nSaved to: {out / 'trajectory.json'}")


if __name__ == "__main__":
    main()
