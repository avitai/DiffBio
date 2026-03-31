#!/usr/bin/env python3
"""Trajectory benchmark: Pseudotime + Velocity on pancreas.

Evaluates DiffBio's DifferentiablePseudotime on PCA embeddings and
DifferentiableVelocity on spliced/unspliced counts from the scVelo
pancreas endocrinogenesis dataset (3,696 cells, 27,998 genes).

Velocity parameters (kinetics rates and time encoder) are optimised
via gradient descent on VelocityConsistencyLoss (unsupervised).

Usage:
    python benchmarks/singlecell/bench_trajectory.py
    python benchmarks/singlecell/bench_trajectory.py --quick
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.trajectory import TRAJECTORY_BASELINES
from diffbio.losses.singlecell_losses import VelocityConsistencyLoss
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

_CONFIG = DiffBioBenchmarkConfig(
    name="singlecell/trajectory",
    domain="singlecell",
    quick_subsample=500,
    n_iterations_quick=5,
    n_iterations_full=20,
)


class TrajectoryBenchmark(DiffBioBenchmark):
    """Evaluate trajectory inference on the pancreas dataset."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Data/scvelo",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Run pseudotime and velocity, compute metrics."""
        subsample = self.config.quick_subsample if self.quick else None

        # 1. Load dataset
        logger.info("Loading pancreas dataset...")
        source = PancreasSource(PancreasConfig(data_dir=self.data_dir, subsample=subsample))
        data = source.load()
        n_cells = data["n_cells"]
        n_genes = data["n_genes"]
        embeddings = data["embeddings"]
        spliced = data["spliced"]
        unspliced = data["unspliced"]

        logger.info("  %d cells, %d genes", n_cells, n_genes)

        rngs = nnx.Rngs(42)
        quality: dict[str, float] = {}

        # 2. Pseudotime inference
        logger.info("Running DifferentiablePseudotime...")
        pt_config = PseudotimeConfig(
            n_neighbors=min(15, n_cells - 1),
            n_diffusion_components=min(10, n_cells - 1),
        )
        pt_op = DifferentiablePseudotime(pt_config, rngs=rngs)
        pt_input = {"embeddings": embeddings}

        pt_result, _, _ = pt_op.apply(pt_input, {}, None)

        pseudotime = pt_result["pseudotime"]
        pt_range = float(jnp.max(pseudotime) - jnp.min(pseudotime))
        pt_finite = bool(jnp.all(jnp.isfinite(pseudotime)))

        logger.info("  Range: %.4f, Finite: %s", pt_range, pt_finite)

        quality["pseudotime_range"] = pt_range
        quality["pseudotime_finite"] = 1.0 if pt_finite else 0.0

        # 3. Velocity inference (subset of genes for speed)
        logger.info("Running DifferentiableVelocity...")
        n_vel_genes = min(200, n_genes) if self.quick else min(2000, n_genes)
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

        # Train velocity parameters (unsupervised consistency loss)
        n_steps = 100 if self.quick else 300
        logger.info("Training velocity params (%d steps)...", n_steps)
        vel_loss = VelocityConsistencyLoss(rngs=rngs)
        vel_opt = nnx.Optimizer(vel_op, optax.adam(1e-3), wrt=nnx.Param)

        @nnx.jit
        def _vel_step(
            model: DifferentiableVelocity,
            optimizer: nnx.Optimizer,
            data: dict[str, jax.Array],
        ) -> jax.Array:
            def _loss(m: DifferentiableVelocity) -> jax.Array:
                res, _, _ = m.apply(data, {}, None)
                return vel_loss(
                    data["spliced"],
                    res["velocity"],
                    res["projected_spliced"],
                )

            loss, grads = nnx.value_and_grad(_loss)(model)
            optimizer.update(model, grads)
            return loss

        for step in range(n_steps):
            loss_val = _vel_step(vel_op, vel_opt, vel_input)
            if (step + 1) % 50 == 0:
                logger.info("  step %d/%d  loss=%.4f", step + 1, n_steps, float(loss_val))

        vel_result, _, _ = vel_op.apply(vel_input, {}, None)

        velocity = vel_result["velocity"]
        vel_shape_ok = velocity.shape == (n_cells, n_vel_genes)
        vel_finite = bool(jnp.all(jnp.isfinite(velocity)))

        logger.info(
            "  Shape correct: %s, Finite: %s",
            vel_shape_ok,
            vel_finite,
        )

        quality["velocity_shape_correct"] = 1.0 if vel_shape_ok else 0.0
        quality["velocity_finite"] = 1.0 if vel_finite else 0.0

        # Loss function for gradient check (velocity has params)
        def loss_fn(model: DifferentiableVelocity, d: dict[str, Any]) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["velocity"])

        return {
            "metrics": quality,
            "operator": vel_op,
            "input_data": vel_input,
            "loss_fn": loss_fn,
            "n_items": n_cells,
            "iterate_fn": lambda: pt_op.apply(pt_input, {}, None),
            "baselines": TRAJECTORY_BASELINES,
            "dataset_info": {
                "n_cells": n_cells,
                "n_genes": n_genes,
            },
            "operator_config": {
                "n_neighbors": pt_config.n_neighbors,
                "n_vel_genes": n_vel_genes,
            },
            "operator_name": ("DifferentiablePseudotime,DifferentiableVelocity"),
            "dataset_name": "pancreas",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        TrajectoryBenchmark,
        _CONFIG,
        data_dir="/media/mahdi/ssd23/Data/scvelo",
    )


if __name__ == "__main__":
    main()
