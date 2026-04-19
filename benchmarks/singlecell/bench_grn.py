#!/usr/bin/env python3
"""GRN inference benchmark: DifferentiableGRN on mESC ground truth.

Evaluates DiffBio's DifferentiableGRN operator against ChIP+Perturb
ground truth regulatory edges from the benGRN benchmark framework
(Stone & Sroy gold standards).

The GRN operator's attention weights are optimised via gradient
descent on an unsupervised expression reconstruction loss
(tf_activity @ grn_matrix should predict counts) with L1 sparsity.

Metrics: AUPRC, precision, recall, EPR (early precision rank).
Baselines: GENIE3, pySCENIC, GRNBoost2 published AUPRC values.

Usage:
    python benchmarks/singlecell/bench_grn.py
    python benchmarks/singlecell/bench_grn.py --quick
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.grn import GRN_BASELINES
from benchmarks._metrics.grn import evaluate_grn
from benchmarks._optimizers import create_benchmark_optimizer
from diffbio.operators.singlecell.grn_inference import (
    DifferentiableGRN,
    GRNInferenceConfig,
)
from diffbio.sources.bengrn_ground_truth import (
    BenGRNConfig,
    BenGRNSource,
)

logger = logging.getLogger(__name__)

_DATA_DIR = "/media/mahdi/ssd23/Works/benGRN/data/GroundTruth/stone_and_sroy"

_CONFIG = DiffBioBenchmarkConfig(
    name="singlecell/grn",
    domain="singlecell",
    quick_subsample=500,
    n_iterations_quick=5,
    n_iterations_full=20,
)


class GRNBenchmark(DiffBioBenchmark):
    """Evaluate DifferentiableGRN on mESC ChIP+Perturb ground truth."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = _DATA_DIR,
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Load GRN data, run operator, evaluate against GT."""
        max_genes = 500 if self.quick else 2000

        # 1. Load data
        logger.info("Loading benGRN mESC ground truth...")
        source = BenGRNSource(
            BenGRNConfig(
                data_dir=self.data_dir,
                species="mouse",
                expression_dataset="duren",
                ground_truth="chipunion_KDUnion_intersect",
                max_genes=max_genes,
            )
        )
        data = source.load()

        n_cells = data["n_cells"]
        n_genes = data["n_genes"]
        n_tfs = data["n_tfs"]
        n_edges = data["n_edges"]
        counts = data["counts"]
        tf_indices = data["tf_indices"]
        gt_matrix = data["ground_truth_matrix"]

        logger.info(
            "  %d cells, %d genes, %d TFs, %d GT edges",
            n_cells,
            n_genes,
            n_tfs,
            n_edges,
        )

        # 2. Run DifferentiableGRN
        logger.info("Running DifferentiableGRN...")
        op_config = GRNInferenceConfig(
            n_genes=n_genes,
            n_tfs=n_tfs,
            hidden_dim=32 if self.quick else 64,
            num_heads=4,
        )
        rngs = nnx.Rngs(42)
        operator = DifferentiableGRN(op_config, rngs=rngs)

        input_data = {
            "counts": counts,
            "tf_indices": jnp.array(tf_indices),
        }

        # Train GRN weights via unsupervised expression reconstruction:
        # predicted_expression = tf_activity @ grn_matrix should reconstruct counts.
        n_steps = 100 if self.quick else 300
        logger.info("Training GRN params (%d steps, unsupervised)...", n_steps)
        opt = nnx.Optimizer(
            operator,
            create_benchmark_optimizer(learning_rate=1e-3),
            wrt=nnx.Param,
        )

        # Normalise counts for reconstruction target
        log_counts = jnp.log1p(counts)
        target = log_counts / (jnp.max(log_counts) + 1e-8)

        @nnx.jit
        def _grn_step(
            model: DifferentiableGRN,
            optimizer: nnx.Optimizer,
            data: dict[str, jax.Array],
            tgt: jax.Array,
        ) -> jax.Array:
            def _loss(m: DifferentiableGRN) -> jax.Array:
                res, _, _ = m.apply(data, {}, None)
                # Reconstruct expression from TF activity and GRN matrix
                reconstructed = res["tf_activity"] @ res["grn_matrix"]
                recon_loss = jnp.mean((reconstructed - tgt) ** 2)
                # L1 sparsity on GRN for biological realism
                sparsity = 0.01 * jnp.mean(jnp.abs(res["grn_matrix"]))
                return recon_loss + sparsity

            loss, grads = nnx.value_and_grad(_loss)(model)
            optimizer.update(model, grads)
            return loss

        for step in range(n_steps):
            loss_val = _grn_step(operator, opt, input_data, target)
            if (step + 1) % 50 == 0:
                logger.info("  step %d/%d  loss=%.4f", step + 1, n_steps, float(loss_val))

        result, _, _ = operator.apply(input_data, {}, None)

        grn_matrix = result["grn_matrix"]
        logger.info("  GRN shape: %s", grn_matrix.shape)

        # 3. Evaluate against ground truth
        logger.info("Evaluating against ChIP+Perturb ground truth...")
        pred_full = np.zeros((n_genes, n_genes), dtype=np.float32)
        grn_np = np.asarray(jnp.abs(grn_matrix))
        for i, tf_idx in enumerate(tf_indices):
            if i < grn_np.shape[0]:
                pred_full[tf_idx] = grn_np[i]

        grn_metrics = evaluate_grn(pred_full, gt_matrix)

        for key, value in sorted(grn_metrics.items()):
            logger.info("  %s: %.6f", key, value)

        # Loss function for gradient check
        def loss_fn(model: DifferentiableGRN, d: dict[str, Any]) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["grn_matrix"])

        return {
            "metrics": grn_metrics,
            "operator": operator,
            "input_data": input_data,
            "loss_fn": loss_fn,
            "n_items": n_cells,
            "iterate_fn": lambda: operator.apply(input_data, {}, None),
            "baselines": GRN_BASELINES,
            "dataset_info": {
                "n_cells": n_cells,
                "n_genes": n_genes,
                "n_tfs": n_tfs,
                "n_gt_edges": n_edges,
            },
            "operator_config": {
                "n_genes": n_genes,
                "n_tfs": n_tfs,
                "hidden_dim": op_config.hidden_dim,
            },
            "operator_name": "DifferentiableGRN",
            "dataset_name": "bengrn_mesc",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        GRNBenchmark,
        _CONFIG,
        data_dir=_DATA_DIR,
    )


if __name__ == "__main__":
    main()
