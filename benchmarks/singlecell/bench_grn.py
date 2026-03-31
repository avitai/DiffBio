#!/usr/bin/env python3
"""GRN inference benchmark: DifferentiableGRN on mESC ground truth.

Evaluates DiffBio's DifferentiableGRN operator against ChIP+Perturb
ground truth regulatory edges from the benGRN benchmark framework
(Stone & Sroy gold standards).

Metrics: AUPRC, precision, recall, EPR (early precision rank).
Baselines: GENIE3, pySCENIC, GRNBoost2 published AUPRC values.

Usage:
    python benchmarks/singlecell/bench_grn.py
    python benchmarks/singlecell/bench_grn.py --quick
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.grn import GRN_BASELINES
from benchmarks._metrics.grn import evaluate_grn
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
