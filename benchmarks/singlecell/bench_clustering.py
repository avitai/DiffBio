#!/usr/bin/env python3
"""Clustering benchmark: SoftKMeansClustering on immune_human.

Evaluates DiffBio's SoftKMeansClustering against ground-truth cell
type labels on the scib immune human dataset using ARI, NMI, and
silhouette score.

The benchmark showcases DiffBio's differentiable clustering: centroids
are optimised via gradient descent on an unsupervised compactness +
separation loss (no label information), then evaluated against the
ground-truth cell-type labels.

Usage:
    python benchmarks/singlecell/bench_clustering.py
    python benchmarks/singlecell/bench_clustering.py --quick
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.clustering import CLUSTERING_BASELINES
from benchmarks._optimizers import create_benchmark_optimizer
from diffbio.losses.singlecell_losses import ClusteringCompactnessLoss
from diffbio.operators.singlecell import (
    SoftClusteringConfig,
    SoftKMeansClustering,
)
from diffbio.sources.immune_human import (
    ImmuneHumanConfig,
    ImmuneHumanSource,
)

logger = logging.getLogger(__name__)

_CONFIG = DiffBioBenchmarkConfig(
    name="singlecell/clustering",
    domain="singlecell",
    quick_subsample=2000,
)

# Training hyper-parameters (unsupervised — no label information used)
_N_STEPS_QUICK = 200
_N_STEPS_FULL = 500
_LEARNING_RATE = 1e-2
_TEMPERATURE = 0.5
_SEPARATION_WEIGHT = 1.0
_MIN_SEPARATION = 2.0


def _train_centroids(
    operator: SoftKMeansClustering,
    embeddings: jax.Array,
    n_steps: int,
) -> list[float]:
    """Optimise cluster centroids via unsupervised compactness loss.

    Uses gradient descent on ``ClusteringCompactnessLoss`` which combines
    within-cluster compactness and between-cluster separation.  No label
    information is used — this is purely unsupervised.

    Args:
        operator: Clustering operator whose centroids will be updated.
        embeddings: Cell embeddings ``(n_cells, n_features)``.
        n_steps: Number of gradient-descent steps.

    Returns:
        Loss history (one float per step).
    """
    loss_module = ClusteringCompactnessLoss(
        separation_weight=_SEPARATION_WEIGHT,
        min_separation=_MIN_SEPARATION,
    )
    opt = nnx.Optimizer(
        operator,
        create_benchmark_optimizer(learning_rate=_LEARNING_RATE),
        wrt=nnx.Param,
    )

    @nnx.jit
    def _step(
        model: SoftKMeansClustering,
        optimizer: nnx.Optimizer,
        data: jax.Array,
    ) -> jax.Array:
        def _loss(m: SoftKMeansClustering) -> jax.Array:
            result, _, _ = m.apply({"embeddings": data}, {}, None)
            return loss_module(
                data,
                result["cluster_assignments"],
                result["centroids"],
            )

        loss, grads = nnx.value_and_grad(_loss)(model)
        optimizer.update(model, grads)
        return loss

    history: list[float] = []
    for step in range(n_steps):
        loss = _step(operator, opt, embeddings)
        history.append(float(loss))
        if (step + 1) % 50 == 0:
            logger.info("  train step %d/%d  loss=%.4f", step + 1, n_steps, float(loss))

    return history


class ClusteringBenchmark(DiffBioBenchmark):
    """Evaluate SoftKMeansClustering on immune_human dataset."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Data/scib",
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)

    def _run_core(self) -> dict[str, Any]:
        """Load immune_human, train clustering, compute metrics."""
        import scib_metrics  # noqa: PLC0415

        subsample = self.config.quick_subsample if self.quick else None

        # 1. Load dataset
        logger.info("Loading immune_human dataset...")
        source = ImmuneHumanSource(ImmuneHumanConfig(data_dir=self.data_dir, subsample=subsample))
        data = source.load()
        n_cells = data["n_cells"]
        n_types = data["n_types"]
        embeddings = data["embeddings"]
        labels = np.array(data["cell_type_labels"], dtype=np.int32, copy=True)

        logger.info("  %d cells, %d types", n_cells, n_types)

        # 2. Create operator and train centroids (unsupervised)
        n_features = embeddings.shape[1]
        op_config = SoftClusteringConfig(
            n_clusters=n_types,
            n_features=n_features,
            temperature=_TEMPERATURE,
        )
        rngs = nnx.Rngs(42)
        operator = SoftKMeansClustering(op_config, rngs=rngs)

        n_steps = _N_STEPS_QUICK if self.quick else _N_STEPS_FULL
        logger.info("Training centroids (%d steps, unsupervised)...", n_steps)
        loss_history = _train_centroids(operator, embeddings, n_steps)
        logger.info(
            "  final loss: %.4f (initial: %.4f)",
            loss_history[-1],
            loss_history[0],
        )

        # 3. Evaluate trained operator
        input_data = {"embeddings": embeddings}
        result, _, _ = operator.apply(input_data, {}, None)

        pred_labels = np.array(
            jnp.argmax(result["cluster_assignments"], axis=-1),
            dtype=np.int32,
            copy=True,
        )

        # 4. Compute metrics
        logger.info("Computing metrics...")
        x_np = np.array(embeddings, dtype=np.float32, copy=True)
        sil = float(scib_metrics.silhouette_label(x_np, labels))

        nmi_ari_pred = scib_metrics.nmi_ari_cluster_labels_kmeans(x_np, pred_labels)

        ari = float(nmi_ari_pred["ari"])
        nmi = float(nmi_ari_pred["nmi"])

        logger.info("  ARI: %.4f, NMI: %.4f, Sil: %.4f", ari, nmi, sil)

        # Loss function for gradient check
        def loss_fn(model: SoftKMeansClustering, d: dict[str, Any]) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["cluster_assignments"])

        quality = {
            "ari_kmeans": ari,
            "nmi_kmeans": nmi,
            "silhouette_label": sil,
        }

        return {
            "metrics": quality,
            "operator": operator,
            "input_data": input_data,
            "loss_fn": loss_fn,
            "n_items": n_cells,
            "iterate_fn": lambda: operator.apply(input_data, {}, None),
            "baselines": CLUSTERING_BASELINES,
            "dataset_info": {
                "n_cells": n_cells,
                "n_types": n_types,
                "n_train_steps": n_steps,
            },
            "operator_config": {
                "n_clusters": n_types,
                "temperature": _TEMPERATURE,
                "learning_rate": _LEARNING_RATE,
                "n_steps": n_steps,
            },
            "operator_name": "SoftKMeansClustering",
            "dataset_name": "immune_human",
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        ClusteringBenchmark,
        _CONFIG,
        data_dir="/media/mahdi/ssd23/Data/scib",
    )


if __name__ == "__main__":
    main()
