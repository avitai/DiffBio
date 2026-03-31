#!/usr/bin/env python3
"""Clustering benchmark: SoftKMeansClustering on immune_human.

Evaluates DiffBio's SoftKMeansClustering against ground-truth cell
type labels on the scib immune human dataset using ARI, NMI, and
silhouette score.

Usage:
    python benchmarks/singlecell/bench_clustering.py
    python benchmarks/singlecell/bench_clustering.py --quick
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.clustering import CLUSTERING_BASELINES
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
        """Load immune_human, run clustering, compute metrics."""
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

        # 2. Run SoftKMeansClustering
        n_features = embeddings.shape[1]
        op_config = SoftClusteringConfig(
            n_clusters=n_types,
            n_features=n_features,
            temperature=0.5,
        )
        rngs = nnx.Rngs(42)
        operator = SoftKMeansClustering(op_config, rngs=rngs)

        input_data = {"embeddings": embeddings}
        result, _, _ = operator.apply(input_data, {}, None)

        # Extract hard assignments
        assignments = result["cluster_assignments"]
        pred_labels = np.array(
            jnp.argmax(assignments, axis=-1),
            dtype=np.int32,
            copy=True,
        )

        # 3. Compute metrics
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
            },
            "operator_config": {
                "n_clusters": n_types,
                "temperature": 0.5,
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
