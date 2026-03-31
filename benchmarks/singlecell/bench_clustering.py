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
import sys
import time
from typing import Any

import jax.numpy as jnp
import numpy as np
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult
from calibrax.profiling.timing import TimingCollector
from flax import nnx

from benchmarks._baselines.clustering import CLUSTERING_BASELINES
from benchmarks._gradient import check_gradient_flow
from diffbio.operators.singlecell import (
    SoftClusteringConfig,
    SoftKMeansClustering,
)
from diffbio.sources.immune_human import ImmuneHumanConfig, ImmuneHumanSource

logger = logging.getLogger(__name__)

_BASELINES = CLUSTERING_BASELINES


class ClusteringBenchmark:
    """Evaluate SoftKMeansClustering on immune_human dataset.

    Args:
        quick: If True, subsample to 2000 cells.
        data_dir: Directory containing the downloaded h5ad file.
    """

    def __init__(
        self,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Data/scib",
    ) -> None:
        self.quick = quick
        self.data_dir = data_dir

    def run(self) -> BenchmarkResult:
        """Execute the clustering benchmark."""
        import scib_metrics  # noqa: PLC0415
        from scib_metrics.nearest_neighbors import pynndescent  # noqa: PLC0415

        subsample = 2000 if self.quick else None
        n_iterations = 10 if self.quick else 50

        # 1. Load dataset
        print("Loading immune_human dataset...")
        source = ImmuneHumanSource(
            ImmuneHumanConfig(
                data_dir=self.data_dir, subsample=subsample
            )
        )
        data = source.load()
        n_cells = data["n_cells"]
        n_types = data["n_types"]
        embeddings = data["embeddings"]
        labels = np.array(data["cell_type_labels"], dtype=np.int32, copy=True)

        print(f"  {n_cells} cells, {n_types} types")

        # 2. Run SoftKMeansClustering
        n_features = embeddings.shape[1]
        config = SoftClusteringConfig(
            n_clusters=n_types,
            n_features=n_features,
            temperature=0.5,
        )
        rngs = nnx.Rngs(42)
        operator = SoftKMeansClustering(config, rngs=rngs)

        print("Running SoftKMeansClustering...")
        input_data = {"embeddings": embeddings}

        start = time.perf_counter()
        result, _, _ = operator.apply(input_data, {}, None)
        wall_time = time.perf_counter() - start
        print(f"  Completed in {wall_time:.2f}s")

        # Extract hard assignments
        assignments = result["cluster_assignments"]
        pred_labels = np.array(
            jnp.argmax(assignments, axis=-1), dtype=np.int32, copy=True
        )

        # 3. Compute metrics
        print("Computing metrics...")
        X = np.array(embeddings, dtype=np.float32, copy=True)

        nmi_ari = scib_metrics.nmi_ari_cluster_labels_kmeans(X, labels)
        sil = float(scib_metrics.silhouette_label(X, labels))

        # Also compute metrics on our predicted clusters
        nmi_ari_pred = scib_metrics.nmi_ari_cluster_labels_kmeans(
            X, pred_labels
        )

        ari = float(nmi_ari_pred["ari"])
        nmi = float(nmi_ari_pred["nmi"])

        print(f"  ARI (DiffBio clusters vs types): {ari:.4f}")
        print(f"  NMI (DiffBio clusters vs types): {nmi:.4f}")
        print(f"  Silhouette (on true labels): {sil:.4f}")

        # 4. Gradient flow
        print("Checking gradient flow...")

        def loss_fn(
            model: SoftKMeansClustering, d: dict[str, Any]
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["cluster_assignments"])

        grad = check_gradient_flow(loss_fn, operator, input_data)
        print(f"  Gradient norm: {grad.gradient_norm:.4f}")

        # 5. Throughput
        print("Measuring throughput...")
        collector = TimingCollector(warmup_iterations=3)
        timing = collector.measure_iteration(
            iterator=iter(range(n_iterations)),
            num_batches=n_iterations,
            process_fn=lambda _: operator.apply(input_data, {}, None),
            count_fn=lambda _: n_cells,
        )
        cells_per_sec = timing.num_elements / timing.wall_clock_sec
        print(f"  {cells_per_sec:.0f} cells/sec")

        # 6. Comparison table
        print("\nComparison:")
        print(f"  {'Method':<22} {'ARI':>6} {'NMI':>6}")
        print(f"  {'DiffBio SoftKMeans':<22} {ari:>6.3f} {nmi:>6.3f}")
        for name, point in _BASELINES.items():
            bl_ari = point.metrics["ari_kmeans"].value
            bl_nmi = point.metrics["nmi_kmeans"].value
            print(f"  {name:<22} {bl_ari:>6.3f} {bl_nmi:>6.3f}")

        # 7. Build result
        metrics = {
            "ari_kmeans": Metric(value=ari),
            "nmi_kmeans": Metric(value=nmi),
            "silhouette_label": Metric(value=sil),
            "gradient_norm": Metric(value=grad.gradient_norm),
            "gradient_nonzero": Metric(
                value=1.0 if grad.gradient_nonzero else 0.0
            ),
            "cells_per_sec": Metric(value=cells_per_sec),
        }

        return BenchmarkResult(
            name="singlecell/clustering",
            domain="diffbio_benchmarks",
            tags={
                "operator": "SoftKMeansClustering",
                "dataset": "immune_human",
                "framework": "diffbio",
            },
            timing=timing,
            metrics=metrics,
            config={
                "n_clusters": n_types,
                "temperature": 0.5,
                "quick": self.quick,
            },
            metadata={
                "dataset_info": {
                    "n_cells": n_cells,
                    "n_types": n_types,
                },
                "baselines": {k: p.to_dict() for k, p in _BASELINES.items()},
            },
        )


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("DiffBio Benchmark: Clustering")
    print("=" * 60)

    bench = ClusteringBenchmark(quick=quick)
    result = bench.run()

    from pathlib import Path  # noqa: PLC0415

    out = Path("benchmarks/results/singlecell")
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "clustering.json")
    print(f"\nSaved to: {out / 'clustering.json'}")


if __name__ == "__main__":
    main()
