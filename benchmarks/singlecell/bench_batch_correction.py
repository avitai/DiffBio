#!/usr/bin/env python3
"""Batch correction benchmark: DifferentiableHarmony on immune_human.

Evaluates DiffBio's DifferentiableHarmony operator on the scib
immune human integration benchmark (33,506 cells, 10 batches,
16 cell types) using the full scib-metrics suite.

Results are compared against published baselines from Luecken et al.
2022: scVI, Harmony (R), Scanorama, BBKNN, and Unintegrated.

Usage:
    python benchmarks/singlecell/bench_batch_correction.py
    python benchmarks/singlecell/bench_batch_correction.py --quick
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

from benchmarks._baselines.scib import INTEGRATION_BASELINES
from benchmarks._gradient import check_gradient_flow
from benchmarks._metrics.scib_bridge import evaluate_integration
from diffbio.operators.singlecell import (
    BatchCorrectionConfig,
    DifferentiableHarmony,
)
from diffbio.sources.immune_human import ImmuneHumanConfig, ImmuneHumanSource

logger = logging.getLogger(__name__)


class BatchCorrectionBenchmark:
    """Evaluate DifferentiableHarmony on the scib immune human dataset.

    Computes the full scib-metrics integration suite and compares
    against published baselines.

    Args:
        quick: If True, subsample to 2000 cells for fast CI runs.
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
        """Execute the benchmark and return a calibrax result."""
        subsample = 2000 if self.quick else None
        n_iterations = 10 if self.quick else 50

        # 1. Load dataset via DataSource
        print("Loading immune_human dataset...")
        source_config = ImmuneHumanConfig(
            data_dir=self.data_dir,
            subsample=subsample,
        )
        source = ImmuneHumanSource(source_config)
        data = source.load()

        n_cells = data["n_cells"]
        n_genes = data["n_genes"]
        n_batches = data["n_batches"]
        n_types = data["n_types"]
        embeddings = data["embeddings"]
        batch_labels = data["batch_labels"]
        cell_type_labels = data["cell_type_labels"]

        print(
            f"  {n_cells} cells, {n_genes} genes, "
            f"{n_batches} batches, {n_types} types"
        )

        # 2. Create operator
        n_features = embeddings.shape[1]
        op_config = BatchCorrectionConfig(
            n_clusters=min(20, n_cells // 10),
            n_features=n_features,
            n_batches=n_batches,
            temperature=1.0,
        )
        rngs = nnx.Rngs(42)
        operator = DifferentiableHarmony(op_config, rngs=rngs)

        # 3. Run operator with profiling
        print("Running DifferentiableHarmony...")
        input_data = {
            "embeddings": embeddings,
            "batch_labels": jnp.array(batch_labels),
        }

        start = time.perf_counter()
        result, _, _ = operator.apply(input_data, {}, None)
        wall_time = time.perf_counter() - start

        corrected = result["corrected_embeddings"]
        print(f"  Completed in {wall_time:.2f}s")

        # 4. Compute scib-metrics
        print("Computing scib-metrics...")
        quality = evaluate_integration(
            corrected_embeddings=np.asarray(corrected),
            labels=np.asarray(cell_type_labels),
            batch=np.asarray(batch_labels),
        )

        for key, value in sorted(quality.items()):
            print(f"  {key}: {value:.4f}")

        # 5. Check gradient flow
        print("Checking gradient flow...")

        def loss_fn(
            model: DifferentiableHarmony, d: dict[str, Any]
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["corrected_embeddings"])

        grad = check_gradient_flow(loss_fn, operator, input_data)
        print(
            f"  Gradient norm: {grad.gradient_norm:.4f}, "
            f"nonzero: {grad.gradient_nonzero}"
        )

        # 6. Measure throughput
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

        # 7. Print comparison table
        baselines = INTEGRATION_BASELINES.get("immune_human", {})
        print("\nComparison Table:")
        header = f"{'Method':<20} {'Aggregate':>10} {'Sil.Lab':>8}"
        print(header)
        print("-" * len(header))
        print(
            f"{'DiffBio Harmony':<20} "
            f"{quality['aggregate_score']:>10.4f} "
            f"{quality['silhouette_label']:>8.4f}"
        )
        for name, point in baselines.items():
            agg = point.metrics.get(
                "aggregate_score", Metric(value=0)
            ).value
            sil = point.metrics.get(
                "silhouette_label", Metric(value=0)
            ).value
            print(f"{name:<20} {agg:>10.4f} {sil:>8.4f}")

        # 8. Build calibrax BenchmarkResult
        metrics = {
            k: Metric(value=v) for k, v in quality.items()
        }
        metrics["gradient_norm"] = Metric(value=grad.gradient_norm)
        metrics["gradient_nonzero"] = Metric(
            value=1.0 if grad.gradient_nonzero else 0.0
        )
        metrics["cells_per_sec"] = Metric(value=cells_per_sec)

        return BenchmarkResult(
            name="singlecell/batch_correction",
            domain="diffbio_benchmarks",
            tags={
                "operator": "DifferentiableHarmony",
                "dataset": "immune_human",
                "framework": "diffbio",
            },
            timing=timing,
            metrics=metrics,
            config={
                "n_clusters": op_config.n_clusters,
                "n_features": n_features,
                "n_batches": n_batches,
                "temperature": op_config.temperature,
                "quick": self.quick,
                "subsample": subsample,
            },
            metadata={
                "dataset_info": {
                    "name": "immune_human",
                    "n_cells": n_cells,
                    "n_genes": n_genes,
                    "n_batches": n_batches,
                    "n_types": n_types,
                },
            },
        )


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("DiffBio Benchmark: Batch Correction")
    print(f"Mode: {'quick (2K cells)' if quick else 'full (33K cells)'}")
    print("=" * 60)

    bench = BatchCorrectionBenchmark(quick=quick)
    result = bench.run()

    # Save result
    from pathlib import Path  # noqa: PLC0415

    output_dir = Path("benchmarks/results/singlecell")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(output_dir / "batch_correction.json")
    print(f"\nResult saved to: {output_dir / 'batch_correction.json'}")

    print("\n" + "=" * 60)
    print(
        f"Aggregate score: "
        f"{result.metrics['aggregate_score'].value:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
