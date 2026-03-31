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
import sys
import time
from typing import Any

import jax.numpy as jnp
import numpy as np
from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult
from calibrax.profiling.timing import TimingCollector
from flax import nnx

from benchmarks._gradient import check_gradient_flow
from benchmarks._metrics.grn import evaluate_grn
from diffbio.operators.singlecell.grn_inference import (
    DifferentiableGRN,
    GRNInferenceConfig,
)
from diffbio.sources.bengrn_ground_truth import BenGRNConfig, BenGRNSource

logger = logging.getLogger(__name__)

# Published baselines (approximate AUPRC from benGRN/Beeline papers)
_BASELINES = {
    "GENIE3": {"auprc": 0.08},
    "GRNBoost2": {"auprc": 0.07},
    "pySCENIC": {"auprc": 0.06},
    "Random": {"auprc": 0.02},
}


class GRNBenchmark:
    """Evaluate DifferentiableGRN on mESC ChIP+Perturb ground truth.

    Args:
        quick: If True, limit to 500 genes for speed.
        data_dir: Path to benGRN ground truth data.
    """

    def __init__(
        self,
        *,
        quick: bool = False,
        data_dir: str = str(
            "/media/mahdi/ssd23/Works/benGRN/data/"
            "GroundTruth/stone_and_sroy"
        ),
    ) -> None:
        self.quick = quick
        self.data_dir = data_dir

    def run(self) -> BenchmarkResult:
        """Execute the GRN benchmark."""
        max_genes = 500 if self.quick else 2000
        n_iters = 5 if self.quick else 20

        # 1. Load data
        print("Loading benGRN mESC ground truth...")
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

        print(
            f"  {n_cells} cells, {n_genes} genes, "
            f"{n_tfs} TFs, {n_edges} GT edges"
        )

        # 2. Run DifferentiableGRN
        print("Running DifferentiableGRN...")
        config = GRNInferenceConfig(
            n_genes=n_genes,
            n_tfs=n_tfs,
            hidden_dim=32 if self.quick else 64,
            num_heads=4,
        )
        rngs = nnx.Rngs(42)
        operator = DifferentiableGRN(config, rngs=rngs)

        input_data = {
            "counts": counts,
            "tf_indices": jnp.array(tf_indices),
        }

        start = time.perf_counter()
        result, _, _ = operator.apply(input_data, {}, None)
        wall_time = time.perf_counter() - start

        grn_matrix = result["grn_matrix"]
        print(
            f"  GRN shape: {grn_matrix.shape}, "
            f"Time: {wall_time:.2f}s"
        )

        # 3. Evaluate against ground truth
        print("Evaluating against ChIP+Perturb ground truth...")

        # Expand GRN from (n_tfs, n_genes) to (n_genes, n_genes)
        pred_full = np.zeros((n_genes, n_genes), dtype=np.float32)
        grn_np = np.asarray(jnp.abs(grn_matrix))
        for i, tf_idx in enumerate(tf_indices):
            if i < grn_np.shape[0]:
                pred_full[tf_idx] = grn_np[i]

        grn_metrics = evaluate_grn(pred_full, gt_matrix)

        for key, value in sorted(grn_metrics.items()):
            print(f"  {key}: {value:.6f}")

        # 4. Gradient flow
        print("Checking gradient flow...")

        def loss_fn(
            model: DifferentiableGRN, d: dict[str, Any]
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["grn_matrix"])

        grad = check_gradient_flow(loss_fn, operator, input_data)
        print(f"  Gradient norm: {grad.gradient_norm:.4f}")

        # 5. Throughput
        print("Measuring throughput...")
        collector = TimingCollector(warmup_iterations=2)
        timing = collector.measure_iteration(
            iterator=iter(range(n_iters)),
            num_batches=n_iters,
            process_fn=lambda _: operator.apply(
                input_data, {}, None
            ),
            count_fn=lambda _: n_cells,
        )
        cells_per_sec = timing.num_elements / timing.wall_clock_sec
        print(f"  {cells_per_sec:.0f} cells/sec")

        # 6. Comparison
        print("\nComparison (AUPRC):")
        print(
            f"  DiffBio GRN: {grn_metrics['auprc']:.6f}"
        )
        for name, bl in _BASELINES.items():
            print(f"  {name}: {bl['auprc']:.6f}")

        # 7. Build result
        metrics = {
            k: Metric(value=v) for k, v in grn_metrics.items()
        }
        metrics["gradient_norm"] = Metric(
            value=grad.gradient_norm
        )
        metrics["gradient_nonzero"] = Metric(
            value=1.0 if grad.gradient_nonzero else 0.0
        )
        metrics["cells_per_sec"] = Metric(value=cells_per_sec)

        return BenchmarkResult(
            name="singlecell/grn",
            domain="diffbio_benchmarks",
            tags={
                "operator": "DifferentiableGRN",
                "dataset": "bengrn_mesc",
                "ground_truth": "chipunion_KDUnion_intersect",
                "framework": "diffbio",
            },
            timing=timing,
            metrics=metrics,
            config={
                "n_genes": n_genes,
                "n_tfs": n_tfs,
                "hidden_dim": config.hidden_dim,
                "quick": self.quick,
            },
            metadata={
                "dataset_info": {
                    "n_cells": n_cells,
                    "n_genes": n_genes,
                    "n_tfs": n_tfs,
                    "n_gt_edges": n_edges,
                },
                "baselines": _BASELINES,
            },
        )


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("DiffBio Benchmark: GRN Inference")
    print("=" * 60)

    bench = GRNBenchmark(quick=quick)
    result = bench.run()

    from pathlib import Path  # noqa: PLC0415

    out = Path("benchmarks/results/singlecell")
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "grn.json")
    print(f"\nSaved to: {out / 'grn.json'}")


if __name__ == "__main__":
    main()
