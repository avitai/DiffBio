#!/usr/bin/env python3
"""RNA folding benchmark: DifferentiableRNAFold on ArchiveII.

Evaluates DiffBio's DifferentiableRNAFold operator on ArchiveII RNA
secondary structures using base pair F1, sensitivity, and PPV.

Results are compared against published baselines: ViennaRNA (~0.72 F1),
LinearFold (~0.73), EternaFold (~0.75), CONTRAfold (~0.70).

Usage:
    python benchmarks/rna_structure/bench_rna_fold.py
    python benchmarks/rna_structure/bench_rna_fold.py --quick
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

from benchmarks._baselines.rna import RNA_FOLD_BASELINES
from benchmarks._gradient import check_gradient_flow
from benchmarks._metrics.structure import base_pair_metrics
from diffbio.operators.rna_structure.rna_folding import (
    DifferentiableRNAFold,
    RNAFoldConfig,
)
from diffbio.sources.archive_ii import ArchiveIIConfig, ArchiveIISource

logger = logging.getLogger(__name__)

# RNA nucleotide mapping for one-hot encoding
_NUC_TO_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}
_RNA_ALPHABET_SIZE = 4


def encode_rna_sequence(sequence: str) -> jnp.ndarray:
    """Encode an RNA sequence string as a one-hot JAX array.

    Maps A=0, C=1, G=2, U=3. T is treated as U for DNA/RNA
    compatibility. Unknown nucleotides are encoded as uniform
    distributions (0.25 for each).

    Args:
        sequence: RNA sequence string (e.g. ``"GUCUACC"``).

    Returns:
        One-hot encoded array of shape ``(length, 4)``.
    """
    indices = []
    for nuc in sequence.upper():
        if nuc == "T":
            nuc = "U"
        indices.append(_NUC_TO_IDX.get(nuc, -1))

    idx_array = jnp.array(indices, dtype=jnp.int32)
    # Use one_hot; unknown nucleotides (idx=-1) get zeros, then
    # replace with uniform distribution
    one_hot = jax.nn.one_hot(idx_array, _RNA_ALPHABET_SIZE)
    unknown_mask = (idx_array < 0).astype(jnp.float32)
    uniform = jnp.full((_RNA_ALPHABET_SIZE,), 1.0 / _RNA_ALPHABET_SIZE)
    one_hot = jnp.where(
        unknown_mask[:, None] > 0.5,
        uniform[None, :],
        one_hot,
    )
    return one_hot


class RNAFoldBenchmark:
    """Evaluate DifferentiableRNAFold on the ArchiveII dataset.

    Runs the RNA fold operator on each sequence, scores predicted
    base pair probabilities against known DBN structures, and
    compares against published baselines.

    Args:
        quick: If True, use at most 5 sequences for fast CI runs.
        data_dir: Directory containing the ArchiveII CSV data.
    """

    def __init__(
        self,
        *,
        quick: bool = False,
        data_dir: str = ("/media/mahdi/ssd23/Works/RNAFoldAssess/tutorial/processed_data"),
    ) -> None:
        self.quick = quick
        self.data_dir = data_dir

    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return a calibrax result."""
        max_sequences = 5 if self.quick else None
        n_timing_iters = 5 if self.quick else 20

        # 1. Load dataset via DataSource
        print("Loading ArchiveII dataset...")
        source_config = ArchiveIIConfig(
            data_dir=self.data_dir,
            max_sequences=max_sequences,
        )
        source = ArchiveIISource(source_config)
        data = source.load()

        entries = data["entries"]
        n_sequences = data["n_sequences"]
        print(f"  {n_sequences} sequences loaded")

        # 2. Create operator
        op_config = RNAFoldConfig(temperature=1.0)
        rngs = nnx.Rngs(42)
        operator = DifferentiableRNAFold(op_config, rngs=rngs)

        # 3. Fold each sequence and compute metrics
        print("Running DifferentiableRNAFold...")
        all_metrics: list[dict[str, float]] = []

        start = time.perf_counter()
        for entry in entries:
            seq_onehot = encode_rna_sequence(entry["sequence"])
            input_data = {"sequence": seq_onehot}
            result, _, _ = operator.apply(input_data, {}, None)
            bp_probs_np = np.asarray(result["bp_probs"])

            metrics = base_pair_metrics(
                bp_probs_np,
                entry["structure"],
            )
            all_metrics.append(metrics)

            logger.debug(
                "  %s: F1=%.4f sens=%.4f ppv=%.4f (len=%d, pairs=%d)",
                entry["name"],
                metrics["f1"],
                metrics["sensitivity"],
                metrics["ppv"],
                len(entry["sequence"]),
                metrics["n_true_pairs"],
            )

        wall_time = time.perf_counter() - start
        print(f"  Completed in {wall_time:.2f}s")

        # 4. Aggregate metrics across all sequences
        avg_sensitivity = float(np.mean([m["sensitivity"] for m in all_metrics]))
        avg_ppv = float(np.mean([m["ppv"] for m in all_metrics]))
        avg_f1 = float(np.mean([m["f1"] for m in all_metrics]))
        total_true = int(sum(m["n_true_pairs"] for m in all_metrics))
        total_predicted = int(sum(m["n_predicted_pairs"] for m in all_metrics))

        quality: dict[str, float] = {
            "sensitivity": avg_sensitivity,
            "ppv": avg_ppv,
            "f1": avg_f1,
        }

        for key, value in sorted(quality.items()):
            print(f"  {key}: {value:.4f}")

        # 5. Check gradient flow
        print("Checking gradient flow...")
        first_seq = encode_rna_sequence(entries[0]["sequence"])
        grad_input = {"sequence": first_seq}

        def loss_fn(
            model: DifferentiableRNAFold,
            d: dict[str, Any],
        ) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["bp_probs"])

        grad = check_gradient_flow(loss_fn, operator, grad_input)
        print(f"  Gradient norm: {grad.gradient_norm:.4f}, nonzero: {grad.gradient_nonzero}")

        # 6. Measure throughput
        print("Measuring throughput...")

        def _fold_one(_: Any) -> Any:
            return operator.apply(grad_input, {}, None)

        collector = TimingCollector(warmup_iterations=2)
        timing = collector.measure_iteration(
            iterator=iter(range(n_timing_iters)),
            num_batches=n_timing_iters,
            process_fn=_fold_one,
            count_fn=lambda _: len(entries[0]["sequence"]),
        )
        nucs_per_sec = timing.num_elements / timing.wall_clock_sec
        print(f"  {nucs_per_sec:.0f} nucleotides/sec")

        # 7. Print comparison table
        print("\nComparison Table:")
        header = f"{'Method':<20} {'F1':>8} {'Sens.':>8} {'PPV':>8}"
        print(header)
        print("-" * len(header))
        print(f"{'DiffBio RNAFold':<20} {avg_f1:>8.4f} {avg_sensitivity:>8.4f} {avg_ppv:>8.4f}")
        for name, point in RNA_FOLD_BASELINES.items():
            bl_f1 = point.metrics.get("f1", Metric(value=0)).value
            bl_sens = point.metrics.get("sensitivity", Metric(value=0)).value
            bl_ppv = point.metrics.get("ppv", Metric(value=0)).value
            print(f"{name:<20} {bl_f1:>8.4f} {bl_sens:>8.4f} {bl_ppv:>8.4f}")

        # 8. Build calibrax BenchmarkResult
        metrics_out = {k: Metric(value=v) for k, v in quality.items()}
        metrics_out["gradient_norm"] = Metric(value=grad.gradient_norm)
        metrics_out["gradient_nonzero"] = Metric(value=1.0 if grad.gradient_nonzero else 0.0)
        metrics_out["nucleotides_per_sec"] = Metric(value=nucs_per_sec)

        return BenchmarkResult(
            name="rna_structure/rna_fold",
            domain="diffbio_benchmarks",
            tags={
                "operator": "DifferentiableRNAFold",
                "dataset": "archiveII",
                "framework": "diffbio",
            },
            timing=timing,
            metrics=metrics_out,
            config={
                "temperature": op_config.temperature,
                "min_hairpin_loop": op_config.min_hairpin_loop,
                "quick": self.quick,
                "max_sequences": max_sequences,
            },
            metadata={
                "dataset_info": {
                    "name": "archiveII",
                    "n_sequences": n_sequences,
                    "total_true_pairs": total_true,
                    "total_predicted_pairs": total_predicted,
                },
            },
        )


def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    quick = "--quick" in sys.argv

    print("=" * 60)
    print("DiffBio Benchmark: RNA Secondary Structure Prediction")
    mode = "quick (5 seqs)" if quick else "full"
    print(f"Mode: {mode}")
    print("=" * 60)

    bench = RNAFoldBenchmark(quick=quick)
    result = bench.run()

    # Save result
    from pathlib import Path  # noqa: PLC0415

    output_dir = Path("benchmarks/results/rna_structure")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(output_dir / "rna_fold.json")
    print(f"\nResult saved to: {output_dir / 'rna_fold.json'}")

    print("\n" + "=" * 60)
    print(f"F1 score: {result.metrics['f1'].value:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
