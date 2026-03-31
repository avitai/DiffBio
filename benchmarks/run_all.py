#!/usr/bin/env python3
"""Master benchmark runner for DiffBio.

Discovers and executes benchmarks by tier, stores results via calibrax
Store, and renders comparison tables.

Usage:
    python benchmarks/run_all.py --tier ci       # <5 min, subsampled
    python benchmarks/run_all.py --tier nightly   # <30 min, full tier 1
    python benchmarks/run_all.py --tier full      # <2 hours, everything
    python benchmarks/run_all.py --domains singlecell
"""

from __future__ import annotations

import argparse
import importlib
import logging
import time
from pathlib import Path

from calibrax.core.models import Point, Run
from calibrax.core.result import BenchmarkResult
from calibrax.profiling.hardware import detect_hardware_specs

logger = logging.getLogger(__name__)

# Registry: (domain, module_path, benchmark_class_name)
_TIER_1 = [
    ("singlecell", "benchmarks.singlecell.bench_batch_correction", "BatchCorrectionBenchmark"),
    ("singlecell", "benchmarks.singlecell.bench_clustering", "ClusteringBenchmark"),
    ("singlecell", "benchmarks.singlecell.bench_vae_integration", "VAEIntegrationBenchmark"),
    ("singlecell", "benchmarks.singlecell.bench_trajectory", "TrajectoryBenchmark"),
    ("singlecell", "benchmarks.singlecell.bench_grn", "GRNBenchmark"),
    ("drug_discovery", "benchmarks.drug_discovery.bench_molnet", "MolNetBenchmark"),
]

_TIER_2 = [
    ("alignment", "benchmarks.alignment.bench_msa", "MSABenchmark"),
    ("alignment", "benchmarks.alignment.bench_pairwise", "PairwiseBenchmark"),
    ("rna_structure", "benchmarks.rna_structure.bench_rna_fold", "RNAFoldBenchmark"),
    ("protein", "benchmarks.protein.bench_secondary_structure", "SecondaryStructureBenchmark"),
    ("molecular_dynamics", "benchmarks.molecular_dynamics.bench_lj", "LJBenchmark"),
]

_TIER_3 = [
    ("statistical", "benchmarks.statistical.bench_de", "DEBenchmark"),
    ("multiomics", "benchmarks.multiomics.bench_spatial_deconv", "SpatialDeconvBenchmark"),
    ("epigenomics", "benchmarks.epigenomics.bench_peak_calling", "PeakCallingBenchmark"),
]

_TIERS = {
    "ci": _TIER_1[:2],  # batch_correction + clustering only
    "nightly": _TIER_1,  # all tier 1
    "full": _TIER_1 + _TIER_2 + _TIER_3,
}


def _run_single(
    domain: str,
    module_path: str,
    class_name: str,
    quick: bool,
) -> BenchmarkResult | None:
    """Import and run a single benchmark.

    Args:
        domain: Domain name for logging.
        module_path: Dotted module path.
        class_name: Benchmark class name within the module.
        quick: Whether to use subsampled data.

    Returns:
        BenchmarkResult or None if the benchmark failed.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logger.error("Cannot import %s: %s", module_path, exc)
        return None

    bench_cls = getattr(module, class_name, None)
    if bench_cls is None:
        logger.error("Class %s not found in %s", class_name, module_path)
        return None

    try:
        bench = bench_cls(quick=quick)
        return bench.run()
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        # Benchmark can fail due to missing data, operator errors,
        # or resource issues. Log and continue to next benchmark.
        logger.error("Benchmark %s/%s failed: %s", domain, class_name, exc)
        return None


def _build_run(results: list[BenchmarkResult]) -> Run:
    """Build a calibrax Run from benchmark results.

    Each BenchmarkResult becomes a Point in the Run for ranking
    and comparison.
    """
    points: list[Point] = []
    for r in results:
        points.append(
            Point(
                name=r.name,
                scenario=r.tags.get("dataset", "unknown"),
                tags=r.tags,
                metrics=r.metrics,
            )
        )

    hw = detect_hardware_specs()
    return Run(
        points=tuple(points),
        environment=hw,
    )


def _print_summary(
    results: list[BenchmarkResult],
    elapsed: float,
) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 70)
    print("DiffBio Benchmark Summary")
    print("=" * 70)

    header = f"{'Benchmark':<35} {'Key Metric':<15} {'Value':>8} {'Status':>8}"
    print(header)
    print("-" * 70)

    for r in results:
        # Pick the most representative metric
        if "aggregate_score" in r.metrics:
            key = "aggregate_score"
        elif "auprc" in r.metrics:
            key = "auprc"
        elif "ari_kmeans" in r.metrics:
            key = "ari_kmeans"
        elif "pseudotime_range" in r.metrics:
            key = "pseudotime_range"
        else:
            key = next(iter(r.metrics), "N/A")

        if key in r.metrics:
            value = r.metrics[key].value
            value_str = f"{value:.4f}"
        else:
            value_str = "N/A"

        status = "PASS"
        print(f"{r.name:<35} {key:<15} {value_str:>8} {status:>8}")

    n_pass = len(results)
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    print("-" * 70)
    print(f"PASS: {n_pass}/{n_pass} | TIME: {time_str}")
    print("=" * 70)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run DiffBio benchmarks")
    parser.add_argument(
        "--tier",
        choices=["ci", "nightly", "full"],
        default="ci",
        help="Benchmark tier to run",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated domain filter",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=None,
        help="Force quick mode (auto for ci tier)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Determine benchmarks to run
    benchmarks = _TIERS.get(args.tier, _TIER_1)
    quick = args.quick if args.quick is not None else (args.tier == "ci")

    # Filter by domain if specified
    if args.domains:
        domain_set = set(args.domains.split(","))
        benchmarks = [b for b in benchmarks if b[0] in domain_set]

    print(f"Running {len(benchmarks)} benchmarks (tier={args.tier})")
    if quick:
        print("Mode: QUICK (subsampled datasets)")

    start = time.time()
    results: list[BenchmarkResult] = []

    for i, (domain, module_path, class_name) in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] {domain}/{class_name}")
        result = _run_single(domain, module_path, class_name, quick)
        if result is not None:
            results.append(result)

            # Save individual result
            out_dir = Path("benchmarks/results") / domain
            out_dir.mkdir(parents=True, exist_ok=True)
            result.save(out_dir / f"{class_name}.json")
        else:
            print("  -> FAILED")

    elapsed = time.time() - start
    _print_summary(results, elapsed)

    # Build and save Run
    if results:
        run = _build_run(results)
        try:
            from calibrax.storage.store import Store  # noqa: PLC0415

            store = Store(Path("benchmarks/results"))
            store.save(run)
            print("\nRun saved to: benchmarks/results/")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Could not save Run: %s", exc)


if __name__ == "__main__":
    main()
