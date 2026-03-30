#!/usr/bin/env python3
"""Master runner for all DiffBio benchmarks.

Discovers and executes all benchmark scripts, collects results,
renders a dashboard, and optionally saves baselines or checks
for regressions.

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --domains variant,singlecell
    python benchmarks/run_all.py --quick
    python benchmarks/run_all.py --save-baseline
    python benchmarks/run_all.py --check-regression
    python benchmarks/run_all.py --show-coverage
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks._common import collect_platform_info
from benchmarks.dashboard import render_full_dashboard
from benchmarks.regression import (
    RegressionThresholds,
    detect_regressions,
    load_baseline,
    save_baseline,
)
from benchmarks.schema import BenchmarkEnvelope, save_envelope

logger = logging.getLogger(__name__)

# Benchmark registry: (domain, module_path)
_BENCHMARK_REGISTRY: list[tuple[str, str]] = [
    ("alignment", "benchmarks.alignment.alignment_benchmark"),
    ("drug_discovery", "benchmarks.drug_discovery.molnet_benchmark"),
    ("drug_discovery", "benchmarks.drug_discovery.fingerprint_benchmark"),
    ("drug_discovery", "benchmarks.drug_discovery.circular_fingerprint_benchmark"),
    ("singlecell", "benchmarks.singlecell.singlecell_benchmark"),
    ("singlecell", "benchmarks.singlecell.scvi_benchmark"),
    ("singlecell", "benchmarks.singlecell.trajectory_benchmark"),
    ("singlecell", "benchmarks.singlecell.grn_benchmark"),
    ("variant", "benchmarks.variant.variant_calling_benchmark"),
    ("epigenomics", "benchmarks.epigenomics.epigenomics_benchmark"),
    ("protein", "benchmarks.protein.protein_structure_benchmark"),
    ("rna_structure", "benchmarks.rna_structure.rna_structure_benchmark"),
    ("molecular_dynamics", "benchmarks.molecular_dynamics.molecular_dynamics_benchmark"),
    ("multiomics", "benchmarks.multiomics.multiomics_benchmark"),
    ("normalization", "benchmarks.normalization.dimreduction_benchmark"),
    ("language_models", "benchmarks.language_models.language_model_benchmark"),
    ("preprocessing", "benchmarks.preprocessing.preprocessing_benchmark"),
    ("assembly", "benchmarks.assembly.assembly_benchmark"),
    ("specialized", "benchmarks.specialized.crispr_benchmark"),
    ("specialized", "benchmarks.specialized.population_benchmark"),
    ("specialized", "benchmarks.specialized.metabolomics_benchmark"),
]


def discover_benchmarks(
    domains: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Filter the benchmark registry by domain.

    Args:
        domains: If provided, only include benchmarks from these domains.

    Returns:
        Filtered list of (domain, module_path) tuples.
    """
    if domains is None:
        return list(_BENCHMARK_REGISTRY)
    domain_set = set(domains)
    return [(d, m) for d, m in _BENCHMARK_REGISTRY if d in domain_set]


def run_single_benchmark(
    domain: str,
    module_path: str,
    quick: bool = False,
) -> BenchmarkEnvelope | None:
    """Import and run a single benchmark module.

    Looks for ``run_benchmark(quick=...)`` in the module. If the module
    also has ``to_envelope(result)``, uses it to produce a
    :class:`BenchmarkEnvelope`. Otherwise, creates a minimal envelope.

    Args:
        domain: Benchmark domain name.
        module_path: Dotted Python import path.
        quick: Whether to use reduced data sizes.

    Returns:
        BenchmarkEnvelope or None if the benchmark failed.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logger.warning("Could not import %s: %s", module_path, exc)
        return None

    run_fn = getattr(module, "run_benchmark", None)
    if run_fn is None:
        logger.warning("Module %s has no run_benchmark function", module_path)
        return None

    try:
        if quick:
            result = run_fn(quick=True)
        else:
            result = run_fn()
    except TypeError:
        # Older benchmarks may not accept quick parameter
        try:
            result = run_fn()
        except Exception as exc:
            logger.error("Benchmark %s failed: %s", module_path, exc)
            return _error_envelope(domain, module_path, str(exc))
    except Exception as exc:
        logger.error("Benchmark %s failed: %s", module_path, exc)
        return _error_envelope(domain, module_path, str(exc))

    # Try to_envelope if available
    to_envelope_fn = getattr(module, "to_envelope", None)
    if to_envelope_fn is not None:
        try:
            return to_envelope_fn(result)
        except Exception as exc:
            logger.debug("to_envelope failed for %s: %s", module_path, exc)

    # Create minimal envelope from result
    return _minimal_envelope(domain, module_path, result)


def _error_envelope(domain: str, module_path: str, error: str) -> BenchmarkEnvelope:
    """Create an error envelope for a failed benchmark."""
    return BenchmarkEnvelope(
        benchmark_id=f"{domain}/{module_path.rsplit('.', 1)[-1]}",
        domain=domain,
        operators_tested=[],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        platform=collect_platform_info(),
        status="error",
        domain_metrics={"error": error},
    )


def _minimal_envelope(
    domain: str,
    module_path: str,
    result: Any,
) -> BenchmarkEnvelope:
    """Create a minimal envelope from a raw benchmark result."""
    from dataclasses import asdict  # noqa: PLC0415

    name = module_path.rsplit(".", 1)[-1]

    # Try to extract fields from dataclass result
    result_dict: dict[str, Any] = {}
    if hasattr(result, "__dataclass_fields__"):
        result_dict = asdict(result)

    return BenchmarkEnvelope(
        benchmark_id=f"{domain}/{name}",
        domain=domain,
        operators_tested=[],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        platform=collect_platform_info(),
        status="pass",
        domain_metrics=result_dict,
    )


def main() -> None:
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description="Run DiffBio benchmarks")
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated list of domains to run",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced data sizes for faster CI runs",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as a baseline for regression detection",
    )
    parser.add_argument(
        "--check-regression",
        type=str,
        default=None,
        metavar="BASELINE",
        help="Path to baseline JSON for regression checking",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory for result output",
    )
    parser.add_argument(
        "--show-coverage",
        action="store_true",
        help="Show scBench/spatialBench task coverage and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    domains = args.domains.split(",") if args.domains else None
    benchmarks = discover_benchmarks(domains)

    print(f"\nDiscovered {len(benchmarks)} benchmarks")
    if args.quick:
        print("Running in QUICK mode (reduced data sizes)")

    start_time = time.time()
    envelopes: list[BenchmarkEnvelope] = []

    for i, (domain, module_path) in enumerate(benchmarks, 1):
        name = module_path.rsplit(".", 1)[-1]
        print(f"\n[{i}/{len(benchmarks)}] Running {domain}/{name}...")

        envelope = run_single_benchmark(domain, module_path, quick=args.quick)
        if envelope is not None:
            envelopes.append(envelope)
            save_envelope(envelope, args.output_dir / domain)
            status = envelope.status.upper()
            print(f"  -> {status}")
        else:
            print("  -> SKIPPED")

    elapsed = time.time() - start_time

    # Render dashboard
    print("\n")
    print(render_full_dashboard(envelopes, elapsed))

    # Save baseline if requested
    if args.save_baseline:
        path = save_baseline(envelopes)
        print(f"\nBaseline saved to: {path}")

    # Check regressions if requested
    if args.check_regression:
        baseline_path = Path(args.check_regression)
        if baseline_path.exists():
            baseline = load_baseline(baseline_path)
            regressions = detect_regressions(
                envelopes, baseline, RegressionThresholds()
            )
            if regressions:
                print(f"\n{len(regressions)} REGRESSIONS DETECTED:")
                for r in regressions:
                    print(f"  [{r.severity.upper()}] {r.message}")
                sys.exit(1)
            else:
                print("\nNo regressions detected.")
        else:
            print(f"\nBaseline not found: {baseline_path}")
            sys.exit(1)


if __name__ == "__main__":
    main()
