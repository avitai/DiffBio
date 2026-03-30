#!/usr/bin/env python3
"""Generate a Markdown benchmark report for DiffBio documentation.

Reads all benchmark result envelopes from ``benchmarks/results/`` and
produces a Markdown file suitable for the mkdocs documentation site.

Usage:
    python benchmarks/generate_report.py
    python benchmarks/generate_report.py --output docs/development/benchmark-results.md
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from benchmarks._common import collect_platform_info
from benchmarks.schema import BenchmarkEnvelope, load_all_envelopes

logger = logging.getLogger(__name__)

# scBench/spatialBench task types for coverage mapping
_EVAL_TASK_TYPES = [
    "qc_filtering",
    "normalization",
    "clustering",
    "differential_expression",
    "batch_correction",
    "trajectory",
    "spatial_analysis",
    "cell_annotation",
]


def _collect_results(results_dir: Path) -> list[BenchmarkEnvelope]:
    """Collect all envelopes from results directory and subdirectories.

    Args:
        results_dir: Root results directory.

    Returns:
        List of all envelopes found.
    """
    envelopes: list[BenchmarkEnvelope] = []
    if not results_dir.exists():
        return envelopes

    # Load from root
    envelopes.extend(load_all_envelopes(results_dir))

    # Load from subdirectories
    for subdir in sorted(results_dir.iterdir()):
        if subdir.is_dir():
            envelopes.extend(load_all_envelopes(subdir))

    return envelopes


def _generate_header(info: dict[str, str]) -> str:
    """Generate report header with environment info."""
    date = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "# Benchmark Results",
        "",
        f"> Auto-generated on {date}. Do not edit manually.",
        "",
        "## Environment",
        "",
        "| Component | Value |",
        "|-----------|-------|",
        f"| DiffBio | {info.get('diffbio_version', 'unknown')} |",
        f"| JAX | {info.get('jax_version', 'unknown')} |",
        f"| Platform | {info.get('platform', 'unknown')} |",
        f"| Device | {info.get('device', 'unknown')} |",
        f"| Python | {info.get('python_version', 'unknown')} |",
        "",
    ]
    return "\n".join(lines)


def _generate_capabilities_matrix(envelopes: list[BenchmarkEnvelope]) -> str:
    """Generate the capabilities matrix table."""
    if not envelopes:
        return "No benchmark results available.\n"

    lines = [
        "## Capabilities Matrix",
        "",
        "| Domain | Operator | Correct | Differentiable | Throughput | Status |",
        "|--------|----------|---------|----------------|------------|--------|",
    ]

    for e in sorted(envelopes, key=lambda x: (x.domain, x.benchmark_id)):
        tests = e.correctness.get("tests", [])
        n_passed = sum(1 for t in tests if t.get("passed", True))
        n_total = len(tests)
        correct = f"{n_passed}/{n_total}" if n_total else "-"

        diff_ok = e.differentiability.get("gradient_nonzero", False)
        diff_icon = ":white_check_mark:" if diff_ok else ":x:"

        tp = e.performance.get("throughput", 0)
        unit = e.performance.get("throughput_unit", "")
        throughput = f"{tp:.1f} {unit}" if tp else "-"

        status_icon = {
            "pass": ":white_check_mark:",
            "fail": ":x:",
            "error": ":warning:",
        }.get(e.status, ":question:")

        operators = ", ".join(e.operators_tested[:3])
        lines.append(
            f"| {e.domain} | {operators} | {correct} | "
            f"{diff_icon} | {throughput} | {status_icon} |"
        )

    lines.append("")
    return "\n".join(lines)


def _generate_task_coverage(envelopes: list[BenchmarkEnvelope]) -> str:
    """Generate scBench/spatialBench task coverage table."""
    covered: dict[str, list[str]] = defaultdict(list)
    for e in envelopes:
        for task in e.evaluation_task_types:
            covered[task].extend(e.operators_tested)

    lines = [
        "## Evaluation Task Coverage (scBench / spatialBench)",
        "",
        "| Task Type | DiffBio Operators | Status |",
        "|-----------|-------------------|--------|",
    ]

    for task in _EVAL_TASK_TYPES:
        if task in covered:
            ops = ", ".join(sorted(set(covered[task]))[:3])
            status = ":white_check_mark:"
        else:
            ops = "-"
            status = ":x:"
        lines.append(f"| {task} | {ops} | {status} |")

    lines.append("")
    return "\n".join(lines)


def _generate_domain_details(envelopes: list[BenchmarkEnvelope]) -> str:
    """Generate per-domain detail sections."""
    domains: dict[str, list[BenchmarkEnvelope]] = defaultdict(list)
    for e in envelopes:
        domains[e.domain].append(e)

    lines: list[str] = ["## Domain Details", ""]

    for domain in sorted(domains):
        lines.append(f"### {domain.replace('_', ' ').title()}")
        lines.append("")

        for e in domains[domain]:
            ops = ", ".join(e.operators_tested) if e.operators_tested else "N/A"
            lines.append(f"**{ops}**")
            lines.append("")

            # Correctness tests
            tests = e.correctness.get("tests", [])
            if tests:
                lines.append("| Test | Value | Status |")
                lines.append("|------|-------|--------|")
                for t in tests:
                    name = t.get("name", "?")
                    value = t.get("value", "?")
                    passed = t.get("passed", True)
                    icon = ":white_check_mark:" if passed else ":x:"
                    lines.append(f"| {name} | {value} | {icon} |")
                lines.append("")

            # Differentiability
            grad_norm = e.differentiability.get("gradient_norm", 0)
            grad_ok = e.differentiability.get("gradient_nonzero", False)
            icon = ":white_check_mark:" if grad_ok else ":x:"
            lines.append(f"Gradient norm: {grad_norm:.4f} {icon}")
            lines.append("")

            # Performance
            tp = e.performance.get("throughput", 0)
            unit = e.performance.get("throughput_unit", "")
            latency = e.performance.get("latency_ms", 0)
            lines.append(f"Throughput: {tp:.1f} {unit} ({latency:.1f} ms/item)")
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def generate_report(
    results_dir: Path = Path("benchmarks/results"),
) -> str:
    """Generate the complete Markdown report.

    Args:
        results_dir: Directory containing benchmark result JSONs.

    Returns:
        Complete Markdown report string.
    """
    envelopes = _collect_results(results_dir)
    info = collect_platform_info()

    sections = [
        _generate_header(info),
        _generate_capabilities_matrix(envelopes),
        _generate_task_coverage(envelopes),
        _generate_domain_details(envelopes),
    ]

    return "\n".join(sections)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark report for docs"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/development/benchmark-results.md"),
        help="Output Markdown file path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    report = generate_report(args.results_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written to: {args.output}")
    print(f"  {len(report)} characters, {report.count(chr(10))} lines")


if __name__ == "__main__":
    main()
