"""Terminal dashboard rendering for DiffBio benchmarks.

Uses ``tabulate`` to render capabilities matrices, regression tables,
and summary lines from :class:`BenchmarkEnvelope` results.
"""

from __future__ import annotations

from benchmarks.schema import BenchmarkEnvelope

# All 8 evaluation task types from scBench / spatialBench
_ALL_EVAL_TASKS = [
    "qc_filtering",
    "clustering",
    "differential_expression",
    "batch_correction",
    "normalization",
    "trajectory",
    "spatial_analysis",
    "cell_annotation",
]


def render_capabilities_matrix(envelopes: list[BenchmarkEnvelope]) -> str:
    """Render a capabilities matrix table.

    Columns: Domain, Operator, Correct, Diff, Status, Throughput.

    Args:
        envelopes: List of benchmark results to display.

    Returns:
        Formatted table string.
    """
    from tabulate import tabulate  # noqa: PLC0415

    if not envelopes:
        return "(no benchmark results)"

    rows: list[list[str]] = []
    for e in sorted(envelopes, key=lambda x: (x.domain, x.benchmark_id)):
        # Correctness summary
        tests = e.correctness.get("tests", [])
        n_passed = sum(1 for t in tests if t.get("passed", True))
        n_total = len(tests)
        correctness_str = f"{n_passed}/{n_total}" if n_total > 0 else "-"

        # Differentiability
        diff_ok = e.differentiability.get("gradient_nonzero", False)
        diff_str = "PASS" if diff_ok else "FAIL"

        # Status
        status_str = e.status.upper()

        # Throughput
        perf = e.performance
        tp = perf.get("throughput", 0)
        unit = perf.get("throughput_unit", "")
        if tp >= 1000:
            throughput_str = f"{tp / 1000:.1f}k {unit}"
        else:
            throughput_str = f"{tp:.1f} {unit}"

        operators = ", ".join(e.operators_tested[:2])
        if len(e.operators_tested) > 2:
            operators += f" +{len(e.operators_tested) - 2}"

        rows.append([e.domain, operators, correctness_str, diff_str, status_str, throughput_str])

    headers = ["Domain", "Operator", "Correct", "Diff", "Status", "Throughput"]
    return tabulate(rows, headers=headers, tablefmt="grid")


def render_summary_line(
    envelopes: list[BenchmarkEnvelope],
    elapsed_seconds: float = 0.0,
) -> str:
    """Render a one-line summary of benchmark results.

    Args:
        envelopes: List of benchmark results.
        elapsed_seconds: Total wall time for the benchmark run.

    Returns:
        Summary string like ``"PASS: 42/48 | FAIL: 0 | ERROR: 6 | TIME: 4m 23s"``.
    """
    n_pass = sum(1 for e in envelopes if e.status == "pass")
    n_fail = sum(1 for e in envelopes if e.status == "fail")
    n_error = sum(1 for e in envelopes if e.status == "error")
    n_total = len(envelopes)

    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

    return f"PASS: {n_pass}/{n_total} | FAIL: {n_fail} | ERROR: {n_error} | TIME: {time_str}"


def render_task_coverage_table(envelopes: list[BenchmarkEnvelope]) -> str:
    """Render a table showing scBench/spatialBench task type coverage.

    Shows which evaluation task types are covered by at least one
    benchmark.

    Args:
        envelopes: List of benchmark results.

    Returns:
        Formatted table string.
    """
    from tabulate import tabulate  # noqa: PLC0415

    # Collect all covered task types
    covered: dict[str, list[str]] = {}
    for e in envelopes:
        for task in e.evaluation_task_types:
            if task not in covered:
                covered[task] = []
            covered[task].extend(e.operators_tested)

    rows: list[list[str]] = []
    for task in _ALL_EVAL_TASKS:
        if task in covered:
            operators = ", ".join(sorted(set(covered[task]))[:3])
            status = "YES"
        else:
            operators = "-"
            status = "NO"
        rows.append([task, operators, status])

    headers = ["Task Type", "DiffBio Operators", "Ready"]
    return tabulate(rows, headers=headers, tablefmt="grid")


def render_full_dashboard(
    envelopes: list[BenchmarkEnvelope],
    elapsed_seconds: float = 0.0,
) -> str:
    """Render the complete dashboard combining all sections.

    Args:
        envelopes: List of benchmark results.
        elapsed_seconds: Total wall time.

    Returns:
        Complete dashboard string ready for printing.
    """
    from benchmarks._common import collect_platform_info  # noqa: PLC0415

    info = collect_platform_info()
    width = 80

    lines: list[str] = []
    lines.append("=" * width)
    lines.append("DiffBio Benchmark Dashboard".center(width))
    lines.append(
        f"{info.get('diffbio_version', '?')} | {info.get('device', '?')}".center(width)
    )
    lines.append("=" * width)
    lines.append("")
    lines.append("CAPABILITIES MATRIX")
    lines.append(render_capabilities_matrix(envelopes))
    lines.append("")
    lines.append("EVALUATION TASK COVERAGE (scBench / spatialBench)")
    lines.append(render_task_coverage_table(envelopes))
    lines.append("")
    lines.append(render_summary_line(envelopes, elapsed_seconds))
    lines.append("=" * width)

    return "\n".join(lines)
