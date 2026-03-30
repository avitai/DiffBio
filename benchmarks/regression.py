"""Regression detection for DiffBio benchmarks.

Compares current benchmark results against a saved baseline and reports
any regressions in correctness, differentiability, or throughput.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from benchmarks.schema import BenchmarkEnvelope

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class RegressionThresholds:
    """Thresholds for detecting regressions.

    Attributes:
        correctness_must_pass: If True, a test flipping pass->fail is a regression.
        throughput_drop_pct: Percentage throughput drop that triggers a regression.
        gradient_must_stay_nonzero: If True, gradient going to zero is a regression.
    """

    correctness_must_pass: bool = True
    throughput_drop_pct: float = 10.0
    gradient_must_stay_nonzero: bool = True


@dataclass(frozen=True, kw_only=True)
class RegressionReport:
    """A single detected regression.

    Attributes:
        benchmark_id: The benchmark that regressed.
        metric_name: Which metric regressed.
        baseline_value: The baseline value.
        current_value: The current value.
        change_pct: Percentage change (negative = decrease).
        severity: ``"error"`` (blocking) or ``"warning"`` (non-blocking).
        message: Human-readable description.
    """

    benchmark_id: str
    metric_name: str
    baseline_value: float
    current_value: float
    change_pct: float
    severity: str  # "error" | "warning"
    message: str


def detect_regressions(
    current: list[BenchmarkEnvelope],
    baseline: list[BenchmarkEnvelope],
    thresholds: RegressionThresholds | None = None,
) -> list[RegressionReport]:
    """Compare current results against baseline and detect regressions.

    Only compares envelopes with matching ``benchmark_id``. Envelopes in
    *current* without a baseline match are silently skipped (new benchmarks
    cannot regress).

    Args:
        current: Current benchmark results.
        baseline: Baseline results to compare against.
        thresholds: Detection thresholds. Defaults to
            :class:`RegressionThresholds` defaults.

    Returns:
        List of detected regressions, possibly empty.
    """
    if thresholds is None:
        thresholds = RegressionThresholds()

    baseline_map: dict[str, BenchmarkEnvelope] = {e.benchmark_id: e for e in baseline}
    regressions: list[RegressionReport] = []

    for cur in current:
        base = baseline_map.get(cur.benchmark_id)
        if base is None:
            continue

        # Correctness regression
        if thresholds.correctness_must_pass:
            base_passed = base.correctness.get("passed", True)
            cur_passed = cur.correctness.get("passed", True)
            if base_passed and not cur_passed:
                regressions.append(
                    RegressionReport(
                        benchmark_id=cur.benchmark_id,
                        metric_name="correctness",
                        baseline_value=1.0,
                        current_value=0.0,
                        change_pct=-100.0,
                        severity="error",
                        message=f"{cur.benchmark_id}: correctness flipped from PASS to FAIL",
                    )
                )

        # Throughput regression
        base_tp = base.performance.get("throughput", 0)
        cur_tp = cur.performance.get("throughput", 0)
        if base_tp > 0:
            change_pct = ((cur_tp - base_tp) / base_tp) * 100
            if change_pct < -thresholds.throughput_drop_pct:
                regressions.append(
                    RegressionReport(
                        benchmark_id=cur.benchmark_id,
                        metric_name="throughput",
                        baseline_value=float(base_tp),
                        current_value=float(cur_tp),
                        change_pct=change_pct,
                        severity="error",
                        message=(
                            f"{cur.benchmark_id}: throughput dropped "
                            f"{abs(change_pct):.1f}% ({base_tp:.1f} -> {cur_tp:.1f})"
                        ),
                    )
                )

        # Gradient regression
        if thresholds.gradient_must_stay_nonzero:
            base_nonzero = base.differentiability.get("gradient_nonzero", True)
            cur_nonzero = cur.differentiability.get("gradient_nonzero", True)
            if base_nonzero and not cur_nonzero:
                regressions.append(
                    RegressionReport(
                        benchmark_id=cur.benchmark_id,
                        metric_name="gradient_nonzero",
                        baseline_value=1.0,
                        current_value=0.0,
                        change_pct=-100.0,
                        severity="error",
                        message=f"{cur.benchmark_id}: gradient went to zero (was nonzero)",
                    )
                )

    return regressions


def save_baseline(
    envelopes: list[BenchmarkEnvelope],
    baselines_dir: Path = Path("benchmarks/baselines"),
) -> Path:
    """Save current results as a baseline snapshot.

    Writes an aggregated JSON file containing all envelopes.

    Args:
        envelopes: List of current benchmark envelopes.
        baselines_dir: Directory to store baselines.

    Returns:
        Path to the saved baseline file.
    """
    baselines_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d")
    path = baselines_dir / f"baseline_{ts}.json"

    data = [asdict(e) for e in envelopes]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Saved baseline with %d envelopes to %s", len(envelopes), path)
    return path


def load_baseline(path: Path) -> list[BenchmarkEnvelope]:
    """Load a baseline snapshot from a JSON file.

    Args:
        path: Path to the baseline JSON.

    Returns:
        List of envelopes from the baseline.
    """
    with open(path) as f:
        data = json.load(f)

    return [
        BenchmarkEnvelope(**{k: v for k, v in d.items() if k != "schema_version"})
        for d in data
    ]
