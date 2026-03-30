"""Unified result schema for DiffBio benchmarks.

Every benchmark produces a :class:`BenchmarkEnvelope` that wraps
domain-specific metrics in a standardised envelope with common metadata
(platform, correctness, differentiability, performance). This enables
uniform dashboard rendering, regression detection, and report generation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class CorrectnessTest:
    """A single correctness assertion within a benchmark.

    Attributes:
        name: Human-readable test name.
        value: Observed numeric value.
        expected_range: Optional ``(low, high)`` bounds for regression
            detection. ``None`` means no regression check.
        passed: Whether this test passed.
    """

    name: str
    value: float
    expected_range: tuple[float, float] | None = None
    passed: bool = True


@dataclass(frozen=True, kw_only=True)
class BenchmarkEnvelope:
    """Standardised result envelope for a single benchmark run.

    Attributes:
        schema_version: Envelope format version (currently ``"1.0"``).
        benchmark_id: Stable identifier ``"<domain>/<snake_case_operator>"``.
        domain: Operator domain matching ``src/diffbio/operators/<domain>/``.
        operators_tested: List of operator class names exercised.
        timestamp: ISO 8601 datetime of the run.
        platform: Runtime platform info dict.
        status: Overall pass/fail/error verdict.
        correctness: Dict with ``passed`` (bool) and ``tests``
            (list of :class:`CorrectnessTest` dicts).
        differentiability: Dict with ``passed``, ``gradient_norm``,
            ``gradient_nonzero``.
        performance: Dict with ``throughput``, ``throughput_unit``,
            ``latency_ms``.
        domain_metrics: Free-form dict for operator-specific metrics.
        configuration: Operator configuration used in the run.
        evaluation_task_types: scBench/spatialBench task types this
            benchmark covers (e.g. ``["clustering", "trajectory"]``).
    """

    schema_version: str = "1.0"
    benchmark_id: str = ""
    domain: str = ""
    operators_tested: list[str] = field(default_factory=list)
    timestamp: str = ""
    platform: dict[str, str] = field(default_factory=dict)
    status: str = "pass"
    correctness: dict[str, Any] = field(default_factory=dict)
    differentiability: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    domain_metrics: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    evaluation_task_types: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _envelope_to_dict(envelope: BenchmarkEnvelope) -> dict[str, Any]:
    """Convert envelope to a JSON-safe dictionary."""
    return asdict(envelope)


def _dict_to_envelope(data: dict[str, Any]) -> BenchmarkEnvelope:
    """Reconstruct a BenchmarkEnvelope from a parsed JSON dict."""
    # Convert CorrectnessTest dicts back (stored as plain dicts in JSON)
    return BenchmarkEnvelope(**{k: v for k, v in data.items() if k != "schema_version"})


def save_envelope(
    envelope: BenchmarkEnvelope,
    output_dir: Path,
) -> Path:
    """Save a BenchmarkEnvelope as a timestamped JSON file.

    Args:
        envelope: The envelope to save.
        output_dir: Directory to write the JSON file into.

    Returns:
        Path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a filename-safe stem from the benchmark_id
    safe_name = envelope.benchmark_id.replace("/", "_") or "benchmark"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{ts}.json"
    path = output_dir / filename

    with open(path, "w") as f:
        json.dump(_envelope_to_dict(envelope), f, indent=2, default=str)

    return path


def load_envelope(path: Path) -> BenchmarkEnvelope:
    """Load a BenchmarkEnvelope from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed BenchmarkEnvelope.
    """
    with open(path) as f:
        data = json.load(f)
    return _dict_to_envelope(data)


def load_all_envelopes(results_dir: Path) -> list[BenchmarkEnvelope]:
    """Load all envelope JSON files from a directory (non-recursive).

    Skips files that are not valid JSON or do not parse as envelopes.

    Args:
        results_dir: Directory containing JSON result files.

    Returns:
        List of parsed envelopes, sorted by timestamp descending.
    """
    envelopes: list[BenchmarkEnvelope] = []

    if not results_dir.exists():
        return envelopes

    for path in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            envelopes.append(load_envelope(path))
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as exc:
            logger.debug("Skipping %s: %s", path, exc)

    return envelopes
