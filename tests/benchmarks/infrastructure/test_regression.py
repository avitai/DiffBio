"""Tests for benchmarks.regression detection."""

from benchmarks.regression import (
    RegressionReport,
    RegressionThresholds,
    detect_regressions,
)
from benchmarks.schema import BenchmarkEnvelope


def _make_envelope(**overrides) -> BenchmarkEnvelope:
    defaults = {
        "benchmark_id": "test/op",
        "domain": "test",
        "operators_tested": ["TestOp"],
        "timestamp": "2026-01-01T00:00:00",
        "platform": {"device": "cpu"},
        "status": "pass",
        "correctness": {"passed": True, "tests": []},
        "differentiability": {"passed": True, "gradient_norm": 1.0, "gradient_nonzero": True},
        "performance": {"throughput": 100.0, "throughput_unit": "items/sec", "latency_ms": 10.0},
        "domain_metrics": {},
        "configuration": {},
    }
    defaults.update(overrides)
    return BenchmarkEnvelope(**defaults)


class TestDetectRegressions:
    """Tests for regression detection logic."""

    def test_no_regressions_when_identical(self) -> None:
        e = _make_envelope()
        result = detect_regressions([e], [e])
        assert result == []

    def test_detects_correctness_regression(self) -> None:
        baseline = _make_envelope(correctness={"passed": True, "tests": []})
        current = _make_envelope(correctness={"passed": False, "tests": []})
        result = detect_regressions([current], [baseline])
        assert len(result) == 1
        assert result[0].severity == "error"
        assert "correctness" in result[0].metric_name

    def test_detects_throughput_regression(self) -> None:
        baseline = _make_envelope(
            performance={"throughput": 100.0, "throughput_unit": "items/sec", "latency_ms": 10.0}
        )
        current = _make_envelope(
            performance={"throughput": 80.0, "throughput_unit": "items/sec", "latency_ms": 12.5}
        )
        thresholds = RegressionThresholds(throughput_drop_pct=10.0)
        result = detect_regressions([current], [baseline], thresholds)
        assert any(r.metric_name == "throughput" for r in result)

    def test_no_regression_for_small_throughput_change(self) -> None:
        baseline = _make_envelope(
            performance={"throughput": 100.0, "throughput_unit": "items/sec", "latency_ms": 10.0}
        )
        current = _make_envelope(
            performance={"throughput": 95.0, "throughput_unit": "items/sec", "latency_ms": 10.5}
        )
        thresholds = RegressionThresholds(throughput_drop_pct=10.0)
        result = detect_regressions([current], [baseline], thresholds)
        throughput_regressions = [r for r in result if r.metric_name == "throughput"]
        assert throughput_regressions == []

    def test_detects_gradient_zero_regression(self) -> None:
        baseline = _make_envelope(
            differentiability={"passed": True, "gradient_norm": 1.0, "gradient_nonzero": True}
        )
        current = _make_envelope(
            differentiability={"passed": False, "gradient_norm": 0.0, "gradient_nonzero": False}
        )
        result = detect_regressions([current], [baseline])
        assert any(r.metric_name == "gradient_nonzero" for r in result)

    def test_matches_by_benchmark_id(self) -> None:
        baseline = _make_envelope(benchmark_id="domain/op1")
        current = _make_envelope(benchmark_id="domain/op2")
        # Different IDs should not be compared
        result = detect_regressions([current], [baseline])
        assert result == []

    def test_empty_lists(self) -> None:
        assert detect_regressions([], []) == []


class TestRegressionThresholds:
    """Tests for threshold configuration."""

    def test_defaults(self) -> None:
        t = RegressionThresholds()
        assert t.correctness_must_pass is True
        assert t.throughput_drop_pct == 10.0
        assert t.gradient_must_stay_nonzero is True

    def test_custom_thresholds(self) -> None:
        t = RegressionThresholds(throughput_drop_pct=5.0)
        assert t.throughput_drop_pct == 5.0


class TestRegressionReport:
    """Tests for the regression report dataclass."""

    def test_frozen(self) -> None:
        import pytest  # noqa: PLC0415

        r = RegressionReport(
            benchmark_id="test/op",
            metric_name="throughput",
            baseline_value=100.0,
            current_value=80.0,
            change_pct=-20.0,
            severity="error",
            message="Throughput dropped by 20%",
        )
        with pytest.raises(AttributeError):
            r.severity = "warning"  # type: ignore[misc]
