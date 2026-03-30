"""Tests for benchmarks.dashboard terminal rendering."""

from benchmarks.dashboard import (
    render_capabilities_matrix,
    render_summary_line,
    render_task_coverage_table,
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
        "correctness": {"passed": True, "tests": [{"name": "t1", "value": 1.0, "passed": True}]},
        "differentiability": {"passed": True, "gradient_norm": 1.0, "gradient_nonzero": True},
        "performance": {"throughput": 100.0, "throughput_unit": "items/sec", "latency_ms": 10.0},
        "domain_metrics": {},
        "configuration": {},
    }
    defaults.update(overrides)
    return BenchmarkEnvelope(**defaults)


class TestRenderCapabilitiesMatrix:
    """Tests for the capabilities matrix renderer."""

    def test_returns_string(self) -> None:
        envelopes = [_make_envelope()]
        result = render_capabilities_matrix(envelopes)
        assert isinstance(result, str)

    def test_contains_domain_name(self) -> None:
        envelopes = [_make_envelope(domain="alignment")]
        result = render_capabilities_matrix(envelopes)
        assert "alignment" in result.lower()

    def test_contains_pass_status(self) -> None:
        envelopes = [_make_envelope(status="pass")]
        result = render_capabilities_matrix(envelopes)
        assert "PASS" in result

    def test_contains_fail_status(self) -> None:
        envelopes = [_make_envelope(status="fail")]
        result = render_capabilities_matrix(envelopes)
        assert "FAIL" in result

    def test_empty_list(self) -> None:
        result = render_capabilities_matrix([])
        assert isinstance(result, str)


class TestRenderSummaryLine:
    """Tests for the summary line renderer."""

    def test_counts_pass_fail(self) -> None:
        envelopes = [
            _make_envelope(status="pass"),
            _make_envelope(status="pass"),
            _make_envelope(status="fail"),
        ]
        result = render_summary_line(envelopes, elapsed_seconds=5.0)
        assert "2" in result  # 2 passed
        assert "1" in result  # 1 failed

    def test_includes_time(self) -> None:
        result = render_summary_line([_make_envelope()], elapsed_seconds=42.5)
        assert "42" in result


class TestRenderTaskCoverageTable:
    """Tests for the evaluation task coverage table."""

    def test_returns_string(self) -> None:
        envelopes = [_make_envelope(evaluation_task_types=["clustering"])]
        result = render_task_coverage_table(envelopes)
        assert isinstance(result, str)

    def test_shows_covered_task(self) -> None:
        envelopes = [_make_envelope(evaluation_task_types=["clustering"])]
        result = render_task_coverage_table(envelopes)
        assert "clustering" in result

    def test_empty_envelopes(self) -> None:
        result = render_task_coverage_table([])
        assert isinstance(result, str)
