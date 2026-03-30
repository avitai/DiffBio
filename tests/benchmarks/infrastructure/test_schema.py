"""Tests for benchmarks.schema unified result format."""

import json

import pytest

from benchmarks.schema import (
    BenchmarkEnvelope,
    CorrectnessTest,
    load_all_envelopes,
    load_envelope,
    save_envelope,
)


def _make_envelope(**overrides) -> BenchmarkEnvelope:
    """Create a minimal valid BenchmarkEnvelope for testing."""
    defaults = {
        "benchmark_id": "test/test_operator",
        "domain": "test",
        "operators_tested": ["TestOperator"],
        "timestamp": "2026-01-01T00:00:00",
        "platform": {"device": "cpu", "jax_version": "0.4.0"},
        "status": "pass",
        "correctness": {
            "passed": True,
            "tests": [{"name": "basic", "value": 1.0, "passed": True}],
        },
        "differentiability": {
            "passed": True,
            "gradient_norm": 1.5,
            "gradient_nonzero": True,
        },
        "performance": {
            "throughput": 100.0,
            "throughput_unit": "items/sec",
            "latency_ms": 10.0,
        },
        "domain_metrics": {},
        "configuration": {},
    }
    defaults.update(overrides)
    return BenchmarkEnvelope(**defaults)


class TestCorrectnessTest:
    """Tests for CorrectnessTest dataclass."""

    def test_frozen(self) -> None:
        ct = CorrectnessTest(name="test", value=1.0, passed=True)
        with pytest.raises(AttributeError):
            ct.name = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        ct = CorrectnessTest(name="test", value=1.0)
        assert ct.passed is True
        assert ct.expected_range is None

    def test_with_expected_range(self) -> None:
        ct = CorrectnessTest(name="test", value=1.0, expected_range=(0.5, 1.5))
        assert ct.expected_range == (0.5, 1.5)


class TestBenchmarkEnvelope:
    """Tests for BenchmarkEnvelope dataclass."""

    def test_frozen(self) -> None:
        envelope = _make_envelope()
        with pytest.raises(AttributeError):
            envelope.domain = "changed"  # type: ignore[misc]

    def test_schema_version_default(self) -> None:
        envelope = _make_envelope()
        assert envelope.schema_version == "1.0"

    def test_evaluation_task_types_default(self) -> None:
        envelope = _make_envelope()
        assert envelope.evaluation_task_types == []

    def test_with_evaluation_task_types(self) -> None:
        envelope = _make_envelope(evaluation_task_types=["clustering", "trajectory"])
        assert envelope.evaluation_task_types == ["clustering", "trajectory"]


class TestSaveAndLoadEnvelope:
    """Tests for envelope serialization."""

    def test_round_trip(self, tmp_path) -> None:
        original = _make_envelope()
        path = save_envelope(original, tmp_path)
        loaded = load_envelope(path)
        assert loaded.benchmark_id == original.benchmark_id
        assert loaded.domain == original.domain
        assert loaded.status == original.status
        assert loaded.schema_version == original.schema_version

    def test_saved_file_is_valid_json(self, tmp_path) -> None:
        envelope = _make_envelope()
        path = save_envelope(envelope, tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert data["schema_version"] == "1.0"
        assert data["benchmark_id"] == "test/test_operator"

    def test_file_has_timestamp_in_name(self, tmp_path) -> None:
        envelope = _make_envelope()
        path = save_envelope(envelope, tmp_path)
        assert "test_operator" in path.stem

    def test_creates_output_directory(self, tmp_path) -> None:
        envelope = _make_envelope()
        nested = tmp_path / "sub" / "dir"
        path = save_envelope(envelope, nested)
        assert path.exists()


class TestLoadAllEnvelopes:
    """Tests for batch loading of envelopes."""

    def test_loads_multiple_envelopes(self, tmp_path) -> None:
        e1 = _make_envelope(benchmark_id="domain/op1")
        e2 = _make_envelope(benchmark_id="domain/op2")
        save_envelope(e1, tmp_path)
        save_envelope(e2, tmp_path)
        loaded = load_all_envelopes(tmp_path)
        ids = {e.benchmark_id for e in loaded}
        assert "domain/op1" in ids
        assert "domain/op2" in ids

    def test_empty_directory(self, tmp_path) -> None:
        loaded = load_all_envelopes(tmp_path)
        assert loaded == []

    def test_skips_non_json_files(self, tmp_path) -> None:
        (tmp_path / "readme.txt").write_text("not json")
        envelope = _make_envelope()
        save_envelope(envelope, tmp_path)
        loaded = load_all_envelopes(tmp_path)
        assert len(loaded) == 1
