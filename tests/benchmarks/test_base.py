"""Tests for benchmarks._base DiffBioBenchmark base class.

TDD: Define expected behavior of the base class before implementation.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from diffbio.operators.foundation_models import (
    AdapterMode,
    FoundationArtifactSpec,
    FoundationModelKind,
    PoolingStrategy,
    build_foundation_model_metadata,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result


class TestDiffBioBenchmarkConfig:
    """Tests for the benchmark config dataclass."""

    def test_frozen(self) -> None:
        config = DiffBioBenchmarkConfig(name="test", domain="test_domain")
        with pytest.raises(FrozenInstanceError):
            config.name = "changed"  # type: ignore[misc]

    def test_required_fields(self) -> None:
        config = DiffBioBenchmarkConfig(name="test/bench", domain="test")
        assert config.name == "test/bench"
        assert config.domain == "test"

    def test_quick_subsample_default(self) -> None:
        config = DiffBioBenchmarkConfig(name="test", domain="test")
        assert config.quick_subsample == 2000

    def test_kw_only(self) -> None:
        """Config should require keyword arguments."""
        with pytest.raises(TypeError):
            DiffBioBenchmarkConfig("test", "domain")  # type: ignore[misc]


class _DummyBenchmark(DiffBioBenchmark):
    """Minimal benchmark for testing shared result construction."""

    def _run_core(self) -> dict[str, object]:
        foundation_spec = FoundationArtifactSpec(
            model_family=FoundationModelKind.SEQUENCE_TRANSFORMER,
            artifact_id="diffbio.sequence.smoke",
            preprocessing_version="one_hot_v1",
            adapter_mode=AdapterMode.FROZEN_ENCODER,
            pooling_strategy=PoolingStrategy.CLS,
        )
        return {
            "metrics": {"score": 0.75},
            "operator": object(),
            "input_data": {},
            "loss_fn": lambda model, data: 0.0,
            "n_items": 4,
            "iterate_fn": lambda: None,
            "result_data": {"foundation_model": build_foundation_model_metadata(foundation_spec)},
            "operator_name": "DummyOperator",
            "dataset_name": "dummy_dataset",
        }


class TestDiffBioBenchmarkResultContract:
    """Tests for shared BenchmarkResult construction."""

    def test_base_result_includes_task_tag(self) -> None:
        """Every benchmark result should expose a canonical task tag."""
        bench = _DummyBenchmark(
            DiffBioBenchmarkConfig(name="singlecell/foundation_smoke", domain="singlecell"),
            quick=True,
        )

        result = bench.run()

        assert_valid_benchmark_result(result, expected_name="singlecell/foundation_smoke")
        assert result.tags["task"] == "foundation_smoke"

    def test_base_result_includes_foundation_tags(self) -> None:
        """Foundation metadata should be promoted to benchmark tags and metadata."""
        bench = _DummyBenchmark(
            DiffBioBenchmarkConfig(name="singlecell/foundation_smoke", domain="singlecell"),
            quick=True,
        )

        result = bench.run()

        assert result.tags["model_family"] == "sequence_transformer"
        assert result.tags["adapter_mode"] == "frozen_encoder"
        assert result.tags["artifact_id"] == "diffbio.sequence.smoke"
        assert result.tags["preprocessing_version"] == "one_hot_v1"
        assert result.metadata["foundation_model"]["pooling_strategy"] == "cls"
        assert result.metadata["comparison_axes"] == [
            "dataset",
            "task",
            "model_family",
            "adapter_mode",
            "artifact_id",
            "preprocessing_version",
        ]
