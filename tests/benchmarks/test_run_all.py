"""Tests for the benchmark run aggregation path."""

from __future__ import annotations

from calibrax.core.models import Metric
from calibrax.core.result import BenchmarkResult

from benchmarks.run_all import _build_run


class TestBuildRun:
    """Tests for calibrax Run construction from benchmark results."""

    def test_build_run_preserves_foundation_tags(self) -> None:
        """Foundation-model tags should survive into the aggregated Run points."""
        result = BenchmarkResult(
            name="singlecell/foundation_smoke",
            domain="diffbio_benchmarks",
            tags={
                "framework": "diffbio",
                "operator": "DummyOperator",
                "dataset": "dummy_dataset",
                "task": "foundation_smoke",
                "model_family": "sequence_transformer",
                "adapter_mode": "frozen_encoder",
                "artifact_id": "diffbio.sequence.smoke",
                "preprocessing_version": "one_hot_v1",
            },
            metrics={"score": Metric(value=0.75)},
        )

        run = _build_run([result])
        point = run.points[0]

        assert point.name == "singlecell/foundation_smoke"
        assert point.scenario == "dummy_dataset"
        assert point.tags["task"] == "foundation_smoke"
        assert point.tags["model_family"] == "sequence_transformer"
        assert point.tags["adapter_mode"] == "frozen_encoder"
        assert point.tags["artifact_id"] == "diffbio.sequence.smoke"
        assert point.tags["preprocessing_version"] == "one_hot_v1"
