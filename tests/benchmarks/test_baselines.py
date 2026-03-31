"""Tests for benchmarks._baselines.scib published baselines.

TDD: Verify baseline data structure before implementation.
"""

from __future__ import annotations


from benchmarks._baselines.scib import INTEGRATION_BASELINES


class TestIntegrationBaselines:
    """Tests for the published scib integration baselines."""

    def test_immune_human_key_exists(self) -> None:
        assert "immune_human" in INTEGRATION_BASELINES

    def test_has_at_least_3_methods(self) -> None:
        methods = INTEGRATION_BASELINES["immune_human"]
        assert len(methods) >= 3

    def test_each_method_has_aggregate(self) -> None:
        for name, point in INTEGRATION_BASELINES["immune_human"].items():
            assert "aggregate_score" in point.metrics, f"{name} missing aggregate_score"

    def test_each_method_has_bio_metrics(self) -> None:
        for name, point in INTEGRATION_BASELINES["immune_human"].items():
            assert "silhouette_label" in point.metrics, f"{name} missing silhouette_label"

    def test_values_are_calibrax_metrics(self) -> None:
        from calibrax.core.models import Metric  # noqa: PLC0415

        for name, point in INTEGRATION_BASELINES["immune_human"].items():
            for metric_name, metric in point.metrics.items():
                assert isinstance(metric, Metric), f"{name}/{metric_name} is not a Metric"

    def test_values_in_valid_range(self) -> None:
        for name, point in INTEGRATION_BASELINES["immune_human"].items():
            for metric_name, metric in point.metrics.items():
                assert 0.0 <= metric.value <= 1.0, (
                    f"{name}/{metric_name}={metric.value} out of [0,1]"
                )

    def test_has_unintegrated_baseline(self) -> None:
        methods = INTEGRATION_BASELINES["immune_human"]
        assert "Unintegrated" in methods

    def test_has_source_tag(self) -> None:
        for name, point in INTEGRATION_BASELINES["immune_human"].items():
            assert "source" in point.tags, f"{name} missing source tag"
