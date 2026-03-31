"""Tests for benchmarks._metrics.scib_bridge.

TDD: These tests define the expected behavior of the scib-metrics
bridge before implementation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks._metrics.scib_bridge import evaluate_integration


@pytest.fixture()
def synthetic_integration_data() -> dict[str, np.ndarray]:
    """Create synthetic data with known cluster + batch structure."""
    rng = np.random.default_rng(42)
    n_cells = 200
    n_features = 20
    n_types = 4
    n_batches = 3

    # Generate embeddings with cluster structure
    centers = rng.normal(size=(n_types, n_features)) * 3.0
    labels = np.repeat(np.arange(n_types), n_cells // n_types)
    embeddings = centers[labels] + rng.normal(size=(n_cells, n_features)) * 0.5

    # Assign batches
    batch = np.tile(np.arange(n_batches), n_cells // n_batches + 1)[:n_cells]

    return {
        "embeddings": embeddings.astype(np.float32),
        "labels": labels.astype(np.int32),
        "batch": batch.astype(np.int32),
    }


class TestEvaluateIntegration:
    """Tests for the evaluate_integration bridge function."""

    def test_returns_dict(self, synthetic_integration_data: dict[str, np.ndarray]) -> None:
        d = synthetic_integration_data
        result = evaluate_integration(d["embeddings"], d["labels"], d["batch"])
        assert isinstance(result, dict)

    def test_required_metric_keys(self, synthetic_integration_data: dict[str, np.ndarray]) -> None:
        d = synthetic_integration_data
        result = evaluate_integration(d["embeddings"], d["labels"], d["batch"])
        required_keys = {
            "silhouette_label",
            "silhouette_batch",
            "nmi_kmeans",
            "ari_kmeans",
            "ilisi",
            "clisi",
            "graph_connectivity",
            "isolated_labels",
            "bio_score",
            "batch_score",
            "aggregate_score",
        }
        assert required_keys.issubset(result.keys())

    def test_values_are_finite_floats(
        self, synthetic_integration_data: dict[str, np.ndarray]
    ) -> None:
        d = synthetic_integration_data
        result = evaluate_integration(d["embeddings"], d["labels"], d["batch"])
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float"
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_aggregate_is_weighted_average(
        self, synthetic_integration_data: dict[str, np.ndarray]
    ) -> None:
        d = synthetic_integration_data
        result = evaluate_integration(d["embeddings"], d["labels"], d["batch"])
        # aggregate = 0.6 * bio + 0.4 * batch
        expected = 0.6 * result["bio_score"] + 0.4 * result["batch_score"]
        assert abs(result["aggregate_score"] - expected) < 1e-6

    def test_silhouette_label_positive_for_good_clusters(
        self, synthetic_integration_data: dict[str, np.ndarray]
    ) -> None:
        d = synthetic_integration_data
        result = evaluate_integration(d["embeddings"], d["labels"], d["batch"])
        # Well-separated clusters should give positive silhouette
        assert result["silhouette_label"] > 0.0

    def test_accepts_jax_arrays(self, synthetic_integration_data: dict[str, np.ndarray]) -> None:
        d = synthetic_integration_data
        result = evaluate_integration(
            jnp.array(d["embeddings"]),
            d["labels"],
            d["batch"],
        )
        assert "aggregate_score" in result
