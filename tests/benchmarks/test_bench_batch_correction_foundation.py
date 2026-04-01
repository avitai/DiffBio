"""Tests for batch-correction benchmark integration with foundation adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_batch_correction import BatchCorrectionBenchmark
from diffbio.operators.foundation_models import GeneformerPrecomputedAdapter
from tests.benchmarks.conftest import assert_valid_benchmark_result


class _SyntheticSource:
    """Minimal source wrapper for batch-correction benchmark tests."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def load(self) -> dict[str, Any]:
        return self._data


def _make_synthetic_batch_data() -> dict[str, Any]:
    """Create a compact synthetic integration dataset."""
    rng = np.random.default_rng(42)
    n_cells = 24
    n_features = 6
    n_genes = 12
    n_types = 3
    n_batches = 2

    embeddings = np.vstack(
        [
            np.full((8, n_features), fill_value=0.0, dtype=np.float32),
            np.full((8, n_features), fill_value=1.0, dtype=np.float32),
            np.full((8, n_features), fill_value=2.0, dtype=np.float32),
        ]
    )
    embeddings += rng.normal(scale=0.05, size=embeddings.shape).astype(np.float32)

    return {
        "counts": jnp.asarray(rng.poisson(lam=4.0, size=(n_cells, n_genes)).astype(np.float32)),
        "batch_labels": np.tile(np.array([0, 1], dtype=np.int32), n_cells // 2),
        "cell_type_labels": np.repeat(np.arange(n_types, dtype=np.int32), 8),
        "cell_ids": [f"cell_{index}" for index in range(n_cells)],
        "embeddings": jnp.asarray(embeddings),
        "gene_names": [f"Gene_{index}" for index in range(n_genes)],
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_batches": n_batches,
        "n_types": n_types,
    }


class TestBatchCorrectionBenchmarkFoundationAdapter:
    """Tests for foundation-model-aware batch correction benchmarking."""

    def test_external_adapter_is_tagged(self, tmp_path: Path, monkeypatch: Any) -> None:
        data = _make_synthetic_batch_data()
        shuffled_indices = np.array(
            [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 15, 16, 17, 12, 13, 14, 21, 22, 23, 18, 19, 20]
        )
        artifact_path = tmp_path / "geneformer_batch_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.asarray(data["embeddings"])[shuffled_indices],
            cell_ids=np.asarray(data["cell_ids"])[shuffled_indices],
        )
        adapter = GeneformerPrecomputedAdapter(
            artifact_path=artifact_path,
            artifact_id="geneformer.v1",
            preprocessing_version="rank_value_v1",
        )

        def _fake_evaluate_integration(
            corrected_embeddings: Any,
            labels: Any,
            batch: Any,
            *,
            n_neighbors: int = 90,
        ) -> dict[str, float]:
            del corrected_embeddings, labels, batch, n_neighbors
            return {
                "aggregate_score": 0.51,
                "silhouette_label": 0.62,
                "nmi_kmeans": 0.57,
                "ari_kmeans": 0.53,
                "clisi": 0.48,
                "isolated_labels": 0.44,
                "silhouette_batch": 0.39,
                "ilisi": 0.41,
                "graph_connectivity": 0.58,
                "bio_score": 0.53,
                "batch_score": 0.46,
            }

        monkeypatch.setattr(
            "benchmarks.singlecell.bench_batch_correction.evaluate_integration",
            _fake_evaluate_integration,
        )

        bench = BatchCorrectionBenchmark(
            quick=True,
            source_factory=lambda subsample: _SyntheticSource(data),
            embedding_adapter=adapter,
        )

        result = bench.run()

        assert isinstance(result, BenchmarkResult)
        assert_valid_benchmark_result(
            result,
            expected_name="singlecell/batch_correction",
            required_metric_keys=["aggregate_score", "silhouette_label", "nmi_kmeans"],
        )
        assert result.tags["model_family"] == "single_cell_transformer"
        assert result.tags["adapter_mode"] == "precomputed"
        assert result.tags["artifact_id"] == "geneformer.v1"
        assert result.metadata["embedding_source"] == "external_artifact"
