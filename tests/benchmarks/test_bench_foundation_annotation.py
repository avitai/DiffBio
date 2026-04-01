"""Tests for the single-cell foundation annotation benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_foundation_annotation import (
    SingleCellFoundationAnnotationBenchmark,
)
from diffbio.operators.foundation_models import (
    GeneformerPrecomputedAdapter,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result


class _SyntheticSource:
    """Minimal source wrapper for benchmark tests."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def load(self) -> dict[str, Any]:
        return self._data


def _make_synthetic_singlecell_data() -> dict[str, Any]:
    """Create a linearly separable single-cell dataset."""
    rng = np.random.default_rng(42)
    n_types = 3
    cells_per_type = 6
    n_cells = n_types * cells_per_type
    n_features = 4

    class_centers = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.repeat(np.arange(n_types, dtype=np.int32), cells_per_type)
    embeddings = np.vstack(
        [class_centers[label] + rng.normal(scale=0.05, size=n_features) for label in labels]
    ).astype(np.float32)

    return {
        "counts": jnp.asarray(rng.poisson(lam=3.0, size=(n_cells, 8)).astype(np.float32)),
        "batch_labels": np.repeat(np.array([0, 1], dtype=np.int32), n_cells // 2),
        "cell_type_labels": labels,
        "cell_ids": [f"cell_{index}" for index in range(n_cells)],
        "embeddings": jnp.asarray(embeddings),
        "gene_names": [f"Gene_{index}" for index in range(8)],
        "n_cells": n_cells,
        "n_genes": 8,
        "n_batches": 2,
        "n_types": n_types,
    }


class TestSingleCellFoundationAnnotationBenchmark:
    """Tests for the single-cell foundation annotation benchmark."""

    def test_standard_contract_with_synthetic_source(self) -> None:
        bench = SingleCellFoundationAnnotationBenchmark(
            quick=True,
            source_factory=lambda subsample: _SyntheticSource(_make_synthetic_singlecell_data()),
        )

        result = bench.run()

        assert_valid_benchmark_result(
            result,
            expected_name="singlecell/foundation_annotation",
            required_metric_keys=["accuracy", "macro_f1", "train_loss"],
        )
        assert result.tags["task"] == "cell_annotation"
        assert result.tags["dataset"] == "immune_human"
        assert result.metrics["accuracy"].value >= 0.95
        assert result.metrics["macro_f1"].value >= 0.95
        assert result.metadata["baseline_families"] == [
            "diffbio_native",
            "geneformer_precomputed",
            "scgpt_precomputed",
        ]
        assert result.metadata["suite_scenarios"] == {
            "cell_annotation": "singlecell/foundation_annotation",
            "batch_correction": "singlecell/batch_correction",
            "grn_transfer": "singlecell/grn",
        }
        assert result.metadata["dataset_contract_keys"] == [
            "counts",
            "batch_labels",
            "cell_type_labels",
            "cell_ids",
            "embeddings",
            "gene_names",
        ]

    def test_external_embedding_artifact_is_aligned_and_tagged(self, tmp_path: Path) -> None:
        data = _make_synthetic_singlecell_data()
        shuffled_indices = [4, 5, 0, 1, 2, 3, 10, 11, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17]
        artifact_path = tmp_path / "geneformer_embeddings.npz"
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

        bench = SingleCellFoundationAnnotationBenchmark(
            quick=True,
            source_factory=lambda subsample: _SyntheticSource(data),
            embedding_adapter=adapter,
        )

        result = bench.run()

        assert isinstance(result, BenchmarkResult)
        assert result.tags["model_family"] == "single_cell_transformer"
        assert result.tags["adapter_mode"] == "precomputed"
        assert result.tags["artifact_id"] == "geneformer.v1"
        assert result.tags["preprocessing_version"] == "rank_value_v1"
        assert result.metrics["accuracy"].value >= 0.95
