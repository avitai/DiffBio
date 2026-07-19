"""Tests for the frozen single-cell annotation baseline benchmark (T01 harness)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from benchmarks.singlecell.bench_frozen_annotation import (
    FrozenAnnotationBaselineBenchmark,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result


class _SyntheticSource:
    """Minimal in-memory source wrapper for benchmark tests."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def load(self) -> dict[str, Any]:
        return self._data


def _make_separable_singlecell_payload(seed: int = 0) -> dict[str, Any]:
    """Create an ImmuneHumanSource-shaped payload with separable count blocks."""
    rng = np.random.default_rng(seed)
    n_types = 4
    cells_per_type = 20
    n_genes = 100
    genes_per_type = n_genes // n_types

    count_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for type_index in range(n_types):
        block = rng.poisson(1.0, size=(cells_per_type, n_genes)).astype(np.float32)
        low = type_index * genes_per_type
        block[:, low : low + genes_per_type] += rng.poisson(
            20.0, size=(cells_per_type, genes_per_type)
        )
        count_blocks.append(block)
        label_blocks.append(np.full(cells_per_type, type_index, dtype=np.int32))

    counts = np.concatenate(count_blocks)
    labels = np.concatenate(label_blocks)
    return {
        "counts": counts,
        "cell_type_labels": labels,
        "n_cells": int(counts.shape[0]),
        "n_genes": n_genes,
        "n_types": n_types,
    }


class TestFrozenAnnotationBaselineBenchmark:
    """Tests for the frozen classic-preprocessing annotation benchmark."""

    def _benchmark(self, seed: int) -> FrozenAnnotationBaselineBenchmark:
        data = _make_separable_singlecell_payload(seed=seed)
        return FrozenAnnotationBaselineBenchmark(
            quick=True,
            source_factory=lambda _subsample: _SyntheticSource(data),
            n_top_genes=40,
            n_components=10,
        )

    def test_standard_contract_with_synthetic_source(self) -> None:
        result = self._benchmark(seed=0).run()

        assert_valid_benchmark_result(
            result,
            expected_name="singlecell/frozen_annotation",
            required_metric_keys=["macro_f1", "balanced_accuracy"],
        )
        assert result.tags["task"] == "cell_annotation"
        assert result.tags["dataset"] == "immune_human"
        assert result.metrics["macro_f1"].value > 0.8

    def test_reports_gradient_flow_through_probe(self) -> None:
        result = self._benchmark(seed=1).run()
        assert result.metrics["gradient_nonzero"].value == 1.0

    def test_runs_against_real_immune_human_source(self, tmp_path: Path) -> None:
        payload = _make_separable_singlecell_payload(seed=0)
        counts = payload["counts"]
        labels = payload["cell_type_labels"]
        n_cells = int(counts.shape[0])

        adata = ad.AnnData(X=counts)
        adata.obs["batch"] = ["b0" if index % 2 == 0 else "b1" for index in range(n_cells)]
        adata.obs["final_annotation"] = [f"type_{int(label)}" for label in labels]
        adata.obsm["X_pca"] = np.zeros((n_cells, 10), dtype=np.float32)
        adata.write_h5ad(tmp_path / "Immune_ALL_human.h5ad")

        bench = FrozenAnnotationBaselineBenchmark(
            quick=True,
            data_dir=str(tmp_path),
            n_top_genes=40,
            n_components=10,
        )
        result = bench.run()

        assert_valid_benchmark_result(
            result,
            expected_name="singlecell/frozen_annotation",
            required_metric_keys=["macro_f1", "balanced_accuracy"],
        )
        assert result.tags["dataset"] == "immune_human"
        assert result.metrics["macro_f1"].value > 0.8
