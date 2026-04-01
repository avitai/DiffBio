"""Tests for shared single-cell foundation comparison harnesses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from benchmarks.singlecell.bench_foundation_annotation import (
    build_foundation_annotation_report,
    run_foundation_annotation_suite,
)
from diffbio.operators.foundation_models import (
    GeneformerPrecomputedAdapter,
    ScGPTPrecomputedAdapter,
)


class _SyntheticSource:
    """Minimal source wrapper for comparison-harness tests."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def load(self) -> dict[str, Any]:
        return self._data


def _make_synthetic_singlecell_data() -> dict[str, Any]:
    """Create a linearly separable synthetic single-cell dataset."""
    rng = np.random.default_rng(7)
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


class TestSingleCellFoundationComparisonHarness:
    """Tests for reproducible multi-model comparison reports."""

    def test_report_is_reproducible_and_artifact_aware(self, tmp_path: Path) -> None:
        data = _make_synthetic_singlecell_data()
        reversed_indices = np.arange(data["n_cells"] - 1, -1, -1)

        geneformer_path = tmp_path / "geneformer_embeddings.npz"
        np.savez(
            geneformer_path,
            embeddings=np.asarray(data["embeddings"])[reversed_indices],
            cell_ids=np.asarray(data["cell_ids"])[reversed_indices],
        )
        scgpt_path = tmp_path / "scgpt_embeddings.npz"
        np.savez(
            scgpt_path,
            embeddings=np.asarray(data["embeddings"])[reversed_indices],
            cell_ids=np.asarray(data["cell_ids"])[reversed_indices],
        )

        adapters = {
            "geneformer_precomputed": GeneformerPrecomputedAdapter(
                artifact_path=geneformer_path,
                artifact_id="geneformer.v1",
                preprocessing_version="rank_value_v1",
            ),
            "scgpt_precomputed": ScGPTPrecomputedAdapter(
                artifact_path=scgpt_path,
                artifact_id="scgpt.v1",
                preprocessing_version="gene_vocab_v1",
            ),
        }

        results_a = run_foundation_annotation_suite(
            quick=True,
            source_factory=lambda subsample: _SyntheticSource(data),
            adapters=adapters,
        )
        results_b = run_foundation_annotation_suite(
            quick=True,
            source_factory=lambda subsample: _SyntheticSource(data),
            adapters=adapters,
        )

        report_a = build_foundation_annotation_report(results_a)
        report_b = build_foundation_annotation_report(results_b)

        assert report_a == report_b
        assert tuple(report_a["model_order"]) == (
            "diffbio_native",
            "geneformer_precomputed",
            "scgpt_precomputed",
        )
        assert (
            report_a["models"]["geneformer_precomputed"]["tags"]["artifact_id"]
            == "geneformer.v1"
        )
        assert report_a["models"]["scgpt_precomputed"]["tags"]["artifact_id"] == "scgpt.v1"
        assert report_a["models"]["scgpt_precomputed"]["tags"]["preprocessing_version"] == (
            "gene_vocab_v1"
        )
