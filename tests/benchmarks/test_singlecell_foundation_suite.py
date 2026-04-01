"""Tests for the full single-cell foundation benchmark suite harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from benchmarks.singlecell.foundation_suite import (
    build_singlecell_foundation_suite_report,
    run_singlecell_foundation_suite,
)
from diffbio.operators.foundation_models import (
    GeneformerPrecomputedAdapter,
    ScGPTPrecomputedAdapter,
)


class _SyntheticSource:
    """Minimal source wrapper for suite tests."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def load(self) -> dict[str, Any]:
        return self._data


def _make_synthetic_singlecell_data() -> dict[str, Any]:
    """Create a linearly separable synthetic single-cell dataset."""
    rng = np.random.default_rng(19)
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


class TestSingleCellFoundationSuiteHarness:
    """Tests for the quick single-cell suite across all three model families."""

    def test_suite_report_is_reproducible_and_task_aware(
        self,
        tmp_path: Path,
        monkeypatch: Any,
    ) -> None:
        data = _make_synthetic_singlecell_data()
        shuffled_indices = np.arange(data["n_cells"] - 1, -1, -1)

        geneformer_path = tmp_path / "geneformer_embeddings.npz"
        np.savez(
            geneformer_path,
            embeddings=np.asarray(data["embeddings"])[shuffled_indices],
            cell_ids=np.asarray(data["cell_ids"])[shuffled_indices],
        )
        scgpt_path = tmp_path / "scgpt_embeddings.npz"
        np.savez(
            scgpt_path,
            embeddings=np.asarray(data["embeddings"])[shuffled_indices],
            cell_ids=np.asarray(data["cell_ids"])[shuffled_indices],
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
                batch_key="batch",
                context_version="obs_batch_v1",
            ),
        }

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

        results_a = run_singlecell_foundation_suite(
            quick=True,
            source_factory=lambda subsample: _SyntheticSource(data),
            adapters=adapters,
        )
        results_b = run_singlecell_foundation_suite(
            quick=True,
            source_factory=lambda subsample: _SyntheticSource(data),
            adapters=adapters,
        )

        report_a = build_singlecell_foundation_suite_report(results_a)
        report_b = build_singlecell_foundation_suite_report(results_b)

        assert report_a == report_b
        assert tuple(report_a["task_order"]) == ("cell_annotation", "batch_correction")
        assert tuple(report_a["tasks"]["cell_annotation"]["model_order"]) == (
            "diffbio_native",
            "geneformer_precomputed",
            "scgpt_precomputed",
        )
        assert tuple(report_a["tasks"]["batch_correction"]["model_order"]) == (
            "diffbio_native",
            "geneformer_precomputed",
            "scgpt_precomputed",
        )
        assert (
            report_a["tasks"]["batch_correction"]["models"]["scgpt_precomputed"]["metadata"][
                "requires_batch_context"
            ]
            is True
        )
        assert (
            report_a["tasks"]["batch_correction"]["models"]["scgpt_precomputed"]["metadata"][
                "batch_key"
            ]
            == "batch"
        )
        assert (
            report_a["tasks"]["batch_correction"]["models"]["scgpt_precomputed"]["metadata"][
                "context_version"
            ]
            == "obs_batch_v1"
        )
