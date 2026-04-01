"""Tests for precomputed single-cell foundation-model adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from diffbio.operators.foundation_models import (
    GeneformerPrecomputedAdapter,
    ScGPTPrecomputedAdapter,
    decode_foundation_text,
)


class TestGeneformerPrecomputedAdapter:
    """Tests for strict Geneformer artifact handling."""

    def test_loads_aligned_embeddings_from_cell_ids(self, tmp_path: Path) -> None:
        embeddings = np.array(
            [
                [3.0, 3.1],
                [1.0, 1.1],
                [2.0, 2.1],
            ],
            dtype=np.float32,
        )
        cell_ids = np.array(["cell_c", "cell_a", "cell_b"])
        artifact_path = tmp_path / "geneformer_embeddings.npz"
        np.savez(artifact_path, embeddings=embeddings, cell_ids=cell_ids)

        adapter = GeneformerPrecomputedAdapter(
            artifact_path=artifact_path,
            artifact_id="geneformer.v1",
            preprocessing_version="rank_value_v1",
        )

        aligned = adapter.load_aligned_embeddings(
            reference_cell_ids=["cell_a", "cell_b", "cell_c"],
        )

        np.testing.assert_allclose(
            np.asarray(aligned),
            np.array(
                [
                    [1.0, 1.1],
                    [2.0, 2.1],
                    [3.0, 3.1],
                ],
                dtype=np.float32,
            ),
        )

    def test_exposes_foundation_model_metadata(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "geneformer_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.ones((2, 3), dtype=np.float32),
            cell_ids=np.array(["cell_a", "cell_b"]),
        )
        adapter = GeneformerPrecomputedAdapter(
            artifact_path=artifact_path,
            artifact_id="geneformer.v1",
            preprocessing_version="rank_value_v1",
        )

        metadata = adapter.result_data()["foundation_model"]

        assert decode_foundation_text(metadata["model_family"]) == "single_cell_transformer"
        assert decode_foundation_text(metadata["adapter_mode"]) == "precomputed"
        assert decode_foundation_text(metadata["artifact_id"]) == "geneformer.v1"
        assert decode_foundation_text(metadata["preprocessing_version"]) == "rank_value_v1"
        assert decode_foundation_text(metadata["pooling_strategy"]) == "mean"


class TestScGPTPrecomputedAdapter:
    """Tests for strict scGPT artifact handling."""

    def test_exposes_foundation_model_metadata(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "scgpt_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.ones((2, 3), dtype=np.float32),
            cell_ids=np.array(["cell_a", "cell_b"]),
        )
        adapter = ScGPTPrecomputedAdapter(
            artifact_path=artifact_path,
            artifact_id="scgpt.v1",
            preprocessing_version="gene_vocab_v1",
        )

        metadata = adapter.result_data()["foundation_model"]

        assert decode_foundation_text(metadata["model_family"]) == "single_cell_transformer"
        assert decode_foundation_text(metadata["adapter_mode"]) == "precomputed"
        assert decode_foundation_text(metadata["artifact_id"]) == "scgpt.v1"
        assert decode_foundation_text(metadata["preprocessing_version"]) == "gene_vocab_v1"
        assert decode_foundation_text(metadata["pooling_strategy"]) == "mean"
