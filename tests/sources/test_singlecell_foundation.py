"""Tests for shared single-cell foundation embedding sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffbio.sources.singlecell_foundation import (
    align_singlecell_embeddings,
    load_singlecell_embedding_artifact,
)


class TestLoadSingleCellEmbeddingArtifact:
    """Tests for loading single-cell embedding artifacts."""

    def test_load_npz_with_cell_ids(self, tmp_path: Path) -> None:
        embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
        cell_ids = np.array(["cell_a", "cell_b", "cell_c"])
        path = tmp_path / "artifact.npz"
        np.savez(path, embeddings=embeddings, cell_ids=cell_ids)

        artifact = load_singlecell_embedding_artifact(path)

        np.testing.assert_allclose(np.asarray(artifact.embeddings), embeddings)
        assert artifact.cell_ids == ("cell_a", "cell_b", "cell_c")


class TestAlignSingleCellEmbeddings:
    """Tests for strict cell-level embedding alignment."""

    def test_reorders_embeddings_to_reference_cell_order(self, tmp_path: Path) -> None:
        embeddings = np.array(
            [
                [30.0, 31.0],
                [10.0, 11.0],
                [20.0, 21.0],
            ],
            dtype=np.float32,
        )
        cell_ids = np.array(["cell_c", "cell_a", "cell_b"])
        path = tmp_path / "shuffled_embeddings.npz"
        np.savez(path, embeddings=embeddings, cell_ids=cell_ids)

        aligned = align_singlecell_embeddings(
            reference_cell_ids=["cell_a", "cell_b", "cell_c"],
            artifact_path=path,
        )

        np.testing.assert_allclose(
            np.asarray(aligned),
            np.array(
                [
                    [10.0, 11.0],
                    [20.0, 21.0],
                    [30.0, 31.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_requires_cell_ids_by_default(self, tmp_path: Path) -> None:
        embeddings = np.arange(6, dtype=np.float32).reshape(3, 2)
        path = tmp_path / "embeddings.npy"
        np.save(path, embeddings)

        with pytest.raises(ValueError, match="must include cell_ids"):
            align_singlecell_embeddings(
                reference_cell_ids=["cell_a", "cell_b", "cell_c"],
                artifact_path=path,
            )

    def test_rejects_mismatched_cell_id_sets(self, tmp_path: Path) -> None:
        embeddings = np.arange(6, dtype=np.float32).reshape(3, 2)
        cell_ids = np.array(["cell_a", "cell_b", "cell_extra"])
        path = tmp_path / "mismatch.npz"
        np.savez(path, embeddings=embeddings, cell_ids=cell_ids)

        with pytest.raises(ValueError, match="Cell ID mismatch"):
            align_singlecell_embeddings(
                reference_cell_ids=["cell_a", "cell_b", "cell_c"],
                artifact_path=path,
            )

    def test_allows_positional_alignment_only_when_explicit(self, tmp_path: Path) -> None:
        embeddings = np.arange(6, dtype=np.float32).reshape(3, 2)
        path = tmp_path / "embeddings.npy"
        np.save(path, embeddings)

        aligned = align_singlecell_embeddings(
            reference_cell_ids=["cell_a", "cell_b", "cell_c"],
            artifact_path=path,
            require_cell_ids=False,
        )

        np.testing.assert_allclose(np.asarray(aligned), embeddings)
