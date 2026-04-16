"""Tests for shared single-cell foundation embedding sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from datarax.core.data_source import DataSourceModule

from diffbio.sources.singlecell_foundation import (
    align_singlecell_embeddings,
    load_singlecell_embedding_source,
)


class TestLoadSingleCellEmbeddingSource:
    """Tests for loading single-cell embedding sources."""

    def test_load_npz_with_cell_ids(self, tmp_path: Path) -> None:
        embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
        cell_ids = np.array(["cell_a", "cell_b", "cell_c"])
        path = tmp_path / "artifact.npz"
        np.savez(path, embeddings=embeddings, cell_ids=cell_ids)

        source = load_singlecell_embedding_source(path)

        assert isinstance(source, DataSourceModule)
        np.testing.assert_allclose(np.asarray(source.embeddings), embeddings)
        assert source.cell_ids == ("cell_a", "cell_b", "cell_c")

    def test_load_pt_with_cell_ids(self, tmp_path: Path) -> None:
        """Load a ``.pt`` artifact that carries embeddings plus ``cell_ids``."""
        torch = pytest.importorskip("torch")
        embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = tmp_path / "artifact.pt"
        torch.save(
            {
                "embeddings": torch.from_numpy(embeddings),
                "cell_ids": ["cell_a", "cell_b", "cell_c"],
            },
            path,
        )

        source = load_singlecell_embedding_source(path)

        np.testing.assert_allclose(np.asarray(source.embeddings), embeddings)
        assert source.cell_ids == ("cell_a", "cell_b", "cell_c")

    def test_rejects_cell_id_length_mismatch(self, tmp_path: Path) -> None:
        """Reject artifacts whose ``cell_ids`` count does not match row count."""
        embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = tmp_path / "artifact.npz"
        np.savez(path, embeddings=embeddings, cell_ids=np.array(["cell_a", "cell_b"]))

        with pytest.raises(ValueError, match="cell_ids must contain one value per embedding row"):
            load_singlecell_embedding_source(path)


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

    def test_reorders_pt_embeddings_to_reference_cell_order(self, tmp_path: Path) -> None:
        """Reorder ``.pt`` artifacts using persisted ``cell_ids`` metadata."""
        torch = pytest.importorskip("torch")
        embeddings = np.array(
            [
                [30.0, 31.0],
                [10.0, 11.0],
                [20.0, 21.0],
            ],
            dtype=np.float32,
        )
        path = tmp_path / "shuffled_embeddings.pt"
        torch.save(
            {
                "embeddings": torch.from_numpy(embeddings),
                "cell_ids": ["cell_c", "cell_a", "cell_b"],
            },
            path,
        )

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
