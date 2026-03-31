"""Tests for PancreasSource data source.

Unit tests use mocks; integration tests require the real h5ad file.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from diffbio.sources.pancreas import PancreasConfig, PancreasSource

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scvelo")
_DATA_EXISTS = (_DATA_DIR / "endocrinogenesis_day15.h5ad").exists()


class TestPancreasConfig:
    """Tests for PancreasConfig validation."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Config raises FileNotFoundError when h5ad is absent."""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            PancreasConfig(data_dir=str(tmp_path))

    def test_frozen(self, tmp_path: Path) -> None:
        """Config is a frozen dataclass."""
        # Can't easily test without a real file, but we can test
        # that a successfully-created config is immutable.
        # We must skip if data doesn't exist.
        if not _DATA_EXISTS:
            pytest.skip("Dataset not available")
        config = PancreasConfig(data_dir=str(_DATA_DIR))
        with pytest.raises(AttributeError):
            config.data_dir = "/other"  # type: ignore[misc]


@pytest.mark.skipif(not _DATA_EXISTS, reason="Pancreas h5ad not available")
class TestPancreasSourceIntegration:
    """Integration tests requiring the real pancreas dataset."""

    @pytest.fixture()
    def source(self) -> PancreasSource:
        """Load pancreas with small subsample for speed."""
        config = PancreasConfig(data_dir=str(_DATA_DIR), subsample=50)
        return PancreasSource(config)

    def test_load_returns_dict(self, source: PancreasSource) -> None:
        """load() returns a dictionary."""
        data = source.load()
        assert isinstance(data, dict)

    def test_required_keys_present(self, source: PancreasSource) -> None:
        """Returned dict has all required keys."""
        data = source.load()
        required = {
            "counts",
            "spliced",
            "unspliced",
            "cell_type_labels",
            "embeddings",
        }
        assert required.issubset(data.keys())

    def test_shapes_consistent(self, source: PancreasSource) -> None:
        """counts, spliced, unspliced share (n_cells, n_genes) shape."""
        data = source.load()
        n_cells = data["n_cells"]
        n_genes = data["n_genes"]
        assert data["counts"].shape == (n_cells, n_genes)
        assert data["spliced"].shape == (n_cells, n_genes)
        assert data["unspliced"].shape == (n_cells, n_genes)

    def test_spliced_non_negative(self, source: PancreasSource) -> None:
        """Spliced counts are non-negative."""
        data = source.load()
        assert float(jnp.min(data["spliced"])) >= 0.0

    def test_unspliced_non_negative(self, source: PancreasSource) -> None:
        """Unspliced counts are non-negative."""
        data = source.load()
        assert float(jnp.min(data["unspliced"])) >= 0.0

    def test_subsample_limits_cells(self, source: PancreasSource) -> None:
        """Subsample reduces cell count to requested size."""
        data = source.load()
        assert data["n_cells"] == 50

    def test_embeddings_shape(self, source: PancreasSource) -> None:
        """Embeddings have (n_cells, dim) shape."""
        data = source.load()
        assert data["embeddings"].shape[0] == data["n_cells"]
        assert data["embeddings"].ndim == 2

    def test_cell_type_labels_length(self, source: PancreasSource) -> None:
        """Cell type labels array matches n_cells."""
        data = source.load()
        assert len(data["cell_type_labels"]) == data["n_cells"]

    def test_len_matches_n_cells(self, source: PancreasSource) -> None:
        """__len__ returns n_cells."""
        data = source.load()
        assert len(source) == data["n_cells"]
