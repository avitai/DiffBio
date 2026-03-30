"""Tests for ImmuneHumanSource benchmark DataSource.

TDD: These tests define the expected behavior of ImmuneHumanSource
before implementation.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from diffbio.sources.immune_human import ImmuneHumanConfig, ImmuneHumanSource

# The dataset must be downloaded to this path before running tests.
# See: benchmarks/README.md for instructions.
_DATA_DIR = Path("/media/mahdi/ssd23/Data/scib")
_SKIP_REASON = "Immune human dataset not downloaded"


def _dataset_available() -> bool:
    return (_DATA_DIR / "Immune_ALL_human.h5ad").exists()


@pytest.mark.skipif(not _dataset_available(), reason=_SKIP_REASON)
class TestImmuneHumanConfig:
    """Tests for the frozen config dataclass."""

    def test_default_config(self) -> None:
        config = ImmuneHumanConfig(data_dir=str(_DATA_DIR))
        assert config.data_dir == str(_DATA_DIR)
        assert config.subsample is None
        assert config.stochastic is False

    def test_frozen(self) -> None:
        config = ImmuneHumanConfig(data_dir=str(_DATA_DIR))
        with pytest.raises(AttributeError):
            config.data_dir = "/other"  # type: ignore[misc]

    def test_subsample_config(self) -> None:
        config = ImmuneHumanConfig(
            data_dir=str(_DATA_DIR), subsample=2000
        )
        assert config.subsample == 2000


@pytest.mark.skipif(not _dataset_available(), reason=_SKIP_REASON)
class TestImmuneHumanSource:
    """Tests for the DataSource loading and access."""

    @pytest.fixture()
    def source(self) -> ImmuneHumanSource:
        """Load with subsample for fast tests."""
        config = ImmuneHumanConfig(
            data_dir=str(_DATA_DIR), subsample=500
        )
        return ImmuneHumanSource(config)

    def test_load_returns_dict(self, source: ImmuneHumanSource) -> None:
        data = source.load()
        assert isinstance(data, dict)

    def test_required_keys(self, source: ImmuneHumanSource) -> None:
        data = source.load()
        required = {
            "counts", "batch_labels", "cell_type_labels",
            "embeddings", "gene_names",
        }
        assert required.issubset(data.keys())

    def test_counts_shape(self, source: ImmuneHumanSource) -> None:
        data = source.load()
        assert data["counts"].ndim == 2
        assert data["counts"].shape[0] == 500  # subsampled
        assert isinstance(data["counts"], jnp.ndarray)

    def test_batch_labels_shape(self, source: ImmuneHumanSource) -> None:
        data = source.load()
        assert data["batch_labels"].shape == (500,)

    def test_cell_type_labels_shape(
        self, source: ImmuneHumanSource
    ) -> None:
        data = source.load()
        assert data["cell_type_labels"].shape == (500,)

    def test_embeddings_shape(self, source: ImmuneHumanSource) -> None:
        data = source.load()
        assert data["embeddings"].ndim == 2
        assert data["embeddings"].shape[0] == 500
        assert isinstance(data["embeddings"], jnp.ndarray)

    def test_gene_names_are_strings(
        self, source: ImmuneHumanSource
    ) -> None:
        data = source.load()
        assert isinstance(data["gene_names"], list)
        assert all(isinstance(g, str) for g in data["gene_names"])

    def test_counts_nonnegative(self, source: ImmuneHumanSource) -> None:
        data = source.load()
        assert jnp.all(data["counts"] >= 0)

    def test_multiple_batches(self, source: ImmuneHumanSource) -> None:
        data = source.load()
        unique_batches = np.unique(np.asarray(data["batch_labels"]))
        assert len(unique_batches) >= 2

    def test_multiple_cell_types(
        self, source: ImmuneHumanSource
    ) -> None:
        data = source.load()
        unique_types = np.unique(np.asarray(data["cell_type_labels"]))
        assert len(unique_types) >= 2

    def test_len(self, source: ImmuneHumanSource) -> None:
        assert len(source) == 500

    def test_full_dataset_size(self) -> None:
        """Full dataset should have ~33K cells."""
        config = ImmuneHumanConfig(data_dir=str(_DATA_DIR))
        source = ImmuneHumanSource(config)
        assert len(source) > 30000

    def test_metadata(self, source: ImmuneHumanSource) -> None:
        data = source.load()
        assert "n_cells" in data
        assert "n_genes" in data
        assert "n_batches" in data
        assert "n_types" in data
        assert data["n_cells"] == 500
        assert data["n_genes"] > 0
        assert data["n_batches"] >= 2
        assert data["n_types"] >= 2
