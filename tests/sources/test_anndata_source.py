"""Tests for AnnDataSource.

Test-driven development: These tests define the expected behavior.
Implementation must pass these tests without modification.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def anndata_module():
    """Import anndata, skipping tests if not installed."""
    return pytest.importorskip("anndata")


@pytest.fixture
def sample_h5ad_dense(anndata_module, tmp_path):
    """Create a sample .h5ad file with dense count matrix."""
    ad = anndata_module
    n_cells, n_genes = 5, 10

    counts = np.random.default_rng(42).poisson(lam=3.0, size=(n_cells, n_genes)).astype(np.float32)
    obs = ad.AnnData(
        X=counts,
        obs={"cell_type": [f"type_{i % 3}" for i in range(n_cells)]},
        var={"gene_name": [f"gene_{j}" for j in range(n_genes)]},
    )
    obs.obsm["X_pca"] = np.random.default_rng(7).standard_normal((n_cells, 3)).astype(np.float32)
    obs.obsm["X_umap"] = np.random.default_rng(8).standard_normal((n_cells, 2)).astype(np.float32)

    file_path = tmp_path / "dense.h5ad"
    obs.write_h5ad(file_path)
    return file_path, counts, n_cells, n_genes


@pytest.fixture
def sample_h5ad_sparse(anndata_module, tmp_path):
    """Create a sample .h5ad file with sparse count matrix."""
    import scipy.sparse

    ad = anndata_module
    n_cells, n_genes = 8, 12

    dense_counts = (
        np.random.default_rng(99).poisson(lam=1.0, size=(n_cells, n_genes)).astype(np.float32)
    )
    sparse_counts = scipy.sparse.csr_matrix(dense_counts)

    adata = ad.AnnData(
        X=sparse_counts,
        obs={"batch": [f"batch_{i % 2}" for i in range(n_cells)]},
        var={"gene_id": [f"ENSG{j:05d}" for j in range(n_genes)]},
    )

    file_path = tmp_path / "sparse.h5ad"
    adata.write_h5ad(file_path)
    return file_path, dense_counts, n_cells, n_genes


@pytest.fixture
def sample_h5ad_no_obsm(anndata_module, tmp_path):
    """Create a sample .h5ad file without obsm embeddings."""
    ad = anndata_module
    n_cells, n_genes = 4, 6

    counts = np.ones((n_cells, n_genes), dtype=np.float32)
    adata = ad.AnnData(
        X=counts,
        obs={"label": ["a", "b", "a", "b"]},
        var={"name": [f"g{j}" for j in range(n_genes)]},
    )
    # Explicitly do NOT set obsm

    file_path = tmp_path / "no_obsm.h5ad"
    adata.write_h5ad(file_path)
    return file_path, n_cells, n_genes


# =============================================================================
# Tests for AnnDataSource Import and Config
# =============================================================================


class TestAnnDataSourceImport:
    """Tests for AnnDataSource module imports."""

    def test_import_from_sources(self):
        """Test that AnnDataSource can be imported from diffbio.sources."""
        from diffbio.sources import AnnDataSource

        assert AnnDataSource is not None

    def test_import_config_from_sources(self):
        """Test that AnnDataSourceConfig can be imported from diffbio.sources."""
        from diffbio.sources import AnnDataSourceConfig

        assert AnnDataSourceConfig is not None

    def test_import_from_module(self):
        """Test that AnnDataSource can be imported from its module."""
        from diffbio.sources.anndata_source import AnnDataSource

        assert AnnDataSource is not None

    def test_import_config_from_module(self):
        """Test that AnnDataSourceConfig can be imported from its module."""
        from diffbio.sources.anndata_source import AnnDataSourceConfig

        assert AnnDataSourceConfig is not None


# =============================================================================
# Tests for AnnDataSourceConfig
# =============================================================================


class TestAnnDataSourceConfig:
    """Tests for configuration validation."""

    def test_config_requires_file_path(self):
        """Test that config raises when file_path is None."""
        from diffbio.sources.anndata_source import AnnDataSourceConfig

        with pytest.raises(ValueError, match="file_path is required"):
            AnnDataSourceConfig()

    def test_config_accepts_file_path(self, sample_h5ad_dense):
        """Test that config accepts a valid file path."""
        from diffbio.sources.anndata_source import AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        assert config.file_path == str(file_path)

    def test_config_defaults(self, sample_h5ad_dense):
        """Test that config has correct defaults."""
        from diffbio.sources.anndata_source import AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        assert config.backed is False
        assert config.shuffle is False
        assert config.seed == 42
        assert config.split is None


# =============================================================================
# Tests for AnnDataSource Loading
# =============================================================================


class TestAnnDataSourceLoad:
    """Tests for loading .h5ad files."""

    def test_load_returns_dict(self, sample_h5ad_dense):
        """Test that load() returns a dictionary."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        assert isinstance(result, dict)

    def test_load_has_required_keys(self, sample_h5ad_dense):
        """Test that load() result contains all required keys."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        assert "counts" in result
        assert "obs" in result
        assert "var" in result
        assert "obsm" in result

    def test_load_counts_is_jax_array(self, sample_h5ad_dense):
        """Test that counts are returned as a JAX array."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        assert isinstance(result["counts"], jnp.ndarray)

    def test_load_counts_shape(self, sample_h5ad_dense):
        """Test that counts have correct shape (n_cells, n_genes)."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, n_cells, n_genes = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        assert result["counts"].shape == (n_cells, n_genes)

    def test_load_counts_values_match(self, sample_h5ad_dense):
        """Test that loaded counts match the original data."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, expected_counts, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        np.testing.assert_array_almost_equal(
            np.array(result["counts"]),
            expected_counts,
            decimal=5,
        )

    def test_load_obs_is_dict(self, sample_h5ad_dense):
        """Test that obs is a dictionary of metadata arrays."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        assert isinstance(result["obs"], dict)
        assert "cell_type" in result["obs"]

    def test_load_var_is_dict(self, sample_h5ad_dense):
        """Test that var is a dictionary of gene metadata."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        assert isinstance(result["var"], dict)
        assert "gene_name" in result["var"]

    def test_load_obsm_contains_embeddings(self, sample_h5ad_dense):
        """Test that obsm contains embedding arrays."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, n_cells, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        assert isinstance(result["obsm"], dict)
        assert "X_pca" in result["obsm"]
        assert "X_umap" in result["obsm"]
        assert isinstance(result["obsm"]["X_pca"], jnp.ndarray)
        assert result["obsm"]["X_pca"].shape[0] == n_cells


# =============================================================================
# Tests for Sparse Matrix Handling
# =============================================================================


class TestAnnDataSourceSparse:
    """Tests for sparse matrix conversion."""

    def test_sparse_converted_to_dense_jax(self, sample_h5ad_sparse):
        """Test that sparse counts are converted to dense JAX arrays."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_sparse
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        assert isinstance(result["counts"], jnp.ndarray)

    def test_sparse_shape_matches(self, sample_h5ad_sparse):
        """Test that sparse conversion preserves shape."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, n_cells, n_genes = sample_h5ad_sparse
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        assert result["counts"].shape == (n_cells, n_genes)

    def test_sparse_values_match(self, sample_h5ad_sparse):
        """Test that sparse conversion preserves values."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, expected_counts, _, _ = sample_h5ad_sparse
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        np.testing.assert_array_almost_equal(
            np.array(result["counts"]),
            expected_counts,
            decimal=5,
        )


# =============================================================================
# Tests for Missing obsm Handling
# =============================================================================


class TestAnnDataSourceMissingObsm:
    """Tests for handling files without obsm embeddings."""

    def test_missing_obsm_returns_empty_dict(self, sample_h5ad_no_obsm):
        """Test that missing obsm yields an empty dict."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _ = sample_h5ad_no_obsm
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        result = source.load()

        assert isinstance(result["obsm"], dict)
        assert len(result["obsm"]) == 0


# =============================================================================
# Tests for __len__ and __getitem__
# =============================================================================


class TestAnnDataSourceIndexing:
    """Tests for length and item access."""

    def test_len_returns_cell_count(self, sample_h5ad_dense):
        """Test that __len__ returns the number of cells."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, n_cells, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        assert len(source) == n_cells

    def test_getitem_returns_dict(self, sample_h5ad_dense):
        """Test that __getitem__ returns a dict for a single cell."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        cell = source[0]

        assert isinstance(cell, dict)

    def test_getitem_has_counts_key(self, sample_h5ad_dense):
        """Test that single-cell dict has counts."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, n_genes = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        cell = source[0]

        assert "counts" in cell
        assert isinstance(cell["counts"], jnp.ndarray)
        assert cell["counts"].shape == (n_genes,)

    def test_getitem_has_obs_key(self, sample_h5ad_dense):
        """Test that single-cell dict has observation metadata."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        cell = source[0]

        assert "obs" in cell
        assert isinstance(cell["obs"], dict)
        assert "cell_type" in cell["obs"]

    def test_getitem_has_obsm_key(self, sample_h5ad_dense):
        """Test that single-cell dict has embeddings."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        cell = source[0]

        assert "obsm" in cell
        assert isinstance(cell["obsm"], dict)
        assert "X_pca" in cell["obsm"]
        assert "X_umap" in cell["obsm"]

    def test_getitem_obsm_shape_single_cell(self, sample_h5ad_dense):
        """Test that obsm arrays for a single cell are 1D."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        cell = source[0]

        # PCA was 3-dimensional, UMAP was 2-dimensional
        assert cell["obsm"]["X_pca"].shape == (3,)
        assert cell["obsm"]["X_umap"].shape == (2,)

    def test_getitem_out_of_bounds_raises(self, sample_h5ad_dense):
        """Test that out-of-bounds index raises IndexError."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, n_cells, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        with pytest.raises(IndexError):
            source[n_cells]

        with pytest.raises(IndexError):
            source[-n_cells - 1]

    def test_getitem_negative_indexing(self, sample_h5ad_dense):
        """Test that negative indexing works correctly."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, n_cells, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        last_cell = source[-1]
        assert isinstance(last_cell, dict)
        assert "counts" in last_cell

    def test_getitem_sparse_single_cell(self, sample_h5ad_sparse):
        """Test single-cell access with sparse underlying matrix."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, expected_counts, _, n_genes = sample_h5ad_sparse
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)
        cell = source[0]

        assert cell["counts"].shape == (n_genes,)
        np.testing.assert_array_almost_equal(
            np.array(cell["counts"]),
            expected_counts[0],
            decimal=5,
        )


# =============================================================================
# Tests for Iteration
# =============================================================================


class TestAnnDataSourceIteration:
    """Tests for iteration protocol."""

    def test_iteration_yields_all_cells(self, sample_h5ad_dense):
        """Test that iteration yields all cells."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, n_cells, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        cells = list(source)
        assert len(cells) == n_cells

    def test_iteration_cell_has_counts(self, sample_h5ad_dense):
        """Test that iterated cells have counts."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, n_genes = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        first_cell = next(iter(source))
        assert "counts" in first_cell
        assert first_cell["counts"].shape == (n_genes,)

    def test_iteration_can_repeat(self, sample_h5ad_dense):
        """Test that iteration can be repeated (epoch increments)."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, n_cells, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        epoch1_cells = list(source)
        epoch2_cells = list(source)
        assert len(epoch1_cells) == n_cells
        assert len(epoch2_cells) == n_cells


# =============================================================================
# Tests for Batch Retrieval
# =============================================================================


class TestAnnDataSourceBatch:
    """Tests for batch retrieval."""

    def test_get_batch_returns_dict(self, sample_h5ad_dense):
        """Test that get_batch returns a dictionary."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        batch = source.get_batch(2)
        assert isinstance(batch, dict)

    def test_get_batch_counts_shape(self, sample_h5ad_dense):
        """Test that batched counts have correct shape."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, n_genes = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        batch_size = 3
        batch = source.get_batch(batch_size)
        assert batch["counts"].shape == (batch_size, n_genes)

    def test_get_batch_has_obs(self, sample_h5ad_dense):
        """Test that batch has obs metadata."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        batch = source.get_batch(2)
        assert "obs" in batch
        assert "cell_type" in batch["obs"]

    def test_get_batch_stateless_with_key(self, sample_h5ad_dense):
        """Test stateless batch retrieval with RNG key."""
        import jax

        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, n_genes = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        key = jax.random.key(0)
        batch = source.get_batch(2, key=key)
        assert batch["counts"].shape == (2, n_genes)


# =============================================================================
# Tests for Error Handling
# =============================================================================


class TestAnnDataSourceErrors:
    """Tests for error handling."""

    def test_file_not_found_raises(self, anndata_module):
        """Test that missing file raises FileNotFoundError."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        config = AnnDataSourceConfig(file_path=str(Path("/nonexistent/path.h5ad")))
        with pytest.raises(FileNotFoundError):
            AnnDataSource(config)

    def test_accepts_string_path(self, sample_h5ad_dense):
        """Test that string paths are accepted."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, n_cells, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        assert len(source) == n_cells


# =============================================================================
# Tests for Dataset Info
# =============================================================================


class TestAnnDataSourceInfo:
    """Tests for dataset info retrieval."""

    def test_get_dataset_info(self, sample_h5ad_dense):
        """Test that dataset info contains expected fields."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, n_cells, n_genes = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        info = source.get_dataset_info()
        assert info["n_cells"] == n_cells
        assert info["n_genes"] == n_genes

    def test_repr(self, sample_h5ad_dense):
        """Test string representation."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        repr_str = repr(source)
        assert "AnnDataSource" in repr_str
        assert "length=" in repr_str


# =============================================================================
# Tests for DataSourceModule conformance
# =============================================================================


class TestAnnDataSourceConformance:
    """Tests for DataSourceModule subclass conformance."""

    def test_is_data_source_module(self, sample_h5ad_dense):
        """Test that AnnDataSource is a DataSourceModule."""
        from datarax.core.data_source import DataSourceModule

        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        assert isinstance(source, DataSourceModule)

    def test_reset_resets_state(self, sample_h5ad_dense):
        """Test that reset brings iteration back to beginning."""
        from diffbio.sources.anndata_source import AnnDataSource, AnnDataSourceConfig

        file_path, _, _, _ = sample_h5ad_dense
        config = AnnDataSourceConfig(file_path=str(file_path))
        source = AnnDataSource(config)

        # Iterate once
        list(source)
        assert source.epoch.get_value() > 0

        # Reset
        source.reset()
        assert source.index.get_value() == 0
        assert source.epoch.get_value() == 0
