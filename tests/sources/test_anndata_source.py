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

    def test_import_from_module(self):
        """Test that AnnDataSource can be imported from its module."""
        from diffbio.sources.anndata_source import AnnDataSource

        assert AnnDataSource is not None


# =============================================================================
# Tests for AnnDataSource Loading
# =============================================================================


class TestAnnDataSourceLoad:
    """Tests for loading .h5ad files."""

    def test_load_returns_dict(self, sample_h5ad_dense):
        """Test that load() returns a dictionary."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)
        result = source.load()

        assert isinstance(result, dict)

    def test_load_has_required_keys(self, sample_h5ad_dense):
        """Test that load() result contains all required keys."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)
        result = source.load()

        assert "counts" in result
        assert "obs" in result
        assert "var" in result
        assert "obsm" in result

    def test_load_counts_is_jax_array(self, sample_h5ad_dense):
        """Test that counts are returned as a JAX array."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)
        result = source.load()

        assert isinstance(result["counts"], jnp.ndarray)

    def test_load_counts_shape(self, sample_h5ad_dense):
        """Test that counts have correct shape (n_cells, n_genes)."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, n_cells, n_genes = sample_h5ad_dense
        source = AnnDataSource(file_path)
        result = source.load()

        assert result["counts"].shape == (n_cells, n_genes)

    def test_load_counts_values_match(self, sample_h5ad_dense):
        """Test that loaded counts match the original data."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, expected_counts, _, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)
        result = source.load()

        np.testing.assert_array_almost_equal(
            np.array(result["counts"]),
            expected_counts,
            decimal=5,
        )

    def test_load_obs_is_dict(self, sample_h5ad_dense):
        """Test that obs is a dictionary of metadata arrays."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)
        result = source.load()

        assert isinstance(result["obs"], dict)
        assert "cell_type" in result["obs"]

    def test_load_var_is_dict(self, sample_h5ad_dense):
        """Test that var is a dictionary of gene metadata."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)
        result = source.load()

        assert isinstance(result["var"], dict)
        assert "gene_name" in result["var"]

    def test_load_obsm_contains_embeddings(self, sample_h5ad_dense):
        """Test that obsm contains embedding arrays."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, n_cells, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)
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
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _, _ = sample_h5ad_sparse
        source = AnnDataSource(file_path)
        result = source.load()

        assert isinstance(result["counts"], jnp.ndarray)

    def test_sparse_shape_matches(self, sample_h5ad_sparse):
        """Test that sparse conversion preserves shape."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, n_cells, n_genes = sample_h5ad_sparse
        source = AnnDataSource(file_path)
        result = source.load()

        assert result["counts"].shape == (n_cells, n_genes)

    def test_sparse_values_match(self, sample_h5ad_sparse):
        """Test that sparse conversion preserves values."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, expected_counts, _, _ = sample_h5ad_sparse
        source = AnnDataSource(file_path)
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
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _ = sample_h5ad_no_obsm
        source = AnnDataSource(file_path)
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
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, n_cells, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)

        assert len(source) == n_cells

    def test_getitem_returns_dict(self, sample_h5ad_dense):
        """Test that __getitem__ returns a dict for a single cell."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)
        cell = source[0]

        assert isinstance(cell, dict)

    def test_getitem_has_counts_key(self, sample_h5ad_dense):
        """Test that single-cell dict has counts."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _, n_genes = sample_h5ad_dense
        source = AnnDataSource(file_path)
        cell = source[0]

        assert "counts" in cell
        assert isinstance(cell["counts"], jnp.ndarray)
        assert cell["counts"].shape == (n_genes,)

    def test_getitem_has_obs_key(self, sample_h5ad_dense):
        """Test that single-cell dict has observation metadata."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)
        cell = source[0]

        assert "obs" in cell
        assert isinstance(cell["obs"], dict)
        assert "cell_type" in cell["obs"]

    def test_getitem_has_obsm_key(self, sample_h5ad_dense):
        """Test that single-cell dict has embeddings."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)
        cell = source[0]

        assert "obsm" in cell
        assert isinstance(cell["obsm"], dict)
        assert "X_pca" in cell["obsm"]
        assert "X_umap" in cell["obsm"]

    def test_getitem_obsm_shape_single_cell(self, sample_h5ad_dense):
        """Test that obsm arrays for a single cell are 1D."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, _, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)
        cell = source[0]

        # PCA was 3-dimensional, UMAP was 2-dimensional
        assert cell["obsm"]["X_pca"].shape == (3,)
        assert cell["obsm"]["X_umap"].shape == (2,)

    def test_getitem_out_of_bounds_raises(self, sample_h5ad_dense):
        """Test that out-of-bounds index raises IndexError."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, n_cells, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)

        with pytest.raises(IndexError):
            source[n_cells]

        with pytest.raises(IndexError):
            source[-n_cells - 1]

    def test_getitem_negative_indexing(self, sample_h5ad_dense):
        """Test that negative indexing works correctly."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, n_cells, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)

        last_cell = source[-1]
        assert isinstance(last_cell, dict)
        assert "counts" in last_cell

    def test_getitem_sparse_single_cell(self, sample_h5ad_sparse):
        """Test single-cell access with sparse underlying matrix."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, expected_counts, _, n_genes = sample_h5ad_sparse
        source = AnnDataSource(file_path)
        cell = source[0]

        assert cell["counts"].shape == (n_genes,)
        np.testing.assert_array_almost_equal(
            np.array(cell["counts"]),
            expected_counts[0],
            decimal=5,
        )


# =============================================================================
# Tests for Error Handling
# =============================================================================


class TestAnnDataSourceErrors:
    """Tests for error handling."""

    def test_file_not_found_raises(self, anndata_module):
        """Test that missing file raises FileNotFoundError."""
        from diffbio.sources.anndata_source import AnnDataSource

        with pytest.raises(FileNotFoundError):
            AnnDataSource(Path("/nonexistent/path.h5ad"))

    def test_accepts_string_path(self, sample_h5ad_dense):
        """Test that string paths are accepted."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, n_cells, _ = sample_h5ad_dense
        source = AnnDataSource(str(file_path))

        assert len(source) == n_cells

    def test_accepts_path_object(self, sample_h5ad_dense):
        """Test that Path objects are accepted."""
        from diffbio.sources.anndata_source import AnnDataSource

        file_path, _, n_cells, _ = sample_h5ad_dense
        source = AnnDataSource(file_path)

        assert len(source) == n_cells
