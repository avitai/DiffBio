"""Tests for AnnData interop layer (to_anndata / from_anndata).

Test-driven development: These tests define the expected behavior.
Implementation must pass these tests without modification.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def anndata_module():
    """Import anndata, skipping tests if not installed."""
    return pytest.importorskip("anndata")


@pytest.fixture
def sample_data_dict() -> dict:
    """Create a sample DiffBio data dict with all standard keys."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 5, 10
    return {
        "counts": jnp.array(rng.poisson(lam=3.0, size=(n_cells, n_genes)).astype(np.float32)),
        "obs": {
            "cell_type": np.array(["typeA", "typeB", "typeA", "typeC", "typeB"]),
            "quality": np.array([0.9, 0.8, 0.7, 0.6, 0.95]),
        },
        "var": {
            "gene_name": np.array([f"gene_{j}" for j in range(n_genes)]),
            "is_highly_variable": np.array([True, False] * 5),
        },
        "obsm": {
            "X_pca": jnp.array(rng.standard_normal((n_cells, 3)).astype(np.float32)),
            "X_umap": jnp.array(rng.standard_normal((n_cells, 2)).astype(np.float32)),
        },
    }


@pytest.fixture
def sample_adata(anndata_module):
    """Create a sample AnnData object."""
    ad = anndata_module
    rng = np.random.default_rng(7)
    n_cells, n_genes = 6, 8

    counts = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(
        X=counts,
        obs={"cell_type": [f"type_{i % 2}" for i in range(n_cells)]},
        var={"gene_name": [f"gene_{j}" for j in range(n_genes)]},
    )
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, 4)).astype(np.float32)
    return adata, counts


@pytest.fixture
def sparse_adata(anndata_module):
    """Create a sample AnnData object with a sparse count matrix."""
    import scipy.sparse

    ad = anndata_module
    rng = np.random.default_rng(99)
    n_cells, n_genes = 8, 12

    dense_counts = rng.poisson(lam=1.0, size=(n_cells, n_genes)).astype(np.float32)
    sparse_counts = scipy.sparse.csr_matrix(dense_counts)

    adata = ad.AnnData(
        X=sparse_counts,
        obs={"batch": [f"batch_{i % 2}" for i in range(n_cells)]},
        var={"gene_id": [f"ENSG{j:05d}" for j in range(n_genes)]},
    )
    return adata, dense_counts


# =============================================================================
# Tests for to_anndata
# =============================================================================


class TestToAnnData:
    """Tests for converting DiffBio data dict to AnnData."""

    def test_returns_anndata_object(self, anndata_module, sample_data_dict):
        """Test that to_anndata returns an AnnData instance."""
        from diffbio.sources.anndata_interop import to_anndata

        result = to_anndata(sample_data_dict)
        assert isinstance(result, anndata_module.AnnData)

    def test_counts_stored_in_x(self, anndata_module, sample_data_dict):
        """Test that counts are stored in .X as a numpy array."""
        from diffbio.sources.anndata_interop import to_anndata

        result = to_anndata(sample_data_dict)
        expected = np.asarray(sample_data_dict["counts"])

        assert result.X is not None
        np.testing.assert_array_almost_equal(np.asarray(result.X), expected, decimal=5)

    def test_x_shape_matches(self, anndata_module, sample_data_dict):
        """Test that .X shape matches original counts shape."""
        from diffbio.sources.anndata_interop import to_anndata

        result = to_anndata(sample_data_dict)
        assert result.X.shape == (5, 10)

    def test_obs_converted_to_dataframe(self, anndata_module, sample_data_dict):
        """Test that obs dict becomes a pandas DataFrame."""
        import pandas as pd

        from diffbio.sources.anndata_interop import to_anndata

        result = to_anndata(sample_data_dict)
        assert isinstance(result.obs, pd.DataFrame)
        assert "cell_type" in result.obs.columns
        assert "quality" in result.obs.columns
        assert len(result.obs) == 5

    def test_var_converted_to_dataframe(self, anndata_module, sample_data_dict):
        """Test that var dict becomes a pandas DataFrame."""
        import pandas as pd

        from diffbio.sources.anndata_interop import to_anndata

        result = to_anndata(sample_data_dict)
        assert isinstance(result.var, pd.DataFrame)
        assert "gene_name" in result.var.columns
        assert "is_highly_variable" in result.var.columns
        assert len(result.var) == 10

    def test_obsm_converted(self, anndata_module, sample_data_dict):
        """Test that obsm dict entries are stored in .obsm."""
        from diffbio.sources.anndata_interop import to_anndata

        result = to_anndata(sample_data_dict)
        assert "X_pca" in result.obsm
        assert "X_umap" in result.obsm
        np.testing.assert_array_almost_equal(
            result.obsm["X_pca"],
            np.asarray(sample_data_dict["obsm"]["X_pca"]),
            decimal=5,
        )

    def test_obsm_are_numpy_arrays(self, anndata_module, sample_data_dict):
        """Test that obsm values are numpy arrays (not JAX)."""
        from diffbio.sources.anndata_interop import to_anndata

        result = to_anndata(sample_data_dict)
        assert isinstance(result.obsm["X_pca"], np.ndarray)
        assert isinstance(result.obsm["X_umap"], np.ndarray)

    def test_empty_obsm(self, anndata_module):
        """Test that empty obsm dict is handled."""
        from diffbio.sources.anndata_interop import to_anndata

        data_dict = {
            "counts": jnp.ones((3, 4)),
            "obs": {"label": np.array(["a", "b", "c"])},
            "var": {"name": np.array(["g0", "g1", "g2", "g3"])},
            "obsm": {},
        }
        result = to_anndata(data_dict)
        assert len(result.obsm) == 0

    def test_uses_string_indexes_without_implicit_modification_warning(
        self,
        anndata_module,
        sample_data_dict,
    ):
        """AnnData conversion should provide string indexes up front."""
        from diffbio.sources.anndata_interop import to_anndata

        warning_cls = anndata_module.ImplicitModificationWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", warning_cls)
            result = to_anndata(sample_data_dict)

        assert not any(issubclass(w.category, warning_cls) for w in caught)
        assert list(result.obs_names) == [str(i) for i in range(result.n_obs)]
        assert list(result.var_names) == [str(i) for i in range(result.n_vars)]


# =============================================================================
# Tests for from_anndata
# =============================================================================


class TestFromAnnData:
    """Tests for converting AnnData to DiffBio data dict."""

    def test_returns_dict(self, sample_adata):
        """Test that from_anndata returns a dictionary."""
        from diffbio.sources.anndata_interop import from_anndata

        adata, _ = sample_adata
        result = from_anndata(adata)
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_adata):
        """Test that result contains all standard keys."""
        from diffbio.sources.anndata_interop import from_anndata

        adata, _ = sample_adata
        result = from_anndata(adata)
        assert "counts" in result
        assert "obs" in result
        assert "var" in result
        assert "obsm" in result

    def test_counts_is_jax_array(self, sample_adata):
        """Test that counts are returned as a JAX array."""
        from diffbio.sources.anndata_interop import from_anndata

        adata, _ = sample_adata
        result = from_anndata(adata)
        assert isinstance(result["counts"], jnp.ndarray)

    def test_counts_values_match(self, sample_adata):
        """Test that counts values match original .X."""
        from diffbio.sources.anndata_interop import from_anndata

        adata, expected_counts = sample_adata
        result = from_anndata(adata)
        np.testing.assert_array_almost_equal(
            np.asarray(result["counts"]),
            expected_counts,
            decimal=5,
        )

    def test_counts_shape_matches(self, sample_adata):
        """Test that counts shape matches .X shape."""
        from diffbio.sources.anndata_interop import from_anndata

        adata, _ = sample_adata
        result = from_anndata(adata)
        assert result["counts"].shape == (6, 8)

    def test_obs_is_dict(self, sample_adata):
        """Test that obs is a plain dict."""
        from diffbio.sources.anndata_interop import from_anndata

        adata, _ = sample_adata
        result = from_anndata(adata)
        assert isinstance(result["obs"], dict)
        assert "cell_type" in result["obs"]

    def test_var_is_dict(self, sample_adata):
        """Test that var is a plain dict."""
        from diffbio.sources.anndata_interop import from_anndata

        adata, _ = sample_adata
        result = from_anndata(adata)
        assert isinstance(result, dict)
        assert isinstance(result["var"], dict)
        assert "gene_name" in result["var"]

    def test_obsm_is_dict_of_jax_arrays(self, sample_adata):
        """Test that obsm values are JAX arrays."""
        from diffbio.sources.anndata_interop import from_anndata

        adata, _ = sample_adata
        result = from_anndata(adata)
        assert isinstance(result["obsm"], dict)
        assert "X_pca" in result["obsm"]
        assert isinstance(result["obsm"]["X_pca"], jnp.ndarray)

    def test_handles_sparse_x(self, sparse_adata):
        """Test that sparse .X is converted to dense JAX array."""
        from diffbio.sources.anndata_interop import from_anndata

        adata, expected_counts = sparse_adata
        result = from_anndata(adata)

        assert isinstance(result["counts"], jnp.ndarray)
        np.testing.assert_array_almost_equal(
            np.asarray(result["counts"]),
            expected_counts,
            decimal=5,
        )

    def test_handles_missing_obsm(self, anndata_module):
        """Test that AnnData with no obsm returns empty obsm dict."""
        from diffbio.sources.anndata_interop import from_anndata

        ad = anndata_module
        adata = ad.AnnData(
            X=np.ones((3, 4), dtype=np.float32),
            obs={"a": [1, 2, 3]},
            var={"b": [1, 2, 3, 4]},
        )
        result = from_anndata(adata)
        assert isinstance(result["obsm"], dict)
        assert len(result["obsm"]) == 0


# =============================================================================
# Tests for roundtrip conversion
# =============================================================================


class TestRoundTrip:
    """Tests for roundtrip conversion: data dict -> AnnData -> data dict."""

    def test_roundtrip_counts_preserved(self, anndata_module, sample_data_dict):
        """Test that counts survive a roundtrip within float tolerance."""
        from diffbio.sources.anndata_interop import from_anndata, to_anndata

        adata = to_anndata(sample_data_dict)
        recovered = from_anndata(adata)

        np.testing.assert_array_almost_equal(
            np.asarray(recovered["counts"]),
            np.asarray(sample_data_dict["counts"]),
            decimal=5,
        )

    def test_roundtrip_obs_keys_preserved(self, anndata_module, sample_data_dict):
        """Test that obs keys survive a roundtrip."""
        from diffbio.sources.anndata_interop import from_anndata, to_anndata

        adata = to_anndata(sample_data_dict)
        recovered = from_anndata(adata)

        assert set(recovered["obs"].keys()) == set(sample_data_dict["obs"].keys())

    def test_roundtrip_var_keys_preserved(self, anndata_module, sample_data_dict):
        """Test that var keys survive a roundtrip."""
        from diffbio.sources.anndata_interop import from_anndata, to_anndata

        adata = to_anndata(sample_data_dict)
        recovered = from_anndata(adata)

        assert set(recovered["var"].keys()) == set(sample_data_dict["var"].keys())

    def test_roundtrip_obsm_values_preserved(self, anndata_module, sample_data_dict):
        """Test that obsm arrays survive a roundtrip within float tolerance."""
        from diffbio.sources.anndata_interop import from_anndata, to_anndata

        adata = to_anndata(sample_data_dict)
        recovered = from_anndata(adata)

        assert set(recovered["obsm"].keys()) == set(sample_data_dict["obsm"].keys())
        for key in sample_data_dict["obsm"]:
            np.testing.assert_array_almost_equal(
                np.asarray(recovered["obsm"][key]),
                np.asarray(sample_data_dict["obsm"][key]),
                decimal=5,
            )

    def test_roundtrip_obs_values_preserved(self, anndata_module, sample_data_dict):
        """Test that obs values survive a roundtrip."""
        from diffbio.sources.anndata_interop import from_anndata, to_anndata

        adata = to_anndata(sample_data_dict)
        recovered = from_anndata(adata)

        np.testing.assert_array_equal(
            recovered["obs"]["cell_type"],
            sample_data_dict["obs"]["cell_type"],
        )


# =============================================================================
# Tests for edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_single_cell(self, anndata_module):
        """Test conversion with a single cell."""
        from diffbio.sources.anndata_interop import from_anndata, to_anndata

        data_dict = {
            "counts": jnp.array([[1.0, 2.0, 3.0]]),
            "obs": {"label": np.array(["only_cell"])},
            "var": {"name": np.array(["g0", "g1", "g2"])},
            "obsm": {},
        }
        adata = to_anndata(data_dict)
        assert adata.n_obs == 1
        assert adata.n_vars == 3

        recovered = from_anndata(adata)
        assert recovered["counts"].shape == (1, 3)

    def test_string_obs_columns(self, anndata_module):
        """Test that string observation columns survive roundtrip."""
        from diffbio.sources.anndata_interop import from_anndata, to_anndata

        data_dict = {
            "counts": jnp.ones((4, 5)),
            "obs": {
                "name": np.array(["alice", "bob", "carol", "dave"]),
                "group": np.array(["ctrl", "treat", "ctrl", "treat"]),
            },
            "var": {"id": np.array([f"v{i}" for i in range(5)])},
            "obsm": {},
        }
        adata = to_anndata(data_dict)
        recovered = from_anndata(adata)

        np.testing.assert_array_equal(recovered["obs"]["name"], data_dict["obs"]["name"])
        np.testing.assert_array_equal(recovered["obs"]["group"], data_dict["obs"]["group"])

    def test_no_obsm_key_in_data_dict(self, anndata_module):
        """Test that missing obsm key defaults to empty."""
        from diffbio.sources.anndata_interop import to_anndata

        data_dict = {
            "counts": jnp.ones((2, 3)),
            "obs": {"a": np.array([1, 2])},
            "var": {"b": np.array([1, 2, 3])},
        }
        result = to_anndata(data_dict)
        assert len(result.obsm) == 0

    def test_numeric_obs_columns(self, anndata_module):
        """Test that numeric obs columns survive roundtrip."""
        from diffbio.sources.anndata_interop import from_anndata, to_anndata

        data_dict = {
            "counts": jnp.ones((3, 4)),
            "obs": {"score": np.array([0.1, 0.5, 0.9])},
            "var": {"name": np.array(["a", "b", "c", "d"])},
            "obsm": {},
        }
        adata = to_anndata(data_dict)
        recovered = from_anndata(adata)

        np.testing.assert_array_almost_equal(
            recovered["obs"]["score"],
            data_dict["obs"]["score"],
            decimal=5,
        )


# =============================================================================
# Tests for import guarding
# =============================================================================


class TestImports:
    """Tests for module import behavior."""

    def test_module_imports(self, anndata_module):
        """Test that the interop module can be imported."""
        from diffbio.sources.anndata_interop import from_anndata, to_anndata

        assert callable(to_anndata)
        assert callable(from_anndata)

    def test_available_from_sources_init(self, anndata_module):
        """Test that functions are available from diffbio.sources."""
        from diffbio.sources import from_anndata, to_anndata

        assert callable(to_anndata)
        assert callable(from_anndata)
