"""Tests for shared AnnData source helpers."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.sources._anndata_shared import (
    extract_anndata_annotations,
    initialize_eager_source_state,
    read_h5ad,
    to_dense_array,
)


@pytest.fixture
def anndata_module():
    """Import anndata, skipping tests if unavailable."""
    return pytest.importorskip("anndata")


@pytest.fixture
def sample_h5ad(anndata_module, tmp_path):
    """Create a sample .h5ad file with metadata and embeddings."""
    ad = anndata_module
    counts = np.arange(20, dtype=np.float32).reshape(5, 4)
    adata = ad.AnnData(
        X=counts,
        obs={"cell_type": ["t", "b", "t", "nk", "b"]},
        var={"gene_name": [f"gene_{i}" for i in range(4)]},
    )
    adata.obsm["X_pca"] = np.arange(15, dtype=np.float32).reshape(5, 3)

    file_path = tmp_path / "shared.h5ad"
    adata.write_h5ad(file_path)
    return file_path


def test_read_h5ad_raises_for_missing_files():
    """Reject missing files before attempting AnnData I/O."""
    config = SimpleNamespace(file_path="/tmp/does-not-exist.h5ad", backed=False)

    with pytest.raises(FileNotFoundError, match="AnnData file not found"):
        read_h5ad(config)


def test_to_dense_array_handles_sparse_inputs():
    """Convert sparse matrices to dense float32 arrays."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    sparse = scipy_sparse.csr_matrix(np.eye(3, dtype=np.float64))

    dense = to_dense_array(sparse)

    assert dense.dtype == np.float32
    np.testing.assert_array_equal(dense, np.eye(3, dtype=np.float32))


def test_extract_anndata_annotations_preserves_obs_var_and_obsm(sample_h5ad):
    """Extract annotations into the standardized in-memory representation."""
    config = SimpleNamespace(file_path=str(sample_h5ad), backed=False)
    adata = read_h5ad(config)

    obs, var, obsm = extract_anndata_annotations(adata)

    assert set(obs) == {"cell_type"}
    assert set(var) == {"gene_name"}
    assert set(obsm) == {"X_pca"}
    assert isinstance(obsm["X_pca"], jnp.ndarray)
    assert obsm["X_pca"].shape == (5, 3)


def test_initialize_eager_source_state_sets_required_fields():
    """Populate the common eager-source bookkeeping fields."""
    source = SimpleNamespace()
    data = {"counts": jnp.ones((2, 3))}

    initialize_eager_source_state(
        source,
        data=data,
        length=2,
        seed=7,
        shuffle=True,
        dataset_name="dataset.h5ad",
        split_name="train",
        dataset_info={"n_genes": 3, "n_cells": 2},
    )

    assert source.data is data
    assert source.length == 2
    assert isinstance(source.index, nnx.Variable)
    assert isinstance(source.epoch, nnx.Variable)
    assert source.index.get_value() == 0
    assert source.epoch.get_value() == 0
    assert source._seed == 7
    assert source.shuffle is True
    assert source.dataset_name == "dataset.h5ad"
    assert source.split_name == "train"
    assert source._dataset_info == {"n_genes": 3, "n_cells": 2}
