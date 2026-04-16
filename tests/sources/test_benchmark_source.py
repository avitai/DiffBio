"""Tests for shared benchmark-source helpers and base class."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.config import StructuralConfig

from diffbio.sources._benchmark_source import (
    BenchmarkDataSource,
    encode_label_column,
    load_benchmark_adata,
    load_benchmark_counts,
)


@pytest.fixture
def anndata_module():
    """Import anndata, skipping tests if unavailable."""
    return pytest.importorskip("anndata")


@pytest.fixture
def sample_benchmark_h5ad(anndata_module, tmp_path):
    """Create a small benchmark-style h5ad file for helper tests."""
    ad = anndata_module
    counts = np.arange(30, dtype=np.float32).reshape(6, 5)
    adata = ad.AnnData(
        X=counts,
        obs={"label": ["a", "b", "a", "c", "b", "c"]},
        var={"gene_name": [f"gene_{i}" for i in range(5)]},
    )

    file_path = tmp_path / "benchmark.h5ad"
    adata.write_h5ad(file_path)
    return file_path


def test_load_benchmark_adata_reads_full_dataset(sample_benchmark_h5ad):
    """Load the benchmark h5ad file without subsampling."""
    adata = load_benchmark_adata(
        data_dir=str(sample_benchmark_h5ad.parent),
        filename=sample_benchmark_h5ad.name,
    )

    assert adata.n_obs == 6
    assert adata.n_vars == 5


def test_load_benchmark_adata_subsamples_deterministically(sample_benchmark_h5ad):
    """Apply deterministic subsampling when a smaller sample size is requested."""
    first = load_benchmark_adata(
        data_dir=str(sample_benchmark_h5ad.parent),
        filename=sample_benchmark_h5ad.name,
        subsample=3,
    )
    second = load_benchmark_adata(
        data_dir=str(sample_benchmark_h5ad.parent),
        filename=sample_benchmark_h5ad.name,
        subsample=3,
    )

    assert first.n_obs == 3
    assert second.n_obs == 3
    np.testing.assert_array_equal(np.asarray(first.X), np.asarray(second.X))


def test_load_benchmark_counts_returns_jax_counts(sample_benchmark_h5ad):
    """Load counts through the shared benchmark preamble helper."""
    adata, counts = load_benchmark_counts(
        data_dir=str(sample_benchmark_h5ad.parent),
        filename=sample_benchmark_h5ad.name,
        to_dense=lambda matrix: np.asarray(matrix, dtype=np.float32),
        subsample=4,
    )

    assert adata.n_obs == 4
    assert isinstance(counts, jnp.ndarray)
    assert counts.shape == (4, 5)


def test_encode_label_column_handles_categorical_values():
    """Encode categorical columns into stable int32 codes and names."""
    pandas = pytest.importorskip("pandas")
    column = pandas.Categorical(["beta", "alpha", "beta", "gamma"])

    codes, names = encode_label_column(column, include_names=True)

    assert codes.dtype == np.int32
    assert names == ["alpha", "beta", "gamma"]
    np.testing.assert_array_equal(codes, np.array([1, 0, 1, 2], dtype=np.int32))


def test_encode_label_column_handles_plain_arrays():
    """Encode non-categorical arrays using sorted unique labels."""
    codes, names = encode_label_column(np.array(["zeta", "alpha", "zeta"]), include_names=True)

    assert codes.dtype == np.int32
    assert names == ["alpha", "zeta"]
    np.testing.assert_array_equal(codes, np.array([1, 0, 1], dtype=np.int32))


@dataclass(frozen=True)
class _DummyConfig(StructuralConfig):
    size: int = 3


class _DummyBenchmarkSource(BenchmarkDataSource):
    iter_static_keys = ("gene_names", "cell_type_names")

    def __init__(self) -> None:
        super().__init__(_DummyConfig(), rngs=nnx.Rngs(0), name="dummy")
        self.data = {
            "counts": jnp.arange(6, dtype=jnp.float32).reshape(3, 2),
            "cell_type_labels": np.array([0, 1, 0], dtype=np.int32),
            "cell_type_names": ["a", "b"],
            "gene_names": ["g1", "g2"],
            "n_cells": 3,
        }


def test_benchmark_data_source_load_returns_data_dict():
    """Expose the stored payload through the shared load implementation."""
    source = _DummyBenchmarkSource()

    assert source.load() is source.data


def test_benchmark_data_source_len_uses_n_cells():
    """Expose row count through the shared __len__ implementation."""
    source = _DummyBenchmarkSource()

    assert len(source) == 3


def test_benchmark_data_source_iter_keeps_static_keys_unindexed():
    """Yield row dictionaries while preserving configured static keys."""
    source = _DummyBenchmarkSource()

    rows = list(source)

    assert len(rows) == 3
    assert rows[0]["counts"].shape == (2,)
    assert rows[0]["gene_names"] == ["g1", "g2"]
    assert rows[0]["cell_type_names"] == ["a", "b"]
