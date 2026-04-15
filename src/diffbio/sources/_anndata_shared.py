"""Shared helpers for AnnData-backed DiffBio sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

from diffbio.sources._utils import _require_anndata


def to_dense_array(matrix: Any) -> np.ndarray:
    """Convert a dense or sparse matrix into a float32 NumPy array."""
    import scipy.sparse  # noqa: PLC0415

    if scipy.sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def read_h5ad(config: Any) -> Any:
    """Load an AnnData object after validating the configured file path."""
    anndata_mod = _require_anndata()
    file_path = Path(str(config.file_path))
    if not file_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {file_path}")
    return anndata_mod.read_h5ad(file_path, backed="r" if config.backed else None)


def load_obsm(adata: Any) -> dict[str, jnp.ndarray]:
    """Load AnnData embedding matrices as float32 JAX arrays."""
    if adata.obsm is None or len(adata.obsm) == 0:
        return {}

    return {
        key: jnp.array(np.asarray(adata.obsm[key], dtype=np.float32)) for key in adata.obsm.keys()
    }


def extract_anndata_annotations(
    adata: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, jnp.ndarray]]:
    """Extract obs, var, and obsm tables into the standard in-memory layout."""
    obs = {col: np.asarray(adata.obs[col]) for col in adata.obs.columns}
    var = {col: np.asarray(adata.var[col]) for col in adata.var.columns}
    obsm = load_obsm(adata)
    return obs, var, obsm


def build_anndata_data(
    *,
    counts: jnp.ndarray,
    obs: dict[str, Any],
    var: dict[str, Any],
    obsm: dict[str, jnp.ndarray],
) -> dict[str, Any]:
    """Assemble the canonical in-memory AnnData payload for DiffBio sources."""
    return {
        "counts": counts,
        "obs": obs,
        "var": var,
        "obsm": obsm,
    }


def initialize_eager_source_state(
    source: Any,
    *,
    data: dict[str, Any],
    length: int,
    seed: int,
    shuffle: bool,
    dataset_name: str | None,
    split_name: str | None,
    dataset_info: dict[str, int],
) -> None:
    """Populate the common eager-source bookkeeping fields."""
    source.data = data
    source.length = length
    source.index = nnx.Variable(0)
    source.epoch = nnx.Variable(0)
    source._seed = seed
    source.shuffle = shuffle
    source.dataset_name = dataset_name
    source.split_name = split_name
    source._dataset_info = dataset_info
