"""AnnData interop layer for DiffBio data dictionaries.

Provides bidirectional conversion between DiffBio's standard data dict format
(with keys ``counts``, ``obs``, ``var``, ``obsm``) and AnnData objects.

This enables integration with the broader single-cell ecosystem (scanpy,
scvi-tools, etc.) while keeping DiffBio's internal representation as
JAX-native dictionaries suitable for differentiable pipelines.

Both ``anndata`` and ``pandas`` are optional dependencies. Functions raise
``ImportError`` with installation instructions if they are not available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import anndata


def _require_anndata() -> Any:
    """Import anndata, raising a clear error if not installed.

    Returns:
        The anndata module.

    Raises:
        ImportError: If anndata is not installed.
    """
    try:
        import anndata as ad  # noqa: PLC0415

        return ad
    except ImportError as err:
        raise ImportError(
            "anndata is required for AnnData interop. "
            "Install with: uv pip install anndata"
        ) from err


def _require_pandas() -> Any:
    """Import pandas, raising a clear error if not installed.

    Returns:
        The pandas module.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd  # noqa: PLC0415

        return pd
    except ImportError as err:
        raise ImportError(
            "pandas is required for AnnData interop. "
            "Install with: uv pip install pandas"
        ) from err


def _to_dense_numpy(matrix: Any) -> np.ndarray:
    """Convert a matrix (dense, JAX, or sparse) to a dense numpy array.

    Args:
        matrix: A numpy array, JAX array, or scipy sparse matrix.

    Returns:
        Dense numpy array with float32 dtype.
    """
    try:
        import scipy.sparse  # noqa: PLC0415

        if scipy.sparse.issparse(matrix):
            return np.asarray(matrix.toarray(), dtype=np.float32)
    except ImportError:
        pass

    return np.asarray(matrix, dtype=np.float32)


def to_anndata(data_dict: dict[str, Any]) -> anndata.AnnData:
    """Convert a DiffBio data dict to an AnnData object.

    Translates the standard DiffBio dictionary format (as produced by
    ``AnnDataSource.load()``) into an ``anndata.AnnData`` object for use
    with scanpy, scvi-tools, and other AnnData-based tools.

    JAX arrays in ``counts`` and ``obsm`` are converted to numpy via
    ``np.asarray()``.  The ``obs`` and ``var`` dicts become pandas
    DataFrames.

    Args:
        data_dict: Dictionary with keys:
            - ``counts``: JAX or numpy array of shape (n_cells, n_genes).
            - ``obs``: Dict mapping column names to per-cell arrays.
            - ``var``: Dict mapping column names to per-gene arrays.
            - ``obsm`` (optional): Dict mapping embedding names to arrays.

    Returns:
        AnnData object with ``.X``, ``.obs``, ``.var``, and ``.obsm``
        populated from the input dictionary.

    Raises:
        ImportError: If anndata or pandas is not installed.
    """
    ad = _require_anndata()
    pd = _require_pandas()

    counts_np = _to_dense_numpy(data_dict["counts"])

    obs_df = pd.DataFrame(data_dict.get("obs", {}))
    var_df = pd.DataFrame(data_dict.get("var", {}))

    adata = ad.AnnData(X=counts_np, obs=obs_df, var=var_df)

    obsm = data_dict.get("obsm", {})
    for key, value in obsm.items():
        adata.obsm[key] = np.asarray(value, dtype=np.float32)

    return adata


def from_anndata(adata: anndata.AnnData) -> dict[str, Any]:
    """Convert an AnnData object to a DiffBio data dict.

    Translates an ``anndata.AnnData`` object into the standard DiffBio
    dictionary format compatible with ``AnnDataSource.load()`` output.

    Sparse ``.X`` matrices are converted to dense before wrapping in a
    JAX array. ``.obs`` and ``.var`` DataFrames become plain dicts of
    numpy arrays. ``.obsm`` entries become JAX arrays.

    Args:
        adata: AnnData object to convert.

    Returns:
        Dictionary with keys:
            - ``counts``: Dense JAX array of shape (n_cells, n_genes).
            - ``obs``: Dict mapping column names to numpy arrays.
            - ``var``: Dict mapping column names to numpy arrays.
            - ``obsm``: Dict mapping embedding names to JAX arrays.
    """
    counts = jnp.array(_to_dense_numpy(adata.X))

    obs: dict[str, Any] = {col: np.asarray(adata.obs[col]) for col in adata.obs.columns}
    var: dict[str, Any] = {col: np.asarray(adata.var[col]) for col in adata.var.columns}

    obsm: dict[str, jnp.ndarray] = {}
    if adata.obsm is not None and len(adata.obsm) > 0:
        for key in adata.obsm.keys():
            obsm[key] = jnp.array(np.asarray(adata.obsm[key], dtype=np.float32))

    return {
        "counts": counts,
        "obs": obs,
        "var": var,
        "obsm": obsm,
    }
