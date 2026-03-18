"""AnnData (.h5ad) data source for single-cell genomics.

This module provides AnnDataSource for loading single-cell RNA-seq data from
.h5ad files (AnnData format) and converting them to JAX-compatible data dicts
suitable for DiffBio operators.

Handles both dense and sparse count matrices, cell/gene metadata, and
optional embeddings (PCA, UMAP, etc.).

References:
    - https://anndata.readthedocs.io/
    - Wolf et al. "SCANPY: large-scale single-cell gene expression data analysis"
      Genome Biology, 2018.
"""

# TODO: Migrate to datarax as DataSourceModule subclass

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


def _require_anndata():
    """Import anndata, raising a clear error if not installed.

    Returns:
        The anndata module.

    Raises:
        ImportError: If anndata is not installed.
    """
    try:
        import anndata  # noqa: PLC0415

        return anndata
    except ImportError as err:
        raise ImportError(
            "anndata is required for AnnDataSource. Install with: uv pip install anndata"
        ) from err


def _to_dense_array(matrix: Any) -> np.ndarray:
    """Convert a matrix (dense or sparse) to a dense numpy array.

    Args:
        matrix: A numpy array or scipy sparse matrix.

    Returns:
        Dense numpy array with float32 dtype.
    """
    import scipy.sparse  # noqa: PLC0415

    if scipy.sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


class AnnDataSource:
    """Data source for loading .h5ad files into JAX-compatible data dicts.

    Reads AnnData objects from .h5ad files and provides:
    - Full dataset loading via ``load()``
    - Per-cell indexed access via ``__getitem__``
    - Automatic sparse-to-dense conversion
    - JAX array output for count matrices and embeddings

    Output dictionary keys:
        - ``counts``: Dense JAX array of shape (n_cells, n_genes) from ``.X``
        - ``obs``: Dict of cell metadata columns from ``.obs``
        - ``var``: Dict of gene metadata columns from ``.var``
        - ``obsm``: Dict of embedding JAX arrays from ``.obsm`` (empty if absent)

    Example:
        ```python
        source = AnnDataSource("pbmc3k.h5ad")
        data = source.load()
        print(data["counts"].shape)   # (2700, 32738)
        print(data["obs"].keys())     # dict_keys(['cell_type', ...])

        cell = source[0]
        print(cell["counts"].shape)   # (32738,)
        ```
    """

    # TODO: Migrate to datarax as DataSourceModule subclass

    def __init__(self, file_path: str | Path) -> None:
        """Initialize AnnDataSource from a .h5ad file path.

        Args:
            file_path: Path to the .h5ad file (string or Path object).

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If anndata is not installed.
        """
        self._anndata = _require_anndata()
        self._file_path = Path(file_path)

        if not self._file_path.exists():
            raise FileNotFoundError(f"AnnData file not found: {self._file_path}")

        self._adata = self._anndata.read_h5ad(self._file_path)

    def load(self) -> dict[str, Any]:
        """Load the full dataset as a dictionary of JAX arrays and metadata.

        Returns:
            Dictionary with keys:
                - ``counts``: JAX array of shape (n_cells, n_genes)
                - ``obs``: Dict mapping column names to lists of cell metadata
                - ``var``: Dict mapping column names to lists of gene metadata
                - ``obsm``: Dict mapping embedding names to JAX arrays
        """
        counts = jnp.array(_to_dense_array(self._adata.X))
        obs = {col: list(self._adata.obs[col]) for col in self._adata.obs.columns}
        var = {col: list(self._adata.var[col]) for col in self._adata.var.columns}
        obsm = self._load_obsm()

        return {"counts": counts, "obs": obs, "var": var, "obsm": obsm}

    def _load_obsm(self) -> dict[str, jnp.ndarray]:
        """Load obsm embeddings as a dict of JAX arrays.

        Returns:
            Dict mapping embedding names to JAX arrays,
            or empty dict if no obsm data exists.
        """
        if self._adata.obsm is None or len(self._adata.obsm) == 0:
            return {}

        return {
            key: jnp.array(np.asarray(self._adata.obsm[key], dtype=np.float32))
            for key in self._adata.obsm.keys()
        }

    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return self._adata.n_obs

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get data for a single cell by index.

        Supports negative indexing (e.g., ``source[-1]`` returns the last cell).

        Args:
            idx: Cell index (supports negative indexing).

        Returns:
            Dictionary with keys:
                - ``counts``: JAX array of shape (n_genes,)
                - ``obs``: Dict of scalar metadata for this cell
                - ``obsm``: Dict mapping embedding names to 1D JAX arrays

        Raises:
            IndexError: If idx is out of bounds.
        """
        n_cells = len(self)

        # Normalize negative indices
        if idx < 0:
            idx = n_cells + idx

        if idx < 0 or idx >= n_cells:
            raise IndexError(f"Cell index {idx} out of range for dataset with {n_cells} cells")

        cell_counts = jnp.array(_to_dense_array(self._adata.X[idx]).ravel())
        cell_obs = {col: self._adata.obs[col].iloc[idx] for col in self._adata.obs.columns}
        cell_obsm = self._cell_obsm(idx)

        return {"counts": cell_counts, "obs": cell_obs, "obsm": cell_obsm}

    def _cell_obsm(self, idx: int) -> dict[str, jnp.ndarray]:
        """Extract obsm embeddings for a single cell.

        Args:
            idx: Cell index (already normalized to non-negative).

        Returns:
            Dict mapping embedding names to 1D JAX arrays.
        """
        if self._adata.obsm is None or len(self._adata.obsm) == 0:
            return {}

        return {
            key: jnp.array(np.asarray(self._adata.obsm[key][idx], dtype=np.float32))
            for key in self._adata.obsm.keys()
        }
