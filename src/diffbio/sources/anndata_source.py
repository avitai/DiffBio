"""AnnData (.h5ad) data source for single-cell genomics.

This module provides AnnDataSource for loading single-cell RNA-seq data from
.h5ad files (AnnData format) and converting them to JAX-compatible data dicts
suitable for DiffBio operators.

Follows the datarax eager-loading pattern (same as HFEagerSource): all data is
loaded to JAX arrays at init, then iteration/batching uses pure JAX operations
with O(1) memory shuffling via Grain's index_shuffle.

Handles both dense and sparse count matrices, cell/gene metadata, and
optional embeddings (PCA, UMAP, etc.).

References:
    - https://anndata.readthedocs.io/
    - Wolf et al. "SCANPY: large-scale single-cell gene expression data analysis"
      Genome Biology, 2018.
"""

# TODO: Migrate to datarax

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.sources._eager_source_ops import eager_get_batch, eager_iter, eager_reset


def _require_anndata() -> Any:
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


@dataclass
class AnnDataSourceConfig(StructuralConfig):
    """Configuration for AnnDataSource.

    Attributes:
        file_path: Path to the .h5ad file (string or Path object).
        backed: Whether to open in backed mode (memory-mapped).
        shuffle: Whether to shuffle during iteration.
        seed: Integer seed for Grain's index_shuffle.
        split: Optional split name for pipeline integration.
    """

    file_path: str | None = None
    backed: bool = False
    shuffle: bool = False
    seed: int = 42
    split: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.shuffle:
            object.__setattr__(self, "stochastic", True)
            if self.stream_name is None:
                object.__setattr__(self, "stream_name", "shuffle")
        else:
            object.__setattr__(self, "stochastic", False)

        super().__post_init__()

        if self.file_path is None:
            raise ValueError("file_path is required for AnnDataSourceConfig")


class AnnDataSource(DataSourceModule):
    """Eager-loading AnnData source for single-cell RNA-seq data.

    Loads all data from .h5ad files to JAX arrays at initialization, then
    provides pure JAX iteration, batching, and indexed access. Follows the
    same eager-loading pattern as datarax's HFEagerSource.

    Provides:
        - Full dataset loading via ``load()``
        - Per-cell indexed access via ``__getitem__``
        - Iteration via ``__iter__`` with optional O(1) memory shuffling
        - Batch retrieval via ``get_batch(batch_size)``
        - Automatic sparse-to-dense conversion
        - JAX array output for count matrices and embeddings

    Output dictionary keys:
        - ``counts``: Dense JAX array of shape (n_cells, n_genes) from ``.X``
        - ``obs``: Dict of cell metadata columns from ``.obs``
        - ``var``: Dict of gene metadata columns from ``.var``
        - ``obsm``: Dict of embedding JAX arrays from ``.obsm`` (empty if absent)

    Example:
        ```python
        config = AnnDataSourceConfig(file_path="pbmc3k.h5ad")
        source = AnnDataSource(config)
        print(len(source))                # 2700
        print(source.load()["counts"].shape)  # (2700, 32738)

        for cell in source:
            print(cell["counts"].shape)   # (32738,)
            break

        batch = source.get_batch(32)
        print(batch["counts"].shape)      # (32, 32738)
        ```
    """

    # Annotate data storage for Flax NNX (prevents parameter tracking)
    data: dict[str, Any] = nnx.data()

    def __init__(
        self,
        config: AnnDataSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize AnnDataSource from a .h5ad file.

        Loads all data to JAX arrays at construction time.

        Args:
            config: AnnDataSourceConfig with file path and options.
            rngs: Optional RNG state for shuffling.
            name: Optional module name.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If anndata is not installed.
        """
        if name is None:
            name = f"AnnDataSource({config.file_path})"
        super().__init__(config, rngs=rngs, name=name)

        anndata_mod = _require_anndata()

        file_path = Path(str(config.file_path))
        if not file_path.exists():
            raise FileNotFoundError(f"AnnData file not found: {file_path}")

        adata = anndata_mod.read_h5ad(file_path, backed="r" if config.backed else None)

        # Convert count matrix to JAX array
        counts = jnp.array(_to_dense_array(adata.X))

        # Extract obs metadata as numpy arrays
        obs: dict[str, Any] = {col: np.asarray(adata.obs[col]) for col in adata.obs.columns}

        # Extract var metadata as numpy arrays
        var: dict[str, Any] = {col: np.asarray(adata.var[col]) for col in adata.var.columns}

        # Extract obsm embeddings as JAX arrays
        obsm = _load_obsm(adata)

        # Set required eager source attributes
        self.data = {
            "counts": counts,
            "obs": obs,
            "var": var,
            "obsm": obsm,
        }
        self.length: int = adata.n_obs
        self.index = nnx.Variable(0)
        self.epoch = nnx.Variable(0)
        self._seed: int = config.seed
        self.shuffle: bool = config.shuffle
        self.dataset_name: str | None = str(config.file_path)
        self.split_name: str | None = config.split
        self._dataset_info: dict[str, int] = {
            "n_genes": adata.n_vars,
            "n_cells": adata.n_obs,
        }

    # =================================================================
    # Public API: load / info
    # =================================================================

    def load(self) -> dict[str, Any]:
        """Return the full dataset as a dictionary of JAX arrays and metadata.

        Returns:
            Dictionary with keys ``counts``, ``obs``, ``var``, ``obsm``.
        """
        return dict(self.data)

    def get_dataset_info(self) -> dict[str, int]:
        """Return cached dataset metadata.

        Returns:
            Dict with ``n_genes`` and ``n_cells``.
        """
        return self._dataset_info

    # =================================================================
    # DataSourceModule protocol: __len__, __iter__, __next__, __getitem__
    # =================================================================

    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return self.length

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over cells with optional O(1) memory shuffling.

        Yields:
            Per-cell dictionaries with ``counts``, ``obs``, ``obsm`` keys.
        """
        return eager_iter(
            self.data,
            self.length,
            self.index,
            self.epoch,
            self.shuffle,
            self._seed,
            _build_cell_element,
        )

    def __next__(self) -> dict[str, Any]:
        """Get the next cell element (required by DataSourceModule).

        Raises:
            StopIteration: When iteration is exhausted.
        """
        raise StopIteration

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get data for a single cell by index.

        Supports negative indexing.

        Args:
            idx: Cell index (supports negative indexing).

        Returns:
            Dictionary with ``counts``, ``obs``, ``obsm`` keys.

        Raises:
            IndexError: If idx is out of bounds.
        """
        if idx < 0:
            idx = self.length + idx
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Cell index {idx} out of range for dataset with {self.length} cells")
        return _build_cell_element(self.data, idx)

    # =================================================================
    # Batch retrieval
    # =================================================================

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> dict[str, Any]:
        """Get a batch of cells.

        Stateful (advances internal index) when called without ``key``.
        Stateless (random sampling) when called with ``key``.

        Args:
            batch_size: Number of cells per batch.
            key: Optional RNG key for stateless random sampling.

        Returns:
            Dictionary with batched arrays.
        """

        def _gather(data: dict[str, Any], indices: jax.Array) -> dict[str, Any]:
            counts = data["counts"][indices]
            obs = {col: np.asarray(arr)[np.array(indices)] for col, arr in data["obs"].items()}
            obsm: dict[str, jnp.ndarray] = {}
            for emb_name, emb_arr in data["obsm"].items():
                obsm[emb_name] = emb_arr[indices]
            return {"counts": counts, "obs": obs, "obsm": obsm}

        return eager_get_batch(
            self.data,
            self.length,
            self.index,
            self.epoch,
            self.shuffle,
            self._seed,
            batch_size,
            key,
            _gather,
        )

    # =================================================================
    # State management
    # =================================================================

    def reset(self, seed: int | None = None) -> None:
        """Reset source to the beginning.

        Args:
            seed: Unused (uses config seed).
        """
        del seed
        eager_reset(self.index, self.epoch, self._cache)

    def set_shuffle(self, shuffle: bool) -> None:
        """Enable or disable shuffling.

        Args:
            shuffle: Whether to shuffle data during iteration.
        """
        self.shuffle = shuffle

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AnnDataSource("
            f"dataset={self.dataset_name}, "
            f"length={self.length}, "
            f"shuffle={self.shuffle}, "
            f"epoch={self.epoch.get_value()})"
        )


# =====================================================================
# Module-level helpers (kept outside the class)
# =====================================================================


def _load_obsm(adata: Any) -> dict[str, jnp.ndarray]:
    """Load obsm embeddings as a dict of JAX arrays.

    Args:
        adata: AnnData object.

    Returns:
        Dict mapping embedding names to JAX arrays,
        or empty dict if no obsm data exists.
    """
    if adata.obsm is None or len(adata.obsm) == 0:
        return {}

    return {
        key: jnp.array(np.asarray(adata.obsm[key], dtype=np.float32)) for key in adata.obsm.keys()
    }


def _build_cell_element(data: dict[str, Any], idx: int) -> dict[str, Any]:
    """Build a per-cell dictionary from the full dataset at a given index.

    Args:
        data: The full data dict with ``counts``, ``obs``, ``obsm`` keys.
        idx: Cell index.

    Returns:
        Per-cell dictionary with scalar obs values and 1D arrays.
    """
    cell_counts = data["counts"][idx]
    cell_obs = {col: arr[idx] for col, arr in data["obs"].items()}
    cell_obsm: dict[str, jnp.ndarray] = {}
    for emb_name, emb_arr in data["obsm"].items():
        cell_obsm[emb_name] = emb_arr[idx]
    return {"counts": cell_counts, "obs": cell_obs, "obsm": cell_obsm}
