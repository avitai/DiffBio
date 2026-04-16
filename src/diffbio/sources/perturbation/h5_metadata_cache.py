"""Singleton H5 metadata cache for fast categorical lookups.

Reads perturbation, cell type, and batch metadata directly from H5/H5AD files
via h5py, avoiding the overhead of loading the full AnnData object. Caches
results in a process-global singleton keyed by file path.

References:
    - cell-load/src/cell_load/utils/data_utils.py (H5MetadataCache,
      GlobalH5MetadataCache)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from diffbio.sources.perturbation._utils import safe_decode_array

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CategoricalEncoding:
    """Decoded categorical labels plus per-row integer codes."""

    categories: np.ndarray
    codes: np.ndarray


def _require_h5py() -> Any:
    """Import h5py, raising a clear error if not installed."""
    try:
        import h5py  # noqa: PLC0415

        return h5py
    except ImportError as err:
        raise ImportError(
            "h5py is required for H5MetadataCache. Install with: uv pip install h5py"
        ) from err


def _read_categorical_encoding(obs: Any, column: str) -> _CategoricalEncoding:
    """Load decoded categories and integer codes for one obs column."""
    dataset = obs[column]
    if "categories" in dataset:
        categories = safe_decode_array(dataset["categories"][:])
        codes = dataset["codes"][:].astype(np.int32)
        return _CategoricalEncoding(categories=categories, codes=codes)

    raw = dataset[:]
    unique_values, inverse_indices = np.unique(raw, return_inverse=True)
    return _CategoricalEncoding(
        categories=unique_values.astype(str),
        codes=inverse_indices.astype(np.int32),
    )


class H5MetadataCache:
    """Cache for H5 file metadata to avoid repeated disk reads.

    Extracts and caches categorical encodings (categories + integer codes) for
    perturbation, cell type, and batch columns directly from the H5 file's
    ``obs`` group. Also computes a boolean control mask.

    Public properties:
        pert_categories: Unique perturbation labels (string array).
        pert_codes: Per-cell perturbation integer codes.
        cell_type_categories: Unique cell type labels.
        cell_type_codes: Per-cell cell type integer codes.
        batch_categories: Unique batch labels.
        batch_codes: Per-cell batch integer codes.
        control_mask: Boolean mask, True for control cells.
        control_pert_code: Integer code of the control perturbation.
        n_cells: Total number of cells in the file.

    Args:
        h5_path: Path to the .h5ad or .h5 file.
        pert_col: Obs column name for perturbation identity.
        cell_type_key: Obs column name for cell type.
        control_pert: Label identifying control cells.
        batch_col: Obs column name for batch/plate.

    Raises:
        ValueError: If ``control_pert`` is not found in the perturbation categories.
    """

    __slots__ = (
        "h5_path",
        "_perturbation",
        "_cell_type",
        "_batch",
        "control_mask",
        "control_pert_code",
    )

    def __init__(
        self,
        h5_path: str,
        pert_col: str = "perturbation",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        batch_col: str = "batch",
    ) -> None:
        h5py = _require_h5py()
        self.h5_path = h5_path

        with h5py.File(h5_path, "r") as f:
            obs = f["obs"]
            self._perturbation = _read_categorical_encoding(obs, pert_col)
            self._cell_type = _read_categorical_encoding(obs, cell_type_key)
            self._batch = _read_categorical_encoding(obs, batch_col)

            # -- Control mask --
            idx = np.where(self.pert_categories == control_pert)[0]
            if idx.size == 0:
                raise ValueError(
                    f"control_pert='{control_pert}' not found in {pert_col} "
                    f"categories: {list(self.pert_categories)}"
                )
            self.control_pert_code: int = int(idx[0])
            self.control_mask: np.ndarray = self.pert_codes == self.control_pert_code

    @property
    def pert_categories(self) -> np.ndarray:
        """Return unique perturbation labels."""
        return self._perturbation.categories

    @property
    def pert_codes(self) -> np.ndarray:
        """Return per-cell perturbation integer codes."""
        return self._perturbation.codes

    @property
    def cell_type_categories(self) -> np.ndarray:
        """Return unique cell type labels."""
        return self._cell_type.categories

    @property
    def cell_type_codes(self) -> np.ndarray:
        """Return per-cell integer cell type codes."""
        return self._cell_type.codes

    @property
    def batch_categories(self) -> np.ndarray:
        """Return unique batch labels."""
        return self._batch.categories

    @property
    def batch_codes(self) -> np.ndarray:
        """Return per-cell integer batch codes."""
        return self._batch.codes

    @property
    def n_cells(self) -> int:
        """Return total number of cells represented by the cached metadata."""
        return int(self.pert_codes.shape[0])

    def get_pert_names(self, codes: np.ndarray) -> np.ndarray:
        """Return perturbation labels for the given integer codes."""
        return self.pert_categories[codes]

    def get_cell_type_names(self, codes: np.ndarray) -> np.ndarray:
        """Return cell type labels for the given integer codes."""
        return self.cell_type_categories[codes]

    def get_batch_names(self, codes: np.ndarray) -> np.ndarray:
        """Return batch labels for the given integer codes."""
        return self.batch_categories[codes]


class GlobalH5MetadataCache:
    """Singleton managing a shared dict of H5MetadataCache instances.

    Thread-safe via a lock. Keyed by file path only; the first caller's
    column parameters win for a given path.
    """

    _instance: GlobalH5MetadataCache | None = None
    _lock = threading.Lock()
    _cache: dict[str, H5MetadataCache]

    def __new__(cls) -> GlobalH5MetadataCache:
        """Return the singleton instance, creating it if necessary."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._cache = {}
            return cls._instance

    def get_cache(
        self,
        h5_path: str,
        pert_col: str = "perturbation",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        batch_col: str = "batch",
    ) -> H5MetadataCache:
        """Get or create a metadata cache for the given file.

        Args:
            h5_path: Path to the H5/H5AD file.
            pert_col: Perturbation column name.
            cell_type_key: Cell type column name.
            control_pert: Control perturbation label.
            batch_col: Batch column name.

        Returns:
            Cached H5MetadataCache instance.
        """
        if h5_path not in self._cache:
            self._cache[h5_path] = H5MetadataCache(
                h5_path, pert_col, cell_type_key, control_pert, batch_col
            )
        return self._cache[h5_path]
