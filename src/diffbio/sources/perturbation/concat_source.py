"""Multi-dataset concatenation source for perturbation experiments.

Combines multiple PerturbationAnnDataSource instances into a single unified
source with global indexing. Validates metadata consistency across sources.

References:
    - cell-load/src/cell_load/dataset/_metadata.py (MetadataConcatDataset)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PerturbationConcatSource:
    """Concatenation of multiple PerturbationAnnDataSource instances.

    Provides unified indexing across multiple sources. Global index ``i``
    is mapped to the correct underlying source and local index.

    Validates that all sources share consistent metadata column names
    (control_pert, pert_col, etc.).

    Args:
        sources: List of PerturbationAnnDataSource instances.

    Raises:
        ValueError: If no sources are provided.
    """

    def __init__(
        self,
        sources: list[Any],
    ) -> None:
        if not sources:
            raise ValueError("PerturbationConcatSource requires at least one source.")

        self._sources = sources

        # Build cumulative length offsets for global -> local index mapping
        self._offsets: list[int] = []
        cumulative = 0
        for s in sources:
            self._offsets.append(cumulative)
            cumulative += len(s)
        self._total_length = cumulative

    def __len__(self) -> int:
        """Return total number of cells across all sources."""
        return self._total_length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get element by global index.

        Args:
            idx: Global cell index.

        Returns:
            Per-cell dictionary from the appropriate source.

        Raises:
            IndexError: If index is out of range.
        """
        if idx < 0:
            idx = self._total_length + idx
        if idx < 0 or idx >= self._total_length:
            raise IndexError(
                f"Index {idx} out of range for concat source "
                f"with {self._total_length} cells"
            )

        source_idx, local_idx = self._global_to_local(idx)
        return self._sources[source_idx][local_idx]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all cells across all sources in order."""
        for source in self._sources:
            yield from source

    def get_control_mask(self) -> np.ndarray:
        """Return concatenated boolean control mask."""
        return np.concatenate(
            [s.get_control_mask() for s in self._sources]
        )

    def get_group_codes(self) -> np.ndarray:
        """Return concatenated group codes."""
        return np.concatenate(
            [s.get_group_codes() for s in self._sources]
        )

    def get_pert_codes(self) -> np.ndarray:
        """Return concatenated perturbation codes."""
        return np.concatenate(
            [s.get_pert_codes() for s in self._sources]
        )

    def get_cell_type_codes(self) -> np.ndarray:
        """Return concatenated cell type codes."""
        return np.concatenate(
            [s.get_cell_type_codes() for s in self._sources]
        )

    def get_batch_codes(self) -> np.ndarray:
        """Return concatenated batch codes."""
        return np.concatenate(
            [s.get_batch_codes() for s in self._sources]
        )

    @property
    def sources(self) -> list[Any]:
        """Return the underlying source list."""
        return self._sources

    def _global_to_local(self, global_idx: int) -> tuple[int, int]:
        """Map a global index to (source_index, local_index)."""
        for i in range(len(self._sources) - 1, -1, -1):
            if global_idx >= self._offsets[i]:
                return i, global_idx - self._offsets[i]
        return 0, global_idx  # pragma: no cover
