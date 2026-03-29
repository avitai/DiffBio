"""Control cell mapping strategies for perturbation experiments.

Maps perturbed cells to control cells using batch-based or random strategies.
Mappings are precomputed at setup time as numpy index arrays.

References:
    - cell-load/src/cell_load/mapping_strategies/batch.py
    - cell-load/src/cell_load/mapping_strategies/random.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from datarax.core.config import StructuralConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ControlMappingConfig(StructuralConfig):
    """Configuration for control cell mapping.

    Attributes:
        strategy: Mapping strategy (``"batch"`` or ``"random"``).
        n_basal_samples: Number of control cells per perturbed cell.
        seed: Random seed for reproducibility.
        map_controls: Whether to also map control cells to other controls.
        cache_pairs: Whether to cache the mapping after first computation.
    """

    strategy: str = "random"
    n_basal_samples: int = 1
    seed: int = 42
    map_controls: bool = False
    cache_pairs: bool = False


def _build_ctrl_pools_by_ct(ctrl_mask: np.ndarray, ct_codes: np.ndarray) -> dict[int, np.ndarray]:
    """Build control index pools grouped by cell type."""
    pools: dict[int, list[int]] = {}
    for idx in np.where(ctrl_mask)[0]:
        ct = int(ct_codes[idx])
        if ct not in pools:
            pools[ct] = []
        pools[ct].append(idx)
    return {k: np.array(v) for k, v in pools.items()}


def _map_cells_to_controls(
    cell_indices: np.ndarray,
    ct_codes: np.ndarray,
    ctrl_by_ct: dict[int, np.ndarray],
    n_basal: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Map a set of cell indices to control cells from the same cell type."""
    mapping = np.empty((len(cell_indices), n_basal), dtype=np.int64)
    for i, cidx in enumerate(cell_indices):
        ct = int(ct_codes[cidx])
        pool = ctrl_by_ct.get(ct, np.array([], dtype=np.int64))
        if len(pool) == 0:
            mapping[i] = -1
            continue
        chosen = rng.choice(pool, size=n_basal, replace=len(pool) < n_basal)
        mapping[i] = chosen
    return mapping


class RandomControlMapping:
    """Map perturbed cells to random controls of the same cell type.

    For each perturbed cell, randomly selects ``n_basal_samples`` control cells
    from the same cell type, pooled across all batches.

    When ``map_controls=True``, also maps each control cell to another random
    control of the same cell type.

    When ``cache_pairs=True``, the mapping is computed once and cached for
    subsequent calls.

    Args:
        config: Mapping configuration.
    """

    def __init__(self, config: ControlMappingConfig) -> None:
        self._config = config
        self._cached_mapping: np.ndarray | None = None

    def build_mapping(self, source: Any) -> np.ndarray:
        """Build mapping from cells to control cells.

        Args:
            source: A PerturbationAnnDataSource or PerturbationConcatSource.

        Returns:
            Array of shape ``(n_mapped, n_basal_samples)`` with control
            cell indices. When ``map_controls=False``, ``n_mapped`` equals
            the number of perturbed cells. When ``True``, ``n_mapped``
            equals the total number of cells.
        """
        if self._config.cache_pairs and self._cached_mapping is not None:
            return self._cached_mapping

        ctrl_mask = source.get_control_mask()
        ct_codes = source.get_cell_type_codes()
        n_basal = self._config.n_basal_samples
        rng = np.random.default_rng(self._config.seed)

        ctrl_by_ct = _build_ctrl_pools_by_ct(ctrl_mask, ct_codes)

        if self._config.map_controls:
            # Map ALL cells (perturbed + controls) to controls
            all_indices = np.arange(len(ctrl_mask))
            mapping = _map_cells_to_controls(all_indices, ct_codes, ctrl_by_ct, n_basal, rng)
        else:
            # Map only perturbed cells
            pert_indices = np.where(~ctrl_mask)[0]
            mapping = _map_cells_to_controls(pert_indices, ct_codes, ctrl_by_ct, n_basal, rng)

        if self._config.cache_pairs:
            self._cached_mapping = mapping

        return mapping


class BatchControlMapping:
    """Map perturbed cells to controls within the same batch and cell type.

    Prefers controls from the same (batch, cell_type) group. Falls back to
    all controls from the same cell type if the batch group is empty.

    When ``map_controls=True``, also maps control cells.
    When ``cache_pairs=True``, caches after first computation.

    Args:
        config: Mapping configuration.
    """

    def __init__(self, config: ControlMappingConfig) -> None:
        self._config = config
        self._cached_mapping: np.ndarray | None = None

    def build_mapping(self, source: Any) -> np.ndarray:
        """Build mapping from cells to control cells.

        Args:
            source: A PerturbationAnnDataSource or PerturbationConcatSource.

        Returns:
            Array of shape ``(n_mapped, n_basal_samples)`` with control
            cell indices.
        """
        if self._config.cache_pairs and self._cached_mapping is not None:
            return self._cached_mapping

        ctrl_mask = source.get_control_mask()
        ct_codes = source.get_cell_type_codes()
        batch_codes = source.get_batch_codes()
        n_basal = self._config.n_basal_samples
        rng = np.random.default_rng(self._config.seed)

        # Build control pools by (batch, cell_type) and by cell_type
        ctrl_indices = np.where(ctrl_mask)[0]

        ctrl_by_batch_ct: dict[tuple[int, int], list[int]] = {}
        ctrl_by_ct: dict[int, list[int]] = {}

        for idx in ctrl_indices:
            ct = int(ct_codes[idx])
            batch = int(batch_codes[idx])
            key = (batch, ct)
            if key not in ctrl_by_batch_ct:
                ctrl_by_batch_ct[key] = []
            ctrl_by_batch_ct[key].append(idx)

            if ct not in ctrl_by_ct:
                ctrl_by_ct[ct] = []
            ctrl_by_ct[ct].append(idx)

        ctrl_by_batch_ct_arr = {k: np.array(v) for k, v in ctrl_by_batch_ct.items()}
        ctrl_by_ct_arr = {k: np.array(v) for k, v in ctrl_by_ct.items()}

        # Determine which cells to map
        if self._config.map_controls:
            cells_to_map = np.arange(len(ctrl_mask))
        else:
            cells_to_map = np.where(~ctrl_mask)[0]

        mapping = np.empty((len(cells_to_map), n_basal), dtype=np.int64)

        for i, cidx in enumerate(cells_to_map):
            ct = int(ct_codes[cidx])
            batch = int(batch_codes[cidx])
            key = (batch, ct)

            pool = ctrl_by_batch_ct_arr.get(key)
            if pool is None or len(pool) == 0:
                pool = ctrl_by_ct_arr.get(ct, np.array([], dtype=np.int64))

            if len(pool) == 0:
                mapping[i] = -1
                continue

            chosen = rng.choice(pool, size=n_basal, replace=len(pool) < n_basal)
            mapping[i] = chosen

        if self._config.cache_pairs:
            self._cached_mapping = mapping

        return mapping
