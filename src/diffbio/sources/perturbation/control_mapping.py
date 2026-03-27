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
    """

    strategy: str = "random"
    n_basal_samples: int = 1
    seed: int = 42
    map_controls: bool = False


class RandomControlMapping:
    """Map perturbed cells to random controls of the same cell type.

    For each perturbed cell, randomly selects ``n_basal_samples`` control cells
    from the same cell type, pooled across all batches.

    Args:
        config: Mapping configuration.
    """

    def __init__(self, config: ControlMappingConfig) -> None:
        self._config = config

    def build_mapping(self, source: Any) -> np.ndarray:
        """Build mapping from perturbed cells to control cells.

        Args:
            source: A PerturbationAnnDataSource or PerturbationConcatSource
                with ``get_control_mask()``, ``get_cell_type_codes()`` methods.

        Returns:
            Array of shape ``(n_perturbed, n_basal_samples)`` with control
            cell indices into the source.
        """
        ctrl_mask = source.get_control_mask()
        ct_codes = source.get_cell_type_codes()
        n_basal = self._config.n_basal_samples
        rng = np.random.default_rng(self._config.seed)

        # Build control pool per cell type
        ctrl_indices = np.where(ctrl_mask)[0]
        ctrl_by_ct: dict[int, np.ndarray] = {}
        for idx in ctrl_indices:
            ct = int(ct_codes[idx])
            if ct not in ctrl_by_ct:
                ctrl_by_ct[ct] = []
            ctrl_by_ct[ct].append(idx)
        ctrl_by_ct = {k: np.array(v) for k, v in ctrl_by_ct.items()}

        # Map each perturbed cell
        pert_indices = np.where(~ctrl_mask)[0]
        mapping = np.empty((len(pert_indices), n_basal), dtype=np.int64)

        for i, pidx in enumerate(pert_indices):
            ct = int(ct_codes[pidx])
            pool = ctrl_by_ct.get(ct, np.array([], dtype=np.int64))
            if len(pool) == 0:
                mapping[i] = -1
                continue
            chosen = rng.choice(pool, size=n_basal, replace=len(pool) < n_basal)
            mapping[i] = chosen

        return mapping


class BatchControlMapping:
    """Map perturbed cells to controls within the same batch and cell type.

    Prefers controls from the same (batch, cell_type) group. Falls back to
    all controls from the same cell type if the batch group is empty.

    Args:
        config: Mapping configuration.
    """

    def __init__(self, config: ControlMappingConfig) -> None:
        self._config = config

    def build_mapping(self, source: Any) -> np.ndarray:
        """Build mapping from perturbed cells to control cells.

        Args:
            source: A PerturbationAnnDataSource or PerturbationConcatSource.

        Returns:
            Array of shape ``(n_perturbed, n_basal_samples)`` with control
            cell indices.
        """
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

        # Convert to arrays
        ctrl_by_batch_ct_arr = {k: np.array(v) for k, v in ctrl_by_batch_ct.items()}
        ctrl_by_ct_arr = {k: np.array(v) for k, v in ctrl_by_ct.items()}

        # Map each perturbed cell
        pert_indices = np.where(~ctrl_mask)[0]
        mapping = np.empty((len(pert_indices), n_basal), dtype=np.int64)

        for i, pidx in enumerate(pert_indices):
            ct = int(ct_codes[pidx])
            batch = int(batch_codes[pidx])
            key = (batch, ct)

            # Prefer same batch + cell type
            pool = ctrl_by_batch_ct_arr.get(key)
            if pool is None or len(pool) == 0:
                # Fall back to same cell type
                pool = ctrl_by_ct_arr.get(ct, np.array([], dtype=np.int64))

            if len(pool) == 0:
                mapping[i] = -1
                continue

            chosen = rng.choice(pool, size=n_basal, replace=len(pool) < n_basal)
            mapping[i] = chosen

        return mapping
