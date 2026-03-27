"""Three-stage on-target knockdown quality filter.

Ports cell-load's three-stage filtering for perturbation experiments:
1. Perturbation-level: filter perturbations by average knockdown.
2. Cell-level: filter individual cells by residual expression.
3. Minimum count: remove perturbations with too few remaining cells.

Controls are always preserved.

References:
    - cell-load/src/cell_load/utils/data_utils.py (filter_on_target_knockdown)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from datarax.core.config import StructuralConfig

logger = logging.getLogger(__name__)


def is_on_target_knockdown(
    gene_expression: np.ndarray,
    pert_mask: np.ndarray,
    ctrl_mask: np.ndarray,
    residual_expression: float = 0.30,
) -> bool:
    """Check if a perturbation shows on-target knockdown for a single gene.

    Returns True if the average expression in perturbed cells is below
    ``residual_expression`` times the control mean.

    Args:
        gene_expression: 1D array of expression values for the target gene.
        pert_mask: Boolean mask for perturbed cells.
        ctrl_mask: Boolean mask for control cells.
        residual_expression: Maximum allowed ratio of perturbed to control mean.

    Returns:
        True if knockdown is on-target (expression sufficiently reduced).
    """
    ctrl_mean = float(gene_expression[ctrl_mask].mean())
    if np.isclose(ctrl_mean, 0.0):
        return False

    pert_mean = float(gene_expression[pert_mask].mean())
    return (pert_mean / ctrl_mean) < residual_expression


@dataclass(frozen=True)
class KnockdownFilterConfig(StructuralConfig):
    """Configuration for OnTargetKnockdownFilter.

    Attributes:
        pert_col: Obs column name for perturbation identity.
        control_pert: Label identifying control cells.
        residual_expression: Stage 1 threshold (perturbation-level).
        cell_residual_expression: Stage 2 threshold (per-cell).
        min_cells: Stage 3 minimum cells per perturbation.
        var_gene_col: Column in var for gene names. If None, uses var index.
    """

    pert_col: str = "perturbation"
    control_pert: str = "non-targeting"
    residual_expression: float = 0.30
    cell_residual_expression: float = 0.50
    min_cells: int = 30
    var_gene_col: str | None = None


class OnTargetKnockdownFilter:
    """Three-stage quality control filter for perturbation experiments.

    **Stage 1** (perturbation-level): Filters perturbations where average
    target gene expression in perturbed cells is >= ``residual_expression``
    times the control mean.

    **Stage 2** (cell-level): Filters individual cells where target gene
    expression remains too high relative to the control mean.

    **Stage 3** (count threshold): Removes perturbations with fewer than
    ``min_cells`` remaining after stages 1 and 2.

    Controls are always preserved in all stages.

    Args:
        config: Filter configuration.
    """

    def __init__(self, config: KnockdownFilterConfig) -> None:
        self._config = config

    def process(self, source: Any) -> np.ndarray:
        """Apply three-stage filtering and return a boolean cell mask.

        Args:
            source: A PerturbationAnnDataSource with ``load()``,
                ``get_control_mask()``, and perturbation metadata.

        Returns:
            Boolean array of shape ``(n_cells,)`` — True for cells passing
            all three filter stages.
        """
        data = source.load()
        counts = np.asarray(data["counts"])
        obs = data["obs"]
        var = data["var"]

        pert_labels = np.asarray(obs[self._config.pert_col])
        ctrl_mask = source.get_control_mask()
        n_cells = len(pert_labels)

        # Build gene name -> index mapping
        if self._config.var_gene_col is not None and self._config.var_gene_col in var:
            gene_names = np.asarray(var[self._config.var_gene_col])
        else:
            gene_names = np.asarray(list(var.keys()))

        gene_to_idx: dict[str, int] = {}
        for i, name in enumerate(gene_names):
            name_str = str(name)
            if name_str not in gene_to_idx:
                gene_to_idx[name_str] = i

        unique_perts = set(pert_labels) - {self._config.control_pert}

        # --- Stage 1: Perturbation-level filter ---
        perts_passing_stage1: set[str] = set()
        for pert in unique_perts:
            if pert not in gene_to_idx:
                continue
            gene_idx = gene_to_idx[pert]
            gene_expr = counts[:, gene_idx]
            pert_mask = pert_labels == pert

            if is_on_target_knockdown(
                gene_expr, pert_mask, ctrl_mask, self._config.residual_expression
            ):
                perts_passing_stage1.add(pert)

        # --- Stage 2: Cell-level filter ---
        keep_mask = np.zeros(n_cells, dtype=bool)
        keep_mask[ctrl_mask] = True  # Always keep controls

        ctrl_mean_cache: dict[str, float] = {}

        for pert in perts_passing_stage1:
            if pert not in gene_to_idx:
                continue
            gene_idx = gene_to_idx[pert]

            if pert not in ctrl_mean_cache:
                ctrl_mean_cache[pert] = float(counts[ctrl_mask, gene_idx].mean())
            ctrl_mean = ctrl_mean_cache[pert]

            if np.isclose(ctrl_mean, 0.0):
                continue

            pert_cell_mask = pert_labels == pert
            ratios = counts[pert_cell_mask, gene_idx] / ctrl_mean
            passing = ratios < self._config.cell_residual_expression
            pert_indices = np.where(pert_cell_mask)[0]
            keep_mask[pert_indices[passing]] = True

        # --- Stage 3: Minimum cell filter ---
        for pert in unique_perts:
            pert_cell_mask = (pert_labels == pert) & keep_mask
            if pert_cell_mask.sum() < self._config.min_cells:
                keep_mask[pert_cell_mask] = False

        return keep_mask
