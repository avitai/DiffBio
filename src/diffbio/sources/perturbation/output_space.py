"""Output space management utilities for perturbation data.

Provides functions to select count representations based on the output space
mode.

References:
    - cell-load/src/cell_load/dataset/_perturbation.py (output space logic)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from diffbio.sources.perturbation._types import OutputSpaceMode


def select_output_counts(
    counts: jnp.ndarray,
    hvg_indices: np.ndarray | None,
    mode: OutputSpaceMode | str,
) -> jnp.ndarray:
    """Select count representation based on output space mode.

    Args:
        counts: Full count matrix of shape ``(n_cells, n_genes)``.
        hvg_indices: Integer indices of highly variable genes. Required
            when ``mode`` is ``OutputSpaceMode.GENE``.
        mode: Output space mode (``"gene"``, ``"all"``, or ``"embedding"``).

    Returns:
        Subset or full count matrix. For embedding mode, returns an empty
        array of shape ``(n_cells, 0)`` since counts are not used.

    Raises:
        ValueError: If mode is ``"gene"`` but ``hvg_indices`` is None.
    """
    mode = OutputSpaceMode(mode)

    if mode == OutputSpaceMode.ALL:
        return counts

    if mode == OutputSpaceMode.GENE:
        if hvg_indices is None:
            raise ValueError(
                "hvg_indices must be provided when output_space='gene'. "
                "Set hvg_col in config to specify which var column marks HVGs."
            )
        return counts[:, hvg_indices]

    # EMBEDDING mode: counts are not used
    return jnp.empty((counts.shape[0], 0), dtype=counts.dtype)
