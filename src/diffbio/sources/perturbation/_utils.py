"""Shared utility functions for the perturbation sub-package.

Ports and adapts utility functions from cell-load's data_utils module to work
with JAX arrays and numpy instead of PyTorch tensors.

References:
    - cell-load/src/cell_load/utils/data_utils.py
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp
import numpy as np


def safe_decode_array(arr: Any) -> np.ndarray:
    """Decode byte-string arrays to UTF-8 and cast all entries to Python str.

    Args:
        arr: Array-like of bytes or other objects.

    Returns:
        Numpy string array with all elements decoded to str.
    """
    decoded: list[str] = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            decoded.append(x.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(x))
    return np.array(decoded, dtype=str)


def generate_onehot_map(keys: Iterable[str]) -> dict[str, jnp.ndarray]:
    """Build a map from each unique key to a one-hot JAX array.

    Keys are sorted to ensure deterministic ordering across runs.

    Args:
        keys: Iterable of hashable string items.

    Returns:
        Dict mapping each unique key to a float32 one-hot JAX array
        of length equal to the number of unique keys.
    """
    unique_keys = sorted(set(keys))
    n = len(unique_keys)
    identity = jnp.eye(n, dtype=jnp.float32)
    return {k: identity[i] for i, k in enumerate(unique_keys)}


def is_discrete_counts(x: jnp.ndarray, n_cells: int = 100) -> bool:
    """Detect if data appears to be raw integer counts.

    Checks whether the row sums of the first ``n_cells`` rows are integers
    (fractional part approximately zero).

    Args:
        x: Array of shape ``(n_cells, n_genes)``.
        n_cells: Number of cells to sample for detection.

    Returns:
        True if data appears to be discrete/raw counts.
    """
    top_n = min(x.shape[0], n_cells)
    row_sums = x[:top_n].sum(axis=1)
    frac_part = row_sums - jnp.floor(row_sums)
    return bool(jnp.all(jnp.abs(frac_part) < 1e-7))


def is_log_transformed(x: jnp.ndarray) -> bool:
    """Detect if data is log-transformed by checking the global maximum.

    Log1p-transformed data typically has a maximum below 15.

    Args:
        x: Array of expression values.

    Returns:
        True if data appears to be log-transformed.
    """
    return bool(x.max() < 15.0)


def split_perturbations_by_cell_fraction(
    pert_groups: dict[str, np.ndarray],
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    """Partition perturbations so the val subset approximates a target cell fraction.

    Uses a greedy algorithm: shuffles perturbations, then greedily assigns each
    to the val subset if doing so brings the val cell count closer to the target.

    Args:
        pert_groups: Dict mapping perturbation names to arrays of cell indices.
        val_fraction: Target fraction of total cells to assign to validation.
        rng: Numpy random generator for shuffling.

    Returns:
        Tuple of (train_perturbation_names, val_perturbation_names).
    """
    total_cells = sum(len(indices) for indices in pert_groups.values())
    target_val_cells = val_fraction * total_cells

    pert_size_list = [(p, len(pert_groups[p])) for p in pert_groups]
    rng.shuffle(pert_size_list)

    val_perts: list[str] = []
    current_val_cells = 0

    for pert, size in pert_size_list:
        new_val_cells = current_val_cells + size
        diff_if_add = abs(new_val_cells - target_val_cells)
        diff_if_skip = abs(current_val_cells - target_val_cells)

        if diff_if_add < diff_if_skip:
            val_perts.append(pert)
            current_val_cells = new_val_cells

    train_perts = [p for p, _ in pert_size_list if p not in set(val_perts)]
    return train_perts, val_perts
