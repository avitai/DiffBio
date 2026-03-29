"""Output space management utilities for perturbation data.

Provides functions to select count representations based on the output space
mode and to load external perturbation embeddings.

References:
    - cell-load/src/cell_load/dataset/_perturbation.py (output space logic)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from diffbio.sources.perturbation._types import OutputSpaceMode

logger = logging.getLogger(__name__)


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


def _require_torch() -> Any:
    """Import torch, raising a clear error if not installed."""
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError as err:
        raise ImportError(
            "PyTorch is required to load .pt embedding files. "
            "Install with: uv pip install 'diffbio[torch-io]'"
        ) from err


def load_external_embeddings(path: Path) -> jnp.ndarray:
    """Load perturbation embeddings from a file and convert to JAX array.

    Supports ``.npy``, ``.npz`` (key ``embeddings``), and ``.pt``
    (PyTorch tensor) files. Loading ``.pt`` requires the optional
    ``torch`` dependency.

    Args:
        path: Path to the embedding file.

    Returns:
        JAX array of shape ``(n_perturbations, embed_dim)``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
        ImportError: If loading a ``.pt`` file without torch installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        data = np.load(path)
        return jnp.array(data, dtype=jnp.float32)

    if suffix == ".npz":
        archive = np.load(path)
        if "embeddings" in archive:
            data = archive["embeddings"]
        else:
            data = archive[list(archive.keys())[0]]
        return jnp.array(data, dtype=jnp.float32)

    if suffix == ".pt":
        torch = _require_torch()
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        return jnp.array(tensor.numpy(), dtype=jnp.float32)

    raise ValueError(
        f"Unsupported embedding file extension '{suffix}'. Use .npy, .npz, or .pt format."
    )
