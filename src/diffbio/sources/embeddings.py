"""Generic embedding file loading utilities.

This module centralizes loading of external embedding artifacts that are used
across DiffBio domains. It currently supports NumPy ``.npy`` arrays, NumPy
``.npz`` archives, and PyTorch ``.pt`` tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


def _require_torch() -> Any:
    """Import torch, raising a clear error if it is not installed."""
    try:
        import torch  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        return torch
    except ImportError as err:
        raise ImportError(
            "PyTorch is required to load .pt embedding files. "
            "Install with: uv pip install 'diffbio[torch-io]'"
        ) from err


def _load_npz_array(path: Path) -> np.ndarray:
    """Load the canonical array from a NumPy archive."""
    with np.load(path) as archive:
        if "embeddings" in archive:
            return np.asarray(archive["embeddings"], dtype=np.float32)

        first_key = next(iter(archive.files), None)
        if first_key is None:
            raise ValueError(f"Embedding archive is empty: {path}")

        return np.asarray(archive[first_key], dtype=np.float32)


def _load_pt_array(path: Path) -> np.ndarray:
    """Load a PyTorch tensor embedding artifact as a NumPy array."""
    torch = _require_torch()
    tensor = torch.load(path, map_location="cpu", weights_only=True)

    if hasattr(tensor, "numpy"):
        return np.asarray(tensor.numpy(), dtype=np.float32)

    raise TypeError(
        "Expected a PyTorch tensor in the embedding artifact, "
        f"but received {type(tensor).__name__}."
    )


def load_embedding_array(path: Path | str) -> jnp.ndarray:
    """Load an embedding matrix from a supported file format.

    Supported formats:

    1. ``.npy``: raw NumPy array
    2. ``.npz``: NumPy archive, preferring key ``embeddings``
    3. ``.pt``: PyTorch tensor saved with ``torch.save``

    Args:
        path: Path to the embedding artifact.

    Returns:
        JAX array containing the loaded embedding matrix.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the suffix is unsupported or an ``.npz`` archive is empty.
        TypeError: If a ``.pt`` artifact does not contain a tensor.
        ImportError: If ``torch`` is required but unavailable.
    """
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {resolved_path}")

    suffix = resolved_path.suffix.lower()

    if suffix == ".npy":
        return jnp.asarray(np.load(resolved_path), dtype=jnp.float32)

    if suffix == ".npz":
        return jnp.asarray(_load_npz_array(resolved_path), dtype=jnp.float32)

    if suffix == ".pt":
        return jnp.asarray(_load_pt_array(resolved_path), dtype=jnp.float32)

    raise ValueError(
        f"Unsupported embedding file extension '{suffix}'. Use .npy, .npz, or .pt format."
    )
