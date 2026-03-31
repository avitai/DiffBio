"""Shared utilities for DiffBio data sources."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_dense_float32(matrix: Any) -> np.ndarray:
    """Convert a sparse or dense matrix to a dense float32 numpy array.

    Handles scipy sparse matrices, numpy arrays, and other array-like
    inputs. Always returns a contiguous float32 numpy array.

    Args:
        matrix: Input matrix (sparse or dense).

    Returns:
        Dense numpy array with dtype float32.
    """
    import scipy.sparse  # noqa: PLC0415

    if scipy.sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)
