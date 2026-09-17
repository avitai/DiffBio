"""Neural network utilities for DiffBio.

This module provides shared utility functions for building and initializing
neural network components, ensuring consistency across operators.
"""

from typing import TypedDict

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


class ArtifexMLPKwargs(TypedDict):
    """Typed shared kwargs for direct Artifex MLP construction."""

    activation: str
    output_activation: str | None
    use_batch_norm: bool


ARTIFEX_RELU_MLP_KWARGS: ArtifexMLPKwargs = {
    "activation": "relu",
    "output_activation": "relu",
    "use_batch_norm": False,
}
ARTIFEX_RELU_BATCH_NORM_MLP_KWARGS: ArtifexMLPKwargs = {
    "activation": "relu",
    "output_activation": "relu",
    "use_batch_norm": True,
}
ARTIFEX_GELU_MLP_KWARGS: ArtifexMLPKwargs = {
    "activation": "gelu",
    "output_activation": "gelu",
    "use_batch_norm": False,
}
ARTIFEX_GELU_NO_OUTPUT_MLP_KWARGS: ArtifexMLPKwargs = {
    "activation": "gelu",
    "output_activation": None,
    "use_batch_norm": False,
}


def init_learnable_param(value: float) -> nnx.Param:
    """Initialize a learnable parameter from a scalar value.

    Args:
        value: Initial scalar value for the parameter.

    Returns:
        An nnx.Param wrapping a JAX array containing the value.

    Example:
        ```python
        temperature = init_learnable_param(1.0)
        threshold = init_learnable_param(20.0)
        ```
    """
    return nnx.Param(jnp.array(value))


def extract_windows_1d(
    signal: Array,
    window_size: int,
    pad_mode: str = "edge",
) -> Array:
    """Extract sliding windows from a 1D signal with padding.

    This utility function pads the input signal and extracts overlapping
    windows of the specified size, one centered at each position.

    Args:
        signal: Input signal of shape (length, features).
        window_size: Size of each window (should be odd for symmetric padding).
        pad_mode: Padding mode for boundaries ("edge", "constant", etc.).

    Returns:
        Windows of shape (length, window_size, features).

    Example:
        ```python
        signal = jnp.ones((100, 4))  # 100 positions, 4 features
        windows = extract_windows_1d(signal, window_size=11)
        assert windows.shape == (100, 11, 4)
        ```
    """
    length = signal.shape[0]
    num_features = signal.shape[1]
    half_window = window_size // 2

    # Pad signal for boundary positions
    padded_signal = jnp.pad(
        signal,
        ((half_window, half_window), (0, 0)),
        mode=pad_mode,
    )

    # Extract all windows using vmap
    def extract_single_window(pos: Array | int) -> Array:
        return jax.lax.dynamic_slice(
            padded_signal,
            (pos, 0),
            (window_size, num_features),
        )

    positions = jnp.arange(length)
    all_windows = jax.vmap(extract_single_window)(positions)

    return all_windows
