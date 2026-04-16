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


def ensure_rngs(rngs: nnx.Rngs | None, seed: int = 0) -> nnx.Rngs:
    """Ensure rngs is initialized, creating a default if None.

    Args:
        rngs: Optional Flax NNX random number generators.
        seed: Seed to use if creating new rngs (default: 0).

    Returns:
        The provided rngs if not None, otherwise a new nnx.Rngs instance.

    Example:
        ```python
        rngs = ensure_rngs(rngs)  # Use passed rngs or create default
        layer = nnx.Linear(10, 20, rngs=rngs)
        ```
    """
    if rngs is not None:
        return rngs
    return nnx.Rngs(seed)


def get_rng_key(
    rngs: nnx.Rngs | None,
    stream_name: str = "params",
    fallback_seed: int = 0,
) -> jax.Array:
    """Get an RNG key from rngs with fallback.

    Args:
        rngs: Optional Flax NNX random number generators.
        stream_name: Name of the RNG stream to use.
        fallback_seed: Seed to use if rngs is None.

    Returns:
        A JAX PRNG key.

    Example:
        ```python
        key = get_rng_key(rngs, "sample")
        noise = jax.random.normal(key, shape)
        ```
    """
    if rngs is not None and stream_name in rngs:
        return getattr(rngs, stream_name)()
    return jax.random.key(fallback_seed)


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
