"""Neural network utilities for DiffBio.

This module provides shared utility functions for building and initializing
neural network components, ensuring consistency across operators.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from diffbio.constants import DEFAULT_TEMPERATURE, EPSILON


def init_learnable_param(value: float) -> nnx.Param:
    """Initialize a learnable parameter from a scalar value.

    Args:
        value: Initial scalar value for the parameter.

    Returns:
        An nnx.Param wrapping a JAX array containing the value.

    Example:
        >>> temperature = init_learnable_param(1.0)
        >>> threshold = init_learnable_param(20.0)
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
        >>> rngs = ensure_rngs(rngs)  # Use passed rngs or create default
        >>> layer = nnx.Linear(10, 20, rngs=rngs)
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
        >>> key = get_rng_key(rngs, "sample")
        >>> noise = jax.random.normal(key, shape)
    """
    if rngs is not None and stream_name in rngs:
        return getattr(rngs, stream_name)()
    return jax.random.key(fallback_seed)


def build_mlp_layers(
    in_features: int,
    hidden_dim: int,
    num_layers: int,
    rngs: nnx.Rngs,
    *,
    with_dropout: bool = False,
    dropout_rate: float = 0.1,
) -> tuple[nnx.List, nnx.List | None, int]:
    """Build MLP hidden layers as an nnx.List.

    Creates a list of Linear layers suitable for use in MLP architectures.
    Optionally creates corresponding dropout layers.

    Args:
        in_features: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of hidden layers to create.
        rngs: Flax NNX random number generators.
        with_dropout: Whether to create dropout layers.
        dropout_rate: Dropout rate if creating dropout layers.

    Returns:
        Tuple of (layers, dropout_layers, output_dim):
            - layers: nnx.List of Linear layers
            - dropout_layers: nnx.List of Dropout layers (None if with_dropout=False)
            - output_dim: Output dimension after all layers

    Example:
        >>> layers, dropouts, out_dim = build_mlp_layers(
        ...     in_features=84,
        ...     hidden_dim=64,
        ...     num_layers=2,
        ...     rngs=rngs,
        ...     with_dropout=True,
        ... )
        >>> for layer, dropout in zip(layers, dropouts):
        ...     x = nnx.relu(layer(x))
        ...     x = dropout(x)
    """
    layers: list[nnx.Linear] = []
    dropout_layers: list[nnx.Dropout] | None = [] if with_dropout else None
    current_dim = in_features

    for _ in range(num_layers):
        layers.append(nnx.Linear(current_dim, hidden_dim, rngs=rngs))
        if with_dropout and dropout_layers is not None:
            dropout_layers.append(nnx.Dropout(rate=dropout_rate, rngs=rngs))
        current_dim = hidden_dim

    return (
        nnx.List(layers),
        nnx.List(dropout_layers) if dropout_layers is not None else None,
        current_dim,
    )


def soft_threshold(
    values: Array,
    threshold: Array | float,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Array:
    """Apply soft sigmoid-based thresholding.

    Returns values close to 1 where values > threshold, and close to 0 otherwise.
    The transition is smooth, controlled by temperature.

    Args:
        values: Input values to threshold.
        threshold: Threshold value(s).
        temperature: Controls sharpness of transition (lower = sharper).

    Returns:
        Soft thresholded values in [0, 1].

    Example:
        >>> weights = soft_threshold(quality_scores, threshold=20.0, temperature=1.0)
        >>> filtered = sequence * weights[:, None]
    """
    return jax.nn.sigmoid((values - threshold) / temperature)


def temperature_scaled_softmax(
    logits: Array,
    temperature: float = DEFAULT_TEMPERATURE,
    axis: int = -1,
) -> Array:
    """Apply softmax with temperature scaling.

    Args:
        logits: Input logits.
        temperature: Temperature for scaling (higher = softer distribution).
        axis: Axis along which to compute softmax.

    Returns:
        Probability distribution from temperature-scaled softmax.
    """
    return jax.nn.softmax(logits / temperature, axis=axis)


def sigmoid_blend(
    value_a: Array,
    value_b: Array,
    blend_weight: Array | float,
) -> Array:
    """Blend two values with sigmoid-weighted combination.

    Computes: sigmoid(blend_weight) * value_a + (1 - sigmoid(blend_weight)) * value_b

    Args:
        value_a: First value (selected when blend_weight is high).
        value_b: Second value (selected when blend_weight is low).
        blend_weight: Raw weight (passed through sigmoid).

    Returns:
        Blended result.
    """
    weight = jax.nn.sigmoid(blend_weight)
    return weight * value_a + (1 - weight) * value_b


def safe_divide(
    numerator: Array,
    denominator: Array,
    epsilon: float = EPSILON,
) -> Array:
    """Safely divide two arrays, avoiding division by zero.

    Args:
        numerator: Numerator array.
        denominator: Denominator array.
        epsilon: Small value to add to denominator.

    Returns:
        numerator / (denominator + epsilon)
    """
    return numerator / (denominator + epsilon)


def safe_log(
    x: Array,
    epsilon: float = EPSILON,
) -> Array:
    """Safely compute log, avoiding log(0).

    Args:
        x: Input array (should be non-negative).
        epsilon: Small value to add to input.

    Returns:
        log(x + epsilon)
    """
    return jnp.log(x + epsilon)
