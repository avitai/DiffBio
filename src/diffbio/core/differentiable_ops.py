"""Differentiable operations for soft approximations of discrete operations.

This module provides differentiable replacements for common discrete operations,
enabling gradient flow through operations like argmax, sorting, and thresholding.

Key technique: Replace discrete operations with smooth approximations controlled
by a temperature parameter. As temperature approaches 0, the smooth approximation
approaches the hard discrete operation.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from diffbio.constants import DEFAULT_TEMPERATURE, EPSILON

# Re-export soft_threshold from nn_utils for centralized access
from diffbio.utils.nn_utils import soft_threshold

__all__ = [
    "soft_argmax",
    "soft_sort",
    "soft_threshold",
    "logsumexp_smooth_max",
    "segment_softmax",
    "gumbel_softmax",
    "differentiable_scan",
]


def logsumexp_smooth_max(
    values: Float[Array, "..."],
    temperature: float = DEFAULT_TEMPERATURE,
    axis: int | None = None,
) -> Float[Array, "..."]:
    """Compute smooth maximum using logsumexp.

    This is a differentiable approximation to max():
        smooth_max(x) = temperature * logsumexp(x / temperature)

    As temperature -> 0, this approaches hard max.
    As temperature -> inf, this approaches log(n) + mean.

    Args:
        values: Input array.
        temperature: Controls sharpness. Lower = closer to hard max.
        axis: Axis along which to compute max. None for global max.

    Returns:
        Smooth maximum value(s).

    Example:
        ```python
        values = jnp.array([1.0, 5.0, 2.0])
        logsumexp_smooth_max(values, temperature=0.1)  # Close to 5.0
        logsumexp_smooth_max(values, temperature=10.0)  # More averaged
        ```
    """
    return temperature * jax.scipy.special.logsumexp(values / temperature, axis=axis)


def soft_argmax(
    logits: Float[Array, "..."],
    temperature: float | Float[Array, ""] = DEFAULT_TEMPERATURE,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Compute soft argmax using weighted position sum.

    This is a differentiable approximation to argmax:
        soft_argmax(x) = sum(softmax(x/T) * positions)

    As temperature -> 0, this approaches hard argmax.
    As temperature -> inf, this approaches mean position.

    Args:
        logits: Input logits.
        temperature: Controls sharpness. Lower = closer to hard argmax.
        axis: Axis along which to compute argmax.

    Returns:
        Soft argmax position(s).

    Example:
        ```python
        logits = jnp.array([1.0, 5.0, 2.0])
        soft_argmax(logits, temperature=0.1)  # Close to 1.0
        ```
    """
    # Get weights via softmax
    weights = jax.nn.softmax(logits / temperature, axis=axis)

    # Create position indices
    n = logits.shape[axis]
    positions = jnp.arange(n, dtype=logits.dtype)

    # Reshape positions for broadcasting
    shape = [1] * logits.ndim
    shape[axis] = n
    positions = positions.reshape(shape)

    # Weighted sum of positions
    return jnp.sum(weights * positions, axis=axis)


def soft_sort(
    values: Float[Array, "n"],
    temperature: float = DEFAULT_TEMPERATURE,
) -> Float[Array, "n"]:
    """Differentiable approximation to sorting.

    Uses the method from "Differentiable Sorting Networks" by Cuturi et al.
    Computes a soft permutation matrix and applies it to values.

    Args:
        values: 1D array of values to sort.
        temperature: Controls sorting sharpness.

    Returns:
        Softly sorted values (approximately ascending order).

    Example:
        ```python
        values = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0])
        soft_sort(values, temperature=0.01)  # Close to [1.0, 1.0, 3.0, 4.0, 5.0]
        ```
    """
    n = values.shape[0]

    # Handle edge cases
    if n <= 1:
        return values

    # Compute pairwise comparisons: P[i,j] = prob that values[i] > values[j]
    diff = values[:, None] - values[None, :]  # (n, n)
    comparison_probs = jax.nn.sigmoid(diff / temperature)

    # Compute soft ranks: rank[i] = count of elements that values[i] is greater than
    # For ascending sort, smallest value should have rank ~0, largest ~(n-1)
    soft_ranks = jnp.sum(comparison_probs, axis=1) - 0.5  # (n,) subtract 0.5 for self-comparison

    # Create soft permutation matrix using softmax over rank differences
    # positions[i] is the target position (0 to n-1)
    # We want element with rank ~i to go to position i
    positions = jnp.arange(n, dtype=values.dtype)
    rank_diffs = positions[:, None] - soft_ranks[None, :]  # (n, n)
    perm_matrix = jax.nn.softmax(-jnp.abs(rank_diffs) / temperature, axis=1)

    # Apply permutation to get sorted values
    sorted_values = perm_matrix @ values

    return sorted_values


def segment_softmax(
    logits: Float[Array, "n"],
    segment_ids: Float[Array, "n"] | jnp.ndarray,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Float[Array, "n"]:
    """Compute softmax within segments defined by segment_ids.

    This is useful for variable-length sequences packed into a single array,
    where softmax should normalize within each sequence independently.

    Args:
        logits: Input logits.
        segment_ids: Integer array assigning each element to a segment.
        temperature: Temperature for softmax scaling.

    Returns:
        Softmax probabilities normalized within each segment.

    Example:
        ```python
        logits = jnp.array([1.0, 2.0, 3.0, 1.0, 2.0])
        segment_ids = jnp.array([0, 0, 0, 1, 1])
        result = segment_softmax(logits, segment_ids)
        # result[:3].sum() == 1.0, result[3:].sum() == 1.0
        ```
    """
    # Scale by temperature
    scaled = logits / temperature

    # Compute max per segment for numerical stability
    num_segments = int(jnp.max(segment_ids)) + 1
    segment_max = jax.ops.segment_max(scaled, segment_ids, num_segments=num_segments)

    # Subtract segment max for stability
    scaled_stable = scaled - segment_max[segment_ids]

    # Compute exp
    exp_vals = jnp.exp(scaled_stable)

    # Sum per segment
    segment_sum = jax.ops.segment_sum(exp_vals, segment_ids, num_segments=num_segments)

    # Normalize
    return exp_vals / (segment_sum[segment_ids] + EPSILON)


def gumbel_softmax(
    logits: Float[Array, "... n"],
    key: jax.Array,
    temperature: float = DEFAULT_TEMPERATURE,
    hard: bool = False,
) -> Float[Array, "... n"]:
    """Sample from a categorical distribution using Gumbel-softmax.

    This provides a differentiable approximation to categorical sampling.
    The reparameterization trick allows gradients to flow through the sample.

    Args:
        logits: Unnormalized log-probabilities.
        key: JAX random key.
        temperature: Controls sample sharpness. Lower = more discrete.
        hard: If True, return one-hot samples with straight-through gradient.

    Returns:
        Soft (or hard with soft gradients) categorical samples.

    Example:
        ```python
        logits = jnp.array([1.0, 2.0, 3.0])
        key = jax.random.PRNGKey(42)
        sample = gumbel_softmax(logits, key, temperature=0.5)
        ```
    """
    # Sample Gumbel noise
    gumbel_noise = jax.random.gumbel(key, logits.shape)

    # Add noise to logits and apply softmax
    perturbed = (logits + gumbel_noise) / temperature
    soft_sample = jax.nn.softmax(perturbed, axis=-1)

    if hard:
        # Straight-through estimator: hard forward, soft backward
        hard_sample = jax.nn.one_hot(jnp.argmax(soft_sample, axis=-1), logits.shape[-1])
        # Stop gradient for hard sample, keep gradient for soft
        return hard_sample - jax.lax.stop_gradient(soft_sample) + soft_sample
    else:
        return soft_sample


def differentiable_scan(
    step_fn: Callable[[Array, Array], tuple[Array, Array]],
    init: Array,
    xs: Array,
    unroll: int = 1,
) -> tuple[Array, Array]:
    """Differentiable scan operation for dynamic programming.

    This is a thin wrapper around jax.lax.scan that ensures proper
    gradient flow for DP algorithms like Smith-Waterman, Viterbi, etc.

    The scan applies step_fn sequentially:
        carry, y = step_fn(carry, x)
        outputs.append(y)

    Args:
        step_fn: Function (carry, x) -> (new_carry, output).
        init: Initial carry value.
        xs: Sequence of inputs to scan over.
        unroll: Number of steps to unroll for efficiency.

    Returns:
        Tuple of (final_carry, stacked_outputs).

    Example:
        ```python
        def cumsum_step(carry, x):
            new_carry = carry + x
            return new_carry, new_carry
        init = jnp.array(0.0)
        xs = jnp.array([1.0, 2.0, 3.0])
        final, outputs = differentiable_scan(cumsum_step, init, xs)
        # outputs = [1.0, 3.0, 6.0]
        ```
    """
    return jax.lax.scan(step_fn, init, xs, unroll=unroll)


def soft_one_hot(
    indices: Float[Array, "..."],
    num_classes: int,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Float[Array, "... num_classes"]:
    """Create soft one-hot encodings from continuous indices.

    This allows for differentiable "indexing" operations where the
    index itself may be a continuous value.

    Args:
        indices: Continuous index values.
        num_classes: Number of classes (output dimension).
        temperature: Controls sharpness of encoding.

    Returns:
        Soft one-hot encodings.

    Example:
        ```python
        index = jnp.array(1.5)  # Between class 1 and 2
        soft_one_hot(index, num_classes=4, temperature=0.1)
        # Returns distribution peaked between indices 1 and 2
        ```
    """
    # Create class indices
    class_indices = jnp.arange(num_classes, dtype=indices.dtype)

    # Compute distances to each class
    distances = jnp.abs(indices[..., None] - class_indices)

    # Convert to probabilities (closer = higher probability)
    logits = -distances / temperature
    return jax.nn.softmax(logits, axis=-1)


def soft_attention_weights(
    query: Float[Array, "... d"],
    keys: Float[Array, "n d"],
    temperature: float = DEFAULT_TEMPERATURE,
) -> Float[Array, "... n"]:
    """Compute soft attention weights.

    Standard scaled dot-product attention with temperature control.

    Args:
        query: Query vector(s).
        keys: Key vectors to attend over.
        temperature: Controls attention sharpness.

    Returns:
        Attention weights summing to 1.
    """
    # Compute scaled dot products
    d = query.shape[-1]
    scale = jnp.sqrt(d).astype(query.dtype)

    scores = jnp.einsum("...d,nd->...n", query, keys) / scale
    return jax.nn.softmax(scores / temperature, axis=-1)


def differentiable_topk(
    values: Float[Array, "n"],
    k: int,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Float[Array, "n"]:
    """Differentiable approximation to top-k selection.

    Returns soft weights indicating inclusion in top-k.

    Args:
        values: Input values.
        k: Number of top elements to select.
        temperature: Controls selection sharpness.

    Returns:
        Soft selection weights (approximately 1 for top-k, 0 otherwise).
    """
    n = values.shape[0]

    # Compute soft ranks
    diff = values[:, None] - values[None, :]
    ranks = jnp.sum(jax.nn.sigmoid(diff / temperature), axis=1)

    # Threshold at n-k (higher rank = higher value = should be selected)
    threshold = n - k
    return jax.nn.sigmoid((ranks - threshold) / temperature)
