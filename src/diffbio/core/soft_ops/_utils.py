"""Internal utility functions for soft_ops.

These are shared helpers used across the soft_ops submodules.
Not part of the public API.
"""

from typing import Literal

import jax
import jax.numpy as jnp


def validate_softness(softness: jnp.ndarray) -> None:
    """Raise ``ValueError`` if softness is not positive.

    Validation is skipped inside JAX-traced contexts (jit, grad, vmap)
    since concrete values are not available during tracing.

    Args:
        softness: The softness (temperature) parameter. Must be > 0.
    """
    if isinstance(softness, jax.core.Tracer):
        return
    if float(softness) <= 0:
        msg = f"softness must be positive, got {softness}"
        raise ValueError(msg)


def ensure_float(x: jnp.ndarray) -> jnp.ndarray:
    """Cast to default float dtype if not already floating point.

    Args:
        x: Input array or scalar.

    Returns:
        Array with floating-point dtype.
    """
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.floating):
        return x
    return x.astype(jnp.result_type(float))


def standardize_and_squash(
    x: jnp.ndarray,
    axis: int = -1,
    eps: float = 1e-6,
    temperature: float = 1.0,
    return_mean_std: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Standardize along axis then squash to (0, 1) via sigmoid.

    Steps:
        1. Standardize: ``(x - mean) / std`` along ``axis``
        2. Scale by ``1 / temperature``
        3. Squash: apply sigmoid to map to (0, 1)

    Args:
        x: Input array.
        axis: Axis along which to standardize.
        eps: Epsilon for numerical stability in std computation.
        temperature: Controls sharpness. Lower = sharper.
        return_mean_std: If True, also return mean and std.

    Returns:
        Squashed array in (0, 1). If ``return_mean_std``, returns
        ``(squashed, mean, std)`` tuple.
    """
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=axis, keepdims=True)
    std = jnp.sqrt(var + eps)
    z = (x - mean) / std
    z = z / temperature
    z = jax.nn.sigmoid(z)
    if return_mean_std:
        return z, mean, std
    return z


def unsquash_and_destandardize(
    y: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    eps: float = 1e-10,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Inverse of :func:`standardize_and_squash`.

    Steps:
        1. Unsquash: logit to map from (0, 1) back to R
        2. Scale by ``temperature``
        3. Destandardize: ``z * std + mean``

    Args:
        y: Squashed array in (0, 1).
        mean: Mean from standardization.
        std: Std from standardization.
        eps: Clipping epsilon for logit stability.
        temperature: Must match the value used in squash.

    Returns:
        Recovered array in original scale.
    """
    safe_eps = jnp.maximum(eps, 10 * jnp.finfo(y.dtype).eps)
    y = jnp.clip(y, safe_eps, 1.0 - safe_eps)
    z = jnp.log(y / (1.0 - y))
    z = z * temperature
    return z * std + mean


def quantile_interpolation_params(
    q: jnp.ndarray,
    n: int,
    method: Literal["linear", "lower", "higher", "nearest", "midpoint"],
) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
    """Compute interpolation parameters for quantile computation.

    Maps a quantile ``q`` in [0, 1] to index ``k``, interpolation
    weight ``a``, and whether to take the next element.

    Args:
        q: Quantile value(s) in [0, 1].
        n: Number of elements in the sorted array.
        method: Interpolation method.

    Returns:
        Tuple of (k, a, take_next) where k is the integer index,
        a is the interpolation weight, and take_next indicates
        whether to interpolate with the next element.
    """
    p = q * (n - 1)

    if method == "linear":
        k = jnp.floor(p).astype(jnp.int32)
        a = p - k
        take_next = True
    elif method == "lower":
        k = jnp.floor(p).astype(jnp.int32)
        k = jnp.clip(k, 0, n - 1)
        a = jnp.zeros_like(p)
        take_next = False
    elif method == "higher":
        k = jnp.ceil(p).astype(jnp.int32)
        k = jnp.clip(k, 0, n - 1)
        a = jnp.zeros_like(p)
        take_next = False
    elif method == "nearest":
        flag = jnp.less_equal(p - jnp.floor(p), 0.5)
        k = jnp.where(flag, jnp.floor(p), jnp.ceil(p)).astype(jnp.int32)
        a = jnp.zeros_like(p)
        take_next = False
    elif method == "midpoint":
        k = jnp.floor(p).astype(jnp.int32)
        a = jnp.full_like(p, 0.5)
        is_int = jnp.isclose(p, jnp.round(p))
        a = jnp.where(is_int, 0.0, a)
        take_next = True
    else:
        msg = f"Unknown quantile method: {method!r}"
        raise ValueError(msg)
    return k, a, take_next


def map_in_chunks(
    f,
    xs: jnp.ndarray,
    chunk_size: int,
) -> jnp.ndarray:
    """Map ``f`` row-wise over axis 0 using checkpointed ``lax.scan``.

    ``f`` receives a chunk of shape ``(chunk_size, *rest)`` and must
    return ``(chunk_size, *out_rest)``. Uses ``jax.checkpoint`` for
    O(n) backward memory.

    Args:
        f: Function to apply to each chunk.
        xs: Input array with shape ``(n, *rest)``.
        chunk_size: Number of rows per chunk.

    Returns:
        Output array with shape ``(n, *out_rest)``.
    """
    n = xs.shape[0]
    if chunk_size >= n:
        return f(xs)
    remainder = n % chunk_size
    if remainder:
        pad_size = chunk_size - remainder
        padding = jnp.zeros((pad_size, *xs.shape[1:]), dtype=xs.dtype)
        xs_padded = jnp.concatenate([xs, padding], axis=0)
    else:
        xs_padded = xs
    n_padded = xs_padded.shape[0]
    xs_chunked = xs_padded.reshape(
        n_padded // chunk_size,
        chunk_size,
        *xs.shape[1:],
    )
    f_remat = jax.checkpoint(f)
    _, ys = jax.lax.scan(
        lambda _, chunk: (None, f_remat(chunk)),
        None,
        xs_chunked,
    )
    ys = ys.reshape(n_padded, *ys.shape[2:])
    return ys[:n]


def reduce_in_chunks(
    f,
    xs: jnp.ndarray,
    chunk_size: int,
) -> jnp.ndarray:
    """Apply ``f`` to chunks of ``xs`` along axis 0 and sum results.

    ``f`` receives a chunk of shape ``(chunk_size, *rest)`` and must
    return a result whose shape does **not** include the chunk
    dimension. Remainder rows are zero-padded; the correction
    ``f(zeros)`` is subtracted.

    Uses ``jax.checkpoint`` for memory-efficient backpropagation.

    Args:
        f: Reduction function applied per chunk.
        xs: Input array with shape ``(n, *rest)``.
        chunk_size: Number of rows per chunk.

    Returns:
        Summed result across all chunks.
    """
    n = xs.shape[0]
    if chunk_size >= n:
        return f(xs)
    remainder = n % chunk_size
    if remainder:
        pad_size = chunk_size - remainder
        padding = jnp.zeros((pad_size, *xs.shape[1:]), dtype=xs.dtype)
        xs_padded = jnp.concatenate([xs, padding], axis=0)
    else:
        pad_size = 0
        xs_padded = xs
    n_padded = xs_padded.shape[0]
    xs_chunked = xs_padded.reshape(
        n_padded // chunk_size,
        chunk_size,
        *xs.shape[1:],
    )

    out_struct = jax.eval_shape(f, xs_chunked[0])
    init = jnp.zeros(out_struct.shape, dtype=out_struct.dtype)

    f_remat = jax.checkpoint(f)

    def body(acc: jnp.ndarray, chunk: jnp.ndarray):
        return acc + f_remat(chunk), None

    result, _ = jax.lax.scan(body, init, xs_chunked)

    if remainder:
        zero_pad = jnp.zeros((pad_size, *xs.shape[1:]), dtype=xs.dtype)
        result = result - f(zero_pad)
    return result


def canonicalize_axis(axis: int | None, num_dims: int) -> int:
    """Normalize axis to a positive integer.

    Args:
        axis: Axis index (positive or negative). Must not be None.
        num_dims: Number of dimensions in the array.

    Returns:
        Normalized positive axis index.

    Raises:
        ValueError: If axis is None or out of bounds.
    """
    if axis is None:
        msg = "axis must be specified"
        raise ValueError(msg)
    if not -num_dims <= axis < num_dims:
        msg = f"axis {axis} is out of bounds for array of dimension {num_dims}"
        raise ValueError(msg)
    if axis < 0:
        axis += num_dims
    return axis


def flatten_or_canonicalize_axis(
    x: jnp.ndarray,
    axis: int | None,
) -> tuple[jnp.ndarray, int, int | None]:
    """Flatten ``x`` if ``axis`` is None, otherwise canonicalize.

    Returns ``(x_prepared, resolved_axis, original_ndim_if_flattened)``.
    The third element is None when axis was not None.

    Args:
        x: Input array.
        axis: Axis index or None (flatten).

    Returns:
        Tuple of prepared array, resolved axis, and original ndim
        (only set when the array was flattened).
    """
    if axis is None:
        num_dims = x.ndim
        return jnp.ravel(x), 0, num_dims
    return x, canonicalize_axis(axis, x.ndim), None
