"""Soft selection and indexing operators.

Provides differentiable relaxations of array selection operations using
SoftBool conditions and SoftIndex probability distributions instead of
discrete boolean masks and integer indices.

Key types:
- **SoftBool**: Probability in [0, 1], used in :func:`where`.
- **SoftIndex**: Probability distribution over indices (sums to 1),
  used in :func:`take_along_axis`, :func:`take`, :func:`choose`, etc.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from diffbio.core.soft_ops._types import SoftBool, SoftIndex
from diffbio.core.soft_ops._utils import canonicalize_axis, normalize_axis_argument


def where(condition: SoftBool, x: Array, y: Array) -> Array:
    """Soft where: ``x * condition + y * (1 - condition)``.

    Unlike ``jnp.where``, this smoothly interpolates between ``x`` and
    ``y`` based on the continuous condition value.

    Args:
        condition: SoftBool in [0, 1], same shape as x and y.
        x: Values selected when condition is 1.
        y: Values selected when condition is 0.

    Returns:
        Interpolated array.
    """
    return x * condition + y * (1.0 - condition)


def take_along_axis(
    x: Array,
    soft_index: SoftIndex,
    axis: int | None = -1,
) -> Array:
    """Soft take_along_axis via weighted dot product.

    ``soft_index`` must have one more dimension than ``x``: the extra
    (last) dimension contains the probability distribution over the
    elements along ``axis``.

    Args:
        x: Input array of shape ``(..., n, ...)``.
        soft_index: SoftIndex of shape ``(..., k, ..., [n])`` where
            ``[n]`` is the probability distribution dimension.
        axis: Axis in ``x`` to select from. If None, ``x`` is flattened.

    Returns:
        Array of shape ``(..., k, ...)``.
    """
    x, axis = normalize_axis_argument(x, axis)
    if x.ndim + 1 != soft_index.ndim:
        msg = (
            f"x.ndim + 1 == soft_index.ndim required, "
            f"got x.ndim={x.ndim}, soft_index.ndim={soft_index.ndim}"
        )
        raise ValueError(msg)
    x = jnp.moveaxis(x, axis, -1)
    soft_index = jnp.moveaxis(soft_index, axis, -2)
    dotprod = jnp.einsum("...n,...kn->...k", x, soft_index)
    return jnp.moveaxis(dotprod, -1, axis)


def take(
    x: Array,
    soft_index: SoftIndex,
    axis: int | None = None,
) -> Array:
    """Soft take via weighted dot product.

    Unlike :func:`take_along_axis`, ``soft_index`` is a 2-D matrix
    of shape ``(k, [n])`` applied uniformly across batch dimensions.

    Args:
        x: Input array of shape ``(..., n, ...)``.
        soft_index: SoftIndex of shape ``(k, [n])``.
        axis: Axis to select from. If None, ``x`` is flattened.

    Returns:
        Array of shape ``(..., k, ...)``.
    """
    if soft_index.ndim != 2:
        msg = f"soft_index must be (k, [n]), got shape {soft_index.shape}"
        raise ValueError(msg)
    x, axis = normalize_axis_argument(x, axis)
    if axis != x.ndim - 1:
        x = jnp.moveaxis(x, axis, -1)
    soft_index = jnp.reshape(
        soft_index,
        (1,) * (x.ndim - 1) + soft_index.shape,
    )
    x = jnp.expand_dims(x, axis)
    soft_index = jnp.moveaxis(soft_index, -2, axis)
    return jnp.sum(x * soft_index, axis=-1)


def choose(
    soft_index: SoftIndex,
    choices: Array,
) -> Array:
    """Soft choose among multiple arrays.

    Softly selects among ``choices`` using ``soft_index`` weights.

    Args:
        soft_index: SoftIndex of shape ``(..., [n])``.
        choices: Array of shape ``(n, ...)``.

    Returns:
        Weighted combination of choices.
    """
    if soft_index.ndim != choices.ndim or soft_index.shape[-1] != choices.shape[0]:
        msg = (
            f"Incompatible shapes: soft_index={soft_index.shape}, "
            f"choices={choices.shape}. Need soft_index.shape=(..., [n]) "
            f"and choices.shape=(n, ...)"
        )
        raise ValueError(msg)
    tgt_shape = jnp.broadcast_shapes(choices.shape[1:], soft_index.shape[:-1])
    choices_bcast = jnp.broadcast_to(choices, (choices.shape[0], *tgt_shape))
    choices_bcast = jnp.moveaxis(choices_bcast, 0, -1)
    return jnp.sum(choices_bcast * soft_index, axis=-1)


def dynamic_index_in_dim(
    x: Array,
    soft_index: SoftIndex,
    axis: int = 0,
    keepdims: bool = True,
) -> Array:
    """Soft dynamic indexing along a dimension.

    Selects a single element (weighted combination) along ``axis``
    using the probability distribution ``soft_index``.

    Args:
        x: Input array of shape ``(..., n, ...)``.
        soft_index: SoftIndex of shape ``([n],)``.
        axis: Axis to index.
        keepdims: If True, retains the indexed dimension as size 1.

    Returns:
        Indexed array.
    """
    axis = canonicalize_axis(axis, x.ndim)
    if x.shape[axis] != soft_index.shape[0]:
        msg = (
            f"Dimension mismatch: x.shape[{axis}]={x.shape[axis]} "
            f"vs soft_index.shape[0]={soft_index.shape[0]}"
        )
        raise ValueError(msg)
    x = jnp.moveaxis(x, axis, -1)
    x_reshaped = jnp.reshape(x, (-1, x.shape[-1]))
    dotprod = jnp.sum(x_reshaped * soft_index[None, :], axis=-1)
    y = jnp.reshape(dotprod, x.shape[:-1])
    if keepdims:
        y = jnp.expand_dims(y, axis=axis)
    return y


def dynamic_slice_in_dim(
    x: Array,
    soft_start_index: SoftIndex,
    slice_size: int,
    axis: int = 0,
) -> Array:
    """Soft dynamic slicing along a dimension.

    Extracts a soft slice of ``slice_size`` elements starting at the
    position defined by ``soft_start_index``.

    Args:
        x: Input array of shape ``(..., n, ...)``.
        soft_start_index: SoftIndex of shape ``([n],)``.
        slice_size: Number of elements to extract.
        axis: Axis to slice.

    Returns:
        Array of shape ``(..., slice_size, ...)``.
    """
    axis = canonicalize_axis(axis, x.ndim)
    if not (0 < slice_size <= x.shape[axis]):
        msg = (
            f"slice_size must satisfy 0 < slice_size <= x.shape[axis], "
            f"got slice_size={slice_size}, x.shape[axis]={x.shape[axis]}"
        )
        raise ValueError(msg)

    x_last = jnp.moveaxis(x, axis, -1)
    t_idx = jnp.arange(slice_size)

    def one_step(t: Array) -> Array:
        rolled = jnp.roll(x_last, shift=-t, axis=-1)
        return jnp.einsum("...n,n->...", rolled, soft_start_index)

    y_stack = jax.vmap(one_step)(t_idx)
    y_last = jnp.moveaxis(y_stack, 0, -1)
    return jnp.moveaxis(y_last, -1, axis)


def dynamic_slice(
    x: Array,
    soft_start_indices: Sequence[SoftIndex],
    slice_sizes: Sequence[int],
) -> Array:
    """Soft dynamic slicing across multiple dimensions.

    Applies :func:`dynamic_slice_in_dim` sequentially along each axis.

    Args:
        x: Input array of shape ``(n_1, n_2, ..., n_k)``.
        soft_start_indices: One SoftIndex per dimension.
        slice_sizes: One slice length per dimension.

    Returns:
        Array of shape ``(l_1, l_2, ..., l_k)``.
    """
    if not (len(soft_start_indices) == len(slice_sizes) == x.ndim):
        msg = (
            f"len(soft_start_indices) == len(slice_sizes) == x.ndim required, "
            f"got {len(soft_start_indices)}, {len(slice_sizes)}, {x.ndim}"
        )
        raise ValueError(msg)
    y = x
    for axis, (start, size) in enumerate(zip(soft_start_indices, slice_sizes)):
        y = dynamic_slice_in_dim(y, start, size, axis=axis)
    return y
