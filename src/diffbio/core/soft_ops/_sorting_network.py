"""Soft bitonic sorting network.

Implements differentiable sorting and argsort using a bitonic sorting
network with soft compare-and-swap operations. The network has
O(n log^2 n) comparisons and produces smooth approximations to the
sorted output and permutation matrix.

Input is padded to the nearest power of 2 for the bitonic network,
then truncated back to the original length.
"""

import jax
import jax.numpy as jnp
from jax import Array

from diffbio.core.soft_ops.elementwise import SigmoidalMode


def _soft_compare_and_swap(
    a: Array,
    b: Array,
    softness: float | Array,
    mode: SigmoidalMode,
) -> tuple[Array, Array, Array]:
    """Soft compare-and-swap via sigmoidal mixing.

    Returns ``(soft_min, soft_max, sigma)`` where ``sigma`` is the
    probability that ``a < b``.
    """
    from diffbio.core.soft_ops.elementwise import sigmoidal

    sigma = sigmoidal(a - b, softness=softness, mode=mode)
    soft_min = sigma * b + (1.0 - sigma) * a
    soft_max = sigma * a + (1.0 - sigma) * b
    return soft_min, soft_max, sigma


def _bitonic_sort_ascending(
    x: Array,
    softness: float | Array,
    mode: SigmoidalMode,
) -> Array:
    """1-D ascending bitonic sort. ``x`` must have power-of-2 length."""
    n = x.shape[0]
    num_phases = (n.bit_length() - 1) if n > 1 else 0

    for phase in range(num_phases):
        for sub_step in range(phase + 1):
            d = 1 << (phase - sub_step)
            indices = jnp.arange(n)
            partner = indices ^ d
            block_size = 1 << (phase + 1)
            ascending_block = (indices & block_size) == 0

            soft_min, soft_max, _ = _soft_compare_and_swap(
                x,
                x[partner],
                softness,
                mode,
            )
            x = jnp.where(
                ascending_block,
                jnp.where(indices < partner, soft_min, soft_max),
                jnp.where(indices < partner, soft_max, soft_min),
            )
    return x


def sort_via_sorting_network(
    x: Array,
    softness: float | Array,
    mode: SigmoidalMode,
    descending: bool,
    standardized: bool = False,
) -> Array:
    """Sort along the last axis using a soft bitonic sorting network.

    Args:
        x: Input array of shape ``(..., n)``.
        softness: Controls sharpness of compare-and-swap.
        mode: Smoothness mode for sigmoidal.
        descending: If True, sort in descending order.
        standardized: If True, input is already in (0, 1) from sigmoid.

    Returns:
        Sorted array of shape ``(..., n)``.
    """
    *batch_shape, n = x.shape

    n_padded = 1 << (n - 1).bit_length() if n > 1 else 2
    if n_padded > n:
        pad_val = 1.0 if standardized else jnp.max(x) + 1.0
        pad_width = [(0, 0)] * len(batch_shape) + [(0, n_padded - n)]
        x = jnp.pad(x, pad_width, constant_values=pad_val)

    if batch_shape:
        x_flat = x.reshape(-1, n_padded)
        sorted_flat = jax.vmap(
            lambda row: _bitonic_sort_ascending(row, softness, mode),
        )(x_flat)
        sorted_x = sorted_flat.reshape(*batch_shape, n_padded)
    else:
        sorted_x = _bitonic_sort_ascending(x, softness, mode)

    if n_padded > n:
        sorted_x = sorted_x[..., :n]
    if descending:
        sorted_x = jnp.flip(sorted_x, axis=-1)
    return sorted_x


def _bitonic_argsort_ascending(
    x: Array,
    softness: float | Array,
    mode: SigmoidalMode,
) -> tuple[Array, Array]:
    """1-D ascending bitonic sort with permutation tracking.

    Returns ``(sorted_x, P)`` where ``P`` is an ``(n, n)`` soft
    permutation matrix: ``P[sorted_pos, original_elem]`` is the
    probability that ``original_elem`` ends up at ``sorted_pos``.
    """
    n = x.shape[0]
    perm = jnp.eye(n)
    num_phases = (n.bit_length() - 1) if n > 1 else 0

    for phase in range(num_phases):
        for sub_step in range(phase + 1):
            d = 1 << (phase - sub_step)
            indices = jnp.arange(n)
            partner = indices ^ d
            block_size = 1 << (phase + 1)
            ascending_block = (indices & block_size) == 0

            soft_min, soft_max, sigma = _soft_compare_and_swap(
                x,
                x[partner],
                softness,
                mode,
            )
            x = jnp.where(
                ascending_block,
                jnp.where(indices < partner, soft_min, soft_max),
                jnp.where(indices < partner, soft_max, soft_min),
            )

            gets_min = (ascending_block & (indices < partner)) | (
                ~ascending_block & (indices >= partner)
            )
            mix = jnp.where(gets_min, sigma, 1.0 - sigma)

            perm_partner = perm[partner]
            perm = (1.0 - mix[:, None]) * perm + mix[:, None] * perm_partner

    return x, perm


def argsort_via_sorting_network(
    x: Array,
    softness: float | Array,
    mode: SigmoidalMode,
    descending: bool,
    standardized: bool = False,
) -> Array:
    """Argsort via soft bitonic sorting network.

    Returns a soft permutation matrix ``P`` of shape ``(..., n, n)``
    where ``P[..., sorted_pos, original_elem]`` is the probability
    that ``original_elem`` ends up at ``sorted_pos``.

    Args:
        x: Input array of shape ``(..., n)``.
        softness: Controls sharpness of compare-and-swap.
        mode: Smoothness mode for sigmoidal.
        descending: If True, reverse the sort order.
        standardized: If True, input is already in (0, 1).

    Returns:
        Soft permutation matrix of shape ``(..., n, n)``.
    """
    *batch_shape, n = x.shape

    n_padded = 1 << (n - 1).bit_length() if n > 1 else 2
    if n_padded > n:
        pad_val = 1.0 if standardized else jnp.max(x) + 1.0
        pad_width = [(0, 0)] * len(batch_shape) + [(0, n_padded - n)]
        x = jnp.pad(x, pad_width, constant_values=pad_val)

    if batch_shape:
        x_flat = x.reshape(-1, n_padded)
        _, perm_flat = jax.vmap(
            lambda row: _bitonic_argsort_ascending(row, softness, mode),
        )(x_flat)
        perm = perm_flat.reshape(*batch_shape, n_padded, n_padded)
    else:
        _, perm = _bitonic_argsort_ascending(x, softness, mode)

    if n_padded > n:
        perm = perm[..., :n, :n]
        perm = perm / jnp.clip(jnp.sum(perm, axis=-1, keepdims=True), min=1e-10)

    if descending:
        perm = jnp.flip(perm, axis=-2)
    return perm
