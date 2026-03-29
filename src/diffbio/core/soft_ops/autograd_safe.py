"""Autograd-safe math operations.

Provides NaN-free alternatives to standard JAX math functions by using
the double-where trick: the forward pass computes the correct value
even at domain boundaries, and the backward pass produces finite (zero)
gradients instead of NaN/Inf.

The double-where trick works by:
    1. Replacing problematic inputs with safe values (e.g., 0 -> 1 for sqrt)
    2. Computing the function on the safe input
    3. Using ``jnp.where`` to select the safe output or a fallback (e.g., 0)

Because JAX traces through both branches of ``jnp.where``, step 1
ensures the "unused" branch never produces NaN in its gradient.
"""

import jax.numpy as jnp
from jax import Array


def sqrt(x: Array) -> Array:
    """Autograd-safe square root.

    Returns ``sqrt(x)`` for ``x > 0`` and ``0`` otherwise, without
    producing NaN gradients at ``x = 0``.

    Args:
        x: Input array.

    Returns:
        Elementwise square root, safe for autodiff.
    """
    safe_x = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, jnp.sqrt(safe_x), 0.0)


def arcsin(x: Array) -> Array:
    """Autograd-safe arcsine.

    Returns ``arcsin(x)`` for ``|x| < 1`` and ``+/-pi/2`` at the
    boundary, without producing NaN gradients at ``x = +/-1``.

    Args:
        x: Input array with values in [-1, 1].

    Returns:
        Elementwise arcsine, safe for autodiff.
    """
    interior = jnp.abs(x) < 1
    safe_x = jnp.where(interior, x, 0.0)
    return jnp.where(interior, jnp.arcsin(safe_x), jnp.sign(x) * (jnp.pi / 2))


def arccos(x: Array) -> Array:
    """Autograd-safe arccosine.

    Returns ``arccos(x)`` for ``|x| < 1``, ``0`` at ``x = 1``, and
    ``pi`` at ``x = -1``, without producing NaN gradients at the
    boundary.

    Args:
        x: Input array with values in [-1, 1].

    Returns:
        Elementwise arccosine, safe for autodiff.
    """
    interior = jnp.abs(x) < 1
    safe_x = jnp.where(interior, x, 0.0)
    return jnp.where(interior, jnp.arccos(safe_x), jnp.where(x >= 1, 0.0, jnp.pi))


def div(x: Array, y: Array) -> Array:
    """Autograd-safe division.

    Returns ``x / y`` when ``y != 0`` and ``0`` otherwise, without
    producing NaN gradients at ``y = 0``.

    Args:
        x: Numerator array.
        y: Denominator array.

    Returns:
        Elementwise safe division.
    """
    nonzero = y != 0
    safe_y = jnp.where(nonzero, y, 1.0)
    return jnp.where(nonzero, x / safe_y, 0.0)


def log(x: Array) -> Array:
    """Autograd-safe natural logarithm.

    Returns ``log(x)`` for ``x > 0`` and ``0`` otherwise, without
    producing NaN gradients at ``x = 0``.

    Args:
        x: Input array.

    Returns:
        Elementwise natural logarithm, safe for autodiff.
    """
    safe_x = jnp.where(x > 0, x, 1.0)
    return jnp.where(x > 0, jnp.log(safe_x), 0.0)


def norm(x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
    """Autograd-safe L2 norm.

    Computes ``sqrt(sum(x**2))`` using :func:`sqrt`, avoiding NaN
    gradients when the norm is zero.

    Args:
        x: Input array.
        axis: Axis or axes along which to compute the norm.
        keepdims: If True, retains reduced axes with size 1.

    Returns:
        L2 norm along the given axis, safe for autodiff.
    """
    return sqrt(jnp.sum(x * x, axis=axis, keepdims=keepdims))
