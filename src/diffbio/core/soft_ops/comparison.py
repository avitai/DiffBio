"""Soft comparison operators.

Provides differentiable relaxations of elementwise comparison operations
(greater, less, equal, etc.) returning SoftBool values in [0, 1].

Each function uses :func:`~diffbio.core.soft_ops.elementwise.sigmoidal`
as the underlying smooth step function, inheriting the multi-mode
smoothness options.
"""

import jax.numpy as jnp
from jax import Array

from diffbio.core.soft_ops._types import SoftBool
from diffbio.core.soft_ops._utils import ensure_float
from diffbio.core.soft_ops.elementwise import Mode, abs, sigmoidal
from diffbio.core.soft_ops.logical import logical_not


def greater(
    x: Array,
    y: float | Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Soft ``x > y``.

    Uses sigmoidal on ``x - y - epsilon`` so the output approaches 0
    at equality as softness -> 0.

    Args:
        x: First input array.
        y: Second input array (broadcastable with x).
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.
        epsilon: Small offset for strict inequality at the limit.

    Returns:
        SoftBool in [0, 1].
    """
    x = ensure_float(x)
    if mode == "hard":
        return jnp.greater(x, y).astype(x.dtype)
    return sigmoidal(x - y - epsilon, softness=softness, mode=mode)


def greater_equal(
    x: Array,
    y: float | Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Soft ``x >= y``.

    Uses sigmoidal on ``x - y + epsilon`` so the output approaches 1
    at equality as softness -> 0.

    Args:
        x: First input array.
        y: Second input array.
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.
        epsilon: Small offset for non-strict inequality at the limit.

    Returns:
        SoftBool in [0, 1].
    """
    x = ensure_float(x)
    if mode == "hard":
        return jnp.greater_equal(x, y).astype(x.dtype)
    return sigmoidal(x - y + epsilon, softness=softness, mode=mode)


def less(
    x: Array,
    y: float | Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Soft ``x < y``. Complement of :func:`greater_equal`.

    Args:
        x: First input array.
        y: Second input array.
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.
        epsilon: Small offset.

    Returns:
        SoftBool in [0, 1].
    """
    x = ensure_float(x)
    if mode == "hard":
        return jnp.less(x, y).astype(x.dtype)
    return logical_not(
        greater_equal(x, y, softness=softness, mode=mode, epsilon=epsilon),
    )


def less_equal(
    x: Array,
    y: float | Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Soft ``x <= y``. Complement of :func:`greater`.

    Args:
        x: First input array.
        y: Second input array.
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.
        epsilon: Small offset.

    Returns:
        SoftBool in [0, 1].
    """
    x = ensure_float(x)
    if mode == "hard":
        return jnp.less_equal(x, y).astype(x.dtype)
    return logical_not(
        greater(x, y, softness=softness, mode=mode, epsilon=epsilon),
    )


def equal(
    x: Array,
    y: Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Soft ``x == y``.

    Implemented as soft ``abs(x - y) <= 0``, scaled to [0, 1].

    Args:
        x: First input array.
        y: Second input array.
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.
        epsilon: Small offset.

    Returns:
        SoftBool in [0, 1].
    """
    x = ensure_float(x)
    if mode == "hard":
        return jnp.equal(x, y).astype(x.dtype)
    diff = abs(x - y, softness=softness, mode=mode)
    return 2.0 * less_equal(
        diff,
        jnp.zeros_like(diff),
        mode=mode,
        softness=softness,
        epsilon=epsilon,
    )


def not_equal(
    x: Array,
    y: Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Soft ``x != y``.

    Implemented as soft ``abs(x - y) > 0``, scaled to [0, 1].

    Args:
        x: First input array.
        y: Second input array.
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.
        epsilon: Small offset.

    Returns:
        SoftBool in [0, 1].
    """
    x = ensure_float(x)
    if mode == "hard":
        return jnp.not_equal(x, y).astype(x.dtype)
    diff = abs(x - y, softness=softness, mode=mode)
    tmp = greater(
        diff,
        jnp.zeros_like(diff),
        mode=mode,
        softness=softness,
        epsilon=epsilon,
    )
    return 2.0 * tmp - 1.0


def isclose(
    x: Array,
    y: Array,
    softness: float | Array = 0.1,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    mode: Mode = "smooth",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Soft approximate equality.

    Implements soft ``abs(x - y) <= atol + rtol * abs(y)``.

    Args:
        x: First input array.
        y: Second input array.
        softness: Width of transition (> 0).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        mode: ``"hard"`` or sigmoidal mode.
        epsilon: Small offset.

    Returns:
        SoftBool in [0, 1].
    """
    x = ensure_float(x)
    if mode == "hard":
        return jnp.isclose(x, y, atol=atol, rtol=rtol).astype(x.dtype)
    diff = abs(x - y, softness=softness, mode=mode)
    y_abs = abs(y, softness=softness, mode=mode)
    return 2.0 * less_equal(
        diff,
        atol + rtol * y_abs,
        mode=mode,
        softness=softness,
        epsilon=epsilon,
    )
