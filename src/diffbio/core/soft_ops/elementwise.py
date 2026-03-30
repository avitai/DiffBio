"""Elementwise soft operations.

Provides differentiable relaxations of elementwise non-smooth functions
(abs, sign, relu, clip, round, heaviside) using sigmoidal smoothing
with configurable smoothness modes.

All functions accept a ``mode`` parameter controlling the smoothness:

- ``"hard"``: Exact (non-differentiable) version matching JAX.
- ``"smooth"``: C-infinity smooth via logistic sigmoid.
- ``"c0"``: Continuous (C0) via piecewise linear/quadratic.
- ``"c1"``: Once differentiable (C1) via cubic Hermite polynomial.
- ``"c2"``: Twice differentiable (C2) via quintic Hermite polynomial.

The ``softness`` parameter controls the width of the transition region.
Higher softness = smoother (wider transition). All functions are
JIT-compatible and support ``jax.grad``/``jax.vmap``.
"""

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from diffbio.core.soft_ops._types import SoftBool
from diffbio.core.soft_ops._utils import ensure_float, validate_softness

Mode = Literal["hard", "smooth", "c0", "c1", "c2"]
SigmoidalMode = Literal["smooth", "c0", "c1", "c2"]


def sigmoidal(
    x: Array,
    softness: float | Array = 0.1,
    mode: SigmoidalMode = "smooth",
) -> SoftBool:
    """Sigmoidal S-curve function mapping R -> (0, 1).

    Foundation for all other elementwise operations. Maps input values
    through an S-shaped curve centered at 0, approaching 0 for large
    negative values and 1 for large positive values.

    Args:
        x: Input array.
        softness: Width of transition region (> 0). Higher = smoother.
        mode: Smoothness family: ``"smooth"`` (logistic sigmoid),
            ``"c0"`` (piecewise linear), ``"c1"`` (cubic Hermite),
            ``"c2"`` (quintic Hermite).

    Returns:
        SoftBool array with values in [0, 1].
    """
    validate_softness(softness)
    x = x / softness
    if mode == "smooth":
        return jax.nn.sigmoid(x)

    # Piecewise modes: scale by 1/5 so transition region [-5s, 5s]
    # matches smooth sigmoid's effective range.
    x = x / 5.0
    if mode == "c0":
        y = jnp.polyval(jnp.array([0.5, 0.5], dtype=x.dtype), x)
        return jnp.where(x < -1.0, 0.0, jnp.where(x < 1.0, y, 1.0))
    if mode == "c1":
        y = jnp.polyval(
            jnp.array([-0.25, 0.0, 0.75, 0.5], dtype=x.dtype),
            x,
        )
        return jnp.where(x < -1.0, 0.0, jnp.where(x < 1.0, y, 1.0))
    if mode == "c2":
        y = jnp.polyval(
            jnp.array([0.1875, 0.0, -0.625, 0.0, 0.9375, 0.5], dtype=x.dtype),
            x,
        )
        return jnp.where(x < -1.0, 0.0, jnp.where(x < 1.0, y, 1.0))
    msg = f"Invalid mode: {mode!r}. Must be 'smooth', 'c0', 'c1', or 'c2'."
    raise ValueError(msg)


def softrelu(
    x: Array,
    softness: float | Array = 0.1,
    mode: SigmoidalMode = "smooth",
    gated: bool = False,
) -> Array:
    """Family of soft relaxations to ReLU.

    Two variants:
    - **Non-gated** (default): Antiderivative of :func:`sigmoidal`.
      Smooth analog of ``max(0, x)``.
    - **Gated**: ``x * sigmoidal(x)``. SiLU-style gating.

    Args:
        x: Input array.
        softness: Width of transition region (> 0).
        mode: Smoothness family (see :func:`sigmoidal`).
        gated: If True, use gated version ``x * sigmoidal(x)``.

    Returns:
        Soft ReLU output, same shape as ``x``.
    """
    validate_softness(softness)
    x = x / softness
    if mode == "smooth":
        if gated:
            y = x * sigmoidal(x, softness=1.0, mode="smooth")
        else:
            y = jax.nn.softplus(x)
    else:
        u = x / 5.0
        if mode == "c0":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="c0")
            else:
                y = 5.0 * jnp.polyval(
                    jnp.array([0.25, 0.5, 0.25], dtype=u.dtype),
                    u,
                )
                y = jnp.where(u < -1.0, 0.0, jnp.where(u < 1.0, y, x))
        elif mode == "c1":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="c1")
            else:
                y = 5.0 * jnp.polyval(
                    jnp.array([-0.0625, 0.0, 0.375, 0.5, 0.1875], dtype=u.dtype),
                    u,
                )
                y = jnp.where(u < -1.0, 0.0, jnp.where(u < 1.0, y, x))
        elif mode == "c2":
            if gated:
                y = x * sigmoidal(x, softness=1.0, mode="c2")
            else:
                y = 5.0 * jnp.polyval(
                    jnp.array(
                        [0.03125, 0.0, -0.15625, 0.0, 0.46875, 0.5, 0.15625],
                        dtype=u.dtype,
                    ),
                    u,
                )
                y = jnp.where(u < -1.0, 0.0, jnp.where(u < 1.0, y, x))
        else:
            msg = f"Invalid mode: {mode!r}"
            raise ValueError(msg)
    return y * softness


def heaviside(
    x: Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
) -> SoftBool:
    """Soft Heaviside step function.

    Returns 0 for x < 0, 1 for x > 0, and 0.5 at x = 0 (hard mode).
    Soft modes use :func:`sigmoidal` for smooth transition.

    Args:
        x: Input array.
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.

    Returns:
        SoftBool in [0, 1].
    """
    x = ensure_float(x)
    if mode == "hard":
        return jnp.where(x < 0.0, 0.0, jnp.where(x > 0.0, 1.0, 0.5)).astype(
            x.dtype,
        )
    return sigmoidal(x, softness=softness, mode=mode)


def round(
    x: Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    neighbor_radius: int = 5,
) -> Array:
    """Soft rounding.

    Hard mode returns ``jnp.round(x)``. Soft modes use a weighted sum
    of nearby integers, with weights from :func:`sigmoidal`.

    Args:
        x: Input array.
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.
        neighbor_radius: Number of integer neighbors to consider.

    Returns:
        Soft-rounded values.
    """
    if mode == "hard":
        return jnp.round(x)
    x = ensure_float(x)
    center = jax.lax.stop_gradient(jnp.floor(x))
    offsets = jnp.arange(
        -neighbor_radius,
        neighbor_radius + 1,
        dtype=x.dtype,
    )
    n = center[..., None] + offsets
    w_left = sigmoidal(x[..., None] - (n - 0.5), softness=softness, mode=mode)
    w_right = sigmoidal(x[..., None] - (n + 0.5), softness=softness, mode=mode)
    return jnp.sum(n * (w_left - w_right), axis=-1)


def sign(
    x: Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
) -> Array:
    """Soft sign function.

    Maps to [-1, 1]. Hard mode returns ``jnp.sign(x)``.
    Soft modes use ``2 * sigmoidal(x) - 1``.

    Args:
        x: Input array.
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.

    Returns:
        Values in [-1, 1].
    """
    x = ensure_float(x)
    if mode == "hard":
        return jnp.sign(x).astype(x.dtype)
    return sigmoidal(x, mode=mode, softness=softness) * 2.0 - 1.0


def abs(
    x: Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
) -> Array:
    """Soft absolute value.

    Hard mode returns ``jnp.abs(x)``. Soft modes use
    ``x * sign(x, softness, mode)``.

    Args:
        x: Input array.
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.

    Returns:
        Non-negative values (approximately).
    """
    if mode == "hard":
        return jnp.abs(x)
    return x * sign(x, mode=mode, softness=softness)


def relu(
    x: Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    gated: bool = False,
) -> Array:
    """Soft ReLU.

    Hard mode returns ``jax.nn.relu(x)``. Soft modes delegate to
    :func:`softrelu`.

    Args:
        x: Input array.
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.
        gated: If True, use gated variant.

    Returns:
        Soft ReLU output.
    """
    if mode == "hard":
        return jax.nn.relu(x)
    return softrelu(x, mode=mode, softness=softness, gated=gated)


def clip(
    x: Array,
    a: float | Array,
    b: float | Array,
    softness: float | Array = 0.1,
    mode: Mode = "smooth",
    gated: bool = False,
) -> Array:
    """Soft clipping to [a, b].

    Hard mode returns ``jnp.clip(x, a, b)``. Soft modes use two
    :func:`softrelu` calls: ``a + softrelu(x - a) - softrelu(x - b)``.

    Args:
        x: Input array.
        a: Lower bound.
        b: Upper bound.
        softness: Width of transition (> 0).
        mode: ``"hard"`` or sigmoidal mode.
        gated: If True, use gated softrelu variant.

    Returns:
        Clipped values approximately in [a, b].
    """
    if mode == "hard":
        return jnp.clip(x, a, b)
    tmp1 = softrelu(x - a, mode=mode, softness=softness, gated=gated)
    tmp2 = softrelu(x - b, mode=mode, softness=softness, gated=gated)
    return a + tmp1 - tmp2
