"""Simplex projection with multiple regularization modes.

Projects vectors onto the probability simplex (non-negative, sums to 1)
using different regularizers that control the smoothness of the
resulting gradient:

- **smooth** (C-infinity): Entropic/softmax regularizer. Closed-form via softmax.
- **c0** (continuous): Euclidean/L2 regularizer. Solved via threshold algorithm.
- **c1** (once differentiable): p=3/2 norm regularizer. Closed-form via
  quadratic formula.
- **c2** (twice differentiable): p=4/3 norm regularizer. Closed-form via
  Cardano's cubic formula.

All modes use custom JVP rules for numerically stable gradients.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from diffbio.core.soft_ops._utils import canonicalize_axis, validate_softness


# --------------------------------------------------------------------------- #
# C0 projection: Euclidean regularizer (threshold method)
# --------------------------------------------------------------------------- #


@jax.custom_jvp
def _proj_unit_simplex_q2(values: Array) -> Array:
    """L2 projection onto the unit simplex (1-D, no batch)."""
    n_features = values.shape[0]
    u = jnp.sort(values)[::-1]
    cumsum_u = jnp.cumsum(u)
    ind = jnp.arange(n_features) + 1
    cond = 1.0 / ind + (u - cumsum_u / ind) > 0
    idx = jnp.count_nonzero(cond)
    return jax.nn.relu(1.0 / idx + (values - cumsum_u[idx - 1] / idx))


@_proj_unit_simplex_q2.defjvp
def _proj_unit_simplex_q2_jvp(
    primals: list[Array],
    tangents: list[Array],
) -> tuple[Array, Array]:
    (values,) = primals
    (values_dot,) = tangents
    primal_out = _proj_unit_simplex_q2(values)
    supp = primal_out > 0
    card = jnp.count_nonzero(supp)
    tangent_out = supp * values_dot - (jnp.dot(supp, values_dot) / card) * supp
    return primal_out, tangent_out


# --------------------------------------------------------------------------- #
# C1 projection: p=3/2 norm regularizer (quadratic formula)
# --------------------------------------------------------------------------- #


def _proj_unit_simplex_q3_impl(
    values: Array,
) -> tuple[Array, Array]:
    """Closed-form simplex projection for p=3/2 via quadratic formula."""
    n = values.shape[0]
    u = jnp.sort(values)[::-1]
    u0 = u[0]
    u_shift = u - u0
    s_cum = jnp.cumsum(u_shift)
    m2 = jnp.cumsum(u_shift**2)
    k_arr = jnp.arange(1, n + 1, dtype=values.dtype)

    disc = s_cum**2 - k_arr * (m2 - 1.0)
    theta_k = (s_cum - jnp.sqrt(jnp.maximum(disc, 0.0))) / k_arr

    cond = u_shift > theta_k
    idx = jnp.count_nonzero(cond)
    theta = theta_k[idx - 1] + u0
    y = jnp.maximum(values - theta, 0.0) ** 2
    return y / jnp.sum(y), theta


@jax.custom_jvp
def _proj_unit_simplex_q3(values: Array) -> Array:
    return _proj_unit_simplex_q3_impl(values)[0]


@_proj_unit_simplex_q3.defjvp
def _proj_unit_simplex_q3_jvp(
    primals: list[Array],
    tangents: list[Array],
) -> tuple[Array, Array]:
    (values,) = primals
    (values_dot,) = tangents
    primal_out, theta = _proj_unit_simplex_q3_impl(values)

    supp = (primal_out > 0).astype(values.dtype)
    t = jnp.maximum(values - theta, 0.0)
    w = t * supp
    w_sum = jnp.where(jnp.sum(w) > 0, jnp.sum(w), 1.0)

    raw_tangent = 2.0 * t * (values_dot - jnp.dot(w, values_dot) / w_sum) * supp
    sum_t2 = jnp.sum(t**2)
    sum_t2 = jnp.where(sum_t2 > 0, sum_t2, 1.0)
    tangent_out = raw_tangent / sum_t2 - primal_out * jnp.sum(raw_tangent) / sum_t2
    return primal_out, tangent_out


# --------------------------------------------------------------------------- #
# C2 projection: p=4/3 norm regularizer (Cardano's cubic formula)
# --------------------------------------------------------------------------- #


def _proj_unit_simplex_q4_impl(
    values: Array,
) -> tuple[Array, Array]:
    """Closed-form simplex projection for p=4/3 via Cardano's method."""
    n = values.shape[0]
    dtype = values.dtype
    u = jnp.sort(values)[::-1]
    u0 = u[0]
    u_shift = u - u0
    s_cum = jnp.cumsum(u_shift)
    m2 = jnp.cumsum(u_shift**2)
    m3 = jnp.cumsum(u_shift**3)
    k_arr = jnp.arange(1, n + 1, dtype=dtype)

    c = s_cum / k_arr
    mu2 = m2 - 2.0 * c * s_cum + k_arr * c**2
    mu3 = m3 - 3.0 * c * m2 + 3.0 * c**2 * s_cum - k_arr * c**3

    p_coeff = 3.0 * mu2 / k_arr
    q_coeff = (1.0 - mu3) / k_arr

    sp3 = jnp.sqrt(jnp.maximum(p_coeff / 3.0, 0.0))
    denom = 2.0 * jnp.maximum(p_coeff, jnp.finfo(dtype).tiny) * sp3
    big_a = 3.0 * jnp.abs(q_coeff) / denom
    u_hyp = -jnp.sign(q_coeff) * 2.0 * sp3 * jnp.sinh(jnp.arcsinh(big_a) / 3.0)
    u_cbrt = -jnp.sign(q_coeff) * jnp.abs(q_coeff) ** (1.0 / 3.0)
    u_root = jnp.where(
        p_coeff > jnp.finfo(dtype).eps * jnp.maximum(jnp.abs(q_coeff), 1.0),
        u_hyp,
        u_cbrt,
    )
    theta_k = u_root + c

    cond = u_shift > theta_k
    idx = jnp.count_nonzero(cond)
    theta = theta_k[idx - 1] + u0
    y = jnp.maximum(values - theta, 0.0) ** 3
    return y / jnp.sum(y), theta


@jax.custom_jvp
def _proj_unit_simplex_q4(values: Array) -> Array:
    return _proj_unit_simplex_q4_impl(values)[0]


@_proj_unit_simplex_q4.defjvp
def _proj_unit_simplex_q4_jvp(
    primals: list[Array],
    tangents: list[Array],
) -> tuple[Array, Array]:
    (values,) = primals
    (values_dot,) = tangents
    primal_out, theta = _proj_unit_simplex_q4_impl(values)

    supp = (primal_out > 0).astype(values.dtype)
    t = jnp.maximum(values - theta, 0.0)
    w = t**2 * supp
    w_sum = jnp.where(jnp.sum(w) > 0, jnp.sum(w), 1.0)

    raw_tangent = 3.0 * t**2 * (values_dot - jnp.dot(w, values_dot) / w_sum) * supp
    sum_t3 = jnp.sum(t**3)
    sum_t3 = jnp.where(sum_t3 > 0, sum_t3, 1.0)
    tangent_out = raw_tangent / sum_t3 - primal_out * jnp.sum(raw_tangent) / sum_t3
    return primal_out, tangent_out


# --------------------------------------------------------------------------- #
# Public dispatch function
# --------------------------------------------------------------------------- #


def proj_simplex(
    x: Array,
    axis: int,
    softness: float | Array = 0.1,
    mode: Literal["smooth", "c0", "c1", "c2"] = "smooth",
) -> Array:
    """Project ``x`` onto the unit simplex along ``axis``.

    Solves: ``argmin_y  <x, y> + softness * R(y)``
    subject to ``y >= 0, sum(y) = 1``, where ``R(y)`` is determined
    by ``mode``.

    Args:
        x: Input array of shape ``(..., n, ...)``.
        axis: Axis containing the simplex dimension.
        softness: Regularization strength (> 0). Lower = sharper.
        mode: Regularizer type controlling smoothness:
            ``"smooth"`` (C-inf), ``"c0"`` (continuous),
            ``"c1"`` (once differentiable), ``"c2"`` (twice differentiable).

    Returns:
        Projected array on the probability simplex along ``axis``.
    """
    validate_softness(softness)
    axis = canonicalize_axis(axis, x.ndim)
    scaled = x / softness

    if mode == "smooth":
        return jax.nn.softmax(scaled, axis=axis)

    if mode == "c0":
        proj_fn = _proj_unit_simplex_q2
    elif mode == "c1":
        proj_fn = _proj_unit_simplex_q3
    elif mode == "c2":
        proj_fn = _proj_unit_simplex_q4
    else:
        msg = f"Invalid mode: {mode!r}. Must be 'smooth', 'c0', 'c1', or 'c2'."
        raise ValueError(msg)

    scaled = jnp.moveaxis(scaled, axis, -1)
    *batch_sizes, n = scaled.shape
    scaled = scaled.reshape(-1, n)
    result = jax.vmap(proj_fn)(scaled)
    result = result.reshape(*batch_sizes, n)
    return jnp.moveaxis(result, -1, axis)
