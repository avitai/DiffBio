"""Temperature softmax with jointly evaluated derivative coefficients.

Ordinary division AD forms ``T**-2`` before multiplying the softmax tail.
The product can be finite even when that reciprocal overflows or the tail
underflows. Evaluate probability products, score gaps and reciprocal powers
together in log space. A recursive coefficient JVP preserves mixed derivatives
at zero gaps; differentiating an outer zero selection would lose them.

This requires representable coefficient sums, not merely a representable final
contraction. Work grows with donor count and derivative order: the first JVP
has quadratic donor work. It is intended for small donor panels.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from diffbio.core.soft_ops._utils import canonicalize_axis


type _Specification = tuple[tuple[int, ...], tuple[tuple[int, int], ...], int]
type _Inputs = tuple[Array, Array, Array]


def _logits(scores: Array, temperature: Array, covered: Array) -> Array:
    """Shift over covered donors before division, guarding excluded operands."""
    maximum = jnp.max(jnp.where(covered, scores, -jnp.inf), axis=0, keepdims=True)
    shifted = jnp.where(covered, scores - maximum, 0.0)
    return jnp.where(covered, shifted / temperature, -jnp.inf)


def _coefficient_value(
    scores: Array, temperature: Array, covered: Array, specification: _Specification
) -> Array:
    """Evaluate a probability/gap monomial divided by a temperature power."""
    indices, gaps, power = specification
    log_weights = jax.nn.log_softmax(_logits(scores, temperature, covered), axis=0)
    magnitude = sum(log_weights[index] for index in indices) - power * jnp.log(temperature)
    sign = jnp.ones_like(magnitude)
    nonzero = jnp.ones_like(magnitude, dtype=jnp.bool_)
    for left, right in gaps:
        gap = scores[left] - scores[right]
        nonzero = nonzero & (gap != 0)
        magnitude = magnitude + jnp.log(jnp.where(gap != 0, jnp.abs(gap), 1.0))
        sign = sign * jnp.sign(gap)
    # Guard the exponential too: a zero gap must not create 0 * inf.
    value = sign * jnp.exp(jnp.where(nonzero, magnitude, 0.0))
    return jnp.where(nonzero, value, 0.0)


_coefficient = jax.custom_jvp(_coefficient_value, nondiff_argnums=(3,))


def _coefficient_jvp(
    specification: _Specification, primals: _Inputs, tangents: _Inputs
) -> tuple[Array, Array]:
    """Differentiate complete coefficients, including zero-gap extensions."""
    scores, _, _ = primals
    score_dot, temperature_dot, _ = tangents
    indices, gaps, power = specification
    value = _coefficient(*primals, specification)
    derivative = jnp.zeros_like(value)
    thermal = jnp.zeros_like(value)
    for index in indices:
        for other in range(scores.shape[0]):
            if other != index:
                extended = (*indices, other)
                coefficient = _coefficient(*primals, (extended, gaps, power + 1))
                derivative = derivative + coefficient * (score_dot[index] - score_dot[other])
                thermal = thermal + _coefficient(
                    *primals, (extended, (*gaps, (other, index)), power + 2)
                )
    for position, (left, right) in enumerate(gaps):
        remaining = gaps[:position] + gaps[position + 1 :]
        derivative = derivative + _coefficient(*primals, (indices, remaining, power)) * (
            score_dot[left] - score_dot[right]
        )
    if power:
        thermal = thermal - power * _coefficient(*primals, (indices, gaps, power + 1))
    return value, derivative + thermal * temperature_dot


_coefficient.defjvp(_coefficient_jvp)


@jax.custom_jvp
def _weights(scores: Array, temperature: Array, covered: Array) -> Array:
    """Keep the native shifted forward softmax arithmetic."""
    return jax.nn.softmax(_logits(scores, temperature, covered), axis=0)


def _weights_jvp(primals: _Inputs, tangents: _Inputs) -> tuple[Array, Array]:
    """Contract pairwise sensitivities without rounding either probability first."""
    scores, _, _ = primals
    score_dot, temperature_dot, _ = tangents
    value = _weights(*primals)
    rows = []
    for index in range(scores.shape[0]):
        derivative = jnp.zeros_like(value[index])
        for other in range(scores.shape[0]):
            if other != index:
                probability = (index, other)
                spatial = _coefficient(*primals, (probability, (), 1))
                thermal = _coefficient(*primals, (probability, ((other, index),), 2))
                derivative = derivative + spatial * (score_dot[index] - score_dot[other])
                derivative = derivative + thermal * temperature_dot
        rows.append(derivative)
    return value, jnp.stack(rows)


_weights.defjvp(_weights_jvp)


@partial(jax.jit, static_argnames=("axis",))
def temperature_softmax(
    scores: Array,
    temperature: float | Array = 1.0,
    *,
    axis: int = -1,
    where: Array | None = None,
) -> Array:
    """Normalize small score axes with range-aware temperature derivatives.

    Equivalent to masked ``jax.nn.softmax(scores / temperature, axis=axis)``
    for valid inputs. Shift before division and evaluate complete derivative
    coefficients in signed log space, preserving sensitivities even when a
    probability rounds to zero or one. Supports JVP, VJP and higher derivatives.

    This is an opt-in operation for small axes, not the default sorting kernel:
    first derivatives require quadratic axis work and higher orders cost more.
    Coefficients, score differences and their necessary sums must be representable;
    a representable final contraction alone does not guarantee a finite derivative.

    Args:
        scores: Real floating array. Included scores and their pairwise differences
            must be finite. Excluded scores are ignored, including NaN/infinity.
        temperature: Positive finite real scalar. Callers own runtime validation;
            nonpositive/nonfinite values are outside this numerical contract.
        axis: Static reduction axis. Must be nonempty.
        where: Boolean mask broadcastable to the score shape. Empty slices return
            zero weights and zero derivatives, matching native masked softmax.

    Returns:
        Weights with the score shape and standard floating dtype promotion.

    Raises:
        TypeError: Scores are not floating, temperature is complex, or mask is not boolean.
        ValueError: Temperature is not scalar, the axis is invalid/empty, or mask cannot broadcast.
    """
    scores = jnp.asarray(scores)
    temperature = jnp.asarray(temperature)
    if not jnp.issubdtype(scores.dtype, jnp.floating):
        raise TypeError("scores must have a real floating dtype")
    if jnp.issubdtype(temperature.dtype, jnp.complexfloating):
        raise TypeError("temperature must be real")
    if temperature.ndim != 0:
        raise ValueError("temperature must be scalar")
    temperature = temperature.astype(jnp.result_type(scores, temperature))
    axis = canonicalize_axis(axis, scores.ndim)
    if scores.shape[axis] == 0:
        raise ValueError("the normalization axis must be nonempty")
    covered = jnp.ones_like(scores, dtype=jnp.bool_) if where is None else jnp.asarray(where)
    if covered.dtype != jnp.bool_:
        raise TypeError("where must have boolean dtype")
    covered = jnp.broadcast_to(covered, scores.shape)
    scores, covered = jnp.moveaxis(scores, axis, 0), jnp.moveaxis(covered, axis, 0)
    scores = jnp.where(covered, scores, 0.0)
    populated = jnp.any(covered, axis=0, keepdims=True)
    first = (jnp.arange(scores.shape[0]) == 0).reshape((-1,) + (1,) * (scores.ndim - 1))
    safe_coverage = covered | (first & ~populated)
    value = _weights(scores, temperature, safe_coverage)
    return jnp.moveaxis(jnp.where(populated, value, 0.0), 0, axis)
