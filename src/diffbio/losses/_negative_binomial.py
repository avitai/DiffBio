"""The negative binomial log-probability by Loader's saddle-point method, with closed-form partials.

Catherine Loader, "Fast and Accurate Computation of Binomial Probabilities" (2002, revising
2000) writes the binomial mass through the Stirling remainder ``delta`` (eq. 4) and the
deviance ``bd0`` (eqs. 3, 6), so no two large logarithms are ever subtracted. The negative
binomial follows from the identity

    NB(x; theta, mu) = theta / (x + theta) * b(theta; x + theta, theta / (theta + mu)),

which with ``n = x + theta`` reduces to

    log NB = delta(n) - delta(theta) - delta(x) - bd0(theta, n P) - bd0(x, n Q)
             - (log(2 pi) + log(x) + log1p(x / theta)) / 2,

``P = theta / (theta + mu)`` and ``Q = mu / (theta + mu)``; it is Loader's Poisson form (eq. 7)
as ``theta`` grows. The direct formula subtracts ``lgamma`` values of size ~7e7 at 5e6 counts,
where float32 resolves 4 (issue #36). A zero count takes R's ``dnbinom_mu`` form,
``theta log(theta / (theta + mu))`` through ``log1p`` of the small ratio.

Two choices follow from float32 rather than from the paper, which works in float64:

- The Stirling series is used from the argument where its first omitted term falls below the
  dtype's ``eps`` relative to ``delta``, and ``lgamma`` below it.
- ``bd0`` is Welinder's ``-x log1pmx((m - x) / x)`` (R bug PR#15628), with the difference
  ``x - m`` computed in closed form, ``theta (mu - x) / (theta + mu)``. Loader's direct form
  ``x log(x / m) + m - x`` cancels beyond his series radius and takes the log of a quotient,
  which XLA's GPU float32 division rounds to about one ULP rather than to the nearest.

The derivatives are not obtained by differentiating these branches: every branch would run in
the backward pass, and autodiff of the formula cancels. A ``jax.custom_jvp`` gives the exact
partials instead, each written without a cancelling difference:

    d/dmu    = theta (x - mu) / (mu (theta + mu)),
    d/dtheta = log1pmx(t) + x / (2 theta (theta + x)) + delta'(theta + x) - delta'(theta),
    d/dx     = log1p(s) - (2x + theta) / (2x (x + theta)) + delta'(x + theta) - delta'(x),

with ``t = (x - mu) / (theta + mu)``, ``s = theta (mu - x) / (x (theta + mu))`` and
``log1pmx(t) = log1p(t) - t``. They follow from ``psi(z) = delta'(z) + log(z) - 1 / (2z)``
applied to the textbook partials ``psi(x + theta) - psi(theta) + log(P) + (mu - x) / (theta +
mu)`` and ``psi(x + theta) - psi(x + 1) + log(Q)``, whose terms are of size ``1 / theta`` while
the partial in ``theta`` is of size ``1 / theta**2``. No published source states them; each piece
is published: ``log1pmx`` by its artanh series (as R's ``log1pmx`` evaluates it), and
``delta'`` by the asymptotic series of the digamma function (DLMF 5.11.2).
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero
from jax.scipy.special import digamma, gammaln
from jaxtyping import Array, Float


_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)

# Stirling's series for log(z!) - log(sqrt(2 pi z) (z / e)**z): B_{2k} / (2k (2k - 1)) z**(1-2k)
# (Abramowitz and Stegun 6.1.40; Loader eq. 4 lists the first three).
_STIRLING_COEFFICIENTS = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
)
# The first omitted coefficient, -3617 / 122400, bounds the truncation error of the alternating
# series above (Whittaker and Watson, section 12.33).
_STIRLING_FIRST_OMITTED = 3617.0 / 122400.0

# Integer counts below this, where theta is below delta'(z)'s series start too, take the partial
# in theta as an exact harmonic sum: one division per term, so a few at most.
_HARMONIC_COUNTS = 4

# log1pmx's artanh series runs on this interval of t and log(1 + t) - t outside it, where it
# loses at most log2(0.5 / 0.193), 1.4 bits, at t = -0.5.
_LOG1PMX_SERIES_INTERVAL = (-0.5, 1.0)


def _stirling_series_start(dtype: jnp.dtype) -> int:
    """The smallest ``z`` where the series' truncation error is below ``eps * delta(z)``.

    ``delta(z) > 1 / (12 z + 1)`` and the truncation error is below the first omitted term
    ``c z**-15``, so ``c z**-15 <= eps / (12 z + 1)`` holds from ``z = (13 c / eps)**(1/14)``.
    """
    eps = float(jnp.finfo(dtype).eps)
    return math.ceil((13.0 * _STIRLING_FIRST_OMITTED / eps) ** (1.0 / 14.0))


def _exact_stirling_remainders(dtype: jnp.dtype) -> dict[int, float]:
    """``delta(n)`` at the integers below the series start, evaluated in float64.

    R's ``stirlerr`` keeps such exact values for small arguments: below the series start
    ``delta`` otherwise comes from ``lgamma`` minus terms many times its size (at ``n = 2``,
    0.041 left from 0.69 and 1.73). Rounded once to ``dtype``, each value is correctly rounded.
    """
    return {
        n: math.lgamma(n + 1.0) - (n + 0.5) * math.log(n) + n - _HALF_LOG_TWO_PI
        for n in range(1, _stirling_series_start(dtype))
    }


def _stirling_derivative_series_start(dtype: jnp.dtype) -> int:
    """The smallest ``z`` where the derivative series' truncation is below ``eps * |delta'(z)|``.

    The series of ``delta'`` is that of ``psi(z) - log(z) + 1 / (2z)`` (DLMF 5.11.2), whose
    remainder for real ``z > 0`` is below the first omitted term (DLMF 5.11(ii)),
    ``15 c z**-16``. Its terms alternate and decrease for ``z >= 1``, so ``|delta'(z)| >= (1 -
    1 / 10) / (12 z**2) > 1 / (14 z**2)``, and ``15 c z**-16 <= eps / (14 z**2)`` holds from
    ``z = (210 c / eps)**(1/14)``.
    """
    eps = float(jnp.finfo(dtype).eps)
    order = 2 * len(_STIRLING_COEFFICIENTS) + 1
    return math.ceil((14.0 * order * _STIRLING_FIRST_OMITTED / eps) ** (1.0 / 14.0))


def _log1pmx_truncation(t: float, terms: int) -> float:
    """The bound ``2 |r| y**(K+1) / ((2K + 3)(1 - y))`` on dropping ``sum_{k >= K}``."""
    r = t / (2.0 + t)
    y = r * r
    return 2.0 * abs(r) * y ** (terms + 1) / ((2 * terms + 3) * (1.0 - y))


def _log1pmx_series_terms(dtype: jnp.dtype) -> int:
    """Terms of ``sum_k y**k / (2k + 3)`` that reach ``eps`` relative to ``log1pmx``.

    Checked at the interval's ends, where ``|r|`` and so the truncation are largest.
    """
    eps = float(jnp.finfo(dtype).eps)
    terms = 1
    while any(
        _log1pmx_truncation(t, terms) > eps * abs(math.log1p(t) - t)
        for t in _LOG1PMX_SERIES_INTERVAL
    ):
        terms += 1
    return terms


def stirling_remainder(z: Float[Array, "..."]) -> Float[Array, "..."]:
    """``delta(z) = log(z!) - log(sqrt(2 pi z) (z / e)**z)`` for ``z > 0`` (Loader eq. 4).

    Args:
        z: Positive arguments.

    Returns:
        The remainder of Stirling's approximation, elementwise; exact values at the integers
        below the series start.
    """
    start = _stirling_series_start(z.dtype)
    large = z >= start
    # Each branch sees an argument in its own domain, so the side the where drops stays finite.
    z_series = jnp.where(large, z, start)
    inverse_square = 1.0 / (z_series * z_series)
    series = jnp.zeros_like(z)
    for coefficient in reversed(_STIRLING_COEFFICIENTS):
        series = series * inverse_square + coefficient
    series = series / z_series
    z_direct = jnp.where(large, 1.0, z)
    direct = gammaln(z_direct + 1.0) - (z_direct + 0.5) * jnp.log(z_direct) + z_direct
    result = jnp.where(large, series, direct - _HALF_LOG_TWO_PI)
    for n, exact in _exact_stirling_remainders(z.dtype).items():
        result = jnp.where(z == n, exact, result)
    return result


def stirling_remainder_derivative(z: Float[Array, "..."]) -> Float[Array, "..."]:
    """``delta'(z) = psi(z + 1) - log(z) - 1 / (2z)`` for ``z > 0``.

    Args:
        z: Positive arguments.

    Returns:
        The derivative of the Stirling remainder, by DLMF 5.11.2's series where it reaches the
        dtype's ``eps`` and by the digamma function below.
    """
    start = _stirling_derivative_series_start(z.dtype)
    large = z >= start
    # The second derivatives differentiate both sides, so each stays in its own domain.
    z_series = jnp.where(large, z, start)
    inverse_square = 1.0 / (z_series * z_series)
    series = jnp.zeros_like(z)
    for order, coefficient in reversed(list(enumerate(_STIRLING_COEFFICIENTS, start=1))):
        series = series * inverse_square - (2 * order - 1) * coefficient
    series = series * inverse_square
    z_direct = jnp.where(large, 1.0, z)
    direct = digamma(z_direct + 1.0) - jnp.log(z_direct) - 0.5 / z_direct
    return jnp.where(large, series, direct)


def log1pmx(t: Float[Array, "..."], one_plus_t: Float[Array, "..."]) -> Float[Array, "..."]:
    """``log(1 + t) - t`` for ``t > -1``.

    Inside ``[-0.5, 1]`` it is ``r (2 y S(y) - t)`` with ``r = t / (2 + t)``, ``y = r**2`` and
    ``S(y) = sum_k y**k / (2k + 3)``, from ``log(1 + t) = 2 artanh(r)`` with the cancelling
    ``2r - t`` formed as ``-r t``; outside, ``log(1 + t) - t`` loses at most 1.4 bits.

    Args:
        t: Arguments above -1.
        one_plus_t: ``1 + t`` computed in closed form by the caller, so that it keeps its
            relative accuracy as ``t`` approaches -1.

    Returns:
        ``log(1 + t) - t``, elementwise.
    """
    lower, upper = _LOG1PMX_SERIES_INTERVAL
    inside = (t >= lower) & (t <= upper)
    t_series = jnp.where(inside, t, 0.0)
    r = t_series / (2.0 + t_series)
    y = r * r
    total = jnp.zeros_like(y)
    for k in reversed(range(_log1pmx_series_terms(t.dtype))):
        total = total * y + 1.0 / (2 * k + 3)
    series = r * (2.0 * y * total - t_series)
    one_plus_t_direct = jnp.where(inside, 1.0, one_plus_t)
    return jnp.where(inside, series, jnp.log(one_plus_t_direct) - t)


def binomial_deviance(
    x: Float[Array, "..."], m: Float[Array, "..."], difference: Float[Array, "..."]
) -> Float[Array, "..."]:
    """``bd0(x, m) = x log(x / m) + m - x`` for ``x, m > 0`` (Loader eq. 6).

    It is ``-x log1pmx(t)`` with ``t = (m - x) / x``, M. Welinder's form (R bug PR#15628; DPQ's
    ``bd0_l1pm``): from ``m = x (1 + t)``, ``x log(x / m) = -x log1p(t)`` and ``m - x = x t``.
    Nothing in it cancels, and for small ``|t|`` it is ``log1pmx``'s artanh series, the series of
    Loader's eq. 6. Loader's direct ``x log(x / m) + (m - x)``, used beyond his series radius,
    cancels and takes the log of ``x / m``, a quotient XLA's GPU float32 division rounds to
    about one ULP rather than to the nearest: at ``(x, mu, theta) = (1e5, 5e4, 1e5)`` on an RTX
    4090 that put 20 ULP of error on ``bd0(theta, n P)``, against 0.4 ULP on a CPU.

    Args:
        x: Positive first arguments.
        m: Positive second arguments.
        difference: ``x - m``, computed without subtracting the two.

    Returns:
        The deviance term, elementwise.
    """
    return -x * log1pmx(-difference / x, m / x)


@jax.custom_jvp
def nb_log_prob(
    counts: Float[Array, "..."], mu: Float[Array, "..."], theta: Float[Array, "..."]
) -> Float[Array, "..."]:
    """``log NB(x | mu, theta)`` in the mean and inverse-dispersion parameterisation.

    Differentiable in all three arguments through the closed-form partials of the module
    docstring; in ``counts``, of the continuous extension through the gamma function.

    Args:
        counts: Non-negative counts ``x``.
        mu: Positive means, broadcastable to ``counts``.
        theta: Positive inverse dispersions, broadcastable to ``counts``.

    Returns:
        The log-probability, elementwise.
    """
    # No broadcast up front: a term in theta alone, such as delta(theta), runs at theta's own
    # shape (one value per gene for NegativeBinomialLoss) and broadcasts only when combined.
    positive = counts > 0
    # The saddle-point branch sees a count of one at a zero count, where the where drops it.
    x = jnp.where(positive, counts, 1.0)
    # n P and n Q are each one product, as R's dnbinom_mu forms them: as x + (theta - n P) the
    # mean term would subtract nearly equal numbers and lose mu once mu / theta is below eps.
    # theta - n P = theta (mu - x) / (theta + mu) = x - n Q, so the deviances also see their
    # arguments' difference in closed form rather than by subtraction.
    total = theta + mu
    n = x + theta
    difference = theta * (mu - x) / total
    log_positive = (
        stirling_remainder(n)
        - stirling_remainder(theta)
        - stirling_remainder(x)
        - binomial_deviance(theta, n * (theta / total), difference)
        - binomial_deviance(x, n * (mu / total), -difference)
        - 0.5 * (2.0 * _HALF_LOG_TWO_PI + jnp.log(x) + jnp.log1p(x / theta))
    )
    return jnp.where(positive, log_positive, _log_prob_of_zero(mu, theta))


def _log_prob_of_zero(mu: Float[Array, "..."], theta: Float[Array, "..."]) -> Float[Array, "..."]:
    """``log NB(0) = theta log(theta / (theta + mu))``, as R's ``dnbinom_mu`` forms it.

    The log takes the small one of ``theta / (theta + mu)`` and ``mu / (theta + mu)``, each a
    single quotient: ``log`` of the first when ``theta < mu``, ``log1p`` of minus the second
    otherwise, where the first rounds towards one.
    """
    total = theta + mu
    return theta * jnp.where(theta < mu, jnp.log(theta / total), jnp.log1p(-mu / total))


def _log_mean_fraction(mu: Float[Array, "..."], theta: Float[Array, "..."]) -> Float[Array, "..."]:
    """``log(mu / (theta + mu))``, as ``log1p(-theta / (theta + mu))`` when ``theta < mu``."""
    total = theta + mu
    return jnp.where(theta < mu, jnp.log1p(-theta / total), jnp.log(mu / total))


def _partial_in_mu(
    counts: Float[Array, "..."], mu: Float[Array, "..."], theta: Float[Array, "..."]
) -> Float[Array, "..."]:
    """``theta (x - mu) / (mu (theta + mu))``: ``x / mu - (x + theta) / (theta + mu)`` combined."""
    return theta * (counts - mu) / (mu * (theta + mu))


def _partial_in_theta(
    counts: Float[Array, "..."], mu: Float[Array, "..."], theta: Float[Array, "..."]
) -> Float[Array, "..."]:
    """``log1pmx(t) + x / (2 theta (theta + x)) + delta'(theta + x) - delta'(theta)``.

    At ``x = 0`` the rational term is 0 and the two ``delta'`` agree exactly, leaving
    ``log1pmx(-mu / (theta + mu))``, the derivative of ``theta log(theta / (theta + mu))``.
    The ``delta'`` difference is formed first: added one at a time, each ``delta'`` would absorb
    a ``log1pmx`` far below its own size before the two cancel.

    Below ``delta'``'s series start in ``theta``, ``delta'(theta)`` rests on digamma, whose
    float32 error then bounds the result where it is small. There an integer count below
    ``_HARMONIC_COUNTS`` takes ``psi(theta + x) - psi(theta) = sum_{j < x} 1 / (theta + j)``
    (DLMF 5.5.2), which jax also uses for its hyp1f1 parameter derivatives: the partial is
    ``sum_{j < x} (mu - j) / ((theta + j)(theta + mu)) + log1pmx(-mu / (theta + mu))``, exact
    term by term. From the series start on, the general form is the more accurate: its terms
    are of the size of the score's ``1 / theta**2``, the harmonic terms of ``1 / theta``.
    """
    total = theta + mu
    stirling_difference = stirling_remainder_derivative(
        theta + counts
    ) - stirling_remainder_derivative(theta)
    general = (
        log1pmx((counts - mu) / total, (theta + counts) / total)
        + counts / (2.0 * theta * (theta + counts))
        + stirling_difference
    )
    harmonic = log1pmx(-mu / total, theta / total)
    for j in range(_HARMONIC_COUNTS - 1):
        harmonic = harmonic + jnp.where(j < counts, (mu - j) / ((theta + j) * total), 0.0)
    is_harmonic = (
        (counts < _HARMONIC_COUNTS)
        & (counts == jnp.floor(counts))
        & (theta < _stirling_derivative_series_start(theta.dtype))
    )
    return jnp.where(is_harmonic, harmonic, general)


def _partial_in_counts(
    counts: Float[Array, "..."], mu: Float[Array, "..."], theta: Float[Array, "..."]
) -> Float[Array, "..."]:
    """``psi(x + theta) - psi(x + 1) + log(mu / (theta + mu))``.

    For ``x >= 1``, ``log1p(s) - (2x + theta) / (2x (x + theta)) + delta'(x + theta) -
    delta'(x)``, ``s = theta (mu - x) / (x (theta + mu))``, with ``log`` of ``1 + s = mu (x +
    theta) / (x (theta + mu))`` near ``s = -1`` and the ``delta'`` difference formed first, as in
    ``_partial_in_theta``: ``psi(x + theta)`` and ``log(theta + mu)`` would otherwise cancel to
    leave a result of size one when ``theta`` is large. Below one count, where the ``1 / x``
    terms grow, the digamma functions directly.
    """
    large = counts >= 1.0
    x = jnp.where(large, counts, 1.0)
    total = theta + mu
    ratio_minus_one = theta * (mu - x) / (x * total)
    log_ratio = jnp.where(
        ratio_minus_one < -0.5,
        jnp.log(mu * (x + theta) / (x * total)),
        jnp.log1p(ratio_minus_one),
    )
    stirling_difference = stirling_remainder_derivative(x + theta) - stirling_remainder_derivative(
        x
    )
    identity = log_ratio - (2.0 * x + theta) / (2.0 * x * (x + theta)) + stirling_difference
    x_direct = jnp.where(large, 1.0, counts)
    direct = digamma(x_direct + theta) - digamma(x_direct + 1.0) + _log_mean_fraction(mu, theta)
    return jnp.where(large, identity, direct)


@functools.partial(nb_log_prob.defjvp, symbolic_zeros=True)
def _nb_log_prob_jvp(
    primals: tuple[Array, Array, Array], tangents: tuple[Array, Array, Array]
) -> tuple[Array, Array]:
    counts, mu, theta = primals
    value = nb_log_prob(counts, mu, theta)
    tangent = jnp.zeros_like(value)
    partials = (_partial_in_counts, _partial_in_mu, _partial_in_theta)
    for partial, tangent_in in zip(partials, tangents, strict=True):
        # An argument nobody differentiates arrives as a symbolic zero: its partial is skipped.
        if not isinstance(tangent_in, SymbolicZero):
            tangent = tangent + partial(counts, mu, theta) * tangent_in
    return value, tangent
