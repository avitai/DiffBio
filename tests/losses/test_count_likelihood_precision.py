"""Float32 accuracy of the negative binomial likelihoods (issue #36), in ULPs against mpmath.

The method is jax's numerics precision tests (``tests/numerics/numerics_test_util.py`` at
jax ``ac860cb2b8``): the float32 result is compared with an mpmath reference, the error is
measured in units in the last place of the reference,

    ulp_error = (computed - reference) / 2**(floor(log2|reference|) - 23),

and the worst case over a sample of inputs, at and near interesting points plus random draws, is
held to a per-platform ULP bound. As there, a fixed ill-conditioned region is left out
(``gamma_test.py``'s ``_betaln_ignored``): inputs whose relative condition number exceeds
``2**12``, where one ULP in each input alone moves the result by more than ``2**12`` ULPs, so the
problem has spent over half of float32's 24 significant bits before any implementation runs.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest

from diffbio.losses._negative_binomial import stirling_remainder
from diffbio.losses.statistical_losses import NegativeBinomialLoss


CONDITION_LIMIT = 2.0**12
REFERENCE_DIGITS = 60
RANDOM_SAMPLES = 400

# The worst ULP error of each quantity over each of its sample sets and both transforms, per
# platform, rounded up to a multiple of 0.5 ULP as jax records its bounds (``max(0.5, ceil(2 w)
# / 2)``), keyed ``quantity/sample set``. Recorded from Loader's saddle-point form and its
# closed-form partials (``diffbio.losses._negative_binomial``) with jax 0.11.1 on an Intel
# i7-13700 (cpu) and an RTX 4090 (gpu). The partials in theta and the counts are limited by
# jax 0.11.1's float32 digamma, 51 ULP from correct near its root at 1.4616. A case missing
# here is held to 0.5 ULP, correct rounding, as in jax.
ULP_BOUNDS: dict[str, dict[str, float]] = {
    "cpu": {
        "nb_value/draws": 14.5,
        "nb_value/zero counts": 2.0,
        "nb_value/small counts": 2.0,
        "nb_value/counts near the mean": 2.0,
        "nb_value/counts in the tails": 2.0,
        "nb_value/large theta": 2.5,
        "nb_value/tiny theta": 2.0,
        "nb_value/tiny mean": 1.5,
        "nb_value/non-integer counts": 9.0,
        "nb_grad_mu/draws": 2.5,
        "nb_grad_mu/zero counts": 2.5,
        "nb_grad_mu/small counts": 1.5,
        "nb_grad_mu/counts near the mean": 1.0,
        "nb_grad_mu/counts in the tails": 1.5,
        "nb_grad_mu/large theta": 2.0,
        "nb_grad_mu/tiny theta": 1.5,
        "nb_grad_mu/tiny mean": 2.0,
        "nb_grad_mu/non-integer counts": 1.5,
        "nb_grad_theta/draws": 797.5,
        "nb_grad_theta/zero counts": 2.0,
        "nb_grad_theta/small counts": 29.0,
        "nb_grad_theta/counts near the mean": 10.5,
        "nb_grad_theta/counts in the tails": 38.0,
        "nb_grad_theta/large theta": 3.0,
        "nb_grad_theta/tiny theta": 1.5,
        "nb_grad_theta/tiny mean": 1.5,
        "nb_grad_theta/non-integer counts": 22.0,
        "nb_grad_counts/draws": 267.0,
        "nb_grad_counts/zero counts": 3.5,
        "nb_grad_counts/small counts": 62.0,
        "nb_grad_counts/counts near the mean": 2.0,
        "nb_grad_counts/counts in the tails": 84.0,
        "nb_grad_counts/large theta": 10.5,
        "nb_grad_counts/tiny theta": 8.5,
        "nb_grad_counts/tiny mean": 1.5,
        "nb_grad_counts/non-integer counts": 46.5,
    },
    "gpu": {
        "nb_value/draws": 12.5,
        "nb_value/zero counts": 2.0,
        "nb_value/small counts": 3.0,
        "nb_value/counts near the mean": 2.5,
        "nb_value/counts in the tails": 2.0,
        "nb_value/large theta": 2.5,
        "nb_value/tiny theta": 1.5,
        "nb_value/tiny mean": 1.5,
        "nb_value/non-integer counts": 10.0,
        "nb_grad_mu/draws": 2.5,
        "nb_grad_mu/zero counts": 2.5,
        "nb_grad_mu/small counts": 2.0,
        "nb_grad_mu/counts near the mean": 2.0,
        "nb_grad_mu/counts in the tails": 1.0,
        "nb_grad_mu/large theta": 2.0,
        "nb_grad_mu/tiny theta": 1.5,
        "nb_grad_mu/tiny mean": 2.0,
        "nb_grad_mu/non-integer counts": 1.5,
        "nb_grad_theta/draws": 477.5,
        "nb_grad_theta/zero counts": 2.0,
        "nb_grad_theta/small counts": 29.0,
        "nb_grad_theta/counts near the mean": 23.5,
        "nb_grad_theta/counts in the tails": 93.0,
        "nb_grad_theta/large theta": 3.0,
        "nb_grad_theta/tiny theta": 1.5,
        "nb_grad_theta/tiny mean": 3.5,
        "nb_grad_theta/non-integer counts": 24.0,
        "nb_grad_counts/draws": 344.0,
        "nb_grad_counts/zero counts": 5.0,
        "nb_grad_counts/small counts": 84.0,
        "nb_grad_counts/counts near the mean": 2.0,
        "nb_grad_counts/counts in the tails": 95.5,
        "nb_grad_counts/large theta": 9.0,
        "nb_grad_counts/tiny theta": 14.5,
        "nb_grad_counts/tiny mean": 1.5,
        "nb_grad_counts/non-integer counts": 174.5,
    },
}


def _platform() -> str:
    return jax.devices()[0].platform


def ulp_bound(quantity: str) -> float:
    """The recorded ULP bound of ``quantity`` on this platform; 0.5 when none is recorded."""
    return ULP_BOUNDS.get(_platform(), {}).get(quantity, 0.5)


FLOAT32_MIN_EXPONENT = -126  # the smallest normal float32 is 2**-126
FLOAT32_MANTISSA_BITS = 23


def ulp_error(computed: float, reference: mpmath.mpf) -> float:
    """The signed float32 ULP error of ``computed`` against ``reference``, flushing subnormals.

    jax's ``_ulp_diff_jax`` with ``ftz=True``: ``ulp(reference) = 2**(max(floor(log2|reference|),
    emin) - 23)``; two values both below the smallest normal ``tiny`` differ by 0 ULP, and a
    comparison across the subnormal range first maps each value ``v`` to ``sign(v) * max(0,
    |v| - (tiny - ulp_tiny))``, so that ``-tiny``, ``0`` and ``tiny`` lie 1 ULP apart.
    """
    tiny = mpmath.ldexp(1, FLOAT32_MIN_EXPONENT)
    ulp_tiny = mpmath.ldexp(1, FLOAT32_MIN_EXPONENT - FLOAT32_MANTISSA_BITS)
    value = mpmath.mpf(computed)
    if abs(value) < tiny and abs(reference) < tiny:
        return 0.0
    exponent = FLOAT32_MIN_EXPONENT
    if reference != 0:
        exponent = max(math.floor(float(mpmath.log(abs(reference), 2))), FLOAT32_MIN_EXPONENT)
    if (abs(value) < tiny) != (abs(reference) < tiny) or (value < 0) != (reference < 0):
        value, reference = (
            mpmath.sign(v) * max(mpmath.mpf(0), abs(v) - (tiny - ulp_tiny))
            for v in (value, reference)
        )
    return float((value - reference) / mpmath.ldexp(1, exponent - FLOAT32_MANTISSA_BITS))


@dataclass(frozen=True, slots=True, kw_only=True)
class Reference:
    """An exact quantity at a sample, and its relative condition number in ULPs per input ULP."""

    value: mpmath.mpf
    condition: float


def reference(quantity: Callable[..., mpmath.mpf], point: Sequence[float]) -> Reference:
    """``quantity`` at ``point`` and ``sum_i |dq/dz_i| |z_i| / |q|``, from mpmath.

    Args:
        quantity: An mpmath function of the point's coordinates.
        point: The inputs, each a float32 value.

    Returns:
        The exact value and its relative condition number.
    """
    with mpmath.workdps(REFERENCE_DIGITS):
        inputs = [mpmath.mpf(coordinate) for coordinate in point]
        value = quantity(*inputs)
        sensitivity = mpmath.mpf(0)
        for index, coordinate in enumerate(inputs):
            order = tuple(1 if j == index else 0 for j in range(len(inputs)))
            sensitivity += abs(mpmath.diff(quantity, inputs, order)) * abs(coordinate)
        return Reference(value=+value, condition=float(sensitivity / abs(value)))


# The negative binomial in the mean and inverse-dispersion parameterisation, and its gradients.


def nb_log_prob(x: mpmath.mpf, mu: mpmath.mpf, theta: mpmath.mpf) -> mpmath.mpf:
    """log NB(x; mu, theta)."""
    return (
        mpmath.loggamma(x + theta)
        - mpmath.loggamma(theta)
        - mpmath.loggamma(x + 1)
        + theta * mpmath.log(theta / (theta + mu))
        + x * mpmath.log(mu / (theta + mu))
    )


def nb_d_mu(x: mpmath.mpf, mu: mpmath.mpf, theta: mpmath.mpf) -> mpmath.mpf:
    """d log NB / d mu."""
    return theta * (x - mu) / (mu * (theta + mu))


def nb_d_theta(x: mpmath.mpf, mu: mpmath.mpf, theta: mpmath.mpf) -> mpmath.mpf:
    """d log NB / d theta."""
    return (
        mpmath.digamma(x + theta)
        - mpmath.digamma(theta)
        + mpmath.log(theta / (theta + mu))
        + (mu - x) / (theta + mu)
    )


def nb_d_counts(x: mpmath.mpf, mu: mpmath.mpf, theta: mpmath.mpf) -> mpmath.mpf:
    """d log NB / d x, of the continuous extension through the gamma function."""
    return mpmath.digamma(x + theta) - mpmath.digamma(x + 1) + mpmath.log(mu / (theta + mu))


# Samples: interesting points, then random draws from a fixed seed. Every coordinate is a
# float32 value, so the float32 call and the reference see the same inputs.


def _f32(value: float) -> float:
    return float(np.float32(value))


_THETA_GRID = (1e-2, 0.5, 4.0, 100.0, 1e4)


def nb_samples() -> list[tuple[float, float, float]]:
    """(counts, mu, theta): counts 0 and 1 to 1e7, mu off the mode, theta from 1e-2 to 1e4."""
    points = [
        (_f32(count), _f32(ratio * count), _f32(theta))
        for count in (1, 2, 10, 100, 1e3, 1e4, 1e5, 1e6, 5e6, 1e7)
        for ratio in (0.5, 1.1, 2.0, 10.0)
        for theta in _THETA_GRID
    ]
    points += [
        (0.0, _f32(mu), _f32(theta)) for mu in (0.1, 1.0, 10.0, 1e3) for theta in _THETA_GRID
    ]
    rng = np.random.default_rng(36)
    counts = np.floor(10.0 ** rng.uniform(0.0, 7.0, RANDOM_SAMPLES))
    mus = counts * 10.0 ** rng.uniform(-1.0, 1.0, RANDOM_SAMPLES)
    thetas = 10.0 ** rng.uniform(-2.0, 4.0, RANDOM_SAMPLES)
    points += [(_f32(c), _f32(m), _f32(t)) for c, m, t in zip(counts, mus, thetas, strict=True)]
    return points


_EDGE_THETAS = (1e-2, 1.0, 1e2, 1e4)


def _grid(
    counts: Sequence[float], ratios: Sequence[float], thetas: Sequence[float]
) -> list[tuple[float, float, float]]:
    """(count, ratio * count, theta) over the product; no ratio of 1, where d/dmu is exactly 0."""
    return [(_f32(c), _f32(q * c), _f32(t)) for c in counts for q in ratios for t in thetas]


# Each regime is its own sample set with its own recorded bounds, so that a regime whose
# errors are inherently larger does not widen the bound that guards every other one.
NB_SAMPLE_SETS: dict[str, Callable[[], list[tuple[float, float, float]]]] = {
    "draws": nb_samples,
    "zero counts": lambda: [
        (0.0, _f32(mu), _f32(theta))
        for mu in (1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6)
        for theta in (*_EDGE_THETAS, 1e6)
    ],
    "small counts": lambda: _grid((1, 2, 3, 5), (0.1, 0.5, 2.0, 10.0), _EDGE_THETAS),
    "counts near the mean": lambda: _grid(
        (1e3, 1e5, 1e6, 5e6, 1e7), (0.999, 1.001, 1.1), _EDGE_THETAS
    ),
    "counts in the tails": lambda: _grid((1e3, 1e5, 1e7), (1e-2, 1e2), _EDGE_THETAS),
    "large theta": lambda: _grid((1, 10, 1e3, 1e5), (0.5, 1.1, 2.0), (1e5, 1e6, 1e8)),
    "tiny theta": lambda: _grid((1, 10, 1e3, 1e6), (0.5, 2.0), (1e-4, 1e-3)),
    "tiny mean": lambda: [
        (_f32(c), _f32(mu), _f32(theta))
        for c in (1, 3, 100)
        for mu in (1e-6, 1e-4)
        for theta in _EDGE_THETAS
    ],
    "non-integer counts": lambda: _grid(
        (0.25, 0.5, 1.5, 2.7, 1234.5), (0.5, 2.0), (1e-2, 1.0, 1e2)
    ),
}


# The float32 quantities under test, each a function of one sample's three float32 inputs.


def _nb_loss(x: jax.Array, mu: jax.Array, theta: jax.Array) -> jax.Array:
    return NegativeBinomialLoss()(x[None, None], mu[None, None], theta[None])


QUANTITIES: dict[str, tuple[Callable[..., jax.Array], Callable[..., mpmath.mpf]]] = {
    # name: (float32 function returning the NEGATED quantity, reference)
    "nb_value": (_nb_loss, nb_log_prob),
    "nb_grad_mu": (jax.grad(_nb_loss, argnums=1), nb_d_mu),
    "nb_grad_theta": (jax.grad(_nb_loss, argnums=2), nb_d_theta),
    "nb_grad_counts": (jax.grad(_nb_loss, argnums=0), nb_d_counts),
}

SAMPLE_SETS: dict[str, dict[str, Callable[[], list[tuple[float, float, float]]]]] = {
    **{name: NB_SAMPLE_SETS for name in QUANTITIES if name.startswith("nb_")},
}

# (quantity, sample set): each is held to its own recorded bound.
CASES = [(name, sample_set) for name, sets in SAMPLE_SETS.items() for sample_set in sets]


TRANSFORMS: dict[str, Callable[[Callable[..., jax.Array]], Callable[..., jax.Array]]] = {
    "vmap": jax.vmap,
    "jit_vmap": lambda f: jax.jit(jax.vmap(f)),
}


@functools.cache
def _cached_reference(name: str, point: tuple[float, ...]) -> Reference:
    return reference(QUANTITIES[name][1], point)


def worst_ulp_error(
    name: str, sample_set: str, transform: str
) -> tuple[float, tuple[float, ...], int]:
    """The worst ULP error of quantity ``name`` under ``transform`` over one sample set.

    Returns:
        The worst error, its input, and how many samples were left out as ill conditioned.
    """
    function, _ = QUANTITIES[name]
    points = SAMPLE_SETS[name][sample_set]()
    columns = [jnp.asarray(column, dtype=jnp.float32) for column in zip(*points, strict=True)]
    computed = TRANSFORMS[transform](function)(*columns)
    assert computed.dtype == jnp.float32, f"promoted to {computed.dtype}"
    worst, worst_point, ignored = 0.0, (), 0
    for value, point in zip(np.asarray(computed).tolist(), points, strict=True):
        wanted = _cached_reference(name, point)
        if wanted.condition > CONDITION_LIMIT:
            ignored += 1
            continue
        error = abs(ulp_error(-value, wanted.value)) if math.isfinite(value) else math.inf
        if error > worst:
            worst, worst_point = error, point
    return worst, worst_point, ignored


@pytest.mark.parametrize("transform", list(TRANSFORMS))
@pytest.mark.parametrize(("name", "sample_set"), CASES)
def test_the_float32_error_is_within_the_recorded_ulp_bound(
    name: str, sample_set: str, transform: str
) -> None:
    """The worst ULP error over a sample set meets the platform's recorded bound for it."""
    worst, point, _ = worst_ulp_error(name, sample_set, transform)
    bound = ulp_bound(f"{name}/{sample_set}")
    assert worst <= bound, (
        f"{name} over {sample_set} on {_platform()}: {worst:.2f} ULP at {point} > {bound}"
    )


@pytest.mark.parametrize("name", list(QUANTITIES))
def test_most_samples_are_well_conditioned(name: str) -> None:
    """Fewer than a tenth of a quantity's samples are left out as ill conditioned."""
    total, ignored = 0, 0
    for sample_set, samples in SAMPLE_SETS[name].items():
        total += len(samples())
        ignored += worst_ulp_error(name, sample_set, "vmap")[2]
    assert ignored < total // 10, f"{ignored} of {total} samples ill conditioned"


def stirling_remainder_reference(z: mpmath.mpf) -> mpmath.mpf:
    """delta(z) = log(z!) - (z + 1/2) log(z) + z - log(2 pi) / 2."""
    return (
        mpmath.loggamma(z + 1)
        - (z + mpmath.mpf(1) / 2) * mpmath.log(z)
        + z
        - (mpmath.log(2 * mpmath.pi) / 2)
    )


@pytest.mark.parametrize("transform", ["eager", "jit"])
@pytest.mark.parametrize("z", [1.0, 2.0])
def test_the_stirling_remainder_is_correctly_rounded_at_small_integers(
    z: float, transform: str
) -> None:
    """delta(1) and delta(2), the integer arguments below the series start, are exact values.

    Through ``gammaln`` the remainder at 2 is 0.041 left from terms of 0.69 and 1.73, a 40-fold
    cancellation.
    """
    function = stirling_remainder if transform == "eager" else jax.jit(stirling_remainder)
    computed = float(function(jnp.float32(z)))
    with mpmath.workdps(REFERENCE_DIGITS):
        wanted = stirling_remainder_reference(mpmath.mpf(z))
    assert abs(ulp_error(computed, wanted)) <= 0.5


@pytest.mark.parametrize(
    ("count", "mu", "theta"), [(0.0, 3.0, 2.0), (3.0, 1.5, 2.0), (5e6, 5.5e6, 4.0)]
)
def test_the_second_derivatives_are_finite(count: float, mu: float, theta: float) -> None:
    """``jax.hessian`` (forward over reverse) runs through the custom derivative rule."""
    hessian = jax.jit(jax.hessian(_nb_loss, argnums=(0, 1, 2)))(
        *(jnp.float32(v) for v in (count, mu, theta))
    )
    assert all(bool(jnp.isfinite(entry)) for row in hessian for entry in row)
