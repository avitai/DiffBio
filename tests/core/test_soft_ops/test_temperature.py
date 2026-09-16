"""Range-aware temperature normalization through the public soft-operations API."""

from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.core.soft_ops import temperature_softmax


def _logistic_reference(score, temperature):
    """Independent high-precision first and mixed second derivatives."""
    with localcontext() as context:
        context.prec = 90
        s, t = Decimal(score), Decimal(temperature)
        tail = (-s / t).exp()
        p = 1 / (1 + tail)
        product = tail / (1 + tail) ** 2
        skew = 1 - 2 * p
        mixed = -product / t**2 - product * skew * s / t**3
        return (
            float(p),
            np.array([float(product / t), float(-product * s / t**2)]),
            np.array(
                [
                    [float(product * skew / t**2), float(mixed)],
                    [float(mixed), float(2 * product * s / t**3 + product * skew * s**2 / t**4)],
                ]
            ),
        )


@pytest.mark.parametrize("score,temperature", [(0.0, 0.7), (0.4, 0.7), (8e-158, 1e-160)])
def test_logistic_derivatives_include_ties_and_underflow(score, temperature):
    """Both AD orders preserve representable sensitivities of rounded weights."""
    expected, gradient, hessian = _logistic_reference(score, temperature)
    with jax.enable_x64():

        def weight(parameters):
            s, t = parameters
            return temperature_softmax(jnp.array([s, 0.0]), t)[0]

        def measure(parameters):
            return (
                weight(parameters),
                jax.grad(weight)(parameters),
                jax.jacfwd(jax.grad(weight))(parameters),
                jax.jacrev(jax.jacfwd(weight))(parameters),
                jax.jvp(weight, (parameters,), (jnp.array([0.3, -0.2]),))[1],
            )

        value, actual, forward, reverse, tangent = jax.jit(measure)(jnp.array([score, temperature]))
        np.testing.assert_allclose(value, expected, rtol=2e-12, atol=0)
        np.testing.assert_allclose(actual, gradient, rtol=2e-12, atol=0)
        np.testing.assert_allclose(forward, hessian, rtol=2e-12, atol=0)
        np.testing.assert_allclose(reverse, hessian, rtol=2e-12, atol=0)
        np.testing.assert_allclose(tangent, gradient @ [0.3, -0.2], rtol=2e-12, atol=0)


@pytest.mark.parametrize("dtype,temperature", [(jnp.float32, 1e-30), (jnp.float64, 1e-160)])
def test_saturated_batched_derivatives(dtype, temperature):
    """Saturated derivatives vanish without reciprocal-square NaNs."""
    with jax.enable_x64():

        def weight(t):
            return temperature_softmax(jnp.array([1.0, 0.0], dtype=dtype), t)[0]

        def measure(t):
            return jax.jvp(weight, (t,), (jnp.ones_like(t),))[1], jax.grad(jax.grad(weight))(t)

        result = jax.jit(jax.vmap(measure))(jnp.array([temperature, temperature * 2], dtype=dtype))
        for values in result:
            np.testing.assert_array_equal(values, [0.0, 0.0])


def test_underflowed_tail_cross_derivatives():
    """A zero-valued tail still differentiates with respect to both nearby scores."""
    with localcontext() as context:
        context.prec = 90
        t = Decimal(1e-160)
        scores = [Decimal(x) for x in [0.0, -1e-160, -8e-158]]
        exp = [(s / t).exp() for s in scores]
        probabilities = [e / sum(exp) for e in exp]
        expected = np.array([float(-probabilities[2] * probabilities[j] / t) for j in (0, 1)])
    with jax.enable_x64():
        value, gradient = jax.jit(
            jax.value_and_grad(lambda x: temperature_softmax(x, jnp.array(1e-160))[2])
        )(jnp.array([0.0, -1e-160, -8e-158]))
        np.testing.assert_array_equal(value, 0.0)
        assert np.all(expected != 0)
        np.testing.assert_allclose(gradient[:2], expected, rtol=2e-12, atol=0)
        np.testing.assert_allclose(gradient[2], -expected.sum(), rtol=2e-12, atol=0)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_axes_broadcast_masks_and_empty_slices(axis):
    """Mask broadcasting, excluded NaNs and empty slices match native semantics."""
    with jax.enable_x64():
        scores = jnp.array([[0.2, -0.1, jnp.nan], [0.3, 0.1, jnp.nan]])
        mask = jnp.array([[True, True, False]])
        if axis in (1, -1):
            scores, mask = scores.T, mask.T
        safe = jnp.where(mask, scores, 0.0)
        expected = jax.nn.softmax(safe / 0.7, axis=axis, where=mask)
        run = jax.jit(lambda x, m: temperature_softmax(x, 0.7, axis=axis, where=m))
        np.testing.assert_allclose(run(scores, mask), expected, rtol=2e-12, atol=0)
        objective = lambda x, t: jnp.sum(temperature_softmax(x, t, axis=axis, where=mask) ** 2)
        gradient = jax.jit(jax.grad(objective, argnums=(0, 1)))(scores, jnp.array(0.7))
        assert all(np.isfinite(g).all() for g in gradient)
        np.testing.assert_array_equal(jnp.where(mask, 0.0, gradient[0]), jnp.zeros_like(scores))
        empty = jnp.zeros_like(mask)
        np.testing.assert_array_equal(run(scores, empty), jnp.zeros_like(scores))
        zero = jax.jit(
            jax.grad(lambda t: temperature_softmax(scores, t, axis=axis, where=empty).sum())
        )(jnp.array(0.7))
        np.testing.assert_array_equal(zero, 0.0)


def test_singleton_promotes_output_and_tangent_dtype():
    """A no-pair derivative follows the promoted result dtype."""
    with jax.enable_x64():
        value, gradient = jax.jit(
            jax.value_and_grad(
                lambda t: temperature_softmax(jnp.array([3.0], dtype=jnp.float32), t)[0]
            )
        )(jnp.array(0.7, dtype=jnp.float64))
        assert value.dtype == gradient.dtype == jnp.float64
        np.testing.assert_array_equal(value, 1.0)
        np.testing.assert_array_equal(gradient, 0.0)


def test_nnx_parameter_and_mutable_state():
    """One compiled NNX graph differentiates temperature across numerical regimes."""

    class Model(nnx.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nnx.Param(jnp.array(1.0))
            self.calls = nnx.BatchStat(jnp.array(0))

    traces = []

    def objective(model):
        traces.append(None)
        model.calls[...] += 1
        return temperature_softmax(jnp.array([1.0, 0.0]), model.temperature[...])[0]

    with jax.enable_x64():
        model = Model()
        run = nnx.jit(nnx.value_and_grad(objective))
        value, gradient = run(model)
        np.testing.assert_allclose(value, 0.7310585786300049, rtol=2e-12)
        np.testing.assert_allclose(gradient.temperature[...], -0.19661193324148185, rtol=2e-12)
        model.temperature[...] = jnp.array(1e-160)
        value, gradient = run(model)
        np.testing.assert_array_equal(value, 1.0)
        np.testing.assert_array_equal(gradient.temperature[...], 0.0)
        np.testing.assert_array_equal(model.calls[...], 2)
        assert len(traces) == 1


@pytest.mark.parametrize(
    "scores,temperature,axis,error",
    [
        (jnp.ones(2), jnp.ones(2), 0, ValueError),
        (jnp.ones(2), 1.0, 2, ValueError),
        (jnp.ones((0,)), 1.0, 0, ValueError),
        (jnp.ones(2, dtype=jnp.int32), 1.0, 0, TypeError),
        (jnp.ones(2), 1.0j, 0, TypeError),
    ],
)
def test_static_contract_errors(scores, temperature, axis, error):
    """Shape, axis and real-floating contracts fail consistently during tracing."""
    with pytest.raises(error):
        jax.jit(lambda x, t: temperature_softmax(x, t, axis=axis))(scores, temperature)


def test_ordinary_derivatives_and_permutation_match_native():
    """Joint score/temperature gradients agree before any range loss occurs."""
    with jax.enable_x64():
        scores = jnp.array([0.4, -0.3, 0.8])
        coefficients = jnp.array([2.0, -1.0, 0.3])
        for order in (jnp.array([0, 1, 2]), jnp.array([2, 0, 1])):
            native = lambda x, t: jnp.dot(jax.nn.softmax(x / t), coefficients[order])
            stable = lambda x, t: jnp.dot(temperature_softmax(x, t), coefficients[order])
            expected = jax.value_and_grad(native, argnums=(0, 1))(scores[order], jnp.array(0.7))
            actual = jax.jit(jax.value_and_grad(stable, argnums=(0, 1)))(
                scores[order], jnp.array(0.7)
            )
            for value, reference in zip(
                jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
            ):
                np.testing.assert_allclose(value, reference, rtol=2e-12, atol=0)


def test_mask_requires_boolean_and_broadcastable_shape():
    """Invalid mask metadata fails before numerical execution."""
    with pytest.raises(TypeError, match="boolean"):
        temperature_softmax(jnp.ones(3), where=jnp.ones(3))
    with pytest.raises(ValueError):
        temperature_softmax(jnp.ones(3), where=jnp.ones(2, dtype=jnp.bool_))
