"""Tests for the ArcsinhCofactor operator (B1: CyTOF learnable arcsinh cofactor)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from diffbio.operators.normalization.arcsinh_cofactor import (
    ArcsinhCofactor,
    ArcsinhCofactorConfig,
    arcsinh_transform,
)

_DEFAULT_COFACTOR = 5.0


def _intensities(n_channels: int, seed: int) -> jnp.ndarray:
    """Return non-negative marker intensities for one cell."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.gamma(2.0, 50.0, size=(n_channels,)).astype(np.float32))


def _operator(n_channels: int, *, cofactor_init: float = _DEFAULT_COFACTOR, trainable: bool = True):
    config = ArcsinhCofactorConfig(
        num_channels=n_channels, cofactor_init=cofactor_init, trainable=trainable
    )
    return ArcsinhCofactor(config, rngs=nnx.Rngs(0))


# --- Free-function parity with the fixed arcsinh(x/5) convention ------------------


def test_free_function_matches_fixed_cofactor_formula() -> None:
    intensities = _intensities(24, seed=0)
    output = arcsinh_transform(intensities, _DEFAULT_COFACTOR)
    expected = jnp.arcsinh(intensities / _DEFAULT_COFACTOR)
    np.testing.assert_allclose(np.asarray(output), np.asarray(expected), atol=1e-6)


def test_operator_init_reproduces_frozen_arcsinh() -> None:
    """Init-at-frozen: cofactor_init=5 reproduces the frozen arcsinh(x/5) exactly."""
    intensities = _intensities(24, seed=1)
    operator = _operator(24)
    transformed = operator.apply({"intensities": intensities}, {}, None)[0]["transformed"]
    expected = jnp.arcsinh(intensities / _DEFAULT_COFACTOR)
    np.testing.assert_allclose(np.asarray(transformed), np.asarray(expected), atol=1e-5)


# --- Per-channel behaviour -------------------------------------------------------


def test_per_channel_cofactor_applies_independently() -> None:
    """A distinct cofactor per channel transforms each channel independently."""
    intensities = _intensities(4, seed=2)
    operator = _operator(4)
    cofactors = jnp.asarray([1.0, 2.0, 5.0, 10.0], dtype=jnp.float32)
    operator.raw_cofactor.value = jnp.log(jnp.expm1(cofactors))
    transformed = operator.apply({"intensities": intensities}, {}, None)[0]["transformed"]
    expected = jnp.arcsinh(intensities / cofactors)
    np.testing.assert_allclose(np.asarray(transformed), np.asarray(expected), atol=1e-5)


def test_cofactor_has_one_parameter_per_channel() -> None:
    operator = _operator(37)
    assert operator.raw_cofactor.value.shape == (37,)


# --- Differentiability and positivity --------------------------------------------


def _cofactor_grad(operator: ArcsinhCofactor, intensities: jnp.ndarray) -> jnp.ndarray:
    def loss_fn(op: ArcsinhCofactor) -> jax.Array:
        return op.apply({"intensities": intensities}, {}, None)[0]["transformed"].sum()

    grads = nnx.grad(loss_fn)(operator)
    return grads.raw_cofactor.value


def test_gradient_flows_to_cofactor() -> None:
    intensities = _intensities(16, seed=3)
    grad = _cofactor_grad(_operator(16), intensities)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0.0)


def test_cofactor_stays_positive_after_update() -> None:
    """A gradient step must not drive any cofactor to zero or negative."""
    intensities = _intensities(16, seed=4)
    operator = _operator(16)
    grad = _cofactor_grad(operator, intensities)
    operator.raw_cofactor.value = operator.raw_cofactor.value - 1e3 * grad
    cofactor = jax.nn.softplus(operator.raw_cofactor.value)
    assert jnp.all(cofactor > 0.0)


def test_frozen_arm_has_zero_cofactor_gradient() -> None:
    """With trainable=False the cofactor is stop-gradiented, so its gradient is zero."""
    intensities = _intensities(16, seed=5)
    grad = _cofactor_grad(_operator(16, trainable=False), intensities)
    np.testing.assert_allclose(np.asarray(grad), 0.0, atol=1e-12)


# --- Batching --------------------------------------------------------------------


def test_vmap_over_cells_matches_per_cell() -> None:
    operator = _operator(8)
    cells = jnp.stack([_intensities(8, seed=s) for s in range(5)])
    batched = jax.vmap(
        lambda cell: operator.apply({"intensities": cell}, {}, None)[0]["transformed"]
    )(cells)
    per_cell = jnp.stack(
        [operator.apply({"intensities": c}, {}, None)[0]["transformed"] for c in cells]
    )
    np.testing.assert_allclose(np.asarray(batched), np.asarray(per_cell), atol=1e-6)


# --- Config validation -----------------------------------------------------------


def test_non_positive_cofactor_init_raises() -> None:
    with pytest.raises(ValueError, match="cofactor_init"):
        ArcsinhCofactorConfig(num_channels=4, cofactor_init=0.0)


def test_non_positive_num_channels_raises() -> None:
    with pytest.raises(ValueError, match="num_channels"):
        ArcsinhCofactorConfig(num_channels=0, cofactor_init=5.0)
