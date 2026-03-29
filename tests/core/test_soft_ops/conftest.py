"""Shared fixtures for soft_ops tests."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float


@pytest.fixture
def make_array():
    """Factory fixture to create test arrays with controlled randomness."""

    def _make(shape: tuple[int, ...], seed: int = 0) -> Float[Array, "..."]:
        key = jax.random.key(seed)
        return jax.random.normal(key, shape)

    return _make


def assert_simplex(
    x: Float[Array, "..."],
    axis: int = -1,
    atol: float = 1e-5,
) -> None:
    """Assert array forms a valid probability simplex along axis."""
    assert jnp.all(x >= -atol), f"Simplex has negative values: min={float(x.min())}"
    sums = jnp.sum(x, axis=axis)
    assert jnp.allclose(sums, 1.0, atol=atol), (
        f"Simplex doesn't sum to 1: {sums}"
    )


def assert_softbool(x: Float[Array, "..."], atol: float = 1e-5) -> None:
    """Assert all values are in [0, 1] (valid SoftBool)."""
    assert jnp.all(x >= -atol), f"SoftBool has values < 0: min={float(x.min())}"
    assert jnp.all(x <= 1.0 + atol), (
        f"SoftBool has values > 1: max={float(x.max())}"
    )


def assert_finite_grads(
    fn,
    args: tuple,
    argnums: int | tuple[int, ...] = 0,
) -> None:
    """Assert that gradients are finite (no NaN or Inf)."""
    grad_fn = jax.grad(lambda *a: jnp.sum(fn(*a)), argnums=argnums)
    grads = grad_fn(*args)
    if isinstance(grads, tuple):
        for i, g in enumerate(grads):
            assert jnp.all(jnp.isfinite(g)), (
                f"Gradient {i} has non-finite values"
            )
    else:
        assert jnp.all(jnp.isfinite(grads)), "Gradient has non-finite values"
