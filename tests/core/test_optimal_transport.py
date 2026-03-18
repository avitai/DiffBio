"""Tests for optimal transport components (SinkhornLayer).

These tests define the expected behavior for the Sinkhorn optimal transport
layer, verifying marginal constraints, convergence, differentiability, and
edge cases.
"""

import jax
import jax.numpy as jnp
import pytest

from diffbio.core.optimal_transport import SinkhornLayer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cost_matrix():
    """Provide a 5x6 cost matrix from pairwise distances."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (5, 3))
    y = jax.random.normal(k2, (6, 3))
    # Squared Euclidean distance
    diff = x[:, None, :] - y[None, :, :]
    return jnp.sum(diff**2, axis=-1)


@pytest.fixture
def uniform_marginals():
    """Provide uniform marginal distributions for a 5x6 problem."""
    a = jnp.ones(5) / 5
    b = jnp.ones(6) / 6
    return a, b


@pytest.fixture
def nonuniform_marginals():
    """Provide non-uniform marginal distributions for a 5x6 problem."""
    key = jax.random.key(1)
    k1, k2 = jax.random.split(key)
    a = jax.nn.softmax(jax.random.normal(k1, (5,)))
    b = jax.nn.softmax(jax.random.normal(k2, (6,)))
    return a, b


# =============================================================================
# TestSinkhornLayer
# =============================================================================


class TestSinkhornLayer:
    """Tests for SinkhornLayer marginal constraints, convergence, and gradients."""

    def test_output_shape(self, rngs, cost_matrix, uniform_marginals):
        """Test that the transport plan has correct shape."""
        a, b = uniform_marginals
        layer = SinkhornLayer(epsilon=0.1, num_iters=50, rngs=rngs)
        plan = layer(cost_matrix, a, b)
        assert plan.shape == cost_matrix.shape

    def test_output_finite(self, rngs, cost_matrix, uniform_marginals):
        """Test that all values in the transport plan are finite."""
        a, b = uniform_marginals
        layer = SinkhornLayer(epsilon=0.1, num_iters=50, rngs=rngs)
        plan = layer(cost_matrix, a, b)
        assert jnp.isfinite(plan).all()

    def test_output_nonnegative(self, rngs, cost_matrix, uniform_marginals):
        """Test that the transport plan is non-negative."""
        a, b = uniform_marginals
        layer = SinkhornLayer(epsilon=0.1, num_iters=50, rngs=rngs)
        plan = layer(cost_matrix, a, b)
        assert jnp.all(plan >= -1e-7)

    def test_row_marginal_constraint(self, rngs, cost_matrix, uniform_marginals):
        """Test that row sums of the transport plan match marginal a."""
        a, b = uniform_marginals
        layer = SinkhornLayer(epsilon=0.1, num_iters=100, rngs=rngs)
        plan = layer(cost_matrix, a, b)
        row_sums = plan.sum(axis=1)
        assert jnp.allclose(row_sums, a, atol=1e-3), (
            f"Row marginal mismatch: got {row_sums}, expected {a}"
        )

    def test_col_marginal_constraint(self, rngs, cost_matrix, uniform_marginals):
        """Test that column sums of the transport plan match marginal b."""
        a, b = uniform_marginals
        layer = SinkhornLayer(epsilon=0.1, num_iters=100, rngs=rngs)
        plan = layer(cost_matrix, a, b)
        col_sums = plan.sum(axis=0)
        assert jnp.allclose(col_sums, b, atol=1e-3), (
            f"Column marginal mismatch: got {col_sums}, expected {b}"
        )

    def test_nonuniform_marginals(self, rngs, cost_matrix, nonuniform_marginals):
        """Test marginal constraints with non-uniform distributions."""
        a, b = nonuniform_marginals
        layer = SinkhornLayer(epsilon=0.1, num_iters=100, rngs=rngs)
        plan = layer(cost_matrix, a, b)
        row_sums = plan.sum(axis=1)
        col_sums = plan.sum(axis=0)
        assert jnp.allclose(row_sums, a, atol=1e-3)
        assert jnp.allclose(col_sums, b, atol=1e-3)

    def test_convergence_with_more_iterations(self, rngs, cost_matrix, uniform_marginals):
        """Test that more iterations improve marginal accuracy."""
        a, b = uniform_marginals
        layer_few = SinkhornLayer(epsilon=0.1, num_iters=5, rngs=rngs)
        layer_many = SinkhornLayer(epsilon=0.1, num_iters=100, rngs=rngs)

        plan_few = layer_few(cost_matrix, a, b)
        plan_many = layer_many(cost_matrix, a, b)

        error_few = jnp.abs(plan_few.sum(axis=1) - a).max()
        error_many = jnp.abs(plan_many.sum(axis=1) - a).max()
        assert error_many <= error_few + 1e-7

    def test_gradient_flow_through_cost(self, rngs, cost_matrix, uniform_marginals):
        """Test that gradients flow through the cost matrix."""
        a, b = uniform_marginals
        layer = SinkhornLayer(epsilon=0.1, num_iters=50, rngs=rngs)

        def loss_fn(c):
            plan = layer(c, a, b)
            return (plan * c).sum()

        grads = jax.grad(loss_fn)(cost_matrix)
        assert grads.shape == cost_matrix.shape
        assert jnp.isfinite(grads).all()
        assert jnp.any(grads != 0.0)

    def test_gradient_flow_through_marginals(self, rngs, cost_matrix):
        """Test that gradients flow through marginal distributions."""
        layer = SinkhornLayer(epsilon=0.1, num_iters=50, rngs=rngs)
        a = jnp.ones(5) / 5
        b = jnp.ones(6) / 6

        def loss_fn(marginal_a):
            plan = layer(cost_matrix, marginal_a, b)
            return (plan * cost_matrix).sum()

        grads = jax.grad(loss_fn)(a)
        assert grads.shape == a.shape
        assert jnp.isfinite(grads).all()

    def test_jit_compatible(self, rngs, cost_matrix, uniform_marginals):
        """Test that the layer works under JIT compilation."""
        a, b = uniform_marginals
        layer = SinkhornLayer(epsilon=0.1, num_iters=50, rngs=rngs)

        @jax.jit
        def forward(c, marginal_a, marginal_b):
            return layer(c, marginal_a, marginal_b)

        plan = forward(cost_matrix, a, b)
        assert plan.shape == cost_matrix.shape
        assert jnp.isfinite(plan).all()

    def test_epsilon_effect(self, rngs, cost_matrix, uniform_marginals):
        """Test that smaller epsilon produces sharper transport plans."""
        a, b = uniform_marginals
        layer_large_eps = SinkhornLayer(epsilon=1.0, num_iters=100, rngs=rngs)
        layer_small_eps = SinkhornLayer(epsilon=0.01, num_iters=100, rngs=rngs)

        plan_large = layer_large_eps(cost_matrix, a, b)
        plan_small = layer_small_eps(cost_matrix, a, b)

        # Smaller epsilon -> sharper (lower entropy) transport plan
        entropy_large = -(plan_large * jnp.log(plan_large + 1e-10)).sum()
        entropy_small = -(plan_small * jnp.log(plan_small + 1e-10)).sum()
        assert entropy_small < entropy_large


# =============================================================================
# TestSinkhornEdgeCases
# =============================================================================


class TestSinkhornEdgeCases:
    """Tests for SinkhornLayer edge cases."""

    def test_uniform_weights_symmetric(self, rngs):
        """Test with uniform weights and symmetric cost matrix."""
        n = 4
        cost = jnp.ones((n, n)) - jnp.eye(n)  # 0 on diagonal, 1 elsewhere
        a = jnp.ones(n) / n
        b = jnp.ones(n) / n
        layer = SinkhornLayer(epsilon=0.1, num_iters=100, rngs=rngs)
        plan = layer(cost, a, b)

        # Symmetric cost + uniform marginals -> symmetric plan
        assert jnp.allclose(plan, plan.T, atol=1e-4)

    def test_symmetric_cost_produces_symmetric_plan(self, rngs):
        """Test that symmetric cost matrix with equal marginals gives symmetric plan."""
        key = jax.random.key(10)
        n = 5
        raw = jax.random.normal(key, (n, n))
        cost = raw + raw.T  # Force symmetric
        a = jnp.ones(n) / n
        b = jnp.ones(n) / n
        layer = SinkhornLayer(epsilon=0.1, num_iters=100, rngs=rngs)
        plan = layer(cost, a, b)
        assert jnp.allclose(plan, plan.T, atol=1e-4)

    def test_single_element(self, rngs):
        """Test with 1x1 cost matrix."""
        cost = jnp.array([[1.0]])
        a = jnp.array([1.0])
        b = jnp.array([1.0])
        layer = SinkhornLayer(epsilon=0.1, num_iters=10, rngs=rngs)
        plan = layer(cost, a, b)
        assert plan.shape == (1, 1)
        assert jnp.allclose(plan, jnp.array([[1.0]]), atol=1e-5)

    def test_square_identity_cost(self, rngs):
        """Test with identity cost matrix and uniform marginals."""
        n = 4
        cost = jnp.eye(n)
        a = jnp.ones(n) / n
        b = jnp.ones(n) / n
        layer = SinkhornLayer(epsilon=0.01, num_iters=200, rngs=rngs)
        plan = layer(cost, a, b)

        # With identity cost, off-diagonal has lower cost -> mass flows off-diagonal
        # Plan should concentrate off-diagonal for small epsilon
        row_sums = plan.sum(axis=1)
        assert jnp.allclose(row_sums, a, atol=1e-3)

    def test_large_problem(self, rngs):
        """Test with a larger problem size (50x60)."""
        key = jax.random.key(5)
        k1, k2 = jax.random.split(key)
        n, m = 50, 60
        cost = jnp.abs(jax.random.normal(k1, (n, m)))
        a = jax.nn.softmax(jax.random.normal(k2, (n,)))
        b_key = jax.random.split(k2)[0]
        b = jax.nn.softmax(jax.random.normal(b_key, (m,)))

        layer = SinkhornLayer(epsilon=0.1, num_iters=100, rngs=rngs)
        plan = layer(cost, a, b)
        assert plan.shape == (n, m)
        assert jnp.isfinite(plan).all()
        assert jnp.allclose(plan.sum(axis=1), a, atol=1e-3)
        assert jnp.allclose(plan.sum(axis=0), b, atol=1e-3)

    def test_zero_cost_matrix(self, rngs):
        """Test with zero cost matrix -- plan should be outer product of marginals."""
        n, m = 3, 4
        cost = jnp.zeros((n, m))
        a = jnp.array([0.2, 0.3, 0.5])
        b = jnp.array([0.1, 0.2, 0.3, 0.4])
        layer = SinkhornLayer(epsilon=0.1, num_iters=50, rngs=rngs)
        plan = layer(cost, a, b)

        # With zero cost, the maximum entropy coupling is the outer product
        expected = a[:, None] * b[None, :]
        assert jnp.allclose(plan, expected, atol=1e-4)
