"""Tests for diffbio.losses.metric_losses module.

These tests define the expected behavior of differentiable metric-based
loss functions, starting with the differentiable AUROC approximation.
"""

import jax
import jax.numpy as jnp
import pytest

from diffbio.losses.metric_losses import DifferentiableAUROC, ExactAUROC


class TestDifferentiableAUROCConfig:
    """Tests for DifferentiableAUROC construction and configuration."""

    def test_default_temperature(self) -> None:
        """Default temperature should be 1.0."""
        auroc = DifferentiableAUROC()
        assert float(auroc.temperature[...]) == pytest.approx(1.0)

    def test_custom_temperature(self) -> None:
        """Custom temperature should be stored correctly."""
        auroc = DifferentiableAUROC(temperature=0.5)
        assert float(auroc.temperature[...]) == pytest.approx(0.5)


class TestDifferentiableAUROC:
    """Tests for DifferentiableAUROC forward computation."""

    def test_perfect_separation_high_auroc(self) -> None:
        """All positives scored higher than all negatives gives AUROC near 1."""
        auroc_fn = DifferentiableAUROC(temperature=0.1)
        predictions = jnp.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        labels = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        result = auroc_fn(predictions, labels)
        assert float(result) > 0.95

    def test_random_scores_near_half(self) -> None:
        """Random scores with balanced labels should give AUROC near 0.5."""
        auroc_fn = DifferentiableAUROC(temperature=1.0)
        key = jax.random.PRNGKey(0)
        predictions = jax.random.uniform(key, (200,))
        labels = jnp.concatenate([jnp.ones(100), jnp.zeros(100)])
        result = auroc_fn(predictions, labels)
        assert 0.3 < float(result) < 0.7

    def test_reversed_low_auroc(self) -> None:
        """All positives scored lower than all negatives gives AUROC near 0."""
        auroc_fn = DifferentiableAUROC(temperature=0.1)
        predictions = jnp.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        result = auroc_fn(predictions, labels)
        assert float(result) < 0.05

    def test_output_range(self) -> None:
        """AUROC should always be in [0, 1]."""
        auroc_fn = DifferentiableAUROC(temperature=1.0)
        key = jax.random.PRNGKey(42)
        for i in range(5):
            k1, k2, key = jax.random.split(key, 3)
            predictions = jax.random.normal(k1, (50,))
            labels = jax.random.bernoulli(k2, 0.5, (50,)).astype(jnp.float32)
            result = auroc_fn(predictions, labels)
            assert 0.0 <= float(result) <= 1.0

    def test_temperature_effect(self) -> None:
        """Lower temperature should give a sharper approximation of hard AUROC."""
        predictions = jnp.array([0.9, 0.8, 0.55, 0.45, 0.2, 0.1])
        labels = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        auroc_warm = DifferentiableAUROC(temperature=5.0)
        auroc_cool = DifferentiableAUROC(temperature=0.01)

        result_warm = float(auroc_warm(predictions, labels))
        result_cool = float(auroc_cool(predictions, labels))

        # With perfect separation, cool temp should be closer to 1.0
        assert result_cool > result_warm


class TestGradientFlow:
    """Tests for gradient flow through DifferentiableAUROC."""

    def test_gradient_wrt_predictions(self) -> None:
        """Gradients should flow through predictions."""
        auroc_fn = DifferentiableAUROC(temperature=1.0)
        predictions = jnp.array([0.9, 0.8, 0.1, 0.2])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])

        grad_fn = jax.grad(lambda p: auroc_fn(p, labels))
        grads = grad_fn(predictions)

        assert grads is not None
        assert grads.shape == predictions.shape

    def test_gradient_finite(self) -> None:
        """All gradients should be finite."""
        auroc_fn = DifferentiableAUROC(temperature=1.0)
        predictions = jnp.array([0.9, 0.8, 0.1, 0.2])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])

        grad_fn = jax.grad(lambda p: auroc_fn(p, labels))
        grads = grad_fn(predictions)

        assert jnp.all(jnp.isfinite(grads))

    def test_gradient_nonzero(self) -> None:
        """Gradients should be non-zero for non-degenerate inputs."""
        auroc_fn = DifferentiableAUROC(temperature=1.0)
        predictions = jnp.array([0.6, 0.55, 0.45, 0.4])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])

        grad_fn = jax.grad(lambda p: auroc_fn(p, labels))
        grads = grad_fn(predictions)

        assert jnp.any(grads != 0.0)


class TestJITCompatibility:
    """Tests for JIT compilation of DifferentiableAUROC."""

    def test_jit_forward(self) -> None:
        """JIT compilation should succeed for forward pass."""
        auroc_fn = DifferentiableAUROC(temperature=1.0)
        predictions = jnp.array([0.9, 0.8, 0.1, 0.2])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])

        jit_auroc = jax.jit(auroc_fn)
        result = jit_auroc(predictions, labels)

        assert jnp.isfinite(result)

    def test_jit_gradient(self) -> None:
        """JIT + grad should work together."""
        auroc_fn = DifferentiableAUROC(temperature=1.0)
        predictions = jnp.array([0.9, 0.8, 0.1, 0.2])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])

        jit_grad = jax.jit(jax.grad(lambda p: auroc_fn(p, labels)))
        grads = jit_grad(predictions)

        assert jnp.all(jnp.isfinite(grads))


class TestEdgeCases:
    """Tests for edge-case inputs to DifferentiableAUROC."""

    def test_single_positive_single_negative(self) -> None:
        """Should work with exactly one positive and one negative sample."""
        auroc_fn = DifferentiableAUROC(temperature=1.0)
        predictions = jnp.array([0.9, 0.1])
        labels = jnp.array([1.0, 0.0])
        result = auroc_fn(predictions, labels)
        assert jnp.isfinite(result)
        assert float(result) > 0.5

    def test_all_same_scores(self) -> None:
        """All equal scores should give AUROC = 0.5."""
        auroc_fn = DifferentiableAUROC(temperature=1.0)
        predictions = jnp.array([0.5, 0.5, 0.5, 0.5])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])
        result = auroc_fn(predictions, labels)
        assert float(result) == pytest.approx(0.5, abs=1e-5)


class TestExactAUROC:
    """Tests for ExactAUROC evaluation metric."""

    def test_perfect_separation(self) -> None:
        """All positives scored higher than all negatives gives exactly 1.0."""
        exact = ExactAUROC()
        predictions = jnp.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        labels = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        result = exact(predictions, labels)
        assert float(result) == pytest.approx(1.0)

    def test_random_near_half(self) -> None:
        """Random scores with balanced labels should yield AUROC near 0.5."""
        exact = ExactAUROC()
        key = jax.random.PRNGKey(0)
        predictions = jax.random.uniform(key, (200,))
        labels = jnp.concatenate([jnp.ones(100), jnp.zeros(100)])
        result = exact(predictions, labels)
        assert 0.3 < float(result) < 0.7

    def test_matches_known_value(self) -> None:
        """Verify on a hand-computed case: 3 of 4 pairs correct gives 0.75."""
        exact = ExactAUROC()
        # Pairs: (0.35>0.1)=Y, (0.35>0.4)=N, (0.8>0.1)=Y, (0.8>0.4)=Y => 3/4
        predictions = jnp.array([0.1, 0.4, 0.35, 0.8])
        labels = jnp.array([0.0, 0.0, 1.0, 1.0])
        result = exact(predictions, labels)
        assert float(result) == pytest.approx(0.75)

    def test_gradient_flow(self) -> None:
        """JAX can trace through ExactAUROC without raising errors.

        The trapezoidal-rule implementation uses argsort internally, so
        gradients are zero -- but the trace must succeed.
        """
        exact = ExactAUROC()
        predictions = jnp.array([0.9, 0.8, 0.1, 0.2])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])

        grad_fn = jax.grad(lambda p: exact(p, labels))
        grads = grad_fn(predictions)

        assert grads is not None
        assert grads.shape == predictions.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_jit_compatible(self) -> None:
        """ExactAUROC must be JIT-compilable."""
        exact = ExactAUROC()
        predictions = jnp.array([0.9, 0.8, 0.1, 0.2])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])

        jit_exact = jax.jit(exact)
        result = jit_exact(predictions, labels)

        assert jnp.isfinite(result)
        assert float(result) == pytest.approx(1.0)
