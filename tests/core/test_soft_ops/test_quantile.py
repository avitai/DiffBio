"""Tests for soft quantile, median, and percentile operators."""

import jax.numpy as jnp

from tests.core.test_soft_ops.conftest import assert_finite_grads, assert_simplex


class TestArgquantile:
    """Soft argquantile returning SoftIndex."""

    def test_hard_median_index(self) -> None:
        from diffbio.core.soft_ops.quantile import argquantile

        x = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = argquantile(x, q=jnp.array(0.5), axis=0, mode="hard")
        # Median of sorted [1,2,3,4,5] is index 2 (value 3)
        assert result.shape == (5,)
        assert jnp.allclose(jnp.sum(result), 1.0, atol=1e-5)

    def test_soft_output_is_simplex(self) -> None:
        from diffbio.core.soft_ops.quantile import argquantile

        x = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = argquantile(
            x, q=jnp.array(0.5), axis=0, softness=0.1, mode="smooth",
        )
        assert_simplex(result, axis=-1, atol=0.05)

    def test_vector_q(self) -> None:
        from diffbio.core.soft_ops.quantile import argquantile

        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q = jnp.array([0.25, 0.5, 0.75])
        result = argquantile(x, q=q, axis=0, softness=0.1, mode="smooth")
        # Should have q dimension prepended
        assert result.shape == (3, 5)


class TestQuantile:
    """Soft quantile returning value."""

    def test_hard_median(self) -> None:
        from diffbio.core.soft_ops.quantile import quantile

        x = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = quantile(x, q=jnp.array(0.5), axis=0, mode="hard")
        assert jnp.allclose(result, 3.0, atol=0.1)

    def test_soft_median_approaches_hard(self) -> None:
        from diffbio.core.soft_ops.quantile import quantile

        x = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = quantile(
            x, q=jnp.array(0.5), axis=0, softness=0.01, mode="smooth",
        )
        assert jnp.allclose(result, 3.0, atol=1.0)

    def test_differentiable(self) -> None:
        from diffbio.core.soft_ops.quantile import quantile

        x = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        assert_finite_grads(
            lambda x: quantile(x, q=jnp.array(0.5), axis=0, softness=0.1),
            (x,),
        )


class TestMedian:
    """Soft median (q=0.5 quantile)."""

    def test_hard_mode(self) -> None:
        from diffbio.core.soft_ops.quantile import median

        x = jnp.array([5.0, 1.0, 3.0])
        result = median(x, axis=0, mode="hard")
        assert jnp.allclose(result, 3.0, atol=0.1)


class TestArgmedian:
    """Soft argmedian."""

    def test_hard_mode(self) -> None:
        from diffbio.core.soft_ops.quantile import argmedian

        x = jnp.array([5.0, 1.0, 3.0])
        result = argmedian(x, axis=0, mode="hard")
        assert result.shape == (3,)
        assert jnp.allclose(jnp.sum(result), 1.0, atol=1e-5)


class TestPercentile:
    """Soft percentile (quantile with q in [0, 100])."""

    def test_hard_50th(self) -> None:
        from diffbio.core.soft_ops.quantile import percentile

        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = percentile(x, p=jnp.array(50.0), axis=0, mode="hard")
        assert jnp.allclose(result, 3.0, atol=0.1)


class TestArgpercentile:
    """Soft argpercentile."""

    def test_hard_mode(self) -> None:
        from diffbio.core.soft_ops.quantile import argpercentile

        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = argpercentile(x, p=jnp.array(50.0), axis=0, mode="hard")
        assert result.shape == (5,)
