"""Tests for soft logical operators (fuzzy logic)."""

import jax.numpy as jnp

from tests.core.test_soft_ops.conftest import assert_softbool


class TestLogicalNot:
    """Fuzzy NOT: 1 - x."""

    def test_complement(self) -> None:
        from diffbio.core.soft_ops.logical import logical_not

        x = jnp.array([0.0, 0.3, 0.5, 0.7, 1.0])
        result = logical_not(x)
        expected = jnp.array([1.0, 0.7, 0.5, 0.3, 0.0])
        assert jnp.allclose(result, expected)


class TestAll:
    """Fuzzy AND reduction (product or geometric mean)."""

    def test_all_true(self) -> None:
        from diffbio.core.soft_ops.logical import all

        x = jnp.array([1.0, 1.0, 1.0])
        assert jnp.allclose(all(x, axis=0), 1.0)

    def test_one_false_makes_all_low(self) -> None:
        from diffbio.core.soft_ops.logical import all

        x = jnp.array([1.0, 0.0, 1.0])
        assert float(all(x, axis=0)) < 0.01

    def test_batch_dimensions(self) -> None:
        from diffbio.core.soft_ops.logical import all

        x = jnp.array([[0.9, 0.8], [0.1, 0.2]])
        result = all(x, axis=-1)
        assert result.shape == (2,)

    def test_geometric_mean_mode(self) -> None:
        from diffbio.core.soft_ops.logical import all

        x = jnp.array([0.8, 0.9])
        all(x, axis=0, use_geometric_mean=False)
        geom_result = all(x, axis=0, use_geometric_mean=True)
        assert jnp.isfinite(geom_result)
        assert_softbool(geom_result[None])  # wrap scalar for assert


class TestAny:
    """Fuzzy OR reduction: 1 - all(1 - x)."""

    def test_all_false(self) -> None:
        from diffbio.core.soft_ops.logical import any

        x = jnp.array([0.0, 0.0, 0.0])
        assert float(any(x, axis=0)) < 0.01

    def test_one_true_makes_any_high(self) -> None:
        from diffbio.core.soft_ops.logical import any

        x = jnp.array([0.0, 1.0, 0.0])
        assert float(any(x, axis=0)) > 0.99


class TestLogicalAnd:
    """Fuzzy AND between two SoftBools."""

    def test_truth_table(self) -> None:
        from diffbio.core.soft_ops.logical import logical_and

        # Both true -> ~1
        assert float(logical_and(jnp.array(1.0), jnp.array(1.0))) > 0.99
        # One false -> ~0
        assert float(logical_and(jnp.array(1.0), jnp.array(0.0))) < 0.01
        assert float(logical_and(jnp.array(0.0), jnp.array(1.0))) < 0.01
        # Both false -> ~0
        assert float(logical_and(jnp.array(0.0), jnp.array(0.0))) < 0.01

    def test_output_is_softbool(self) -> None:
        from diffbio.core.soft_ops.logical import logical_and

        x = jnp.array([0.3, 0.7, 0.9])
        y = jnp.array([0.8, 0.2, 0.6])
        assert_softbool(logical_and(x, y))


class TestLogicalOr:
    """Fuzzy OR between two SoftBools."""

    def test_truth_table(self) -> None:
        from diffbio.core.soft_ops.logical import logical_or

        assert float(logical_or(jnp.array(1.0), jnp.array(1.0))) > 0.99
        assert float(logical_or(jnp.array(1.0), jnp.array(0.0))) > 0.99
        assert float(logical_or(jnp.array(0.0), jnp.array(1.0))) > 0.99
        assert float(logical_or(jnp.array(0.0), jnp.array(0.0))) < 0.01


class TestLogicalXor:
    """Fuzzy XOR between two SoftBools."""

    def test_truth_table(self) -> None:
        from diffbio.core.soft_ops.logical import logical_xor

        assert float(logical_xor(jnp.array(1.0), jnp.array(1.0))) < 0.05
        assert float(logical_xor(jnp.array(1.0), jnp.array(0.0))) > 0.95
        assert float(logical_xor(jnp.array(0.0), jnp.array(1.0))) > 0.95
        assert float(logical_xor(jnp.array(0.0), jnp.array(0.0))) < 0.05
