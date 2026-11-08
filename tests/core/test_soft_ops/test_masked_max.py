"""Soft max and argmax over the entries a ``where`` mask includes.

An excluded entry takes no part in the standardization statistics and gets zero weight, as in
``jax.nn.standardize(where=)`` and ``jax.nn.softmax(where=)``, so a padded axis gives the result
of the unpadded one. The inputs are exactly representable and their means, variances and
softmax denominators are exact in float32 under any summation order, so the comparisons are
exact rather than within a tolerance.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffbio.core import soft_ops

REAL = jnp.array([1.0, 3.0])
PADDED = jnp.array([1.0, 3.0, -50.0, 7.0])
MASK = jnp.array([True, True, False, False])


def test_soft_argmax_weights_only_included_entries() -> None:
    index = soft_ops.argmax(PADDED, softness=0.5, where=MASK)

    np.testing.assert_array_equal(
        np.asarray(index[:2]), np.asarray(soft_ops.argmax(REAL, softness=0.5))
    )
    np.testing.assert_array_equal(np.asarray(index[2:]), [0.0, 0.0])


def test_soft_max_equals_the_max_of_the_included_entries() -> None:
    value = soft_ops.max(PADDED, softness=0.5, where=MASK)

    assert float(value) == float(soft_ops.max(REAL, softness=0.5))


def test_soft_max_masks_each_row_along_the_axis() -> None:
    rows = jnp.stack([PADDED, jnp.array([2.0, 6.0, 4.0, 9.0])])
    mask = jnp.stack([MASK, jnp.array([True, True, True, False])])

    values = soft_ops.max(rows.T, axis=0, softness=0.5, where=mask.T)

    assert float(values[0]) == float(soft_ops.max(REAL, softness=0.5))
    assert float(values[1]) == float(soft_ops.max(jnp.array([2.0, 6.0, 4.0]), softness=0.5))


def test_hard_max_ignores_excluded_entries() -> None:
    assert float(soft_ops.max(PADDED, mode="hard", where=MASK)) == 3.0
    np.testing.assert_array_equal(
        np.asarray(soft_ops.argmax(PADDED, mode="hard", where=MASK)), [0.0, 1.0, 0.0, 0.0]
    )


def test_excluded_entries_get_no_gradient() -> None:
    grad = jax.grad(lambda x: soft_ops.max(x, softness=0.5, where=MASK))(PADDED)
    unpadded = jax.grad(lambda x: soft_ops.max(x, softness=0.5))(REAL)

    np.testing.assert_array_equal(np.asarray(grad[:2]), np.asarray(unpadded))
    np.testing.assert_array_equal(np.asarray(grad[2:]), [0.0, 0.0])


@pytest.mark.parametrize("method", ["neuralsort", "sorting_network", "ot"])
def test_a_method_without_a_masked_form_refuses_where(method: str) -> None:
    with pytest.raises(ValueError, match="where"):
        soft_ops.max(PADDED, softness=0.5, method=method, where=MASK)
