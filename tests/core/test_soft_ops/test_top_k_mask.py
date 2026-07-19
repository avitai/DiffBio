"""Tests for the soft top-k selection mask primitive (``top_k_mask``).

The smooth mask is the Euclidean projection onto the capped simplex
``{m : 0 <= m <= 1, sum(m) = k}`` -- the convex-analytic sparse top-k operator
(Sander et al., ICML 2023; Blondel et al., ICML 2020) -- a bounded ``[0, 1]``
membership summing to exactly ``k``. ``"hard"`` mode is the exact indicator and,
via ``top_k_mask_st``, a straight-through estimator (exact indicator forward,
smooth gradient backward).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from diffbio.core.soft_ops import top_k_mask, top_k_mask_st


def _scores(n: int, seed: int) -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(seed).normal(size=(n,)).astype(np.float32))


def _hard_topk_set(scores: jnp.ndarray, k: int) -> set[int]:
    return {int(i) for i in np.argsort(np.asarray(scores))[::-1][:k]}


# --- Hard mode: exact top-k indicator -------------------------------------------


def test_hard_mode_matches_jax_top_k_selection() -> None:
    scores = _scores(40, seed=0)
    k = 12
    mask = top_k_mask(scores, k, mode="hard")
    selected = {int(i) for i in np.flatnonzero(np.asarray(mask) > 0.5)}
    assert selected == _hard_topk_set(scores, k)


def test_hard_mode_is_binary_and_sums_to_k() -> None:
    scores = _scores(30, seed=1)
    mask = top_k_mask(scores, 9, mode="hard")
    assert set(float(v) for v in np.unique(np.asarray(mask))) <= {0.0, 1.0}
    np.testing.assert_allclose(float(jnp.sum(mask)), 9.0)


def test_hard_mode_sums_to_exactly_k_under_boundary_ties() -> None:
    # Ranks break ties by position, so a tie spanning the k/(k+1) boundary still
    # selects exactly k elements rather than over-selecting.
    scores = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0], dtype=jnp.float32)
    for k in (1, 3, 5):
        mask = top_k_mask(scores, k, mode="hard")
        assert float(jnp.sum(mask)) == float(k)
        assert set(float(v) for v in np.unique(np.asarray(mask))) <= {0.0, 1.0}


# --- Smooth mode: bounded [0,1], sharpens ---------------------------------------


def test_smooth_mode_is_bounded_in_unit_interval() -> None:
    scores = _scores(50, seed=2)
    mask = top_k_mask(scores, 15, softness=0.3, mode="smooth")
    # The projection clips exactly onto [0, 1] -- no tolerance needed.
    assert float(jnp.min(mask)) >= 0.0
    assert float(jnp.max(mask)) <= 1.0


def test_smooth_mode_sharpens_to_hard_as_softness_shrinks() -> None:
    scores = _scores(40, seed=3)
    k = 10
    hard = top_k_mask(scores, k, mode="hard")
    sharp = top_k_mask(scores, k, softness=1e-2, mode="smooth")
    np.testing.assert_allclose(np.asarray(sharp), np.asarray(hard), atol=1e-2)


def test_smooth_mode_sums_to_exactly_k() -> None:
    # The capped-simplex projection conserves the budget exactly at any softness.
    scores = _scores(60, seed=4)
    for softness in (0.5, 0.1, 0.02):
        mask = top_k_mask(scores, 20, softness=softness, mode="smooth")
        # Budget is conserved to float32 rounding (~3e-5 measured).
        np.testing.assert_allclose(float(jnp.sum(mask)), 20.0, atol=1e-4)


def test_smooth_mode_is_sparse() -> None:
    # Genuine 0s and 1s: most genes are exactly selected/rejected, only a few
    # boundary elements are fractional.
    scores = _scores(50, seed=14)
    mask = np.asarray(top_k_mask(scores, 15, softness=0.1, mode="smooth"))
    saturated = np.sum((mask < 1e-4) | (mask > 1.0 - 1e-4))
    assert saturated >= 40


def test_smooth_mode_arbitrary_axis_sums_to_k() -> None:
    scores = jnp.asarray(np.random.default_rng(15).normal(size=(3, 16)).astype(np.float32))
    mask = top_k_mask(scores, 6, axis=1, softness=0.1, mode="smooth")
    assert mask.shape == (3, 16)
    np.testing.assert_allclose(np.asarray(jnp.sum(mask, axis=1)), np.full(3, 6.0), atol=1e-4)


# --- Straight-through: exact forward, smooth gradient ---------------------------


def test_st_forward_is_exact_hard_mask() -> None:
    scores = _scores(35, seed=5)
    st_mask = top_k_mask_st(scores, 11, softness=0.2)
    hard = top_k_mask(scores, 11, mode="hard")
    np.testing.assert_array_equal(np.asarray(st_mask), np.asarray(hard))


def test_st_gradient_is_finite_and_nonzero() -> None:
    scores = _scores(30, seed=6)

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(top_k_mask_st(x, 8, softness=0.5) * jnp.arange(30, dtype=jnp.float32))

    grad = jax.grad(loss)(scores)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0


def test_smooth_gradient_couples_across_elements() -> None:
    # The mask redistributes a fixed budget across the boundary block, so a
    # non-uniform objective couples the gradient across several elements. (A plain
    # ``sum(mask)`` would be the constant budget k, hence zero gradient.)
    scores = _scores(25, seed=7)

    def objective(x: jnp.ndarray) -> jnp.ndarray:
        mask = top_k_mask(x, 8, softness=0.3, mode="smooth")
        return jnp.sum(mask * jnp.arange(25, dtype=x.dtype))

    grad = jax.grad(objective)(scores)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert int(jnp.sum(jnp.abs(grad) > 1e-6)) > 1


# --- Edge cases -----------------------------------------------------------------


def test_k_at_least_n_selects_all() -> None:
    scores = _scores(10, seed=8)
    np.testing.assert_array_equal(np.asarray(top_k_mask(scores, 10, mode="hard")), np.ones(10))
    np.testing.assert_array_equal(np.asarray(top_k_mask(scores, 25, mode="hard")), np.ones(10))


def test_k_zero_or_negative_selects_none() -> None:
    scores = _scores(10, seed=9)
    np.testing.assert_array_equal(np.asarray(top_k_mask(scores, 0, mode="hard")), np.zeros(10))
    np.testing.assert_array_equal(np.asarray(top_k_mask(scores, -3, mode="smooth")), np.zeros(10))


def test_k_equals_one_selects_the_max() -> None:
    scores = _scores(20, seed=10)
    mask = top_k_mask(scores, 1, mode="hard")
    np.testing.assert_allclose(float(jnp.sum(mask)), 1.0)
    assert int(np.argmax(np.asarray(mask))) == int(jnp.argmax(scores))


def test_arbitrary_axis() -> None:
    scores = jnp.asarray(np.random.default_rng(11).normal(size=(4, 12)).astype(np.float32))
    mask = top_k_mask(scores, 5, axis=1, mode="hard")
    assert mask.shape == (4, 12)
    np.testing.assert_allclose(np.asarray(jnp.sum(mask, axis=1)), np.full(4, 5.0))


# --- Transform compatibility ----------------------------------------------------


def test_jit_matches_eager() -> None:
    scores = _scores(30, seed=12)
    eager = top_k_mask(scores, 9, softness=0.2, mode="smooth")
    jitted = jax.jit(top_k_mask, static_argnums=(1,), static_argnames=("axis", "softness", "mode"))
    result = jitted(scores, 9, axis=-1, softness=0.2, mode="smooth")
    np.testing.assert_allclose(np.asarray(result), np.asarray(eager), atol=1e-6)


def test_vmap_over_batch() -> None:
    batch = jnp.asarray(np.random.default_rng(13).normal(size=(5, 20)).astype(np.float32))
    masks = jax.vmap(lambda row: top_k_mask(row, 6, mode="hard"))(batch)
    assert masks.shape == (5, 20)
    np.testing.assert_allclose(np.asarray(jnp.sum(masks, axis=1)), np.full(5, 6.0))


@settings(max_examples=30, deadline=None)
@given(n=st.integers(min_value=2, max_value=50), seed=st.integers(min_value=0, max_value=2**16))
def test_property_smooth_mask_bounded(n: int, seed: int) -> None:
    scores = _scores(n, seed)
    k = max(1, n // 2)
    mask = top_k_mask(scores, k, softness=0.3, mode="smooth")
    assert bool(jnp.all(jnp.isfinite(mask)))
    assert float(jnp.min(mask)) >= 0.0
    assert float(jnp.max(mask)) <= 1.0
