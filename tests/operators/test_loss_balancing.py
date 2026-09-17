"""Tests for shared operator loss-balancing helpers."""

from typing import Any

import jax.numpy as jnp
import pytest
from flax import nnx

import diffbio.operators._loss_balancing as loss_balancing


class _DummyConfig:
    """Minimal config stub exposing the loss-balancing flag."""

    def __init__(self, *, use_gradnorm: bool) -> None:
        self.use_gradnorm = use_gradnorm


class _DummyOperator(loss_balancing.LossBalancingMixin):
    """Minimal operator stub used to exercise the mixin."""

    def __init__(self, *, use_gradnorm: bool) -> None:
        self.config = _DummyConfig(use_gradnorm=use_gradnorm)
        self.rngs = nnx.Rngs(123)


class TestCombineScalarLosses:
    """Tests for shared scalar-loss aggregation."""

    def test_sums_losses_when_gradnorm_disabled(self) -> None:
        """Losses are summed directly when GradNorm is disabled."""
        combined = loss_balancing.combine_scalar_losses(
            {
                "reconstruction": jnp.array(1.5),
                "kl": jnp.array(0.25),
                "auxiliary": jnp.array(2.25),
            },
            balancer=None,
        )

        assert combined.shape == ()
        assert jnp.allclose(combined, 4.0)

    def test_rejects_empty_loss_mapping(self) -> None:
        """Empty loss mappings fail fast with a clear error."""
        with pytest.raises(ValueError, match="at least one"):
            loss_balancing.combine_scalar_losses({}, balancer=None)

    def test_uses_the_balancer_when_given(self) -> None:
        """A balancer combines the losses through its ``compute_weighted_loss``."""
        calls: dict[str, Any] = {}

        class _DummyBalancer:
            def compute_weighted_loss(self, loss_values: jnp.ndarray) -> jnp.ndarray:
                calls["loss_values"] = loss_values
                return jnp.array(7.0)

        combined = loss_balancing.combine_scalar_losses(
            {
                "reconstruction": jnp.array(1.0),
                "regularizer": jnp.array(2.0),
            },
            balancer=_DummyBalancer(),  # type: ignore[arg-type]
        )

        assert jnp.allclose(combined, 7.0)
        assert calls["loss_values"].shape == (2,)

    def test_real_gradnorm_balancer_combines_without_error(self) -> None:
        """Integration: the real GradNormBalancer path runs (regression for the
        missing ``__call__`` that a mock had hidden). A fresh balancer has unit
        weights, so the result is the equal-weighted sum."""
        combined = loss_balancing.combine_scalar_losses(
            {"a": jnp.array(1.0), "b": jnp.array(2.0)},
            balancer=loss_balancing.GradNormBalancer(num_losses=2, rngs=nnx.Rngs(0)),
        )
        assert combined.shape == ()
        assert jnp.allclose(combined, 3.0)


class TestLossBalancingMixin:
    """Tests for the reusable operator mixin."""

    def test_mixin_uses_config_flag(self) -> None:
        """Mixin delegates to the shared helper using config.use_gradnorm."""
        operator = _DummyOperator(use_gradnorm=False)

        combined = operator.compute_balanced_loss(
            {
                "primary": jnp.array(3.0),
                "secondary": jnp.array(4.0),
            }
        )

        assert jnp.allclose(combined, 7.0)

    def test_mixin_preserves_fail_fast_validation(self) -> None:
        """Mixin surfaces the shared helper's empty-input validation."""
        operator = _DummyOperator(use_gradnorm=False)

        with pytest.raises(ValueError, match="at least one"):
            operator.compute_balanced_loss({})

    def test_mixin_builds_the_balancer_from_the_operator_rngs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the flag on, the mixin builds the balancer from the operator's own ``rngs``."""
        calls: dict[str, Any] = {}

        class _DummyBalancer:
            def __init__(self, *, num_losses: int, rngs: nnx.Rngs) -> None:
                calls["num_losses"] = num_losses
                calls["rngs"] = rngs

            def compute_weighted_loss(self, loss_values: jnp.ndarray) -> jnp.ndarray:
                return jnp.sum(loss_values)

        monkeypatch.setattr(loss_balancing, "GradNormBalancer", _DummyBalancer)
        operator = _DummyOperator(use_gradnorm=True)

        combined = operator.compute_balanced_loss({"a": jnp.array(1.0), "b": jnp.array(2.0)})

        assert jnp.allclose(combined, 3.0)
        assert calls["num_losses"] == 2
        assert calls["rngs"] is operator.rngs
