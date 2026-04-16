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
            use_gradnorm=False,
        )

        assert combined.shape == ()
        assert jnp.allclose(combined, 4.0)

    def test_rejects_empty_loss_mapping(self) -> None:
        """Empty loss mappings fail fast with a clear error."""
        with pytest.raises(ValueError, match="at least one"):
            loss_balancing.combine_scalar_losses({}, use_gradnorm=False)

    def test_uses_gradnorm_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GradNorm path delegates to the balancer with deterministic inputs."""
        calls: dict[str, Any] = {}

        class _DummyBalancer:
            def __init__(self, *, num_losses: int, rngs: nnx.Rngs) -> None:
                calls["num_losses"] = num_losses
                calls["rngs_type"] = type(rngs)

            def __call__(self, loss_values: list[jnp.ndarray]) -> jnp.ndarray:
                calls["loss_values"] = loss_values
                return jnp.array(7.0)

        monkeypatch.setattr(loss_balancing, "GradNormBalancer", _DummyBalancer)

        combined = loss_balancing.combine_scalar_losses(
            {
                "reconstruction": jnp.array(1.0),
                "regularizer": jnp.array(2.0),
            },
            use_gradnorm=True,
            rngs=nnx.Rngs(123),
        )

        assert jnp.allclose(combined, 7.0)
        assert calls["num_losses"] == 2
        assert calls["rngs_type"] is nnx.Rngs
        assert len(calls["loss_values"]) == 2


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
