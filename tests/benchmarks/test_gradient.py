"""Tests for benchmarks._gradient module.

Verifies gradient flow checking and the GradientFlowResult dataclass.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import warnings

import jax.numpy as jnp
import pytest
from flax import nnx

from benchmarks._gradient import GradientFlowResult, check_gradient_flow


class TestGradientFlowResult:
    """Tests for the GradientFlowResult frozen dataclass."""

    def test_fields_accessible(self) -> None:
        """Result exposes gradient_norm and gradient_nonzero."""
        result = GradientFlowResult(gradient_norm=1.5, gradient_nonzero=True)
        assert result.gradient_norm == 1.5
        assert result.gradient_nonzero is True

    def test_frozen(self) -> None:
        """Result is immutable (frozen dataclass)."""
        result = GradientFlowResult(gradient_norm=0.0, gradient_nonzero=False)
        with pytest.raises(FrozenInstanceError):
            result.gradient_norm = 42.0  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two results with identical fields are equal."""
        a = GradientFlowResult(gradient_norm=1.0, gradient_nonzero=True)
        b = GradientFlowResult(gradient_norm=1.0, gradient_nonzero=True)
        assert a == b


class TestCheckGradientFlow:
    """Tests for check_gradient_flow with simple NNX models."""

    @pytest.fixture()
    def linear_model(self) -> nnx.Linear:
        """Create a simple Linear model with known weights."""
        return nnx.Linear(in_features=4, out_features=2, rngs=nnx.Rngs(0))

    def test_returns_gradient_flow_result(self, linear_model: nnx.Linear) -> None:
        """Return type is GradientFlowResult."""
        x = jnp.ones((1, 4))

        def loss_fn(model: nnx.Module, inputs: jnp.ndarray) -> jnp.ndarray:
            return model(inputs).sum()

        result = check_gradient_flow(loss_fn, linear_model, x)
        assert isinstance(result, GradientFlowResult)

    def test_gradient_norm_positive(self, linear_model: nnx.Linear) -> None:
        """Gradient norm is positive for a differentiable model."""
        x = jnp.ones((1, 4))

        def loss_fn(model: nnx.Module, inputs: jnp.ndarray) -> jnp.ndarray:
            return model(inputs).sum()

        result = check_gradient_flow(loss_fn, linear_model, x)
        assert result.gradient_norm > 0.0

    def test_gradient_nonzero_true(self, linear_model: nnx.Linear) -> None:
        """gradient_nonzero is True for a differentiable model."""
        x = jnp.ones((1, 4))

        def loss_fn(model: nnx.Module, inputs: jnp.ndarray) -> jnp.ndarray:
            return model(inputs).sum()

        result = check_gradient_flow(loss_fn, linear_model, x)
        assert result.gradient_nonzero is True

    def test_zero_input_still_has_bias_gradient(self) -> None:
        """With zero input, bias gradient is nonzero."""
        model = nnx.Linear(
            in_features=3,
            out_features=2,
            use_bias=True,
            rngs=nnx.Rngs(0),
        )
        x = jnp.zeros((1, 3))

        def loss_fn(model: nnx.Module, inputs: jnp.ndarray) -> jnp.ndarray:
            return model(inputs).sum()

        result = check_gradient_flow(loss_fn, model, x)
        # Bias gradient should still be nonzero even with zero input
        assert result.gradient_norm > 0.0

    def test_multiple_args_forwarded(self, linear_model: nnx.Linear) -> None:
        """Extra positional args are forwarded to the loss function."""
        x = jnp.ones((1, 4))
        scale = jnp.array(2.0)

        def loss_fn(
            model: nnx.Module,
            inputs: jnp.ndarray,
            multiplier: jnp.ndarray,
        ) -> jnp.ndarray:
            return (model(inputs) * multiplier).sum()

        result = check_gradient_flow(loss_fn, linear_model, x, scale)
        assert result.gradient_norm > 0.0

    def test_no_nnx_value_deprecation_warning(self, linear_model: nnx.Linear) -> None:
        """Gradient checks should avoid deprecated .value access on NNX variables."""
        x = jnp.ones((1, 4))

        def loss_fn(model: nnx.Module, inputs: jnp.ndarray) -> jnp.ndarray:
            return model(inputs).sum()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            result = check_gradient_flow(loss_fn, linear_model, x)

        assert result.gradient_norm > 0.0
        assert not any(
            ".value access is now deprecated" in str(w.message) for w in caught
        )
