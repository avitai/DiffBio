"""Tests for FNO-based peak calling operator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.epigenomics.fno_peak_calling import (
    FNOPeakCaller,
    FNOPeakCallerConfig,
)


@pytest.fixture()
def rngs() -> nnx.Rngs:
    """Shared RNG."""
    return nnx.Rngs(42)


@pytest.fixture()
def coverage_data() -> dict[str, jnp.ndarray]:
    """Synthetic coverage signal (batch=2, length=128)."""
    key = jax.random.key(0)
    coverage = jax.nn.relu(jax.random.normal(key, (2, 128))) * 10
    return {"coverage": coverage}


class TestFNOPeakCallerConfig:
    """Tests for FNO peak caller configuration."""

    def test_defaults(self) -> None:
        """Config has sensible defaults."""
        config = FNOPeakCallerConfig()
        assert config.hidden_channels == 32
        assert config.modes == 16
        assert config.num_layers == 4

    def test_frozen(self) -> None:
        """Config is immutable."""
        config = FNOPeakCallerConfig()
        with pytest.raises(AttributeError):
            config.modes = 8  # type: ignore[misc]


class TestFNOPeakCaller:
    """Tests for the FNO peak calling operator."""

    def test_output_keys(self, rngs: nnx.Rngs, coverage_data: dict) -> None:
        """Output contains expected keys."""
        config = FNOPeakCallerConfig(hidden_channels=16, modes=8, num_layers=2)
        op = FNOPeakCaller(config, rngs=rngs)

        result, state, meta = op.apply(coverage_data, {}, None)
        assert "peak_probabilities" in result
        assert "peak_scores" in result
        assert "coverage" in result

    def test_output_shape(self, rngs: nnx.Rngs, coverage_data: dict) -> None:
        """Peak probabilities have same length as input coverage."""
        config = FNOPeakCallerConfig(hidden_channels=16, modes=8, num_layers=2)
        op = FNOPeakCaller(config, rngs=rngs)

        result, _, _ = op.apply(coverage_data, {}, None)
        assert result["peak_probabilities"].shape == coverage_data["coverage"].shape

    def test_probabilities_in_unit_interval(self, rngs: nnx.Rngs, coverage_data: dict) -> None:
        """Peak probabilities are in [0, 1]."""
        config = FNOPeakCallerConfig(hidden_channels=16, modes=8, num_layers=2)
        op = FNOPeakCaller(config, rngs=rngs)

        result, _, _ = op.apply(coverage_data, {}, None)
        probs = result["peak_probabilities"]
        assert jnp.all(probs >= 0.0)
        assert jnp.all(probs <= 1.0)

    def test_differentiable(self, rngs: nnx.Rngs, coverage_data: dict) -> None:
        """Operator is differentiable w.r.t. input."""
        config = FNOPeakCallerConfig(hidden_channels=16, modes=8, num_layers=2)
        op = FNOPeakCaller(config, rngs=rngs)

        def loss_fn(cov: jnp.ndarray) -> jnp.ndarray:
            result, _, _ = op.apply({"coverage": cov}, {}, None)
            return jnp.sum(result["peak_probabilities"])

        grads = jax.grad(loss_fn)(coverage_data["coverage"])
        assert grads.shape == coverage_data["coverage"].shape
        assert jnp.all(jnp.isfinite(grads))

    def test_unbatched_input(self, rngs: nnx.Rngs) -> None:
        """Handles single (unbatched) coverage signal."""
        config = FNOPeakCallerConfig(hidden_channels=16, modes=8, num_layers=2)
        op = FNOPeakCaller(config, rngs=rngs)

        key = jax.random.key(1)
        coverage = jax.nn.relu(jax.random.normal(key, (128,))) * 10
        result, _, _ = op.apply({"coverage": coverage}, {}, None)
        assert result["peak_probabilities"].shape == (128,)

    def test_jit_compatible(self, rngs: nnx.Rngs, coverage_data: dict) -> None:
        """Operator works under jax.jit."""
        config = FNOPeakCallerConfig(hidden_channels=16, modes=8, num_layers=2)
        op = FNOPeakCaller(config, rngs=rngs)

        @nnx.jit
        def forward(operator: FNOPeakCaller, data: dict) -> dict:
            result, _, _ = operator.apply(data, {}, None)
            return result

        result = forward(op, coverage_data)
        assert "peak_probabilities" in result
        assert jnp.all(jnp.isfinite(result["peak_probabilities"]))
