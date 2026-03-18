"""Tests for diffbio.operators.singlecell.switch_de module.

These tests define the expected behavior of the DifferentiableSwitchDE
operator for sigmoidal switch differential expression analysis.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.switch_de import (
    DifferentiableSwitchDE,
    SwitchDEConfig,
)

N_CELLS = 100
N_GENES = 50


@pytest.fixture
def default_config() -> SwitchDEConfig:
    """Provide default SwitchDE configuration."""
    return SwitchDEConfig(n_genes=N_GENES)


@pytest.fixture
def sample_data() -> dict[str, jax.Array]:
    """Provide sample single-cell data with pseudotime."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    counts = jax.random.uniform(k1, (N_CELLS, N_GENES), minval=0.0, maxval=10.0)
    pseudotime = jnp.linspace(0.0, 1.0, N_CELLS)
    return {"counts": counts, "pseudotime": pseudotime}


class TestSwitchDEConfig:
    """Tests for SwitchDEConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SwitchDEConfig()
        assert config.n_genes == 2000
        assert config.temperature == 1.0
        assert config.learnable_temperature is False
        assert config.stochastic is False

    def test_custom_n_genes(self) -> None:
        """Test custom number of genes."""
        config = SwitchDEConfig(n_genes=500)
        assert config.n_genes == 500


class TestDifferentiableSwitchDE:
    """Tests for DifferentiableSwitchDE operator."""

    def test_output_keys(
        self, rngs: nnx.Rngs, default_config: SwitchDEConfig, sample_data: dict
    ) -> None:
        """Test that apply returns expected output keys."""
        op = DifferentiableSwitchDE(default_config, rngs=rngs)
        result, state, metadata = op.apply(sample_data, {}, None)

        expected_keys = {
            "counts",
            "pseudotime",
            "switch_times",
            "switch_scores",
            "predicted_expression",
        }
        assert set(result.keys()) == expected_keys

    def test_output_shapes(
        self, rngs: nnx.Rngs, default_config: SwitchDEConfig, sample_data: dict
    ) -> None:
        """Test correct shapes for all outputs."""
        op = DifferentiableSwitchDE(default_config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)

        assert result["switch_times"].shape == (N_GENES,)
        assert result["switch_scores"].shape == (N_GENES,)
        assert result["predicted_expression"].shape == (N_CELLS, N_GENES)

    def test_recovers_step_function(self, rngs: nnx.Rngs) -> None:
        """With known step-function data, t_switch should be near the true switch point."""
        n_genes = 5
        n_cells = 200
        true_switch = 0.5

        config = SwitchDEConfig(n_genes=n_genes, temperature=0.01)
        op = DifferentiableSwitchDE(config, rngs=rngs)

        pseudotime = jnp.linspace(0.0, 1.0, n_cells)
        # Step function at t=0.5: 0 before, 3.0 after
        counts = jnp.where(
            pseudotime[:, None] > true_switch,
            3.0 * jnp.ones((n_cells, n_genes)),
            0.0 * jnp.ones((n_cells, n_genes)),
        )
        data = {"counts": counts, "pseudotime": pseudotime}

        # Manually set parameters to match the step function
        op.t_switch[...] = jnp.full((n_genes,), true_switch)
        op.amplitude[...] = jnp.full((n_genes,), 3.0)
        op.baseline[...] = jnp.zeros((n_genes,))

        result, _, _ = op.apply(data, {}, None)

        # With correctly set params and low temperature, predicted should
        # approximate a step function near t_switch=0.5
        predicted = result["predicted_expression"]
        # Check early cells (t < 0.3) have low expression
        early_mean = predicted[:50, :].mean()
        # Check late cells (t > 0.7) have high expression
        late_mean = predicted[150:, :].mean()
        assert late_mean > early_mean + 1.0

    def test_monotonic_sigmoid(self, rngs: nnx.Rngs, default_config: SwitchDEConfig) -> None:
        """Predicted expression should be monotonic along pseudotime for positive amplitude."""
        op = DifferentiableSwitchDE(default_config, rngs=rngs)

        # Ensure positive amplitude
        op.amplitude[...] = jnp.ones((N_GENES,)) * 2.0

        n_cells = 200
        pseudotime = jnp.linspace(0.0, 1.0, n_cells)
        key = jax.random.key(10)
        counts = jax.random.uniform(key, (n_cells, N_GENES))
        data = {"counts": counts, "pseudotime": pseudotime}

        result, _, _ = op.apply(data, {}, None)
        predicted = result["predicted_expression"]

        # Check monotonicity: each later cell should have >= expression
        diffs = jnp.diff(predicted, axis=0)
        assert jnp.all(diffs >= -1e-5)


class TestGradientFlow:
    """Tests for gradient flow through the switch DE operator."""

    def test_gradient_wrt_operator_params(
        self, rngs: nnx.Rngs, default_config: SwitchDEConfig, sample_data: dict
    ) -> None:
        """nnx.grad on loss produces non-zero gradients for t_switch, amplitude, baseline."""
        op = DifferentiableSwitchDE(default_config, rngs=rngs)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableSwitchDE) -> jax.Array:
            result, _, _ = model.apply(sample_data, {}, None)
            return jnp.mean(result["predicted_expression"])

        _, grads = loss_fn(op)

        assert hasattr(grads, "t_switch")
        assert hasattr(grads, "amplitude")
        assert hasattr(grads, "baseline")
        assert jnp.any(grads.t_switch[...] != 0.0)
        assert jnp.any(grads.amplitude[...] != 0.0)
        assert jnp.any(grads.baseline[...] != 0.0)

    def test_gradient_wrt_pseudotime(self, rngs: nnx.Rngs, default_config: SwitchDEConfig) -> None:
        """jax.grad on sum of predicted_expression wrt pseudotime is non-zero."""
        op = DifferentiableSwitchDE(default_config, rngs=rngs)

        counts = jnp.ones((N_CELLS, N_GENES))
        pseudotime = jnp.linspace(0.0, 1.0, N_CELLS)

        def loss_fn(pt: jax.Array) -> jax.Array:
            data = {"counts": counts, "pseudotime": pt}
            result, _, _ = op.apply(data, {}, None)
            return jnp.sum(result["predicted_expression"])

        grad = jax.grad(loss_fn)(pseudotime)
        assert grad.shape == pseudotime.shape
        assert jnp.any(grad != 0.0)
        assert jnp.all(jnp.isfinite(grad))


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_jit_apply(
        self, rngs: nnx.Rngs, default_config: SwitchDEConfig, sample_data: dict
    ) -> None:
        """jax.jit(operator.apply) compiles and runs."""
        op = DifferentiableSwitchDE(default_config, rngs=rngs)

        @jax.jit
        def jit_apply(data: dict, state: dict) -> tuple[dict, dict, dict[str, Any] | None]:
            return op.apply(data, state, None)

        result, _, _ = jit_apply(sample_data, {})
        assert jnp.all(jnp.isfinite(result["predicted_expression"]))
        assert jnp.all(jnp.isfinite(result["switch_scores"]))

    def test_jit_gradient(
        self, rngs: nnx.Rngs, default_config: SwitchDEConfig, sample_data: dict
    ) -> None:
        """jax.jit + jax.grad works."""
        op = DifferentiableSwitchDE(default_config, rngs=rngs)

        @jax.jit
        @nnx.value_and_grad
        def jit_loss(model: DifferentiableSwitchDE) -> jax.Array:
            result, _, _ = model.apply(sample_data, {}, None)
            return jnp.mean(result["predicted_expression"])

        loss, grads = jit_loss(op)
        assert jnp.isfinite(loss)
        assert hasattr(grads, "t_switch")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_cell(self, rngs: nnx.Rngs) -> None:
        """Works with n_cells=1."""
        config = SwitchDEConfig(n_genes=10)
        op = DifferentiableSwitchDE(config, rngs=rngs)

        data = {
            "counts": jnp.ones((1, 10)),
            "pseudotime": jnp.array([0.5]),
        }
        result, _, _ = op.apply(data, {}, None)
        assert result["predicted_expression"].shape == (1, 10)
        assert jnp.all(jnp.isfinite(result["predicted_expression"]))

    def test_zero_temperature_sharp(self, rngs: nnx.Rngs) -> None:
        """Very low temp gives near-step-function output."""
        n_genes = 5
        n_cells = 100
        config = SwitchDEConfig(n_genes=n_genes, temperature=1e-4)
        op = DifferentiableSwitchDE(config, rngs=rngs)

        # Set switch at 0.5 with amplitude 1.0
        op.t_switch[...] = jnp.full((n_genes,), 0.5)
        op.amplitude[...] = jnp.ones((n_genes,))
        op.baseline[...] = jnp.zeros((n_genes,))

        pseudotime = jnp.linspace(0.0, 1.0, n_cells)
        data = {
            "counts": jnp.zeros((n_cells, n_genes)),
            "pseudotime": pseudotime,
        }
        result, _, _ = op.apply(data, {}, None)
        predicted = result["predicted_expression"]

        # Before switch: should be near 0 (baseline)
        before_switch = predicted[:40, :]
        assert jnp.allclose(before_switch, 0.0, atol=0.05)

        # After switch: should be near 1 (baseline + amplitude)
        after_switch = predicted[60:, :]
        assert jnp.allclose(after_switch, 1.0, atol=0.05)
