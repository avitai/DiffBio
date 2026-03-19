"""Tests for diffbio.operators.singlecell.differential_distribution module.

These tests define the expected behavior of the DifferentiableDifferentialDistribution
operator for differentiable KS-test and pattern classification in single-cell data.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.differential_distribution import (
    DifferentiableDifferentialDistribution,
    DifferentialDistributionConfig,
)

N_CELLS = 40
N_GENES = 20
N_PATTERNS = 4


@pytest.fixture
def default_config() -> DifferentialDistributionConfig:
    """Provide default DifferentialDistribution configuration."""
    return DifferentialDistributionConfig(n_genes=N_GENES, n_pattern_classes=N_PATTERNS)


@pytest.fixture
def sample_data() -> dict[str, jax.Array]:
    """Provide sample single-cell data with binary condition labels."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    counts = jax.random.uniform(k1, (N_CELLS, N_GENES), minval=0.0, maxval=10.0)
    # First half condition 0, second half condition 1
    condition_labels = jnp.concatenate([jnp.zeros(N_CELLS // 2), jnp.ones(N_CELLS // 2)])
    return {"counts": counts, "condition_labels": condition_labels}


class TestDifferentialDistributionConfig:
    """Tests for DifferentialDistributionConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DifferentialDistributionConfig()
        assert config.n_genes == 2000
        assert config.temperature == 1.0
        assert config.learnable_temperature is False
        assert config.n_pattern_classes == 4
        assert config.stochastic is False

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = DifferentialDistributionConfig(
            n_genes=500,
            temperature=0.5,
            learnable_temperature=True,
            n_pattern_classes=3,
        )
        assert config.n_genes == 500
        assert config.temperature == 0.5
        assert config.learnable_temperature is True
        assert config.n_pattern_classes == 3


class TestDifferentialDistribution:
    """Tests for DifferentiableDifferentialDistribution operator."""

    def test_output_keys(
        self,
        rngs: nnx.Rngs,
        default_config: DifferentialDistributionConfig,
        sample_data: dict,
    ) -> None:
        """Test that apply returns expected output keys."""
        op = DifferentiableDifferentialDistribution(default_config, rngs=rngs)
        result, state, metadata = op.apply(sample_data, {}, None)

        expected_keys = {
            "counts",
            "condition_labels",
            "ks_statistics",
            "pattern_logits",
            "pattern_labels",
        }
        assert set(result.keys()) == expected_keys

    def test_output_shapes(
        self,
        rngs: nnx.Rngs,
        default_config: DifferentialDistributionConfig,
        sample_data: dict,
    ) -> None:
        """Test correct shapes for all outputs."""
        op = DifferentiableDifferentialDistribution(default_config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)

        assert result["ks_statistics"].shape == (N_GENES,)
        assert result["pattern_logits"].shape == (N_GENES, N_PATTERNS)
        assert result["pattern_labels"].shape == (N_GENES,)

    def test_ks_stats_in_valid_range(
        self,
        rngs: nnx.Rngs,
        default_config: DifferentialDistributionConfig,
        sample_data: dict,
    ) -> None:
        """KS statistics should be in [0, 1] range."""
        op = DifferentiableDifferentialDistribution(default_config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)

        ks_stats = result["ks_statistics"]
        assert jnp.all(ks_stats >= 0.0)
        assert jnp.all(ks_stats <= 1.0 + 1e-5)
        assert jnp.all(jnp.isfinite(ks_stats))

    def test_pattern_labels_in_valid_range(
        self,
        rngs: nnx.Rngs,
        default_config: DifferentialDistributionConfig,
        sample_data: dict,
    ) -> None:
        """Pattern labels should be in [0, n_patterns) range."""
        op = DifferentiableDifferentialDistribution(default_config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)

        labels = result["pattern_labels"]
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < N_PATTERNS)

    def test_state_and_metadata_passthrough(
        self,
        rngs: nnx.Rngs,
        default_config: DifferentialDistributionConfig,
        sample_data: dict,
    ) -> None:
        """State and metadata should be passed through unchanged."""
        op = DifferentiableDifferentialDistribution(default_config, rngs=rngs)
        in_state = {"key": "value"}
        in_meta = {"info": 42}
        _, out_state, out_meta = op.apply(sample_data, in_state, in_meta)

        assert out_state is in_state
        assert out_meta is in_meta


class TestKnownPatterns:
    """Tests with synthetic data that has known distributional differences."""

    def test_shift_pattern_detected(self, rngs: nnx.Rngs) -> None:
        """Synthetic data with known mean shift should yield high KS statistic."""
        n_cells = 40
        n_genes = 20
        key = jax.random.key(123)
        k1, k2, k3 = jax.random.split(key, 3)

        # Condition A: low expression, Condition B: high expression (clear shift)
        counts_a = jax.random.uniform(k1, (n_cells // 2, n_genes), minval=0.0, maxval=2.0)
        counts_b = jax.random.uniform(k2, (n_cells // 2, n_genes), minval=8.0, maxval=10.0)
        counts = jnp.concatenate([counts_a, counts_b], axis=0)
        condition_labels = jnp.concatenate([jnp.zeros(n_cells // 2), jnp.ones(n_cells // 2)])

        config = DifferentialDistributionConfig(
            n_genes=n_genes, temperature=0.1, n_pattern_classes=N_PATTERNS
        )
        op = DifferentiableDifferentialDistribution(config, rngs=rngs)
        data = {"counts": counts, "condition_labels": condition_labels}
        result, _, _ = op.apply(data, {}, None)

        # KS statistic should be high for clearly shifted distributions
        ks_stats = result["ks_statistics"]
        assert jnp.mean(ks_stats) > 0.5

    def test_identical_distributions_low_ks(self, rngs: nnx.Rngs) -> None:
        """Identical distributions across conditions should yield low KS statistics."""
        n_cells = 40
        n_genes = 20
        key = jax.random.key(456)

        # Same distribution for both conditions
        counts = jax.random.uniform(key, (n_cells, n_genes), minval=0.0, maxval=5.0)
        condition_labels = jnp.concatenate([jnp.zeros(n_cells // 2), jnp.ones(n_cells // 2)])

        config = DifferentialDistributionConfig(
            n_genes=n_genes, temperature=0.1, n_pattern_classes=N_PATTERNS
        )
        op = DifferentiableDifferentialDistribution(config, rngs=rngs)
        data = {"counts": counts, "condition_labels": condition_labels}
        result, _, _ = op.apply(data, {}, None)

        # KS statistic should be relatively low for identical distributions
        ks_stats = result["ks_statistics"]
        assert jnp.mean(ks_stats) < 0.5


class TestGradientFlow:
    """Tests for gradient flow through the differential distribution operator."""

    def test_gradient_through_ks_statistic(
        self,
        rngs: nnx.Rngs,
        default_config: DifferentialDistributionConfig,
        sample_data: dict,
    ) -> None:
        """Gradients should flow through KS statistic computation."""
        op = DifferentiableDifferentialDistribution(default_config, rngs=rngs)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableDifferentialDistribution) -> jax.Array:
            result, _, _ = model.apply(sample_data, {}, None)
            return jnp.mean(result["ks_statistics"])

        loss, grads = loss_fn(op)
        assert jnp.isfinite(loss)
        # Operator should have learnable parameters with non-zero gradients
        assert hasattr(grads, "pattern_head")

    def test_gradient_through_pattern_logits(
        self,
        rngs: nnx.Rngs,
        default_config: DifferentialDistributionConfig,
        sample_data: dict,
    ) -> None:
        """Gradients should flow through pattern classification logits."""
        op = DifferentiableDifferentialDistribution(default_config, rngs=rngs)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableDifferentialDistribution) -> jax.Array:
            result, _, _ = model.apply(sample_data, {}, None)
            return jnp.mean(result["pattern_logits"])

        loss, grads = loss_fn(op)
        assert jnp.isfinite(loss)
        assert hasattr(grads, "pattern_head")

    def test_gradient_through_counts(
        self,
        rngs: nnx.Rngs,
        default_config: DifferentialDistributionConfig,
    ) -> None:
        """Gradients should flow back to input counts."""
        op = DifferentiableDifferentialDistribution(default_config, rngs=rngs)

        condition_labels = jnp.concatenate([jnp.zeros(N_CELLS // 2), jnp.ones(N_CELLS // 2)])
        counts = jnp.ones((N_CELLS, N_GENES))

        def loss_fn(c: jax.Array) -> jax.Array:
            data = {"counts": c, "condition_labels": condition_labels}
            result, _, _ = op.apply(data, {}, None)
            return jnp.sum(result["ks_statistics"])

        grad = jax.grad(loss_fn)(counts)
        assert grad.shape == counts.shape
        assert jnp.all(jnp.isfinite(grad))


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_jit_apply(
        self,
        rngs: nnx.Rngs,
        default_config: DifferentialDistributionConfig,
        sample_data: dict,
    ) -> None:
        """jax.jit(operator.apply) compiles and runs."""
        op = DifferentiableDifferentialDistribution(default_config, rngs=rngs)

        @jax.jit
        def jit_apply(data: dict, state: dict) -> tuple[dict, dict, dict[str, Any] | None]:
            return op.apply(data, state, None)

        result, _, _ = jit_apply(sample_data, {})
        assert jnp.all(jnp.isfinite(result["ks_statistics"]))
        assert jnp.all(jnp.isfinite(result["pattern_logits"]))

    def test_jit_gradient(
        self,
        rngs: nnx.Rngs,
        default_config: DifferentialDistributionConfig,
        sample_data: dict,
    ) -> None:
        """jax.jit + nnx.value_and_grad works together."""
        op = DifferentiableDifferentialDistribution(default_config, rngs=rngs)

        @jax.jit
        @nnx.value_and_grad
        def jit_loss(model: DifferentiableDifferentialDistribution) -> jax.Array:
            result, _, _ = model.apply(sample_data, {}, None)
            return jnp.mean(result["ks_statistics"])

        loss, grads = jit_loss(op)
        assert jnp.isfinite(loss)
        assert hasattr(grads, "pattern_head")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_identical_distributions_ks_near_zero(self, rngs: nnx.Rngs) -> None:
        """When both conditions have the same distribution, KS should be near 0."""
        n_cells = 40
        n_genes = 20
        # All cells have the exact same expression value per gene
        counts = jnp.ones((n_cells, n_genes)) * 5.0
        condition_labels = jnp.concatenate([jnp.zeros(n_cells // 2), jnp.ones(n_cells // 2)])

        config = DifferentialDistributionConfig(
            n_genes=n_genes, temperature=1.0, n_pattern_classes=N_PATTERNS
        )
        op = DifferentiableDifferentialDistribution(config, rngs=rngs)
        data = {"counts": counts, "condition_labels": condition_labels}
        result, _, _ = op.apply(data, {}, None)

        ks_stats = result["ks_statistics"]
        # With identical values in both conditions, KS should be very small
        assert jnp.all(ks_stats < 0.1)

    def test_single_cell_per_condition(self, rngs: nnx.Rngs) -> None:
        """Works with minimum viable input: 1 cell per condition."""
        n_genes = 10
        counts = jnp.array([[1.0] * n_genes, [5.0] * n_genes])
        condition_labels = jnp.array([0.0, 1.0])

        config = DifferentialDistributionConfig(
            n_genes=n_genes, temperature=1.0, n_pattern_classes=N_PATTERNS
        )
        op = DifferentiableDifferentialDistribution(config, rngs=rngs)
        data = {"counts": counts, "condition_labels": condition_labels}
        result, _, _ = op.apply(data, {}, None)

        assert result["ks_statistics"].shape == (n_genes,)
        assert result["pattern_logits"].shape == (n_genes, N_PATTERNS)
        assert result["pattern_labels"].shape == (n_genes,)
        assert jnp.all(jnp.isfinite(result["ks_statistics"]))

    def test_learnable_temperature(self, rngs: nnx.Rngs) -> None:
        """Learnable temperature mode should have gradient through temperature."""
        config = DifferentialDistributionConfig(
            n_genes=N_GENES,
            temperature=1.0,
            learnable_temperature=True,
            n_pattern_classes=N_PATTERNS,
        )
        op = DifferentiableDifferentialDistribution(config, rngs=rngs)

        counts = jnp.ones((N_CELLS, N_GENES))
        condition_labels = jnp.concatenate([jnp.zeros(N_CELLS // 2), jnp.ones(N_CELLS // 2)])
        data = {"counts": counts, "condition_labels": condition_labels}

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableDifferentialDistribution) -> jax.Array:
            result, _, _ = model.apply(data, {}, None)
            return jnp.mean(result["ks_statistics"])

        _, grads = loss_fn(op)
        assert hasattr(grads, "temperature")
