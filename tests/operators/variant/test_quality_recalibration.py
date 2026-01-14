"""Tests for diffbio.operators.variant.quality_recalibration module.

These tests define the expected behavior of the SoftVariantQualityFilter
operator for VQSR-style variant quality filtering with differentiable GMM.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.variant.quality_recalibration import (
    SoftVariantQualityFilter,
    VariantQualityFilterConfig,
)


class TestVariantQualityFilterConfig:
    """Tests for VariantQualityFilterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VariantQualityFilterConfig()
        assert config.n_components == 3
        assert config.n_features == 4  # depth, qual, strand_bias, mapq
        assert config.threshold == 0.5
        assert config.temperature == 1.0
        assert config.stochastic is False

    def test_custom_components(self):
        """Test custom number of GMM components."""
        config = VariantQualityFilterConfig(n_components=5)
        assert config.n_components == 5

    def test_custom_threshold(self):
        """Test custom filtering threshold."""
        config = VariantQualityFilterConfig(threshold=0.8)
        assert config.threshold == 0.8


class TestSoftVariantQualityFilter:
    """Tests for SoftVariantQualityFilter operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNGs for operator initialization."""
        return nnx.Rngs(42)

    @pytest.fixture
    def sample_variants(self):
        """Provide sample variant features."""
        # Features: depth, quality, strand_bias, mapping_quality
        key = jax.random.key(0)
        n_variants = 100
        features = jax.random.uniform(key, (n_variants, 4))
        return {"variant_features": features}

    def test_initialization(self, rngs):
        """Test operator initialization."""
        config = VariantQualityFilterConfig()
        op = SoftVariantQualityFilter(config, rngs=rngs)
        assert op is not None

    def test_output_contains_scores(self, rngs, sample_variants):
        """Test that output contains quality scores."""
        config = VariantQualityFilterConfig()
        op = SoftVariantQualityFilter(config, rngs=rngs)

        transformed, _, _ = op.apply(sample_variants, {}, None, None)

        assert "quality_scores" in transformed
        assert transformed["quality_scores"].shape == (100,)

    def test_scores_in_valid_range(self, rngs, sample_variants):
        """Test that quality scores are in [0, 1]."""
        config = VariantQualityFilterConfig()
        op = SoftVariantQualityFilter(config, rngs=rngs)

        transformed, _, _ = op.apply(sample_variants, {}, None, None)

        scores = transformed["quality_scores"]
        assert jnp.all(scores >= 0)
        assert jnp.all(scores <= 1)

    def test_output_contains_soft_filter(self, rngs, sample_variants):
        """Test that output contains soft filter weights."""
        config = VariantQualityFilterConfig()
        op = SoftVariantQualityFilter(config, rngs=rngs)

        transformed, _, _ = op.apply(sample_variants, {}, None, None)

        assert "filter_weights" in transformed
        assert transformed["filter_weights"].shape == (100,)

    def test_filter_weights_sigmoid_based(self, rngs, sample_variants):
        """Test that filter weights are sigmoid-based (0 to 1)."""
        config = VariantQualityFilterConfig()
        op = SoftVariantQualityFilter(config, rngs=rngs)

        transformed, _, _ = op.apply(sample_variants, {}, None, None)

        weights = transformed["filter_weights"]
        assert jnp.all(weights >= 0)
        assert jnp.all(weights <= 1)

    def test_component_responsibilities(self, rngs, sample_variants):
        """Test that GMM responsibilities are returned."""
        config = VariantQualityFilterConfig(n_components=3)
        op = SoftVariantQualityFilter(config, rngs=rngs)

        transformed, _, _ = op.apply(sample_variants, {}, None, None)

        assert "component_probs" in transformed
        assert transformed["component_probs"].shape == (100, 3)

    def test_responsibilities_sum_to_one(self, rngs, sample_variants):
        """Test that component responsibilities sum to 1."""
        config = VariantQualityFilterConfig(n_components=3)
        op = SoftVariantQualityFilter(config, rngs=rngs)

        transformed, _, _ = op.apply(sample_variants, {}, None, None)

        probs_sum = jnp.sum(transformed["component_probs"], axis=-1)
        assert jnp.allclose(probs_sum, 1.0, atol=1e-5)


class TestGradientFlow:
    """Tests for gradient flow through quality filter."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_gradient_flows_through_filter(self, rngs):
        """Test that gradients flow through filtering."""
        config = VariantQualityFilterConfig()
        op = SoftVariantQualityFilter(config, rngs=rngs)

        key = jax.random.key(0)
        features = jax.random.uniform(key, (50, 4))

        def loss_fn(feats):
            data = {"variant_features": feats}
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["quality_scores"].sum()

        grad = jax.grad(loss_fn)(features)
        assert grad is not None
        assert grad.shape == features.shape
        assert jnp.isfinite(grad).all()

    def test_gmm_parameters_learnable(self, rngs):
        """Test that GMM parameters are learnable."""
        config = VariantQualityFilterConfig()
        op = SoftVariantQualityFilter(config, rngs=rngs)

        key = jax.random.key(0)
        features = jax.random.uniform(key, (50, 4))
        data = {"variant_features": features}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["quality_scores"].sum()

        loss, grads = loss_fn(op)

        # Check GMM parameters have gradients
        assert hasattr(grads, "means")
        assert hasattr(grads, "log_variances")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_apply_is_jit_compatible(self, rngs):
        """Test that apply method works with JIT."""
        config = VariantQualityFilterConfig()
        op = SoftVariantQualityFilter(config, rngs=rngs)

        key = jax.random.key(0)
        features = jax.random.uniform(key, (50, 4))
        data = {"variant_features": features}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["quality_scores"]).all()


class TestTemperatureControl:
    """Tests for temperature control."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_high_temperature_more_uniform(self, rngs):
        """Test that high temperature gives more uniform weights."""
        key = jax.random.key(0)
        features = jax.random.uniform(key, (50, 4))
        data = {"variant_features": features}

        config_low = VariantQualityFilterConfig(temperature=0.1)
        op_low = SoftVariantQualityFilter(config_low, rngs=rngs)

        config_high = VariantQualityFilterConfig(temperature=10.0)
        op_high = SoftVariantQualityFilter(config_high, rngs=nnx.Rngs(42))

        trans_low, _, _ = op_low.apply(data, {}, None, None)
        trans_high, _, _ = op_high.apply(data, {}, None, None)

        # High temperature should give more uniform component probs
        var_low = jnp.var(trans_low["component_probs"])
        var_high = jnp.var(trans_high["component_probs"])
        assert var_high < var_low


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_single_variant(self, rngs):
        """Test with single variant."""
        config = VariantQualityFilterConfig()
        op = SoftVariantQualityFilter(config, rngs=rngs)

        key = jax.random.key(0)
        features = jax.random.uniform(key, (1, 4))
        data = {"variant_features": features}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["quality_scores"]).all()

    def test_many_variants(self, rngs):
        """Test with many variants."""
        config = VariantQualityFilterConfig()
        op = SoftVariantQualityFilter(config, rngs=rngs)

        key = jax.random.key(0)
        features = jax.random.uniform(key, (10000, 4))
        data = {"variant_features": features}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["quality_scores"]).all()

    def test_many_components(self, rngs):
        """Test with many GMM components."""
        config = VariantQualityFilterConfig(n_components=10)
        op = SoftVariantQualityFilter(config, rngs=rngs)

        key = jax.random.key(0)
        features = jax.random.uniform(key, (100, 4))
        data = {"variant_features": features}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["component_probs"].shape == (100, 10)
