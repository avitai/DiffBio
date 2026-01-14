"""Tests for diffbio.operators.variant.cnv_segmentation module.

These tests define the expected behavior of the DifferentiableCNVSegmentation
operator for soft changepoint detection in copy number analysis.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.variant.cnv_segmentation import (
    DifferentiableCNVSegmentation,
    CNVSegmentationConfig,
)


class TestCNVSegmentationConfig:
    """Tests for CNVSegmentationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CNVSegmentationConfig()
        assert config.max_segments == 100
        assert config.hidden_dim == 64
        assert config.attention_heads == 4
        assert config.temperature == 1.0
        assert config.stochastic is False

    def test_custom_max_segments(self):
        """Test custom max segments."""
        config = CNVSegmentationConfig(max_segments=50)
        assert config.max_segments == 50

    def test_custom_hidden_dim(self):
        """Test custom hidden dimension."""
        config = CNVSegmentationConfig(hidden_dim=128)
        assert config.hidden_dim == 128


class TestDifferentiableCNVSegmentation:
    """Tests for DifferentiableCNVSegmentation operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNGs for operator initialization."""
        return nnx.Rngs(42)

    @pytest.fixture
    def sample_coverage(self):
        """Provide sample coverage signal."""
        # Coverage signal along genome
        key = jax.random.key(0)
        n_positions = 1000
        # Simulate coverage with some copy number changes
        base_coverage = jnp.ones(n_positions)
        # Add a deletion region
        base_coverage = base_coverage.at[300:500].set(0.5)
        # Add a duplication region
        base_coverage = base_coverage.at[700:800].set(2.0)
        # Add noise
        noise = jax.random.normal(key, (n_positions,)) * 0.1
        coverage = base_coverage + noise
        return {"coverage": coverage}

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return CNVSegmentationConfig(
            max_segments=20,
            hidden_dim=32,
            attention_heads=2,
        )

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)
        assert op is not None

    def test_output_contains_segment_means(self, rngs, small_config, sample_coverage):
        """Test that output contains segment mean values."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        assert "segment_means" in transformed

    def test_output_contains_boundaries(self, rngs, small_config, sample_coverage):
        """Test that output contains soft segment boundaries."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        assert "boundary_probs" in transformed
        # Boundary probs should be per position
        assert transformed["boundary_probs"].shape[0] == 1000

    def test_boundary_probs_valid(self, rngs, small_config, sample_coverage):
        """Test that boundary probabilities are in [0, 1]."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        probs = transformed["boundary_probs"]
        assert jnp.all(probs >= 0)
        assert jnp.all(probs <= 1)

    def test_output_contains_segment_assignments(self, rngs, small_config, sample_coverage):
        """Test that output contains soft segment assignments."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        assert "segment_assignments" in transformed
        # Shape: (n_positions, max_segments)
        assert transformed["segment_assignments"].shape == (1000, small_config.max_segments)

    def test_segment_assignments_sum_to_one(self, rngs, small_config, sample_coverage):
        """Test that segment assignments sum to 1 per position."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        assignments_sum = jnp.sum(transformed["segment_assignments"], axis=-1)
        assert jnp.allclose(assignments_sum, 1.0, atol=1e-5)

    def test_smoothed_signal(self, rngs, small_config, sample_coverage):
        """Test that smoothed/segmented signal is returned."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_coverage, {}, None, None)

        assert "smoothed_coverage" in transformed
        assert transformed["smoothed_coverage"].shape == (1000,)


class TestGradientFlow:
    """Tests for gradient flow through CNV segmentation."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def small_config(self):
        return CNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
        )

    def test_gradient_flows_through_segmentation(self, rngs, small_config):
        """Test that gradients flow through segmentation."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (500,))

        def loss_fn(cov):
            data = {"coverage": cov}
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["smoothed_coverage"].sum()

        grad = jax.grad(loss_fn)(coverage)
        assert grad is not None
        assert grad.shape == coverage.shape
        assert jnp.isfinite(grad).all()

    def test_attention_parameters_learnable(self, rngs, small_config):
        """Test that attention parameters are learnable."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (500,))
        data = {"coverage": coverage}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["smoothed_coverage"].sum()

        loss, grads = loss_fn(op)

        # Check attention layers have gradients
        assert hasattr(grads, "query_proj")
        assert hasattr(grads, "key_proj")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    @pytest.fixture
    def small_config(self):
        return CNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = DifferentiableCNVSegmentation(small_config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (500,))
        data = {"coverage": coverage}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_short_signal(self, rngs):
        """Test with short coverage signal."""
        config = CNVSegmentationConfig(
            max_segments=5,
            hidden_dim=16,
            attention_heads=2,
        )
        op = DifferentiableCNVSegmentation(config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (50,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()

    def test_long_signal(self, rngs):
        """Test with long coverage signal."""
        config = CNVSegmentationConfig(
            max_segments=50,
            hidden_dim=32,
            attention_heads=2,
        )
        op = DifferentiableCNVSegmentation(config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (5000,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()

    def test_uniform_signal(self, rngs):
        """Test with uniform coverage (no CNVs)."""
        config = CNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
        )
        op = DifferentiableCNVSegmentation(config, rngs=rngs)

        # Uniform signal should have minimal boundaries
        coverage = jnp.ones(500)
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()

    def test_high_temperature(self, rngs):
        """Test with high temperature."""
        config = CNVSegmentationConfig(
            max_segments=10,
            hidden_dim=16,
            attention_heads=2,
            temperature=10.0,
        )
        op = DifferentiableCNVSegmentation(config, rngs=rngs)

        key = jax.random.key(0)
        coverage = jax.random.uniform(key, (500,))
        data = {"coverage": coverage}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["smoothed_coverage"]).all()
