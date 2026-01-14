"""Tests for diffbio.operators.quality_filter module.

These tests define the expected behavior of the DifferentiableQualityFilter
operator. Implementation should be written to pass these tests.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.quality_filter import (
    DifferentiableQualityFilter,
    QualityFilterConfig,
)
from diffbio.sequences.dna import encode_dna_string


class TestQualityFilterConfig:
    """Tests for QualityFilterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QualityFilterConfig()
        assert config.initial_threshold == 20.0
        assert config.stochastic is False

    def test_custom_threshold(self):
        """Test custom threshold configuration."""
        config = QualityFilterConfig(initial_threshold=30.0)
        assert config.initial_threshold == 30.0


class TestDifferentiableQualityFilter:
    """Tests for DifferentiableQualityFilter operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNGs for operator initialization."""
        return nnx.Rngs(42)

    @pytest.fixture
    def sample_data(self):
        """Provide sample DNA sequence data."""
        sequence = encode_dna_string("ACGTACGT")
        quality = jnp.array([30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0, 40.0])
        return {"sequence": sequence, "quality_scores": quality}

    @pytest.fixture
    def sample_state(self):
        """Provide sample element state."""
        return {}

    def test_initialization(self, rngs):
        """Test operator initialization."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)
        assert op is not None
        assert float(op.threshold[...]) == 20.0

    def test_initialization_custom_threshold(self, rngs):
        """Test initialization with custom threshold."""
        config = QualityFilterConfig(initial_threshold=30.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)
        assert float(op.threshold[...]) == 30.0

    def test_apply_high_quality_passes(self, rngs, sample_state):
        """Test that high quality positions pass through with high weight."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        # Create data with all high quality
        sequence = encode_dna_string("ACGT")
        quality = jnp.array([40.0, 40.0, 40.0, 40.0])  # All Q40
        data = {"sequence": sequence, "quality_scores": quality}

        transformed_data, new_state, new_metadata = op.apply(data, sample_state, None, None)

        # High quality should result in weights close to 1
        # Weighted sequence should be close to original
        assert "sequence" in transformed_data
        # Check that high quality positions retain most of their value
        original_sum = jnp.sum(sequence)
        transformed_sum = jnp.sum(transformed_data["sequence"])
        assert transformed_sum > 0.9 * original_sum

    def test_apply_low_quality_filtered(self, rngs, sample_state):
        """Test that low quality positions are down-weighted."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        # Create data with all low quality
        sequence = encode_dna_string("ACGT")
        quality = jnp.array([0.0, 0.0, 0.0, 0.0])  # All Q0
        data = {"sequence": sequence, "quality_scores": quality}

        transformed_data, new_state, new_metadata = op.apply(data, sample_state, None, None)

        # Low quality should result in weights close to 0
        # Weighted sequence should be significantly reduced
        original_sum = jnp.sum(sequence)
        transformed_sum = jnp.sum(transformed_data["sequence"])
        assert transformed_sum < 0.5 * original_sum

    def test_apply_mixed_quality(self, rngs, sample_data, sample_state):
        """Test mixed quality filtering."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        transformed_data, new_state, new_metadata = op.apply(sample_data, sample_state, None, None)

        # Verify output structure
        assert "sequence" in transformed_data
        assert transformed_data["sequence"].shape == sample_data["sequence"].shape

    def test_apply_preserves_quality_scores(self, rngs, sample_data, sample_state):
        """Test that quality scores are preserved in output."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, sample_state, None, None)

        # Quality scores should be preserved
        assert "quality_scores" in transformed_data
        assert jnp.allclose(transformed_data["quality_scores"], sample_data["quality_scores"])

    def test_threshold_is_learnable(self, rngs):
        """Test that threshold parameter is learnable (has gradients)."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        sequence = encode_dna_string("ACGT")
        quality = jnp.array([25.0, 15.0, 25.0, 15.0])
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        # Use NNX's value_and_grad pattern for module gradient computation
        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["sequence"])

        loss, grads = loss_fn(op)

        # Check that gradients exist for the threshold parameter
        assert hasattr(grads, "threshold")
        grad_value = grads.threshold[...]
        assert grad_value is not None
        # Gradient should be non-zero for mixed quality data
        assert jnp.abs(grad_value) > 0

    def test_output_shape_preserved(self, rngs, sample_data, sample_state):
        """Test that output shape matches input shape."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, sample_state, None, None)

        assert transformed_data["sequence"].shape == sample_data["sequence"].shape


class TestGradientFlow:
    """Tests for gradient flow through quality filter."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_gradient_flows_through_apply(self, rngs):
        """Test that gradients flow through the apply method."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        sequence = encode_dna_string("ACGT")
        quality = jnp.array([25.0, 15.0, 25.0, 15.0])
        state = {}

        def loss_fn(seq):
            data = {"sequence": seq, "quality_scores": quality}
            transformed, _, _ = op.apply(data, state, None, None)
            return jnp.sum(transformed["sequence"])

        grad = jax.grad(loss_fn)(sequence)
        assert grad is not None
        assert grad.shape == sequence.shape

    def test_gradient_wrt_quality(self, rngs):
        """Test that gradients flow with respect to quality scores."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        sequence = encode_dna_string("ACGT")
        quality = jnp.array([25.0, 15.0, 25.0, 15.0])
        state = {}

        def loss_fn(q):
            data = {"sequence": sequence, "quality_scores": q}
            transformed, _, _ = op.apply(data, state, None, None)
            return jnp.sum(transformed["sequence"])

        grad = jax.grad(loss_fn)(quality)
        assert grad is not None
        assert grad.shape == quality.shape
        # Higher quality should increase output, so gradient should be positive
        assert jnp.all(grad >= 0)


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_apply_is_jit_compatible(self, rngs):
        """Test that apply method works with JIT."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        sequence = encode_dna_string("ACGT")
        quality = jnp.array([30.0, 20.0, 30.0, 20.0])
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        # JIT compile the apply method
        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, new_state, metadata = jit_apply(data, state)
        assert transformed["sequence"].shape == data["sequence"].shape

    def test_jit_produces_same_result(self, rngs):
        """Test that JIT produces same result as eager execution."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        sequence = encode_dna_string("ACGT")
        quality = jnp.array([30.0, 20.0, 30.0, 20.0])
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        # Eager execution
        eager_result, _, _ = op.apply(data, state, None, None)

        # JIT execution
        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        jit_result, _, _ = jit_apply(data, state)

        assert jnp.allclose(eager_result["sequence"], jit_result["sequence"])


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_single_position(self, rngs):
        """Test with single position sequence."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        sequence = encode_dna_string("A")
        quality = jnp.array([30.0])
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        transformed, _, _ = op.apply(data, state, None, None)
        assert transformed["sequence"].shape == (1, 4)

    def test_very_high_threshold(self, rngs):
        """Test with threshold higher than all quality scores."""
        config = QualityFilterConfig(initial_threshold=50.0)  # Very high
        op = DifferentiableQualityFilter(config, rngs=rngs)

        sequence = encode_dna_string("ACGT")
        quality = jnp.array([40.0, 40.0, 40.0, 40.0])  # Below threshold
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        transformed, _, _ = op.apply(data, state, None, None)
        # All positions below threshold should be heavily down-weighted
        original_sum = jnp.sum(sequence)
        transformed_sum = jnp.sum(transformed["sequence"])
        assert transformed_sum < 0.5 * original_sum

    def test_very_low_threshold(self, rngs):
        """Test with threshold lower than all quality scores."""
        config = QualityFilterConfig(initial_threshold=0.0)  # Very low
        op = DifferentiableQualityFilter(config, rngs=rngs)

        sequence = encode_dna_string("ACGT")
        quality = jnp.array([10.0, 10.0, 10.0, 10.0])  # Above threshold
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        transformed, _, _ = op.apply(data, state, None, None)
        # All positions above threshold should pass through
        original_sum = jnp.sum(sequence)
        transformed_sum = jnp.sum(transformed["sequence"])
        assert transformed_sum > 0.9 * original_sum

    def test_exact_threshold(self, rngs):
        """Test with quality scores exactly at threshold."""
        config = QualityFilterConfig(initial_threshold=20.0)
        op = DifferentiableQualityFilter(config, rngs=rngs)

        sequence = encode_dna_string("ACGT")
        quality = jnp.array([20.0, 20.0, 20.0, 20.0])  # Exactly at threshold
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        transformed, _, _ = op.apply(data, state, None, None)
        # At threshold, sigmoid(0) = 0.5, so should be around 50%
        original_sum = jnp.sum(sequence)
        transformed_sum = jnp.sum(transformed["sequence"])
        assert 0.4 * original_sum < transformed_sum < 0.6 * original_sum
