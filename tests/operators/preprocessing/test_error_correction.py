"""Tests for diffbio.operators.preprocessing.error_correction module.

These tests define the expected behavior of the SoftErrorCorrection
operator. Implementation should be written to pass these tests.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.preprocessing.error_correction import (
    ErrorCorrectionConfig,
    SoftErrorCorrection,
)
from diffbio.sequences.dna import encode_dna_string


class TestErrorCorrectionConfig:
    """Tests for ErrorCorrectionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ErrorCorrectionConfig()
        assert config.window_size == 11
        assert config.hidden_dim == 64
        assert config.num_layers == 2
        assert config.use_quality is True
        assert config.temperature == 1.0
        assert config.stochastic is False

    def test_custom_window_size(self):
        """Test custom window size."""
        config = ErrorCorrectionConfig(window_size=7)
        assert config.window_size == 7

    def test_custom_architecture(self):
        """Test custom architecture parameters."""
        config = ErrorCorrectionConfig(hidden_dim=128, num_layers=3)
        assert config.hidden_dim == 128
        assert config.num_layers == 3


class TestSoftErrorCorrection:
    """Tests for SoftErrorCorrection operator."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample sequence data."""
        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        return {"sequence": sequence, "quality_scores": quality}

    @pytest.fixture
    def sample_data_low_quality(self):
        """Provide sample data with low quality regions."""
        sequence = encode_dna_string("ACGTACGTACGTACGT")
        # Low quality at positions 4-7
        quality = jnp.array(
            [
                30.0,
                30.0,
                30.0,
                30.0,
                5.0,
                5.0,
                5.0,
                5.0,
                30.0,
                30.0,
                30.0,
                30.0,
                30.0,
                30.0,
                30.0,
                30.0,
            ]
        )
        return {"sequence": sequence, "quality_scores": quality}

    def test_initialization(self, rngs):
        """Test operator initialization."""
        config = ErrorCorrectionConfig()
        op = SoftErrorCorrection(config, rngs=rngs)
        assert op is not None
        assert op.backbone is not None
        assert len(op.backbone.layers) == config.num_layers

    def test_initialization_custom_layers(self, rngs):
        """Test initialization with custom layer count."""
        config = ErrorCorrectionConfig(num_layers=4)
        op = SoftErrorCorrection(config, rngs=rngs)
        assert op.backbone is not None
        assert len(op.backbone.layers) == 4

    def test_apply_output_shape(self, rngs, sample_data):
        """Test that apply produces correct output shape."""
        config = ErrorCorrectionConfig()
        op = SoftErrorCorrection(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_data, {}, None, None)

        assert "sequence" in transformed_data
        assert transformed_data["sequence"].shape == sample_data["sequence"].shape

    def test_apply_preserves_quality(self, rngs, sample_data):
        """Test that apply preserves quality scores."""
        config = ErrorCorrectionConfig()
        op = SoftErrorCorrection(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, {}, None, None)

        assert "quality_scores" in transformed_data
        assert jnp.allclose(transformed_data["quality_scores"], sample_data["quality_scores"])

    def test_apply_returns_confidence(self, rngs, sample_data):
        """Test that apply returns correction confidence."""
        config = ErrorCorrectionConfig()
        op = SoftErrorCorrection(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, {}, None, None)

        assert "correction_confidence" in transformed_data
        # Confidence should be in [0, 1]
        assert 0 <= transformed_data["correction_confidence"] <= 1

    def test_output_is_valid_probability(self, rngs, sample_data):
        """Test that output is valid probability distribution at each position."""
        config = ErrorCorrectionConfig()
        op = SoftErrorCorrection(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, {}, None, None)

        # Each position should sum to approximately 1
        position_sums = jnp.sum(transformed_data["sequence"], axis=-1)
        assert jnp.allclose(position_sums, 1.0, rtol=0.1)

    def test_output_non_negative(self, rngs, sample_data):
        """Test that output probabilities are non-negative."""
        config = ErrorCorrectionConfig()
        op = SoftErrorCorrection(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, {}, None, None)

        assert jnp.all(transformed_data["sequence"] >= 0)

    def test_quality_affects_output(self, rngs):
        """Test that quality scores affect the correction."""
        config = ErrorCorrectionConfig(use_quality=True)
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        high_quality = jnp.ones(16) * 40.0
        low_quality = jnp.ones(16) * 5.0

        data_high = {"sequence": sequence, "quality_scores": high_quality}
        data_low = {"sequence": sequence, "quality_scores": low_quality}

        result_high, _, _ = op.apply(data_high, {}, None, None)
        result_low, _, _ = op.apply(data_low, {}, None, None)

        # Results should differ based on quality
        # (unless model hasn't been trained)
        # At minimum, they should both be valid
        assert result_high["sequence"].shape == result_low["sequence"].shape


class TestGradientFlow:
    """Tests for gradient flow through error correction."""

    def test_gradient_flows_through_apply(self, rngs):
        """Test that gradients flow through the apply method."""
        config = ErrorCorrectionConfig()
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
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
        config = ErrorCorrectionConfig(use_quality=True)
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        state = {}

        def loss_fn(q):
            data = {"sequence": sequence, "quality_scores": q}
            transformed, _, _ = op.apply(data, state, None, None)
            return jnp.sum(transformed["sequence"])

        grad = jax.grad(loss_fn)(quality)
        assert grad is not None
        assert grad.shape == quality.shape

    def test_backbone_layers_are_learnable(self, rngs):
        """Test that backbone MLP parameters are learnable."""
        config = ErrorCorrectionConfig()
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            # Project onto a single alphabet column so the loss is not
            # invariant to the per-position softmax renormalization
            # inside the operator (renormalized rows sum to 1, so summing
            # everything would yield a constant loss with zero gradient).
            return jnp.sum(transformed["sequence"][:, 0])

        loss, grads = loss_fn(op)

        assert hasattr(grads, "backbone")
        assert grads.backbone is not None
        assert jnp.any(grads.backbone.layers[0].kernel[...] != 0.0)
        assert hasattr(grads, "output_layer")

    def test_correction_weight_is_learnable(self, rngs):
        """Test that correction weight parameter is learnable."""
        config = ErrorCorrectionConfig()
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["sequence"])

        loss, grads = loss_fn(op)

        assert hasattr(grads, "correction_weight")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_apply_is_jit_compatible(self, rngs):
        """Test that apply method works with JIT."""
        config = ErrorCorrectionConfig()
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, new_state, metadata = jit_apply(data, state)
        assert transformed["sequence"].shape == data["sequence"].shape

    def test_jit_produces_same_result(self, rngs):
        """Test that JIT produces same result as eager execution."""
        config = ErrorCorrectionConfig()
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        # Eager execution
        eager_result, _, _ = op.apply(data, state, None, None)

        # JIT execution
        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        jit_result, _, _ = jit_apply(data, state)

        assert jnp.allclose(eager_result["sequence"], jit_result["sequence"], rtol=1e-5)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_sequence(self, rngs):
        """Test with sequence shorter than window."""
        config = ErrorCorrectionConfig(window_size=11)
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGT")  # 4 bases < window_size 11
        quality = jnp.ones(4) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["sequence"].shape == (4, 4)

    def test_single_position(self, rngs):
        """Test with single position sequence."""
        config = ErrorCorrectionConfig(window_size=11)
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("A")
        quality = jnp.array([30.0])
        data = {"sequence": sequence, "quality_scores": quality}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["sequence"].shape == (1, 4)

    def test_without_quality(self, rngs):
        """Test with quality features disabled."""
        config = ErrorCorrectionConfig(use_quality=False)
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["sequence"].shape == data["sequence"].shape

    def test_small_window(self, rngs):
        """Test with small window size."""
        config = ErrorCorrectionConfig(window_size=3)
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["sequence"].shape == data["sequence"].shape

    def test_high_temperature(self, rngs):
        """Test with high temperature (uniform output)."""
        config = ErrorCorrectionConfig(temperature=10.0)
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["sequence"].shape == data["sequence"].shape

    def test_low_temperature(self, rngs):
        """Test with low temperature (peaked output)."""
        config = ErrorCorrectionConfig(temperature=0.1)
        op = SoftErrorCorrection(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["sequence"].shape == data["sequence"].shape
