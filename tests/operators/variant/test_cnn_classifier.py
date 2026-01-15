"""Tests for diffbio.operators.variant.cnn_classifier module.

These tests define the expected behavior of the CNNVariantClassifier
operator for DeepVariant-style pileup image classification.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.variant.cnn_classifier import (
    CNNVariantClassifier,
    CNNVariantClassifierConfig,
)


class TestCNNVariantClassifierConfig:
    """Tests for CNNVariantClassifierConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CNNVariantClassifierConfig()
        assert config.num_classes == 3  # REF, SNV, INDEL
        assert config.input_height == 100  # coverage depth
        assert config.input_width == 221  # context window
        assert config.num_channels == 6  # A, C, G, T, quality, strand
        assert config.dropout_rate == 0.1
        assert config.stochastic is True  # Uses dropout
        assert config.stream_name == "dropout"  # Required for stochastic

    def test_custom_num_classes(self):
        """Test custom number of classes."""
        config = CNNVariantClassifierConfig(num_classes=5)
        assert config.num_classes == 5

    def test_custom_input_dimensions(self):
        """Test custom input dimensions."""
        config = CNNVariantClassifierConfig(input_height=50, input_width=101)
        assert config.input_height == 50
        assert config.input_width == 101


class TestCNNVariantClassifier:
    """Tests for CNNVariantClassifier operator."""

    @pytest.fixture
    def sample_pileup(self):
        """Provide sample pileup image."""
        # Shape: (batch, height, width, channels) = (1, 100, 221, 6)
        key = jax.random.key(0)
        pileup = jax.random.uniform(key, (1, 100, 221, 6))
        return {"pileup_image": pileup}

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return CNNVariantClassifierConfig(
            input_height=32,
            input_width=64,
            num_channels=6,
            hidden_channels=[16, 32],
            fc_dims=[32],
            dropout_rate=0.0,  # Disable for deterministic tests
            stochastic=False,
            stream_name=None,  # No stream when deterministic
        )

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = CNNVariantClassifier(small_config, rngs=rngs)
        assert op is not None
        assert op.num_classes == small_config.num_classes

    def test_output_shape(self, rngs, small_config):
        """Test that output has correct shape."""
        op = CNNVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(0)
        pileup = jax.random.uniform(
            key, (1, small_config.input_height, small_config.input_width, 6)
        )
        data = {"pileup_image": pileup}

        transformed, _, _ = op.apply(data, {}, None, None)

        # Output should have class probabilities
        assert "class_probs" in transformed
        assert transformed["class_probs"].shape == (1, small_config.num_classes)

    def test_output_sums_to_one(self, rngs, small_config):
        """Test that class probabilities sum to 1."""
        op = CNNVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(0)
        pileup = jax.random.uniform(
            key, (1, small_config.input_height, small_config.input_width, 6)
        )
        data = {"pileup_image": pileup}

        transformed, _, _ = op.apply(data, {}, None, None)

        prob_sum = jnp.sum(transformed["class_probs"], axis=-1)
        assert jnp.allclose(prob_sum, 1.0, atol=1e-5)

    def test_batch_processing(self, rngs, small_config):
        """Test processing multiple samples in batch."""
        op = CNNVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(0)
        batch_size = 4
        pileup = jax.random.uniform(
            key, (batch_size, small_config.input_height, small_config.input_width, 6)
        )
        data = {"pileup_image": pileup}

        transformed, _, _ = op.apply(data, {}, None, None)

        assert transformed["class_probs"].shape == (batch_size, small_config.num_classes)

    def test_returns_logits(self, rngs, small_config):
        """Test that raw logits are also returned."""
        op = CNNVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(0)
        pileup = jax.random.uniform(
            key, (1, small_config.input_height, small_config.input_width, 6)
        )
        data = {"pileup_image": pileup}

        transformed, _, _ = op.apply(data, {}, None, None)

        assert "logits" in transformed
        assert transformed["logits"].shape == (1, small_config.num_classes)


class TestGradientFlow:
    """Tests for gradient flow through CNN classifier."""

    @pytest.fixture
    def small_config(self):
        return CNNVariantClassifierConfig(
            input_height=32,
            input_width=64,
            num_channels=6,
            hidden_channels=[16, 32],
            fc_dims=[32],
            dropout_rate=0.0,
            stochastic=False,
            stream_name=None,
        )

    def test_gradient_flows_through_classifier(self, rngs, small_config):
        """Test that gradients flow through classification."""
        op = CNNVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(0)
        pileup = jax.random.uniform(
            key, (1, small_config.input_height, small_config.input_width, 6)
        )

        def loss_fn(image):
            data = {"pileup_image": image}
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["class_probs"][:, 0].sum()

        grad = jax.grad(loss_fn)(pileup)
        assert grad is not None
        assert grad.shape == pileup.shape
        assert jnp.isfinite(grad).all()

    def test_model_is_learnable(self, rngs, small_config):
        """Test that model parameters are learnable."""
        op = CNNVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(0)
        pileup = jax.random.uniform(
            key, (1, small_config.input_height, small_config.input_width, 6)
        )
        data = {"pileup_image": pileup}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["class_probs"][:, 0].sum()

        loss, grads = loss_fn(op)

        # Check conv layers have gradients
        assert hasattr(grads, "conv_layers")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def small_config(self):
        return CNNVariantClassifierConfig(
            input_height=32,
            input_width=64,
            num_channels=6,
            hidden_channels=[16, 32],
            fc_dims=[32],
            dropout_rate=0.0,
            stochastic=False,
            stream_name=None,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = CNNVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(0)
        pileup = jax.random.uniform(
            key, (1, small_config.input_height, small_config.input_width, 6)
        )
        data = {"pileup_image": pileup}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["class_probs"]).all()


class TestClassifySingleMethod:
    """Tests for the _classify_single method."""

    def test_classify_single_basic(self, rngs):
        """Test _classify_single method directly."""
        config = CNNVariantClassifierConfig(
            input_height=32,
            input_width=64,
            hidden_channels=[16],
            fc_dims=[16],
            dropout_rate=0.0,
            stochastic=False,
            stream_name=None,
        )
        op = CNNVariantClassifier(config, rngs=rngs)

        key = jax.random.key(0)
        # Single image without batch dimension
        pileup = jax.random.uniform(key, (32, 64, 6))

        logits = op._classify_single(pileup)
        assert logits.shape == (config.num_classes,)
        assert jnp.isfinite(logits).all()

    def test_classify_single_with_dropout(self, rngs):
        """Test _classify_single with dropout enabled."""
        config = CNNVariantClassifierConfig(
            input_height=32,
            input_width=64,
            hidden_channels=[16],
            fc_dims=[16],
            dropout_rate=0.5,
            stochastic=True,
            stream_name="dropout",
        )
        op = CNNVariantClassifier(config, rngs=rngs)

        key = jax.random.key(0)
        pileup = jax.random.uniform(key, (32, 64, 6))

        logits = op._classify_single(pileup)
        assert logits.shape == (config.num_classes,)
        assert jnp.isfinite(logits).all()


class TestInitializationVariants:
    """Tests for different initialization scenarios."""

    def test_initialization_without_rngs(self):
        """Test that operator can be initialized without rngs (uses ensure_rngs)."""
        config = CNNVariantClassifierConfig(
            input_height=32,
            input_width=64,
            hidden_channels=[16],
            fc_dims=[16],
            dropout_rate=0.0,
            stochastic=False,
            stream_name=None,
        )
        # Initialize without rngs - should use ensure_rngs fallback
        op = CNNVariantClassifier(config, rngs=None)
        assert op is not None
        assert op.num_classes == config.num_classes

    def test_initialization_with_dropout(self, rngs):
        """Test initialization with dropout enabled."""
        config = CNNVariantClassifierConfig(
            input_height=32,
            input_width=64,
            hidden_channels=[16],
            fc_dims=[16],
            dropout_rate=0.5,
            stochastic=True,
            stream_name="dropout",
        )
        op = CNNVariantClassifier(config, rngs=rngs)
        assert op.dropout is not None
        assert op.dropout_rate == 0.5

    def test_initialization_without_dropout(self, rngs):
        """Test initialization with dropout disabled."""
        config = CNNVariantClassifierConfig(
            input_height=32,
            input_width=64,
            hidden_channels=[16],
            fc_dims=[16],
            dropout_rate=0.0,
            stochastic=False,
            stream_name=None,
        )
        op = CNNVariantClassifier(config, rngs=rngs)
        assert op.dropout is None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_conv_layer(self, rngs):
        """Test with single convolutional layer."""
        config = CNNVariantClassifierConfig(
            input_height=32,
            input_width=64,
            hidden_channels=[16],
            fc_dims=[32],
            dropout_rate=0.0,
            stochastic=False,
            stream_name=None,
        )
        op = CNNVariantClassifier(config, rngs=rngs)

        key = jax.random.key(0)
        pileup = jax.random.uniform(key, (1, 32, 64, 6))
        data = {"pileup_image": pileup}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["class_probs"]).all()

    def test_deep_network(self, rngs):
        """Test with many layers."""
        config = CNNVariantClassifierConfig(
            input_height=64,
            input_width=128,
            hidden_channels=[16, 32, 64, 128],
            fc_dims=[128, 64, 32],
            dropout_rate=0.0,
            stochastic=False,
            stream_name=None,
        )
        op = CNNVariantClassifier(config, rngs=rngs)

        key = jax.random.key(0)
        pileup = jax.random.uniform(key, (1, 64, 128, 6))
        data = {"pileup_image": pileup}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["class_probs"]).all()

    def test_binary_classification(self, rngs):
        """Test with binary classification."""
        config = CNNVariantClassifierConfig(
            num_classes=2,
            input_height=32,
            input_width=64,
            hidden_channels=[16],
            fc_dims=[16],
            dropout_rate=0.0,
            stochastic=False,
            stream_name=None,
        )
        op = CNNVariantClassifier(config, rngs=rngs)

        key = jax.random.key(0)
        pileup = jax.random.uniform(key, (1, 32, 64, 6))
        data = {"pileup_image": pileup}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["class_probs"].shape == (1, 2)
