"""Tests for diffbio.operators.mapping.neural_mapper module.

These tests define the expected behavior of the NeuralReadMapper
operator for differentiable read-to-reference alignment.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.mapping.neural_mapper import (
    NeuralReadMapper,
    NeuralReadMapperConfig,
)


class TestNeuralReadMapperConfig:
    """Tests for NeuralReadMapperConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NeuralReadMapperConfig()
        assert config.read_length == 150
        assert config.reference_window == 500
        assert config.embedding_dim == 64
        assert config.num_heads == 4
        assert config.stochastic is False

    def test_custom_read_length(self):
        """Test custom read length."""
        config = NeuralReadMapperConfig(read_length=100)
        assert config.read_length == 100

    def test_custom_embedding_dim(self):
        """Test custom embedding dimension."""
        config = NeuralReadMapperConfig(embedding_dim=128)
        assert config.embedding_dim == 128


class TestNeuralReadMapper:
    """Tests for NeuralReadMapper operator."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample read and reference data."""
        key = jax.random.key(0)
        read_length = 100
        ref_length = 300

        # One-hot encoded sequences (batch_size, length, 4)
        key, subkey = jax.random.split(key)
        read_indices = jax.random.randint(subkey, (1, read_length), 0, 4)
        read = jax.nn.one_hot(read_indices, 4)

        key, subkey = jax.random.split(key)
        ref_indices = jax.random.randint(subkey, (1, ref_length), 0, 4)
        reference = jax.nn.one_hot(ref_indices, 4)

        return {"read": read, "reference": reference}

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return NeuralReadMapperConfig(
            read_length=100,
            reference_window=300,
            embedding_dim=32,
            num_heads=2,
            num_layers=2,
        )

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = NeuralReadMapper(small_config, rngs=rngs)
        assert op is not None

    def test_output_contains_alignment_scores(self, rngs, small_config, sample_data):
        """Test that output contains alignment scores."""
        op = NeuralReadMapper(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "alignment_scores" in transformed
        # Scores for each reference position
        assert transformed["alignment_scores"].shape[0] == 1  # batch
        assert transformed["alignment_scores"].shape[1] == 300  # ref positions

    def test_output_contains_position_probs(self, rngs, small_config, sample_data):
        """Test that output contains position probabilities."""
        op = NeuralReadMapper(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "position_probs" in transformed
        # Softmax probabilities over positions
        assert transformed["position_probs"].shape == (1, 300)

    def test_position_probs_sum_to_one(self, rngs, small_config, sample_data):
        """Test that position probabilities sum to 1."""
        op = NeuralReadMapper(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        prob_sum = jnp.sum(transformed["position_probs"], axis=-1)
        assert jnp.allclose(prob_sum, 1.0, atol=1e-5)

    def test_output_contains_best_position(self, rngs, small_config, sample_data):
        """Test that output contains best mapping position."""
        op = NeuralReadMapper(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "best_position" in transformed
        assert transformed["best_position"].shape == (1,)

    def test_best_position_valid(self, rngs, small_config, sample_data):
        """Test that best position is valid index."""
        op = NeuralReadMapper(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        pos = transformed["best_position"]
        assert jnp.all(pos >= 0)
        assert jnp.all(pos < 300)

    def test_output_contains_mapping_quality(self, rngs, small_config, sample_data):
        """Test that output contains mapping quality score."""
        op = NeuralReadMapper(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_data, {}, None, None)

        assert "mapping_quality" in transformed
        assert transformed["mapping_quality"].shape == (1,)


class TestGradientFlow:
    """Tests for gradient flow through neural mapper."""

    @pytest.fixture
    def small_config(self):
        return NeuralReadMapperConfig(
            read_length=50,
            reference_window=100,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
        )

    def test_gradient_flows_through_mapping(self, rngs, small_config):
        """Test that gradients flow through mapping."""
        op = NeuralReadMapper(small_config, rngs=rngs)

        key = jax.random.key(0)
        read_indices = jax.random.randint(key, (1, 50), 0, 4)
        read = jax.nn.one_hot(read_indices, 4)

        key, subkey = jax.random.split(key)
        ref_indices = jax.random.randint(subkey, (1, 100), 0, 4)
        reference = jax.nn.one_hot(ref_indices, 4)

        def loss_fn(read_input):
            data = {"read": read_input, "reference": reference}
            transformed, _, _ = op.apply(data, {}, None, None)
            return transformed["alignment_scores"].sum()

        grad = jax.grad(loss_fn)(read)
        assert grad is not None
        assert grad.shape == read.shape
        assert jnp.isfinite(grad).all()

    def test_model_is_learnable(self, rngs, small_config):
        """Test that model parameters are learnable."""
        op = NeuralReadMapper(small_config, rngs=rngs)

        key = jax.random.key(0)
        read_indices = jax.random.randint(key, (1, 50), 0, 4)
        read = jax.nn.one_hot(read_indices, 4)

        key, subkey = jax.random.split(key)
        ref_indices = jax.random.randint(subkey, (1, 100), 0, 4)
        reference = jax.nn.one_hot(ref_indices, 4)

        data = {"read": read, "reference": reference}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["alignment_scores"].sum()

        loss, grads = loss_fn(op)

        # Check encoder has gradients
        assert hasattr(grads, "read_encoder")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def small_config(self):
        return NeuralReadMapperConfig(
            read_length=50,
            reference_window=100,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = NeuralReadMapper(small_config, rngs=rngs)

        key = jax.random.key(0)
        read_indices = jax.random.randint(key, (1, 50), 0, 4)
        read = jax.nn.one_hot(read_indices, 4)

        key, subkey = jax.random.split(key)
        ref_indices = jax.random.randint(subkey, (1, 100), 0, 4)
        reference = jax.nn.one_hot(ref_indices, 4)

        data = {"read": read, "reference": reference}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["alignment_scores"]).all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_read(self, rngs):
        """Test with short read."""
        config = NeuralReadMapperConfig(
            read_length=25,
            reference_window=100,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
        )
        op = NeuralReadMapper(config, rngs=rngs)

        key = jax.random.key(0)
        read_indices = jax.random.randint(key, (1, 25), 0, 4)
        read = jax.nn.one_hot(read_indices, 4)

        key, subkey = jax.random.split(key)
        ref_indices = jax.random.randint(subkey, (1, 100), 0, 4)
        reference = jax.nn.one_hot(ref_indices, 4)

        data = {"read": read, "reference": reference}
        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["alignment_scores"]).all()

    def test_long_reference(self, rngs):
        """Test with long reference window."""
        config = NeuralReadMapperConfig(
            read_length=50,
            reference_window=500,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
        )
        op = NeuralReadMapper(config, rngs=rngs)

        key = jax.random.key(0)
        read_indices = jax.random.randint(key, (1, 50), 0, 4)
        read = jax.nn.one_hot(read_indices, 4)

        key, subkey = jax.random.split(key)
        ref_indices = jax.random.randint(subkey, (1, 500), 0, 4)
        reference = jax.nn.one_hot(ref_indices, 4)

        data = {"read": read, "reference": reference}
        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["alignment_scores"].shape == (1, 500)

    def test_batch_processing(self, rngs):
        """Test with batch of reads."""
        config = NeuralReadMapperConfig(
            read_length=50,
            reference_window=100,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
        )
        op = NeuralReadMapper(config, rngs=rngs)

        batch_size = 4
        key = jax.random.key(0)
        read_indices = jax.random.randint(key, (batch_size, 50), 0, 4)
        read = jax.nn.one_hot(read_indices, 4)

        key, subkey = jax.random.split(key)
        ref_indices = jax.random.randint(subkey, (batch_size, 100), 0, 4)
        reference = jax.nn.one_hot(ref_indices, 4)

        data = {"read": read, "reference": reference}
        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["alignment_scores"].shape == (batch_size, 100)
