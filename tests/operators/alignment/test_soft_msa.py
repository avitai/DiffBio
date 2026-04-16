"""Tests for diffbio.operators.alignment.soft_msa module.

These tests define the expected behavior of the SoftProgressiveMSA
operator for differentiable multiple sequence alignment.
"""

import jax
import jax.numpy as jnp
import pytest
from artifex.generative_models.core.base import MLP
from flax import nnx

from diffbio.operators.alignment.soft_msa import (
    SoftProgressiveMSA,
    SoftProgressiveMSAConfig,
)


class TestSoftProgressiveMSAConfig:
    """Tests for SoftProgressiveMSAConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SoftProgressiveMSAConfig()
        assert config.max_seq_length == 100
        assert config.hidden_dim == 64
        assert config.temperature == 1.0
        assert config.stochastic is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SoftProgressiveMSAConfig(
            max_seq_length=200,
            hidden_dim=128,
            temperature=0.5,
        )
        assert config.max_seq_length == 200
        assert config.hidden_dim == 128
        assert config.temperature == 0.5

    def test_num_layers_must_be_positive(self):
        """Soft MSA should fail fast without encoder layers."""
        with pytest.raises(ValueError, match="num_layers"):
            SoftProgressiveMSAConfig(num_layers=0)


class TestSoftProgressiveMSA:
    """Tests for SoftProgressiveMSA operator."""

    @pytest.fixture
    def small_config(self):
        """Provide small config for faster tests."""
        return SoftProgressiveMSAConfig(
            max_seq_length=20,
            hidden_dim=32,
            num_layers=2,
            alphabet_size=4,
        )

    @pytest.fixture
    def sample_sequences(self):
        """Provide sample sequences for testing."""
        key = jax.random.key(0)
        n_sequences = 4
        seq_length = 15

        # Generate random one-hot sequences
        key, subkey = jax.random.split(key)
        indices = jax.random.randint(subkey, (n_sequences, seq_length), 0, 4)
        sequences = jax.nn.one_hot(indices, 4)

        return {"sequences": sequences}

    def test_initialization(self, rngs, small_config):
        """Test operator initialization."""
        op = SoftProgressiveMSA(small_config, rngs=rngs)
        assert op is not None
        assert isinstance(op.seq_encoder.backbone, MLP)
        assert isinstance(op.profile_builder.backbone, MLP)
        assert len(op.seq_encoder.backbone.layers) == small_config.num_layers
        assert len(op.profile_builder.backbone.layers) == 2

    def test_output_contains_alignment(self, rngs, small_config, sample_sequences):
        """Test that output contains aligned sequences."""
        op = SoftProgressiveMSA(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_sequences, {}, None, None)

        assert "aligned_sequences" in transformed
        # Aligned sequences should have same batch size
        assert transformed["aligned_sequences"].shape[0] == 4

    def test_output_contains_alignment_scores(self, rngs, small_config, sample_sequences):
        """Test that output contains alignment scores."""
        op = SoftProgressiveMSA(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_sequences, {}, None, None)

        assert "alignment_scores" in transformed

    def test_output_contains_guide_tree(self, rngs, small_config, sample_sequences):
        """Test that output contains guide tree distances."""
        op = SoftProgressiveMSA(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_sequences, {}, None, None)

        assert "pairwise_distances" in transformed
        # Should be (n_sequences, n_sequences)
        assert transformed["pairwise_distances"].shape == (4, 4)

    def test_output_finite(self, rngs, small_config, sample_sequences):
        """Test that outputs are finite."""
        op = SoftProgressiveMSA(small_config, rngs=rngs)

        transformed, _, _ = op.apply(sample_sequences, {}, None, None)

        assert jnp.isfinite(transformed["aligned_sequences"]).all()
        assert jnp.isfinite(transformed["pairwise_distances"]).all()


class TestGradientFlow:
    """Tests for gradient flow through soft MSA."""

    @pytest.fixture
    def small_config(self):
        return SoftProgressiveMSAConfig(
            max_seq_length=15,
            hidden_dim=16,
            num_layers=1,
            alphabet_size=4,
        )

    def test_gradient_flows_through_msa(self, rngs, small_config):
        """Test that gradients flow through MSA."""
        op = SoftProgressiveMSA(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_sequences = 3
        seq_length = 10

        key, subkey = jax.random.split(key)
        indices = jax.random.randint(subkey, (n_sequences, seq_length), 0, 4)
        sequences = jax.nn.one_hot(indices, 4).astype(jnp.float32)

        def loss_fn(seqs):
            data = {"sequences": seqs}
            transformed, _, _ = op.apply(data, {}, None, None)
            return jnp.sum(transformed["pairwise_distances"] ** 2)

        grad = jax.grad(loss_fn)(sequences)
        assert grad is not None
        assert grad.shape == sequences.shape
        assert jnp.isfinite(grad).all()

    def test_model_is_learnable(self, rngs, small_config):
        """Test that model parameters are learnable."""
        op = SoftProgressiveMSA(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_sequences = 3
        seq_length = 10

        key, subkey = jax.random.split(key)
        indices = jax.random.randint(subkey, (n_sequences, seq_length), 0, 4)
        sequences = jax.nn.one_hot(indices, 4).astype(jnp.float32)

        data = {"sequences": sequences}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["pairwise_distances"] ** 2) + jnp.sum(
                transformed["consensus_profile"] ** 2
            )

        loss, grads = loss_fn(op)

        assert loss is not None
        assert hasattr(grads, "seq_encoder")
        assert hasattr(grads, "profile_builder")
        assert jnp.any(grads.seq_encoder.backbone.layers[0].kernel[...] != 0.0)
        assert jnp.any(grads.profile_builder.backbone.layers[0].kernel[...] != 0.0)


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def small_config(self):
        return SoftProgressiveMSAConfig(
            max_seq_length=15,
            hidden_dim=16,
            num_layers=1,
            alphabet_size=4,
        )

    def test_apply_is_jit_compatible(self, rngs, small_config):
        """Test that apply method works with JIT."""
        op = SoftProgressiveMSA(small_config, rngs=rngs)

        key = jax.random.key(0)
        n_sequences = 3
        seq_length = 10

        key, subkey = jax.random.split(key)
        indices = jax.random.randint(subkey, (n_sequences, seq_length), 0, 4)
        sequences = jax.nn.one_hot(indices, 4).astype(jnp.float32)

        data = {"sequences": sequences}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, state)
        assert jnp.isfinite(transformed["aligned_sequences"]).all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_two_sequences(self, rngs):
        """Test with minimum number of sequences."""
        config = SoftProgressiveMSAConfig(
            max_seq_length=10,
            hidden_dim=16,
            num_layers=1,
            alphabet_size=4,
        )
        op = SoftProgressiveMSA(config, rngs=rngs)

        key = jax.random.key(0)
        indices = jax.random.randint(key, (2, 8), 0, 4)
        sequences = jax.nn.one_hot(indices, 4).astype(jnp.float32)

        data = {"sequences": sequences}
        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["aligned_sequences"]).all()

    def test_different_temperature(self, rngs):
        """Test with different temperature values."""
        for temp in [0.1, 1.0, 10.0]:
            config = SoftProgressiveMSAConfig(
                max_seq_length=10,
                hidden_dim=16,
                num_layers=1,
                temperature=temp,
            )
            op = SoftProgressiveMSA(config, rngs=rngs)

            key = jax.random.key(0)
            indices = jax.random.randint(key, (3, 8), 0, 4)
            sequences = jax.nn.one_hot(indices, 4).astype(jnp.float32)

            data = {"sequences": sequences}
            transformed, _, _ = op.apply(data, {}, None, None)
            assert jnp.isfinite(transformed["aligned_sequences"]).all()
