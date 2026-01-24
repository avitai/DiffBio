"""Tests for TransformerSequenceEncoder operator.

This module tests the differentiable transformer-based sequence encoder
for DNA/RNA sequences, following DNABERT/RNA-FM architecture patterns.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.language_models import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
    create_dna_encoder,
    create_rna_encoder,
)


class TestTransformerSequenceEncoderConfig:
    """Tests for TransformerSequenceEncoderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TransformerSequenceEncoderConfig()

        assert config.hidden_dim == 256
        assert config.num_layers == 4
        assert config.num_heads == 4
        assert config.intermediate_dim == 1024
        assert config.max_length == 512
        assert config.alphabet_size == 4
        assert config.dropout_rate == 0.1
        assert config.pooling == "mean"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TransformerSequenceEncoderConfig(
            hidden_dim=640,
            num_layers=12,
            num_heads=20,
            intermediate_dim=5120,
            max_length=1024,
            alphabet_size=5,  # Include N for unknown
            dropout_rate=0.0,
            pooling="cls",
        )

        assert config.hidden_dim == 640
        assert config.num_layers == 12
        assert config.num_heads == 20
        assert config.intermediate_dim == 5120
        assert config.max_length == 1024
        assert config.alphabet_size == 5
        assert config.dropout_rate == 0.0
        assert config.pooling == "cls"


class TestTransformerSequenceEncoder:
    """Tests for TransformerSequenceEncoder operator."""

    @pytest.fixture
    def config(self):
        """Create default config for tests."""
        return TransformerSequenceEncoderConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            intermediate_dim=128,
            max_length=128,
            dropout_rate=0.0,  # Deterministic for testing
        )

    @pytest.fixture
    def encoder(self, config):
        """Create encoder for tests."""
        return TransformerSequenceEncoder(config, rngs=nnx.Rngs(42))

    def test_initialization(self, encoder, config):
        """Test encoder initializes correctly."""
        assert encoder.config.hidden_dim == config.hidden_dim
        assert encoder.config.num_layers == config.num_layers
        assert encoder.config.num_heads == config.num_heads

    def test_forward_pass_shape(self, encoder):
        """Test forward pass produces correct output shapes."""
        # Create one-hot encoded sequence
        seq_len = 50
        batch_size = 2
        sequences = jax.random.uniform(jax.random.PRNGKey(0), (batch_size, seq_len, 4))
        sequences = jax.nn.softmax(sequences, axis=-1)  # Normalize to valid probs

        data = {"sequence": sequences}
        result, state, metadata = encoder.apply(data, {}, None)

        # Check output keys
        assert "sequence" in result
        assert "embedding" in result
        assert "position_embeddings" in result

        # Check shapes
        assert result["sequence"].shape == (batch_size, seq_len, 4)
        assert result["embedding"].shape == (batch_size, 64)
        assert result["position_embeddings"].shape == (batch_size, seq_len, 64)

    def test_single_sequence(self, encoder):
        """Test encoding a single sequence (no batch dimension)."""
        seq_len = 30
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        data = {"sequence": sequence}
        result, _, _ = encoder.apply(data, {}, None)

        # Should handle single sequence
        assert result["embedding"].shape == (64,)
        assert result["position_embeddings"].shape == (seq_len, 64)

    def test_cls_pooling(self):
        """Test CLS token pooling strategy."""
        config = TransformerSequenceEncoderConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            pooling="cls",
            dropout_rate=0.0,
        )
        encoder = TransformerSequenceEncoder(config, rngs=nnx.Rngs(42))

        seq_len = 20
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        data = {"sequence": sequence}
        result, _, _ = encoder.apply(data, {}, None)

        # CLS pooling should use first position
        assert result["embedding"].shape == (32,)

    def test_mean_pooling(self):
        """Test mean pooling strategy."""
        config = TransformerSequenceEncoderConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            pooling="mean",
            dropout_rate=0.0,
        )
        encoder = TransformerSequenceEncoder(config, rngs=nnx.Rngs(42))

        seq_len = 20
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        data = {"sequence": sequence}
        result, _, _ = encoder.apply(data, {}, None)

        assert result["embedding"].shape == (32,)

    def test_attention_mask(self, encoder):
        """Test attention mask handling."""
        seq_len = 40
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        # Create attention mask (first 30 positions valid)
        mask = jnp.concatenate([jnp.ones(30), jnp.zeros(10)])

        data = {"sequence": sequence, "attention_mask": mask}
        result, _, _ = encoder.apply(data, {}, None)

        # Should produce output even with mask
        assert result["embedding"].shape == (64,)

    def test_gradient_flow(self, encoder):
        """Test that gradients flow through the encoder."""
        seq_len = 20
        # Use soft probabilities (not hard one-hot) for proper gradient flow
        logits = jax.random.normal(jax.random.PRNGKey(0), (seq_len, 4))
        sequence = jax.nn.softmax(logits, axis=-1)

        def loss_fn(seq):
            data = {"sequence": seq}
            result, _, _ = encoder.apply(data, {}, None)
            return result["embedding"].sum()

        grads = jax.grad(loss_fn)(sequence)

        # Gradients should exist and be non-zero
        assert grads is not None
        assert grads.shape == sequence.shape
        # Check that at least some gradients are non-zero
        assert jnp.abs(grads).max() > 1e-10

    def test_parameter_gradients(self, encoder):
        """Test gradients with respect to encoder parameters."""
        seq_len = 15
        # Use soft probabilities for proper gradient flow
        logits = jax.random.normal(jax.random.PRNGKey(0), (seq_len, 4))
        sequence = jax.nn.softmax(logits, axis=-1)

        def loss_fn(enc):
            data = {"sequence": sequence}
            result, _, _ = enc.apply(data, {}, None)
            return result["embedding"].sum()

        # Use nnx.value_and_grad for NNX modules
        _, grads = nnx.value_and_grad(loss_fn)(encoder)

        # Should have gradients for parameters
        assert grads is not None

    def test_jit_compatibility(self, encoder):
        """Test JIT compilation works."""
        seq_len = 25
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        @jax.jit
        def encode(seq):
            data = {"sequence": seq}
            result, _, _ = encoder.apply(data, {}, None)
            return result["embedding"]

        # Should compile and run without errors
        embedding = encode(sequence)
        assert embedding.shape == (64,)

        # Second call should be faster (cached)
        embedding2 = encode(sequence)
        assert jnp.allclose(embedding, embedding2)

    def test_deterministic_inference(self, encoder):
        """Test deterministic behavior when dropout is disabled."""
        seq_len = 20
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        data = {"sequence": sequence}

        # Multiple calls should give same result when dropout=0
        result1, _, _ = encoder.apply(data, {}, None)
        result2, _, _ = encoder.apply(data, {}, None)

        assert jnp.allclose(result1["embedding"], result2["embedding"])

    def test_different_sequence_lengths(self, encoder):
        """Test encoder handles different sequence lengths."""
        for seq_len in [10, 50, 100]:
            sequence = jax.nn.one_hot(
                jax.random.randint(jax.random.PRNGKey(seq_len), (seq_len,), 0, 4),
                num_classes=4,
            )

            data = {"sequence": sequence}
            result, _, _ = encoder.apply(data, {}, None)

            assert result["embedding"].shape == (64,)
            assert result["position_embeddings"].shape == (seq_len, 64)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_dna_encoder(self):
        """Test DNA encoder factory function."""
        encoder = create_dna_encoder()

        # Check it creates a valid encoder
        assert isinstance(encoder, TransformerSequenceEncoder)
        assert encoder.config.alphabet_size == 4  # A, C, G, T

    def test_create_dna_encoder_custom_dims(self):
        """Test DNA encoder with custom dimensions."""
        encoder = create_dna_encoder(
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
        )

        assert encoder.config.hidden_dim == 128
        assert encoder.config.num_layers == 3
        assert encoder.config.num_heads == 4

    def test_create_rna_encoder(self):
        """Test RNA encoder factory function."""
        encoder = create_rna_encoder()

        assert isinstance(encoder, TransformerSequenceEncoder)
        assert encoder.config.alphabet_size == 4  # A, C, G, U

    def test_create_rna_encoder_custom_dims(self):
        """Test RNA encoder with custom dimensions."""
        encoder = create_rna_encoder(
            hidden_dim=640,
            num_layers=12,
            num_heads=20,
        )

        assert encoder.config.hidden_dim == 640
        assert encoder.config.num_layers == 12
        assert encoder.config.num_heads == 20


class TestTransformerComponents:
    """Tests for internal transformer components."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for component testing."""
        config = TransformerSequenceEncoderConfig(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            intermediate_dim=64,
            dropout_rate=0.0,
        )
        return TransformerSequenceEncoder(config, rngs=nnx.Rngs(42))

    def test_input_embedding(self, encoder):
        """Test input embedding layer."""
        seq_len = 15
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )

        # The embedding should project from alphabet_size to hidden_dim
        embedded = encoder.input_projection(sequence)
        assert embedded.shape == (seq_len, 32)

    def test_positional_encoding(self, encoder):
        """Test positional encoding is applied."""
        seq_len = 20

        # Get positional encodings
        pos_enc = encoder.get_positional_encoding(seq_len)

        # Should have correct shape
        assert pos_enc.shape == (seq_len, 32)

        # Different positions should have different encodings
        assert not jnp.allclose(pos_enc[0], pos_enc[1])

    def test_transformer_layer(self, encoder):
        """Test transformer layer (uses artifex's TransformerEncoder)."""
        seq_len = 10
        hidden_dim = 32
        batch_size = 1

        # Create mock hidden states with batch dimension
        hidden = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, hidden_dim))

        # Apply transformer
        output = encoder.transformer(hidden, deterministic=True)

        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_cls_token_shape(self, encoder):
        """Test CLS token has correct shape."""
        cls_token = encoder.cls_token[...]
        assert cls_token.shape == (32,)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_sequence(self):
        """Test handling of very short sequences."""
        config = TransformerSequenceEncoderConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            dropout_rate=0.0,
        )
        encoder = TransformerSequenceEncoder(config, rngs=nnx.Rngs(42))

        # Single nucleotide
        sequence = jax.nn.one_hot(jnp.array([0]), num_classes=4)
        data = {"sequence": sequence}

        result, _, _ = encoder.apply(data, {}, None)
        assert result["embedding"].shape == (32,)

    def test_max_length_sequence(self):
        """Test handling of sequences at max length."""
        config = TransformerSequenceEncoderConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            max_length=64,
            dropout_rate=0.0,
        )
        encoder = TransformerSequenceEncoder(config, rngs=nnx.Rngs(42))

        # Exactly at max length
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (64,), 0, 4),
            num_classes=4,
        )
        data = {"sequence": sequence}

        result, _, _ = encoder.apply(data, {}, None)
        assert result["embedding"].shape == (32,)

    def test_soft_one_hot_input(self):
        """Test handling of soft (probabilistic) one-hot input."""
        config = TransformerSequenceEncoderConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            dropout_rate=0.0,
        )
        encoder = TransformerSequenceEncoder(config, rngs=nnx.Rngs(42))

        # Soft probabilities instead of hard one-hot
        seq_len = 20
        sequence = jax.nn.softmax(
            jax.random.normal(jax.random.PRNGKey(0), (seq_len, 4)),
            axis=-1,
        )
        data = {"sequence": sequence}

        result, _, _ = encoder.apply(data, {}, None)
        assert result["embedding"].shape == (32,)

    def test_preserves_input_in_output(self):
        """Test that original input is preserved in output."""
        config = TransformerSequenceEncoderConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=2,
            dropout_rate=0.0,
        )
        encoder = TransformerSequenceEncoder(config, rngs=nnx.Rngs(42))

        seq_len = 20
        sequence = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(0), (seq_len,), 0, 4),
            num_classes=4,
        )
        data = {"sequence": sequence, "other_key": "preserved"}

        result, _, _ = encoder.apply(data, {}, None)

        # Original sequence should be in output
        assert jnp.allclose(result["sequence"], sequence)
        # Other keys should be preserved
        assert result["other_key"] == "preserved"
