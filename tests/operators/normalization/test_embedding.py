"""Tests for diffbio.operators.normalization.embedding module.

These tests define the expected behavior of the SequenceEmbedding
operator. Implementation should be written to pass these tests.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.normalization.embedding import (
    SequenceEmbedding,
    SequenceEmbeddingConfig,
)
from diffbio.sequences.dna import encode_dna_string


class TestSequenceEmbeddingConfig:
    """Tests for SequenceEmbeddingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SequenceEmbeddingConfig()
        assert config.embedding_dim == 64
        assert config.method == "conv"
        assert config.kernel_size == 7
        assert config.num_conv_layers == 3
        assert config.stochastic is False

    def test_custom_embedding_dim(self):
        """Test custom embedding dimension."""
        config = SequenceEmbeddingConfig(embedding_dim=128)
        assert config.embedding_dim == 128

    def test_custom_architecture(self):
        """Test custom architecture parameters."""
        config = SequenceEmbeddingConfig(
            num_conv_layers=5,
            kernel_size=11
        )
        assert config.num_conv_layers == 5
        assert config.kernel_size == 11


class TestSequenceEmbedding:
    """Tests for SequenceEmbedding operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNGs for operator initialization."""
        return nnx.Rngs(42)

    @pytest.fixture
    def sample_data(self):
        """Provide sample sequence data."""
        sequence = encode_dna_string("ACGTACGTACGTACGT")
        return {"sequence": sequence}

    @pytest.fixture
    def long_sequence_data(self):
        """Provide longer sequence data."""
        sequence = encode_dna_string("ACGT" * 25)  # 100 bases
        return {"sequence": sequence}

    def test_initialization(self, rngs):
        """Test operator initialization."""
        config = SequenceEmbeddingConfig()
        op = SequenceEmbedding(config, rngs=rngs)
        assert op is not None
        assert op.embedding_dim == 64

    def test_initialization_custom_dim(self, rngs):
        """Test initialization with custom embedding dimension."""
        config = SequenceEmbeddingConfig(embedding_dim=128)
        op = SequenceEmbedding(config, rngs=rngs)
        assert op.embedding_dim == 128

    def test_apply_output_shape(self, rngs, sample_data):
        """Test that apply produces correct output shape."""
        config = SequenceEmbeddingConfig(embedding_dim=64)
        op = SequenceEmbedding(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_data, {}, None, None)

        assert "embedding" in transformed_data
        assert transformed_data["embedding"].shape == (64,)

    def test_apply_preserves_sequence(self, rngs, sample_data):
        """Test that apply preserves original sequence."""
        config = SequenceEmbeddingConfig()
        op = SequenceEmbedding(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, {}, None, None)

        assert "sequence" in transformed_data
        assert jnp.allclose(transformed_data["sequence"], sample_data["sequence"])

    def test_apply_returns_per_position_features(self, rngs, sample_data):
        """Test that apply returns per-position features."""
        config = SequenceEmbeddingConfig(embedding_dim=64)
        op = SequenceEmbedding(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, {}, None, None)

        assert "position_embeddings" in transformed_data
        # Shape should be (seq_len, embedding_dim)
        assert transformed_data["position_embeddings"].shape[0] == 16
        assert transformed_data["position_embeddings"].shape[1] == 64

    def test_different_sequences_different_embeddings(self, rngs):
        """Test that different sequences produce different embeddings."""
        config = SequenceEmbeddingConfig()
        op = SequenceEmbedding(config, rngs=rngs)

        seq1 = {"sequence": encode_dna_string("AAAAAAAAAAAAAAAA")}
        seq2 = {"sequence": encode_dna_string("TTTTTTTTTTTTTTTT")}

        result1, _, _ = op.apply(seq1, {}, None, None)
        result2, _, _ = op.apply(seq2, {}, None, None)

        # Embeddings should differ
        assert not jnp.allclose(result1["embedding"], result2["embedding"])

    def test_embedding_is_normalized(self, rngs, sample_data):
        """Test that embedding is normalized."""
        config = SequenceEmbeddingConfig()
        op = SequenceEmbedding(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, {}, None, None)

        # Check that embedding has reasonable magnitude
        norm = jnp.linalg.norm(transformed_data["embedding"])
        assert jnp.isfinite(norm)
        assert norm > 0


class TestGradientFlow:
    """Tests for gradient flow through sequence embedding."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_gradient_flows_through_apply(self, rngs):
        """Test that gradients flow through the apply method."""
        config = SequenceEmbeddingConfig()
        op = SequenceEmbedding(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        state = {}

        def loss_fn(seq):
            data = {"sequence": seq}
            transformed, _, _ = op.apply(data, state, None, None)
            return jnp.sum(transformed["embedding"])

        grad = jax.grad(loss_fn)(sequence)
        assert grad is not None
        assert grad.shape == sequence.shape

    def test_conv_layers_are_learnable(self, rngs):
        """Test that conv layer parameters are learnable."""
        config = SequenceEmbeddingConfig()
        op = SequenceEmbedding(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        data = {"sequence": sequence}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["embedding"])

        loss, grads = loss_fn(op)

        assert hasattr(grads, "conv_layers")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_apply_is_jit_compatible(self, rngs):
        """Test that apply method works with JIT."""
        config = SequenceEmbeddingConfig()
        op = SequenceEmbedding(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        data = {"sequence": sequence}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, new_state, metadata = jit_apply(data, state)
        assert transformed["embedding"].shape == (64,)

    def test_jit_produces_same_result(self, rngs):
        """Test that JIT produces same result as eager execution."""
        config = SequenceEmbeddingConfig()
        op = SequenceEmbedding(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        data = {"sequence": sequence}
        state = {}

        # Eager execution
        eager_result, _, _ = op.apply(data, state, None, None)

        # JIT execution
        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        jit_result, _, _ = jit_apply(data, state)

        assert jnp.allclose(
            eager_result["embedding"],
            jit_result["embedding"],
            rtol=1e-5
        )


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_short_sequence(self, rngs):
        """Test with very short sequence."""
        config = SequenceEmbeddingConfig(kernel_size=3)
        op = SequenceEmbedding(config, rngs=rngs)

        sequence = encode_dna_string("ACG")  # Only 3 bases
        data = {"sequence": sequence}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["embedding"].shape == (64,)

    def test_single_base(self, rngs):
        """Test with single base sequence."""
        config = SequenceEmbeddingConfig(kernel_size=3)
        op = SequenceEmbedding(config, rngs=rngs)

        sequence = encode_dna_string("A")
        data = {"sequence": sequence}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["embedding"].shape == (64,)

    def test_long_sequence(self, rngs):
        """Test with long sequence."""
        config = SequenceEmbeddingConfig()
        op = SequenceEmbedding(config, rngs=rngs)

        sequence = encode_dna_string("ACGT" * 100)  # 400 bases
        data = {"sequence": sequence}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["embedding"].shape == (64,)
        assert transformed["position_embeddings"].shape[0] == 400

    def test_small_embedding_dim(self, rngs):
        """Test with small embedding dimension."""
        config = SequenceEmbeddingConfig(embedding_dim=8)
        op = SequenceEmbedding(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        data = {"sequence": sequence}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["embedding"].shape == (8,)

    def test_large_kernel(self, rngs):
        """Test with large kernel size."""
        config = SequenceEmbeddingConfig(kernel_size=15)
        op = SequenceEmbedding(config, rngs=rngs)

        sequence = encode_dna_string("ACGT" * 10)  # 40 bases
        data = {"sequence": sequence}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["embedding"].shape == (64,)
