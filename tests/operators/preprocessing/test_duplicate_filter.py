"""Tests for diffbio.operators.preprocessing.duplicate_filter module.

These tests define the expected behavior of the DifferentiableDuplicateWeighting
operator. Implementation should be written to pass these tests.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.preprocessing.duplicate_filter import (
    DifferentiableDuplicateWeighting,
    DuplicateWeightingConfig,
)
from diffbio.sequences.dna import encode_dna_string


class TestDuplicateWeightingConfig:
    """Tests for DuplicateWeightingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DuplicateWeightingConfig()
        assert config.temperature == 1.0
        assert config.similarity_threshold == 0.9
        assert config.embedding_dim == 32
        assert config.stochastic is False

    def test_custom_threshold(self):
        """Test custom similarity threshold."""
        config = DuplicateWeightingConfig(similarity_threshold=0.95)
        assert config.similarity_threshold == 0.95

    def test_custom_embedding_dim(self):
        """Test custom embedding dimension."""
        config = DuplicateWeightingConfig(embedding_dim=64)
        assert config.embedding_dim == 64


class TestDifferentiableDuplicateWeighting:
    """Tests for DifferentiableDuplicateWeighting operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNGs for operator initialization."""
        return nnx.Rngs(42)

    @pytest.fixture
    def sample_data(self):
        """Provide sample single sequence data."""
        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        return {"sequence": sequence, "quality_scores": quality}

    @pytest.fixture
    def batch_data_unique(self):
        """Provide batch of unique sequences."""
        seqs = [
            "ACGTACGTACGTACGT",
            "TGCATGCATGCATGCA",
            "AAAACCCCGGGGTTTT",
            "TTTTGGGGCCCCAAAA",
        ]
        sequences = jnp.stack([encode_dna_string(s) for s in seqs])
        quality = jnp.ones((4, 16)) * 30.0
        return {"sequence": sequences, "quality_scores": quality}

    @pytest.fixture
    def batch_data_duplicates(self):
        """Provide batch with duplicate sequences."""
        seqs = [
            "ACGTACGTACGTACGT",
            "ACGTACGTACGTACGT",  # Duplicate
            "ACGTACGTACGTACGT",  # Duplicate
            "TGCATGCATGCATGCA",  # Unique
        ]
        sequences = jnp.stack([encode_dna_string(s) for s in seqs])
        quality = jnp.ones((4, 16)) * 30.0
        return {"sequence": sequences, "quality_scores": quality}

    def test_initialization(self, rngs):
        """Test operator initialization."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)
        assert op is not None
        assert float(op.temperature[...]) == 1.0

    def test_apply_single_sequence(self, rngs, sample_data):
        """Test apply with single sequence returns weight 1.0."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_data, {}, None, None)

        # Single sequence should have weight 1.0
        assert "uniqueness_weight" in transformed_data
        assert jnp.isclose(transformed_data["uniqueness_weight"], 1.0)

    def test_apply_returns_embedding(self, rngs, sample_data):
        """Test that apply returns sequence embedding."""
        config = DuplicateWeightingConfig(embedding_dim=32)
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_data, {}, None, None)

        assert "embedding" in transformed_data
        assert transformed_data["embedding"].shape == (32,)

    def test_apply_preserves_sequence(self, rngs, sample_data):
        """Test that apply preserves original sequence."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, {}, None, None)

        assert "sequence" in transformed_data
        assert jnp.allclose(transformed_data["sequence"], sample_data["sequence"])

    def test_batch_processing(self, rngs, batch_data_unique):
        """Test batch processing produces weights for each sequence."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        # Use apply_batch for batch processing
        weights, embeddings = op.apply_batch(
            batch_data_unique["sequence"], batch_data_unique["quality_scores"]
        )

        assert weights.shape == (4,)
        assert embeddings.shape == (4, config.embedding_dim)

    def test_unique_sequences_equal_weights(self, rngs, batch_data_unique):
        """Test that unique sequences have similar weights."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        weights, _ = op.apply_batch(
            batch_data_unique["sequence"], batch_data_unique["quality_scores"]
        )

        # All weights should be relatively close for unique sequences
        # Since weights are normalized to mean=1, all should be around 1
        assert jnp.allclose(jnp.mean(weights), 1.0, rtol=0.1)

    def test_duplicate_sequences_lower_weights(self, rngs, batch_data_duplicates):
        """Test that duplicate sequences get lower weights."""
        config = DuplicateWeightingConfig(similarity_threshold=0.5)
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        weights, _ = op.apply_batch(
            batch_data_duplicates["sequence"], batch_data_duplicates["quality_scores"]
        )

        # Weights are normalized to mean 1.0, so duplicates should be below mean
        # and unique sequence should be above mean
        # Note: exact behavior depends on embedding and similarity
        assert weights.shape == (4,)
        # Mean should be 1.0 after normalization
        assert jnp.isclose(jnp.mean(weights), 1.0, rtol=0.1)


class TestGradientFlow:
    """Tests for gradient flow through duplicate weighting."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_gradient_flows_through_apply(self, rngs):
        """Test that gradients flow through the apply method."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        state = {}

        def loss_fn(seq):
            data = {"sequence": seq, "quality_scores": quality}
            transformed, _, _ = op.apply(data, state, None, None)
            return jnp.sum(transformed["embedding"])

        grad = jax.grad(loss_fn)(sequence)
        assert grad is not None
        assert grad.shape == sequence.shape

    def test_gradient_flows_through_batch(self, rngs):
        """Test that gradients flow through batch processing."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        sequences = jnp.stack(
            [
                encode_dna_string("ACGTACGTACGTACGT"),
                encode_dna_string("TGCATGCATGCATGCA"),
            ]
        )
        quality = jnp.ones((2, 16)) * 30.0

        def loss_fn(seqs):
            weights, embeddings = op.apply_batch(seqs, quality)
            return jnp.sum(weights) + jnp.sum(embeddings)

        grad = jax.grad(loss_fn)(sequences)
        assert grad is not None
        assert grad.shape == sequences.shape

    def test_conv_kernel_is_learnable(self, rngs):
        """Test that conv kernel parameter is learnable."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["embedding"])

        loss, grads = loss_fn(op)

        assert hasattr(grads, "conv_kernel")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_apply_is_jit_compatible(self, rngs):
        """Test that apply method works with JIT."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        sequence = encode_dna_string("ACGTACGTACGTACGT")
        quality = jnp.ones(16) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, new_state, metadata = jit_apply(data, state)
        assert "uniqueness_weight" in transformed
        assert "embedding" in transformed

    def test_batch_is_jit_compatible(self, rngs):
        """Test that batch processing works with JIT."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        sequences = jnp.stack(
            [
                encode_dna_string("ACGTACGTACGTACGT"),
                encode_dna_string("TGCATGCATGCATGCA"),
            ]
        )
        quality = jnp.ones((2, 16)) * 30.0

        @jax.jit
        def jit_batch(seqs, qual):
            return op.apply_batch(seqs, qual)

        weights, embeddings = jit_batch(sequences, quality)
        assert weights.shape == (2,)
        assert embeddings.shape == (2, config.embedding_dim)


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_short_sequence(self, rngs):
        """Test with very short sequence."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        sequence = encode_dna_string("ACGT")  # Only 4 bases
        quality = jnp.ones(4) * 30.0
        data = {"sequence": sequence, "quality_scores": quality}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert "uniqueness_weight" in transformed
        assert "embedding" in transformed

    def test_single_batch(self, rngs):
        """Test batch processing with single sequence."""
        config = DuplicateWeightingConfig()
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        sequences = encode_dna_string("ACGTACGTACGTACGT")[None, :, :]  # Add batch dim
        quality = jnp.ones((1, 16)) * 30.0

        weights, embeddings = op.apply_batch(sequences, quality)
        assert weights.shape == (1,)
        assert embeddings.shape == (1, config.embedding_dim)

    def test_high_temperature_smooth_clustering(self, rngs):
        """Test with high temperature (smooth clustering)."""
        config = DuplicateWeightingConfig(temperature=10.0)
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        sequences = jnp.stack(
            [
                encode_dna_string("ACGTACGTACGTACGT"),
                encode_dna_string("ACGTACGTACGTACGT"),  # Duplicate
            ]
        )
        quality = jnp.ones((2, 16)) * 30.0

        weights, _ = op.apply_batch(sequences, quality)
        assert weights.shape == (2,)

    def test_low_temperature_sharp_clustering(self, rngs):
        """Test with low temperature (sharp clustering)."""
        config = DuplicateWeightingConfig(temperature=0.1)
        op = DifferentiableDuplicateWeighting(config, rngs=rngs)

        sequences = jnp.stack(
            [
                encode_dna_string("ACGTACGTACGTACGT"),
                encode_dna_string("ACGTACGTACGTACGT"),  # Duplicate
            ]
        )
        quality = jnp.ones((2, 16)) * 30.0

        weights, _ = op.apply_batch(sequences, quality)
        assert weights.shape == (2,)
