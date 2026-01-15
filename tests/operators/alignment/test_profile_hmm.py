"""Tests for diffbio.operators.alignment.profile_hmm module.

These tests define the expected behavior of the ProfileHMMSearch
operator for HMMER-style profile search with differentiable scoring.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.alignment.profile_hmm import (
    ProfileHMMSearch,
    ProfileHMMConfig,
)


class TestProfileHMMConfig:
    """Tests for ProfileHMMConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProfileHMMConfig()
        assert config.profile_length == 100
        assert config.alphabet_size == 20  # Amino acids
        assert config.temperature == 1.0
        assert config.learnable_profile is True
        assert config.stochastic is False

    def test_custom_profile_length(self):
        """Test custom profile length."""
        config = ProfileHMMConfig(profile_length=50)
        assert config.profile_length == 50

    def test_dna_alphabet(self):
        """Test DNA alphabet configuration."""
        config = ProfileHMMConfig(alphabet_size=4)
        assert config.alphabet_size == 4


class TestProfileHMMSearch:
    """Tests for ProfileHMMSearch operator."""

    @pytest.fixture
    def sample_sequence(self):
        """Provide sample protein sequence (one-hot encoded)."""
        # Random protein sequence of length 150
        key = jax.random.key(0)
        seq = jax.random.categorical(key, jnp.ones(20), shape=(150,))
        one_hot = jax.nn.one_hot(seq, 20)
        return {"sequence": one_hot}

    @pytest.fixture
    def dna_sequence(self):
        """Provide sample DNA sequence (one-hot encoded)."""
        key = jax.random.key(0)
        seq = jax.random.categorical(key, jnp.ones(4), shape=(100,))
        one_hot = jax.nn.one_hot(seq, 4)
        return {"sequence": one_hot}

    def test_initialization(self, rngs):
        """Test operator initialization."""
        config = ProfileHMMConfig(profile_length=50)
        op = ProfileHMMSearch(config, rngs=rngs)
        assert op is not None
        assert op.profile_length == 50

    def test_match_emissions_shape(self, rngs):
        """Test that match emission matrix has correct shape."""
        config = ProfileHMMConfig(profile_length=50, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        match_emit = op.get_match_emissions()
        # Shape: (profile_length, alphabet_size)
        assert match_emit.shape == (50, 20)

    def test_insert_emissions_shape(self, rngs):
        """Test that insert emission matrix has correct shape."""
        config = ProfileHMMConfig(profile_length=50, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        insert_emit = op.get_insert_emissions()
        # Shape: (profile_length, alphabet_size)
        assert insert_emit.shape == (50, 20)

    def test_transitions_shape(self, rngs):
        """Test that transition parameters have correct shape."""
        config = ProfileHMMConfig(profile_length=50)
        op = ProfileHMMSearch(config, rngs=rngs)

        trans = op.get_transitions()
        # 3 transitions per position: M->M, M->I, M->D
        # Shape depends on implementation
        assert trans is not None

    def test_score_sequence_returns_scalar(self, rngs, sample_sequence):
        """Test that scoring returns a scalar."""
        config = ProfileHMMConfig(profile_length=50, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        score = op.score_sequence(sample_sequence["sequence"])
        assert score.shape == ()

    def test_score_sequence_is_finite(self, rngs, sample_sequence):
        """Test that score is finite."""
        config = ProfileHMMConfig(profile_length=50, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        score = op.score_sequence(sample_sequence["sequence"])
        assert jnp.isfinite(score)

    def test_apply_returns_score(self, rngs, sample_sequence):
        """Test that apply returns alignment score."""
        config = ProfileHMMConfig(profile_length=50, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_sequence, {}, None, None)

        assert "score" in transformed_data
        assert jnp.isfinite(transformed_data["score"])

    def test_apply_returns_state_path(self, rngs, sample_sequence):
        """Test that apply returns soft state path."""
        config = ProfileHMMConfig(profile_length=50, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_sequence, {}, None, None)

        assert "state_posteriors" in transformed_data
        # Shape: (seq_len, profile_length, 3) for M/I/D states
        # Or similar depending on implementation

    def test_dna_profile(self, rngs, dna_sequence):
        """Test with DNA alphabet."""
        config = ProfileHMMConfig(profile_length=30, alphabet_size=4)
        op = ProfileHMMSearch(config, rngs=rngs)

        transformed_data, _, _ = op.apply(dna_sequence, {}, None, None)
        assert jnp.isfinite(transformed_data["score"])


class TestGradientFlow:
    """Tests for gradient flow through profile HMM."""

    def test_gradient_flows_through_score(self, rngs):
        """Test that gradients flow through scoring."""
        config = ProfileHMMConfig(profile_length=20, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        key = jax.random.key(0)
        seq = jax.nn.one_hot(jax.random.categorical(key, jnp.ones(20), shape=(50,)), 20)

        def loss_fn(sequence):
            return op.score_sequence(sequence)

        grad = jax.grad(loss_fn)(seq)
        assert grad is not None
        assert grad.shape == seq.shape

    def test_profile_is_learnable(self, rngs):
        """Test that profile parameters are learnable."""
        config = ProfileHMMConfig(profile_length=20, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        key = jax.random.key(0)
        seq = jax.nn.one_hot(jax.random.categorical(key, jnp.ones(20), shape=(50,)), 20)
        data = {"sequence": seq}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["score"]

        loss, grads = loss_fn(op)

        assert hasattr(grads, "log_match_emissions")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_score_is_jit_compatible(self, rngs):
        """Test that scoring works with JIT."""
        config = ProfileHMMConfig(profile_length=20, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        key = jax.random.key(0)
        seq = jax.nn.one_hot(jax.random.categorical(key, jnp.ones(20), shape=(50,)), 20)

        @jax.jit
        def jit_score(sequence):
            return op.score_sequence(sequence)

        score = jit_score(seq)
        assert jnp.isfinite(score)

    def test_apply_is_jit_compatible(self, rngs):
        """Test that apply method works with JIT."""
        config = ProfileHMMConfig(profile_length=20, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        key = jax.random.key(0)
        seq = jax.nn.one_hot(jax.random.categorical(key, jnp.ones(20), shape=(50,)), 20)
        data = {"sequence": seq}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, new_state, metadata = jit_apply(data, state)
        assert jnp.isfinite(transformed["score"])


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_sequence(self, rngs):
        """Test with sequence shorter than profile."""
        config = ProfileHMMConfig(profile_length=50, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        key = jax.random.key(0)
        seq = jax.nn.one_hot(jax.random.categorical(key, jnp.ones(20), shape=(20,)), 20)
        data = {"sequence": seq}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["score"])

    def test_exact_length_sequence(self, rngs):
        """Test with sequence exactly profile length."""
        config = ProfileHMMConfig(profile_length=50, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        key = jax.random.key(0)
        seq = jax.nn.one_hot(jax.random.categorical(key, jnp.ones(20), shape=(50,)), 20)
        data = {"sequence": seq}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["score"])

    def test_long_sequence(self, rngs):
        """Test with long sequence."""
        config = ProfileHMMConfig(profile_length=20, alphabet_size=20)
        op = ProfileHMMSearch(config, rngs=rngs)

        key = jax.random.key(0)
        seq = jax.nn.one_hot(jax.random.categorical(key, jnp.ones(20), shape=(500,)), 20)
        data = {"sequence": seq}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["score"])

    def test_high_temperature(self, rngs):
        """Test with high temperature."""
        config = ProfileHMMConfig(profile_length=20, alphabet_size=20, temperature=10.0)
        op = ProfileHMMSearch(config, rngs=rngs)

        key = jax.random.key(0)
        seq = jax.nn.one_hot(jax.random.categorical(key, jnp.ones(20), shape=(50,)), 20)
        data = {"sequence": seq}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["score"])

    def test_low_temperature(self, rngs):
        """Test with low temperature."""
        config = ProfileHMMConfig(profile_length=20, alphabet_size=20, temperature=0.1)
        op = ProfileHMMSearch(config, rngs=rngs)

        key = jax.random.key(0)
        seq = jax.nn.one_hot(jax.random.categorical(key, jnp.ones(20), shape=(50,)), 20)
        data = {"sequence": seq}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["score"])
