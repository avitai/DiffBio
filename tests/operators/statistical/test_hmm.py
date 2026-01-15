"""Tests for diffbio.operators.statistical.hmm module.

These tests define the expected behavior of the DifferentiableHMM
operator. Implementation should be written to pass these tests.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.statistical.hmm import (
    DifferentiableHMM,
    HMMConfig,
)
from diffbio.sequences.dna import encode_dna_string


class TestHMMConfig:
    """Tests for HMMConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HMMConfig()
        assert config.n_states == 3
        assert config.n_emissions == 4  # DNA alphabet
        assert config.temperature == 1.0
        assert config.learnable_transitions is True
        assert config.learnable_emissions is True
        assert config.stochastic is False

    def test_custom_states(self):
        """Test custom number of states."""
        config = HMMConfig(n_states=5)
        assert config.n_states == 5

    def test_custom_emissions(self):
        """Test custom number of emissions."""
        config = HMMConfig(n_emissions=20)  # Amino acids
        assert config.n_emissions == 20


class TestDifferentiableHMM:
    """Tests for DifferentiableHMM operator."""

    @pytest.fixture
    def rngs(self):
        """Provide RNGs for operator initialization."""
        return nnx.Rngs(42)

    @pytest.fixture
    def sample_data(self):
        """Provide sample observation data."""
        # DNA sequence encoded as one-hot
        sequence = encode_dna_string("ACGTACGTACGTACGT")
        return {"observations": sequence}

    @pytest.fixture
    def integer_observations(self):
        """Provide integer-encoded observations."""
        # Integer indices: A=0, C=1, G=2, T=3
        obs = jnp.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        return {"observations": obs}

    def test_initialization(self, rngs):
        """Test operator initialization."""
        config = HMMConfig()
        op = DifferentiableHMM(config, rngs=rngs)
        assert op is not None
        assert op.n_states == 3
        assert op.n_emissions == 4

    def test_initialization_custom_states(self, rngs):
        """Test initialization with custom state count."""
        config = HMMConfig(n_states=5)
        op = DifferentiableHMM(config, rngs=rngs)
        assert op.n_states == 5

    def test_log_transition_matrix_shape(self, rngs):
        """Test that transition matrix has correct shape."""
        config = HMMConfig(n_states=3)
        op = DifferentiableHMM(config, rngs=rngs)

        log_trans = op.get_log_transition_matrix()
        assert log_trans.shape == (3, 3)

    def test_log_emission_matrix_shape(self, rngs):
        """Test that emission matrix has correct shape."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        log_emit = op.get_log_emission_matrix()
        assert log_emit.shape == (3, 4)

    def test_transition_matrix_is_valid_log_prob(self, rngs):
        """Test that transition matrix rows sum to 1 in probability space."""
        config = HMMConfig(n_states=3)
        op = DifferentiableHMM(config, rngs=rngs)

        log_trans = op.get_log_transition_matrix()
        # exp(log_trans) should sum to 1 along rows
        row_sums = jnp.sum(jnp.exp(log_trans), axis=1)
        assert jnp.allclose(row_sums, 1.0, rtol=1e-5)

    def test_emission_matrix_is_valid_log_prob(self, rngs):
        """Test that emission matrix rows sum to 1 in probability space."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        log_emit = op.get_log_emission_matrix()
        # exp(log_emit) should sum to 1 along rows
        row_sums = jnp.sum(jnp.exp(log_emit), axis=1)
        assert jnp.allclose(row_sums, 1.0, rtol=1e-5)

    def test_forward_algorithm_output_shape(self, rngs, integer_observations):
        """Test that forward algorithm returns correct shape."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        log_prob = op.forward(integer_observations["observations"])

        # Should return scalar log probability
        assert log_prob.shape == ()

    def test_forward_algorithm_returns_finite(self, rngs, integer_observations):
        """Test that forward algorithm returns finite value."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        log_prob = op.forward(integer_observations["observations"])

        assert jnp.isfinite(log_prob)

    def test_forward_algorithm_negative_log_prob(self, rngs, integer_observations):
        """Test that log probability is non-positive."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        log_prob = op.forward(integer_observations["observations"])

        # Log probability should be <= 0
        assert log_prob <= 0

    def test_apply_returns_log_likelihood(self, rngs, integer_observations):
        """Test that apply returns log likelihood."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(integer_observations, {}, None, None)

        assert "log_likelihood" in transformed_data
        assert jnp.isfinite(transformed_data["log_likelihood"])

    def test_apply_returns_state_posteriors(self, rngs, integer_observations):
        """Test that apply returns state posteriors."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        transformed_data, _, _ = op.apply(integer_observations, {}, None, None)

        assert "state_posteriors" in transformed_data
        # Shape should be (seq_len, n_states)
        assert transformed_data["state_posteriors"].shape == (16, 3)

    def test_state_posteriors_are_valid_probs(self, rngs, integer_observations):
        """Test that state posteriors are valid probabilities."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        transformed_data, _, _ = op.apply(integer_observations, {}, None, None)

        posteriors = transformed_data["state_posteriors"]
        # Should sum to 1 at each position
        row_sums = jnp.sum(posteriors, axis=1)
        assert jnp.allclose(row_sums, 1.0, rtol=1e-4)
        # Should be non-negative
        assert jnp.all(posteriors >= 0)


class TestGradientFlow:
    """Tests for gradient flow through HMM."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_gradient_flows_through_forward(self, rngs):
        """Test that gradients flow through the forward algorithm."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        # Use soft observations for gradient flow
        obs = jax.nn.softmax(jax.random.normal(jax.random.key(0), (16, 4)))

        def loss_fn(observations):
            return op.forward_soft(observations)

        grad = jax.grad(loss_fn)(obs)
        assert grad is not None
        assert grad.shape == obs.shape

    def test_transitions_are_learnable(self, rngs):
        """Test that transition parameters are learnable."""
        config = HMMConfig(n_states=3, n_emissions=4, learnable_transitions=True)
        op = DifferentiableHMM(config, rngs=rngs)

        obs = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
        data = {"observations": obs}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["log_likelihood"]

        loss, grads = loss_fn(op)

        assert hasattr(grads, "log_transition_params")

    def test_emissions_are_learnable(self, rngs):
        """Test that emission parameters are learnable."""
        config = HMMConfig(n_states=3, n_emissions=4, learnable_emissions=True)
        op = DifferentiableHMM(config, rngs=rngs)

        obs = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
        data = {"observations": obs}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return transformed["log_likelihood"]

        loss, grads = loss_fn(op)

        assert hasattr(grads, "log_emission_params")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_forward_is_jit_compatible(self, rngs):
        """Test that forward algorithm works with JIT."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        obs = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])

        @jax.jit
        def jit_forward(observations):
            return op.forward(observations)

        log_prob = jit_forward(obs)
        assert jnp.isfinite(log_prob)

    def test_apply_is_jit_compatible(self, rngs):
        """Test that apply method works with JIT."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        obs = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
        data = {"observations": obs}
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, new_state, metadata = jit_apply(data, state)
        assert jnp.isfinite(transformed["log_likelihood"])


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(42)

    def test_single_observation(self, rngs):
        """Test with single observation."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        obs = jnp.array([0])
        data = {"observations": obs}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["log_likelihood"])
        assert transformed["state_posteriors"].shape == (1, 3)

    def test_two_states(self, rngs):
        """Test with minimal two states."""
        config = HMMConfig(n_states=2, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        obs = jnp.array([0, 1, 2, 3])
        data = {"observations": obs}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["state_posteriors"].shape == (4, 2)

    def test_many_states(self, rngs):
        """Test with many states."""
        config = HMMConfig(n_states=10, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        obs = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
        data = {"observations": obs}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["state_posteriors"].shape == (8, 10)

    def test_long_sequence(self, rngs):
        """Test with long sequence."""
        config = HMMConfig(n_states=3, n_emissions=4)
        op = DifferentiableHMM(config, rngs=rngs)

        obs = jnp.tile(jnp.array([0, 1, 2, 3]), 50)  # 200 observations
        data = {"observations": obs}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["log_likelihood"])
        assert transformed["state_posteriors"].shape == (200, 3)

    def test_high_temperature(self, rngs):
        """Test with high temperature (smoothed)."""
        config = HMMConfig(n_states=3, n_emissions=4, temperature=10.0)
        op = DifferentiableHMM(config, rngs=rngs)

        obs = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
        data = {"observations": obs}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["log_likelihood"])

    def test_low_temperature(self, rngs):
        """Test with low temperature (peaked)."""
        config = HMMConfig(n_states=3, n_emissions=4, temperature=0.1)
        op = DifferentiableHMM(config, rngs=rngs)

        obs = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
        data = {"observations": obs}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isfinite(transformed["log_likelihood"])
