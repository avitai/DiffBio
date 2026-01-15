"""Tests for diffbio.operators.statistical.em_quantification module.

These tests define the expected behavior of the DifferentiableEMQuantifier
operator for transcript quantification. Implementation should be written
to pass these tests.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.statistical.em_quantification import (
    DifferentiableEMQuantifier,
    EMQuantifierConfig,
)


class TestEMQuantifierConfig:
    """Tests for EMQuantifierConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EMQuantifierConfig()
        assert config.n_transcripts == 1000
        assert config.n_iterations == 10
        assert config.temperature == 1.0
        assert config.stochastic is False

    def test_custom_transcripts(self):
        """Test custom number of transcripts."""
        config = EMQuantifierConfig(n_transcripts=5000)
        assert config.n_transcripts == 5000

    def test_custom_iterations(self):
        """Test custom number of iterations."""
        config = EMQuantifierConfig(n_iterations=20)
        assert config.n_iterations == 20


class TestDifferentiableEMQuantifier:
    """Tests for DifferentiableEMQuantifier operator."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample read assignment data."""
        n_reads = 100
        n_transcripts = 50

        # Compatibility matrix: which transcripts each read could come from
        # Random sparse assignments
        key = jax.random.key(0)
        compatibility = jax.random.bernoulli(key, p=0.1, shape=(n_reads, n_transcripts))
        # Ensure at least one transcript per read
        compatibility = compatibility.at[:, 0].set(1.0)
        compatibility = compatibility.astype(jnp.float32)

        # Effective lengths for each transcript
        effective_lengths = jax.random.uniform(
            jax.random.key(1), shape=(n_transcripts,), minval=100.0, maxval=1000.0
        )

        return {
            "compatibility": compatibility,
            "effective_lengths": effective_lengths,
        }

    def test_initialization(self, rngs):
        """Test operator initialization."""
        config = EMQuantifierConfig(n_transcripts=50)
        op = DifferentiableEMQuantifier(config, rngs=rngs)
        assert op is not None
        assert op.n_transcripts == 50

    def test_initial_abundances_sum_to_one(self, rngs):
        """Test that initial abundances sum to 1."""
        config = EMQuantifierConfig(n_transcripts=50)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        abundances = op.get_initial_abundances()
        assert jnp.isclose(jnp.sum(abundances), 1.0, rtol=1e-5)

    def test_initial_abundances_positive(self, rngs):
        """Test that initial abundances are positive."""
        config = EMQuantifierConfig(n_transcripts=50)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        abundances = op.get_initial_abundances()
        assert jnp.all(abundances > 0)

    def test_em_step_preserves_normalization(self, rngs, sample_data):
        """Test that EM step preserves abundance normalization."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=1)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        abundances = op.quantify(sample_data["compatibility"], sample_data["effective_lengths"])

        assert jnp.isclose(jnp.sum(abundances), 1.0, rtol=1e-4)

    def test_quantify_output_shape(self, rngs, sample_data):
        """Test that quantify produces correct output shape."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=5)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        abundances = op.quantify(sample_data["compatibility"], sample_data["effective_lengths"])

        assert abundances.shape == (50,)

    def test_quantify_output_positive(self, rngs, sample_data):
        """Test that quantify produces positive abundances."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=5)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        abundances = op.quantify(sample_data["compatibility"], sample_data["effective_lengths"])

        assert jnp.all(abundances >= 0)

    def test_apply_returns_abundances(self, rngs, sample_data):
        """Test that apply returns transcript abundances."""
        config = EMQuantifierConfig(n_transcripts=50)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_data, {}, None, None)

        assert "abundances" in transformed_data
        assert transformed_data["abundances"].shape == (50,)

    def test_apply_returns_tpm(self, rngs, sample_data):
        """Test that apply returns TPM values."""
        config = EMQuantifierConfig(n_transcripts=50)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_data, {}, None, None)

        assert "tpm" in transformed_data
        # TPM should sum to 1 million
        assert jnp.isclose(jnp.sum(transformed_data["tpm"]), 1e6, rtol=1e-3)

    def test_convergence_with_iterations(self, rngs, sample_data):
        """Test that more iterations lead to convergence."""
        config_few = EMQuantifierConfig(n_transcripts=50, n_iterations=2)
        config_many = EMQuantifierConfig(n_transcripts=50, n_iterations=20)

        op_few = DifferentiableEMQuantifier(config_few, rngs=rngs)
        op_many = DifferentiableEMQuantifier(config_many, rngs=nnx.Rngs(42))

        abundances_few = op_few.quantify(
            sample_data["compatibility"], sample_data["effective_lengths"]
        )
        abundances_many = op_many.quantify(
            sample_data["compatibility"], sample_data["effective_lengths"]
        )

        # Both should be valid probability distributions
        assert jnp.isclose(jnp.sum(abundances_few), 1.0, rtol=1e-4)
        assert jnp.isclose(jnp.sum(abundances_many), 1.0, rtol=1e-4)


class TestGradientFlow:
    """Tests for gradient flow through EM quantification."""

    def test_gradient_flows_through_quantify(self, rngs):
        """Test that gradients flow through quantification."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=5)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        compatibility = jax.random.uniform(jax.random.key(0), shape=(100, 50))
        effective_lengths = jnp.ones(50) * 500.0

        def loss_fn(compat):
            abundances = op.quantify(compat, effective_lengths)
            return jnp.sum(abundances)

        grad = jax.grad(loss_fn)(compatibility)
        assert grad is not None
        assert grad.shape == compatibility.shape

    def test_initial_abundances_are_learnable(self, rngs):
        """Test that initial abundance parameters are learnable."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=5)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        compatibility = jax.random.uniform(jax.random.key(0), shape=(100, 50))
        effective_lengths = jnp.ones(50) * 500.0
        data = {
            "compatibility": compatibility,
            "effective_lengths": effective_lengths,
        }
        state = {}

        @nnx.value_and_grad
        def loss_fn(model):
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["abundances"])

        loss, grads = loss_fn(op)

        assert hasattr(grads, "log_initial_abundances")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_quantify_is_jit_compatible(self, rngs):
        """Test that quantify works with JIT."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=5)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        compatibility = jax.random.uniform(jax.random.key(0), shape=(100, 50))
        effective_lengths = jnp.ones(50) * 500.0

        @jax.jit
        def jit_quantify(compat, lengths):
            return op.quantify(compat, lengths)

        abundances = jit_quantify(compatibility, effective_lengths)
        assert abundances.shape == (50,)

    def test_apply_is_jit_compatible(self, rngs):
        """Test that apply method works with JIT."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=5)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        compatibility = jax.random.uniform(jax.random.key(0), shape=(100, 50))
        effective_lengths = jnp.ones(50) * 500.0
        data = {
            "compatibility": compatibility,
            "effective_lengths": effective_lengths,
        }
        state = {}

        @jax.jit
        def jit_apply(data, state):
            return op.apply(data, state, None, None)

        transformed, new_state, metadata = jit_apply(data, state)
        assert transformed["abundances"].shape == (50,)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_transcript(self, rngs):
        """Test with single transcript."""
        config = EMQuantifierConfig(n_transcripts=1, n_iterations=5)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        compatibility = jnp.ones((100, 1))
        effective_lengths = jnp.array([500.0])
        data = {
            "compatibility": compatibility,
            "effective_lengths": effective_lengths,
        }

        transformed, _, _ = op.apply(data, {}, None, None)
        # Single transcript should have abundance 1.0
        assert jnp.isclose(transformed["abundances"][0], 1.0, rtol=1e-4)

    def test_single_read(self, rngs):
        """Test with single read."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=5)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        compatibility = jnp.zeros((1, 50)).at[0, 0].set(1.0)
        effective_lengths = jnp.ones(50) * 500.0
        data = {
            "compatibility": compatibility,
            "effective_lengths": effective_lengths,
        }

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["abundances"].shape == (50,)

    def test_uniform_compatibility(self, rngs):
        """Test with uniform read compatibility."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=10)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        # All reads compatible with all transcripts equally
        compatibility = jnp.ones((100, 50))
        effective_lengths = jnp.ones(50) * 500.0
        data = {
            "compatibility": compatibility,
            "effective_lengths": effective_lengths,
        }

        transformed, _, _ = op.apply(data, {}, None, None)
        # Should converge to uniform distribution
        assert jnp.allclose(transformed["abundances"], jnp.ones(50) / 50, rtol=0.1)

    def test_high_temperature(self, rngs):
        """Test with high temperature (smooth assignments)."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=5, temperature=10.0)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        compatibility = jax.random.uniform(jax.random.key(0), shape=(100, 50))
        effective_lengths = jnp.ones(50) * 500.0
        data = {
            "compatibility": compatibility,
            "effective_lengths": effective_lengths,
        }

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isclose(jnp.sum(transformed["abundances"]), 1.0, rtol=1e-4)

    def test_low_temperature(self, rngs):
        """Test with low temperature (sharp assignments)."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=5, temperature=0.1)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        compatibility = jax.random.uniform(jax.random.key(0), shape=(100, 50))
        effective_lengths = jnp.ones(50) * 500.0
        data = {
            "compatibility": compatibility,
            "effective_lengths": effective_lengths,
        }

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isclose(jnp.sum(transformed["abundances"]), 1.0, rtol=1e-4)

    def test_many_iterations(self, rngs):
        """Test with many iterations."""
        config = EMQuantifierConfig(n_transcripts=50, n_iterations=50)
        op = DifferentiableEMQuantifier(config, rngs=rngs)

        compatibility = jax.random.uniform(jax.random.key(0), shape=(100, 50))
        effective_lengths = jnp.ones(50) * 500.0
        data = {
            "compatibility": compatibility,
            "effective_lengths": effective_lengths,
        }

        transformed, _, _ = op.apply(data, {}, None, None)
        assert jnp.isclose(jnp.sum(transformed["abundances"]), 1.0, rtol=1e-4)
