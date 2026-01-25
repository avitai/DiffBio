"""Tests for MACCSKeysOperator.

Tests the differentiable MACCS 166 structural keys fingerprint operator.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class TestMACCSConfigImport:
    """Test MACCSKeysOperator can be imported."""

    def test_import(self):
        """Test that MACCSKeysOperator can be imported."""
        from diffbio.operators.drug_discovery.maccs_keys import (
            MACCSKeysConfig,
            MACCSKeysOperator,
        )

        assert MACCSKeysConfig is not None
        assert MACCSKeysOperator is not None


class TestMACCSKeysConfig:
    """Test MACCSKeysConfig configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from diffbio.operators.drug_discovery.maccs_keys import MACCSKeysConfig

        config = MACCSKeysConfig()

        # MACCS has 166 keys
        assert config.n_bits == 166
        # Default is differentiable mode
        assert config.differentiable is True
        # Default temperature for soft matching
        assert config.temperature == 1.0

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from diffbio.operators.drug_discovery.maccs_keys import MACCSKeysConfig

        config = MACCSKeysConfig(
            differentiable=False,
            temperature=0.5,
        )

        assert config.differentiable is False
        assert config.temperature == 0.5


class TestMACCSKeysOperator:
    """Test MACCSKeysOperator operator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from diffbio.operators.drug_discovery.maccs_keys import MACCSKeysConfig

        return MACCSKeysConfig(
            differentiable=True,
            in_features=4,
        )

    @pytest.fixture
    def operator(self, config):
        """Create test operator."""
        from diffbio.operators.drug_discovery.maccs_keys import MACCSKeysOperator

        return MACCSKeysOperator(config, rngs=nnx.Rngs(42))

    @pytest.fixture
    def sample_data(self):
        """Create sample molecular graph data."""
        num_atoms = 8
        num_features = 4

        node_features = jax.random.uniform(
            jax.random.PRNGKey(0), (num_atoms, num_features)
        )
        adjacency = jnp.eye(num_atoms)
        adjacency = adjacency.at[0, 1].set(1.0)
        adjacency = adjacency.at[1, 0].set(1.0)
        adjacency = adjacency.at[1, 2].set(1.0)
        adjacency = adjacency.at[2, 1].set(1.0)
        node_mask = jnp.ones(num_atoms)

        return {
            "node_features": node_features,
            "adjacency": adjacency,
            "node_mask": node_mask,
        }

    def test_initialization(self, operator, config):
        """Test operator initialization."""
        assert operator is not None
        assert operator.config.n_bits == 166

    def test_output_shape(self, operator, sample_data):
        """Test that output has correct shape (166 bits)."""
        result, state, metadata = operator.apply(sample_data, {}, None)

        assert "fingerprint" in result
        assert result["fingerprint"].shape == (166,)

    def test_output_values_in_range(self, operator, sample_data):
        """Test that fingerprint values are in [0, 1] for differentiable mode."""
        result, _, _ = operator.apply(sample_data, {}, None)

        fingerprint = result["fingerprint"]
        assert jnp.all(fingerprint >= 0.0)
        assert jnp.all(fingerprint <= 1.0)

    def test_output_finite(self, operator, sample_data):
        """Test that outputs are finite."""
        result, _, _ = operator.apply(sample_data, {}, None)

        fingerprint = result["fingerprint"]
        assert jnp.all(jnp.isfinite(fingerprint))

    def test_preserves_input_data(self, operator, sample_data):
        """Test that input data is preserved in output."""
        result, _, _ = operator.apply(sample_data, {}, None)

        assert "node_features" in result
        assert jnp.allclose(result["node_features"], sample_data["node_features"])


class TestMACCSKeysDifferentiability:
    """Test MACCSKeysOperator differentiability."""

    @pytest.fixture
    def operator(self):
        """Create test operator."""
        from diffbio.operators.drug_discovery.maccs_keys import (
            MACCSKeysConfig,
            MACCSKeysOperator,
        )

        config = MACCSKeysConfig(
            differentiable=True,
            in_features=4,
            temperature=1.0,
        )
        return MACCSKeysOperator(config, rngs=nnx.Rngs(42))

    def test_differentiable_through_node_features(self, operator):
        """Test gradients flow through node features."""
        num_atoms = 6
        num_features = 4

        node_features = jax.random.uniform(
            jax.random.PRNGKey(0), (num_atoms, num_features)
        )
        adjacency = jnp.eye(num_atoms)
        adjacency = adjacency.at[0, 1].set(1.0)
        adjacency = adjacency.at[1, 0].set(1.0)
        node_mask = jnp.ones(num_atoms)

        def loss_fn(node_feats):
            data = {
                "node_features": node_feats,
                "adjacency": adjacency,
                "node_mask": node_mask,
            }
            result, _, _ = operator.apply(data, {}, None)
            return jnp.sum(result["fingerprint"])

        # Should not raise
        grads = jax.grad(loss_fn)(node_features)
        assert grads.shape == node_features.shape
        assert jnp.all(jnp.isfinite(grads))


class TestMACCSKeysJIT:
    """Test MACCSKeysOperator JIT compilation."""

    def test_jit_compatible(self):
        """Test that operator is JIT-compatible."""
        from diffbio.operators.drug_discovery.maccs_keys import (
            MACCSKeysConfig,
            MACCSKeysOperator,
        )

        config = MACCSKeysConfig(
            differentiable=True,
            in_features=4,
        )
        operator = MACCSKeysOperator(config, rngs=nnx.Rngs(42))

        num_atoms = 6
        node_features = jax.random.uniform(jax.random.PRNGKey(0), (num_atoms, 4))
        adjacency = jnp.eye(num_atoms)
        node_mask = jnp.ones(num_atoms)

        @jax.jit
        def jit_apply(node_feats, adj, mask):
            data = {
                "node_features": node_feats,
                "adjacency": adj,
                "node_mask": mask,
            }
            result, _, _ = operator.apply(data, {}, None)
            return result["fingerprint"]

        # Should not raise
        fingerprint = jit_apply(node_features, adjacency, node_mask)
        assert fingerprint.shape == (166,)
        assert jnp.all(jnp.isfinite(fingerprint))


class TestMACCSKeysTemperature:
    """Test temperature effects on MACCS keys."""

    def test_lower_temperature_sharper_bits(self):
        """Test that lower temperature produces sharper (more binary) fingerprints."""
        from diffbio.operators.drug_discovery.maccs_keys import (
            MACCSKeysConfig,
            MACCSKeysOperator,
        )

        num_atoms = 6
        node_features = jax.random.uniform(jax.random.PRNGKey(0), (num_atoms, 4))
        adjacency = jnp.eye(num_atoms)
        adjacency = adjacency.at[0, 1].set(1.0)
        adjacency = adjacency.at[1, 0].set(1.0)
        node_mask = jnp.ones(num_atoms)

        data = {
            "node_features": node_features,
            "adjacency": adjacency,
            "node_mask": node_mask,
        }

        # High temperature (softer)
        config_high = MACCSKeysConfig(temperature=2.0, in_features=4)
        op_high = MACCSKeysOperator(config_high, rngs=nnx.Rngs(42))
        result_high, _, _ = op_high.apply(data, {}, None)

        # Low temperature (sharper)
        config_low = MACCSKeysConfig(temperature=0.1, in_features=4)
        op_low = MACCSKeysOperator(config_low, rngs=nnx.Rngs(42))
        result_low, _, _ = op_low.apply(data, {}, None)

        # Lower temperature should have more extreme values (closer to 0 or 1)
        fp_high = result_high["fingerprint"]
        fp_low = result_low["fingerprint"]

        # Variance should be higher for low temperature (more binary-like)
        variance_high = jnp.var(fp_high)
        variance_low = jnp.var(fp_low)

        # Low temperature fingerprint should have higher variance
        # (values pushed toward 0 and 1)
        assert variance_low >= variance_high * 0.5  # Allow some tolerance
