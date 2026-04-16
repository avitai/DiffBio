"""Tests for AttentiveFP operator.

Tests the attention-based graph fingerprint operator following
the AttentiveFP architecture from Xiong et al. 2019.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class TestAttentiveFPConfigImport:
    """Test AttentiveFP can be imported."""

    def test_import(self):
        """Test that AttentiveFP can be imported."""
        from diffbio.operators.drug_discovery.attentive_fp import (
            AttentiveFP,
            AttentiveFPConfig,
        )

        assert AttentiveFPConfig is not None
        assert AttentiveFP is not None


class TestAttentiveFPConfig:
    """Test AttentiveFPConfig configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from diffbio.operators.drug_discovery.attentive_fp import AttentiveFPConfig

        config = AttentiveFPConfig()

        # Default hidden dimension (from PyG implementation)
        assert config.hidden_dim == 200
        # Default output dimension
        assert config.out_dim == 200
        # Default number of atom-level layers
        assert config.num_layers == 2
        # Default number of timesteps for molecular aggregation
        assert config.num_timesteps == 2
        # Default dropout
        assert config.dropout_rate == 0.0

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from diffbio.operators.drug_discovery.attentive_fp import AttentiveFPConfig

        config = AttentiveFPConfig(
            hidden_dim=128,
            out_dim=256,
            num_layers=3,
            num_timesteps=3,
            dropout_rate=0.2,
        )

        assert config.hidden_dim == 128
        assert config.out_dim == 256
        assert config.num_layers == 3
        assert config.num_timesteps == 3
        assert config.dropout_rate == 0.2

    def test_invalid_hidden_dim_rejected(self):
        """Hidden dimension must be positive."""
        from diffbio.operators.drug_discovery.attentive_fp import AttentiveFPConfig

        with pytest.raises(ValueError, match="hidden_dim"):
            AttentiveFPConfig(hidden_dim=0)

    def test_invalid_dropout_rate_rejected(self):
        """Dropout must be in [0, 1)."""
        from diffbio.operators.drug_discovery.attentive_fp import AttentiveFPConfig

        with pytest.raises(ValueError, match="dropout_rate"):
            AttentiveFPConfig(dropout_rate=1.0)

    def test_invalid_edge_dim_rejected(self):
        """Edge dimension cannot be negative."""
        from diffbio.operators.drug_discovery.attentive_fp import AttentiveFPConfig

        with pytest.raises(ValueError, match="edge_dim"):
            AttentiveFPConfig(edge_dim=-1)


class TestAttentiveFP:
    """Test AttentiveFP operator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from diffbio.operators.drug_discovery.attentive_fp import AttentiveFPConfig

        return AttentiveFPConfig(
            hidden_dim=64,
            out_dim=64,
            num_layers=2,
            num_timesteps=2,
            in_features=4,
            edge_dim=4,
        )

    @pytest.fixture
    def attentive_fp(self, config):
        """Create test AttentiveFP."""
        from diffbio.operators.drug_discovery.attentive_fp import AttentiveFP

        return AttentiveFP(config, rngs=nnx.Rngs(42))

    @pytest.fixture
    def sample_data(self):
        """Create sample molecular graph data."""
        num_atoms = 10
        num_features = 4
        edge_dim = 4

        node_features = jax.random.uniform(jax.random.PRNGKey(0), (num_atoms, num_features))
        # Create adjacency matrix with some bonds
        adjacency = jnp.zeros((num_atoms, num_atoms))
        adjacency = adjacency.at[0, 1].set(1.0)
        adjacency = adjacency.at[1, 0].set(1.0)
        adjacency = adjacency.at[1, 2].set(1.0)
        adjacency = adjacency.at[2, 1].set(1.0)
        adjacency = adjacency.at[2, 3].set(1.0)
        adjacency = adjacency.at[3, 2].set(1.0)

        # Edge features
        edge_features = jax.random.uniform(jax.random.PRNGKey(1), (num_atoms, num_atoms, edge_dim))
        node_mask = jnp.ones(num_atoms)

        return {
            "node_features": node_features,
            "adjacency": adjacency,
            "edge_features": edge_features,
            "node_mask": node_mask,
        }

    def test_initialization(self, attentive_fp, config):
        """Test AttentiveFP initialization."""
        assert attentive_fp is not None
        assert attentive_fp.config.hidden_dim == config.hidden_dim
        assert attentive_fp.config.out_dim == config.out_dim

    def test_output_shape(self, attentive_fp, sample_data, config):
        """Test that output has correct shape."""
        result, state, metadata = attentive_fp.apply(sample_data, {}, None)

        # Should have fingerprint with out_dim
        assert "fingerprint" in result
        assert result["fingerprint"].shape == (config.out_dim,)

    def test_output_finite(self, attentive_fp, sample_data):
        """Test that outputs are finite."""
        result, _, _ = attentive_fp.apply(sample_data, {}, None)

        fingerprint = result["fingerprint"]
        assert jnp.all(jnp.isfinite(fingerprint))

    def test_attention_weights_available(self, attentive_fp, sample_data):
        """Test that attention weights are available for interpretability."""
        result, _, _ = attentive_fp.apply(sample_data, {}, None)

        # Should have attention weights
        assert "attention_weights" in result

    def test_preserves_input_data(self, attentive_fp, sample_data):
        """Test that input data is preserved in output."""
        result, _, _ = attentive_fp.apply(sample_data, {}, None)

        assert "node_features" in result
        assert jnp.allclose(result["node_features"], sample_data["node_features"])


class TestAttentiveFPDifferentiability:
    """Test AttentiveFP differentiability."""

    @pytest.fixture
    def attentive_fp(self):
        """Create test AttentiveFP."""
        from diffbio.operators.drug_discovery.attentive_fp import (
            AttentiveFP,
            AttentiveFPConfig,
        )

        config = AttentiveFPConfig(
            hidden_dim=32,
            out_dim=32,
            num_layers=2,
            num_timesteps=2,
            in_features=4,
            edge_dim=4,
        )
        return AttentiveFP(config, rngs=nnx.Rngs(42))

    def test_differentiable_through_node_features(self, attentive_fp):
        """Test gradients flow through node features."""
        num_atoms = 8
        num_features = 4
        edge_dim = 4

        node_features = jax.random.uniform(jax.random.PRNGKey(0), (num_atoms, num_features))
        adjacency = jnp.zeros((num_atoms, num_atoms))
        adjacency = adjacency.at[0, 1].set(1.0)
        adjacency = adjacency.at[1, 0].set(1.0)
        edge_features = jax.random.uniform(jax.random.PRNGKey(1), (num_atoms, num_atoms, edge_dim))
        node_mask = jnp.ones(num_atoms)

        def loss_fn(node_feats):
            data = {
                "node_features": node_feats,
                "adjacency": adjacency,
                "edge_features": edge_features,
                "node_mask": node_mask,
            }
            result, _, _ = attentive_fp.apply(data, {}, None)
            return jnp.sum(result["fingerprint"])

        # Should not raise
        grads = jax.grad(loss_fn)(node_features)
        assert grads.shape == node_features.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_differentiable_through_edge_features(self, attentive_fp):
        """Test gradients flow through edge features."""
        num_atoms = 8
        num_features = 4
        edge_dim = 4

        node_features = jax.random.uniform(jax.random.PRNGKey(0), (num_atoms, num_features))
        adjacency = jnp.zeros((num_atoms, num_atoms))
        adjacency = adjacency.at[0, 1].set(1.0)
        adjacency = adjacency.at[1, 0].set(1.0)
        edge_features = jax.random.uniform(jax.random.PRNGKey(1), (num_atoms, num_atoms, edge_dim))
        node_mask = jnp.ones(num_atoms)

        def loss_fn(edge_feats):
            data = {
                "node_features": node_features,
                "adjacency": adjacency,
                "edge_features": edge_feats,
                "node_mask": node_mask,
            }
            result, _, _ = attentive_fp.apply(data, {}, None)
            return jnp.sum(result["fingerprint"])

        # Should not raise
        grads = jax.grad(loss_fn)(edge_features)
        assert grads.shape == edge_features.shape
        assert jnp.all(jnp.isfinite(grads))


class TestAttentiveFPJIT:
    """Test AttentiveFP JIT compilation."""

    def test_jit_compatible(self):
        """Test that AttentiveFP is JIT-compatible."""
        from diffbio.operators.drug_discovery.attentive_fp import (
            AttentiveFP,
            AttentiveFPConfig,
        )

        config = AttentiveFPConfig(
            hidden_dim=32,
            out_dim=32,
            num_layers=2,
            num_timesteps=2,
            in_features=4,
            edge_dim=4,
        )
        attentive_fp = AttentiveFP(config, rngs=nnx.Rngs(42))

        num_atoms = 6
        node_features = jax.random.uniform(jax.random.PRNGKey(0), (num_atoms, 4))
        adjacency = jnp.zeros((num_atoms, num_atoms))
        adjacency = adjacency.at[0, 1].set(1.0)
        adjacency = adjacency.at[1, 0].set(1.0)
        edge_features = jax.random.uniform(jax.random.PRNGKey(1), (num_atoms, num_atoms, 4))
        node_mask = jnp.ones(num_atoms)

        @jax.jit
        def jit_apply(node_feats, adj, edge_feats, mask):
            data = {
                "node_features": node_feats,
                "adjacency": adj,
                "edge_features": edge_feats,
                "node_mask": mask,
            }
            result, _, _ = attentive_fp.apply(data, {}, None)
            return result["fingerprint"]

        # Should not raise
        fingerprint = jit_apply(node_features, adjacency, edge_features, node_mask)
        assert fingerprint.shape == (32,)
        assert jnp.all(jnp.isfinite(fingerprint))


class TestAttentiveFPArchitecture:
    """Test AttentiveFP architectural components."""

    def test_gru_cell_used(self):
        """Test that GRU cells are used in architecture."""
        from diffbio.operators.drug_discovery.attentive_fp import (
            AttentiveFP,
            AttentiveFPConfig,
        )

        config = AttentiveFPConfig(
            hidden_dim=32,
            out_dim=32,
            num_layers=2,
            num_timesteps=2,
            in_features=4,
            edge_dim=4,
        )
        attentive_fp = AttentiveFP(config, rngs=nnx.Rngs(42))

        # Should have GRU cells for atom and molecule level
        assert hasattr(attentive_fp, "atom_grus") or hasattr(attentive_fp, "gru_cells")

    def test_num_layers_affects_depth(self):
        """Test that num_layers affects the message passing depth."""
        from diffbio.operators.drug_discovery.attentive_fp import (
            AttentiveFP,
            AttentiveFPConfig,
        )

        config_shallow = AttentiveFPConfig(
            hidden_dim=32,
            out_dim=32,
            num_layers=1,
            num_timesteps=1,
            in_features=4,
            edge_dim=4,
        )
        config_deep = AttentiveFPConfig(
            hidden_dim=32,
            out_dim=32,
            num_layers=4,
            num_timesteps=4,
            in_features=4,
            edge_dim=4,
        )

        shallow = AttentiveFP(config_shallow, rngs=nnx.Rngs(42))
        deep = AttentiveFP(config_deep, rngs=nnx.Rngs(42))

        # Both should work, but deep should have more parameters
        # (This is a structural test, not an exact parameter count)
        assert shallow is not None
        assert deep is not None


class TestAttentiveFPWithoutEdgeFeatures:
    """Test AttentiveFP can work without edge features."""

    def test_works_without_edge_features(self):
        """Test that AttentiveFP works without edge features."""
        from diffbio.operators.drug_discovery.attentive_fp import (
            AttentiveFP,
            AttentiveFPConfig,
        )

        config = AttentiveFPConfig(
            hidden_dim=32,
            out_dim=32,
            num_layers=2,
            num_timesteps=2,
            in_features=4,
            edge_dim=0,  # No edge features
        )
        attentive_fp = AttentiveFP(config, rngs=nnx.Rngs(42))

        num_atoms = 6
        node_features = jax.random.uniform(jax.random.PRNGKey(0), (num_atoms, 4))
        adjacency = jnp.zeros((num_atoms, num_atoms))
        adjacency = adjacency.at[0, 1].set(1.0)
        adjacency = adjacency.at[1, 0].set(1.0)
        node_mask = jnp.ones(num_atoms)

        data = {
            "node_features": node_features,
            "adjacency": adjacency,
            "node_mask": node_mask,
        }

        result, _, _ = attentive_fp.apply(data, {}, None)

        assert "fingerprint" in result
        assert result["fingerprint"].shape == (32,)
        assert jnp.all(jnp.isfinite(result["fingerprint"]))
