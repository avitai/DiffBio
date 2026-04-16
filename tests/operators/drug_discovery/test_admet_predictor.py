"""Tests for ADMETPredictor operator.

Tests the multi-task ADMET property prediction operator following
the TDC ADMET benchmark with 22 endpoints.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class TestADMETConfigImport:
    """Test ADMETPredictor can be imported."""

    def test_import(self):
        """Test that ADMETPredictor can be imported."""
        from diffbio.operators.drug_discovery.admet_predictor import (
            ADMETConfig,
            ADMETPredictor,
        )

        assert ADMETConfig is not None
        assert ADMETPredictor is not None


class TestADMETConfig:
    """Test ADMETConfig configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from diffbio.operators.drug_discovery.admet_predictor import ADMETConfig

        config = ADMETConfig()

        # Default hidden dimension
        assert config.hidden_dim == 300
        # Default number of message passing steps
        assert config.num_message_passing_steps == 3
        # Default number of ADMET tasks (22 from TDC benchmark)
        assert config.num_tasks == 22
        # Default dropout
        assert config.dropout_rate == 0.0

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from diffbio.operators.drug_discovery.admet_predictor import ADMETConfig

        config = ADMETConfig(
            hidden_dim=256,
            num_message_passing_steps=4,
            num_tasks=10,
            dropout_rate=0.2,
        )

        assert config.hidden_dim == 256
        assert config.num_message_passing_steps == 4
        assert config.num_tasks == 10
        assert config.dropout_rate == 0.2

    def test_config_task_names(self):
        """Test that ADMET task names are available."""
        from diffbio.operators.drug_discovery.admet_predictor import (
            ADMET_TASK_NAMES,
        )

        # Should have 22 standard TDC ADMET tasks
        assert len(ADMET_TASK_NAMES) == 22
        # Check some key tasks exist
        assert "BBB_Martins" in ADMET_TASK_NAMES
        assert "CYP3A4_Veith" in ADMET_TASK_NAMES
        assert "Solubility_AqSolDB" in ADMET_TASK_NAMES
        assert "hERG" in ADMET_TASK_NAMES


class TestADMETPredictor:
    """Test ADMETPredictor operator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from diffbio.operators.drug_discovery.admet_predictor import ADMETConfig

        return ADMETConfig(
            hidden_dim=64,
            num_message_passing_steps=2,
            num_tasks=22,
            in_features=4,
        )

    @pytest.fixture
    def predictor(self, config):
        """Create test predictor."""
        from diffbio.operators.drug_discovery.admet_predictor import ADMETPredictor

        return ADMETPredictor(config, rngs=nnx.Rngs(42))

    @pytest.fixture
    def sample_data(self):
        """Create sample molecular graph data."""
        num_atoms = 10
        num_features = 4

        # Simple molecular graph
        node_features = jax.random.uniform(jax.random.PRNGKey(0), (num_atoms, num_features))
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

    def test_initialization(self, predictor, config):
        """Test predictor initialization."""
        assert predictor is not None
        assert predictor.config.hidden_dim == config.hidden_dim
        assert predictor.config.num_tasks == config.num_tasks
        assert predictor.ffn_backbone is not None
        assert len(predictor.ffn_backbone.layers) == config.ffn_num_layers - 1

    def test_output_shape(self, predictor, sample_data):
        """Test that output has correct shape."""
        result, state, metadata = predictor.apply(sample_data, {}, None)

        # Should have 22 ADMET predictions
        assert "predictions" in result
        assert result["predictions"].shape == (22,)

    def test_output_finite(self, predictor, sample_data):
        """Test that outputs are finite."""
        result, _, _ = predictor.apply(sample_data, {}, None)

        predictions = result["predictions"]
        assert jnp.all(jnp.isfinite(predictions))

    def test_task_specific_outputs(self, predictor, sample_data):
        """Test that task-specific outputs are available."""
        result, _, _ = predictor.apply(sample_data, {}, None)

        # Should have task-specific predictions
        assert "task_predictions" in result
        assert len(result["task_predictions"]) == 22

    def test_preserves_input_data(self, predictor, sample_data):
        """Test that input data is preserved in output."""
        result, _, _ = predictor.apply(sample_data, {}, None)

        assert "node_features" in result
        assert jnp.allclose(result["node_features"], sample_data["node_features"])


class TestADMETPredictorDifferentiability:
    """Test ADMETPredictor differentiability."""

    @pytest.fixture
    def predictor(self):
        """Create test predictor."""
        from diffbio.operators.drug_discovery.admet_predictor import (
            ADMETConfig,
            ADMETPredictor,
        )

        config = ADMETConfig(
            hidden_dim=32,
            num_message_passing_steps=2,
            num_tasks=5,
            in_features=4,
        )
        return ADMETPredictor(config, rngs=nnx.Rngs(42))

    def test_differentiable_through_node_features(self, predictor):
        """Test gradients flow through node features."""
        num_atoms = 8
        num_features = 4

        node_features = jax.random.uniform(jax.random.PRNGKey(0), (num_atoms, num_features))
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
            result, _, _ = predictor.apply(data, {}, None)
            return jnp.sum(result["predictions"])

        # Should not raise
        grads = jax.grad(loss_fn)(node_features)
        assert grads.shape == node_features.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_gradients_reach_shared_ffn_backbone(self, predictor):
        """Gradients should reach the first shared FFN layer."""
        num_atoms = 8
        num_features = 4

        node_features = jax.random.uniform(jax.random.PRNGKey(0), (num_atoms, num_features))
        adjacency = jnp.eye(num_atoms)
        adjacency = adjacency.at[0, 1].set(1.0)
        adjacency = adjacency.at[1, 0].set(1.0)
        node_mask = jnp.ones(num_atoms)
        data = {
            "node_features": node_features,
            "adjacency": adjacency,
            "node_mask": node_mask,
        }

        @nnx.value_and_grad
        def loss_fn(model):
            result, _, _ = model.apply(data, {}, None)
            return jnp.sum(result["predictions"])

        _, grads = loss_fn(predictor)
        assert hasattr(grads, "ffn_backbone")
        assert grads.ffn_backbone is not None
        assert jnp.any(grads.ffn_backbone.layers[0].kernel[...] != 0.0)


class TestADMETPredictorJIT:
    """Test ADMETPredictor JIT compilation."""

    def test_jit_compatible(self):
        """Test that predictor is JIT-compatible."""
        from diffbio.operators.drug_discovery.admet_predictor import (
            ADMETConfig,
            ADMETPredictor,
        )

        config = ADMETConfig(
            hidden_dim=32,
            num_message_passing_steps=2,
            num_tasks=5,
            in_features=4,
        )
        predictor = ADMETPredictor(config, rngs=nnx.Rngs(42))

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
            result, _, _ = predictor.apply(data, {}, None)
            return result["predictions"]

        # Should not raise
        predictions = jit_apply(node_features, adjacency, node_mask)
        assert predictions.shape == (5,)
        assert jnp.all(jnp.isfinite(predictions))


class TestADMETTaskTypes:
    """Test ADMET task type handling."""

    def test_task_types_defined(self):
        """Test that task types (regression/classification) are defined."""
        from diffbio.operators.drug_discovery.admet_predictor import ADMET_TASK_TYPES

        # Should have type info for all 22 tasks
        assert len(ADMET_TASK_TYPES) == 22

        # Check some known task types
        assert ADMET_TASK_TYPES["BBB_Martins"] == "classification"
        assert ADMET_TASK_TYPES["Solubility_AqSolDB"] == "regression"
        assert ADMET_TASK_TYPES["CYP3A4_Veith"] == "classification"
        assert ADMET_TASK_TYPES["Half_Life_Obach"] == "regression"

    def test_classification_output_activation(self):
        """Test classification tasks get sigmoid activation."""
        from diffbio.operators.drug_discovery.admet_predictor import (
            ADMETConfig,
            ADMETPredictor,
        )

        config = ADMETConfig(
            hidden_dim=32,
            num_message_passing_steps=2,
            num_tasks=22,
            in_features=4,
            apply_task_activations=True,
        )
        predictor = ADMETPredictor(config, rngs=nnx.Rngs(42))

        num_atoms = 6
        node_features = jax.random.uniform(jax.random.PRNGKey(0), (num_atoms, 4))
        adjacency = jnp.eye(num_atoms)
        node_mask = jnp.ones(num_atoms)

        data = {
            "node_features": node_features,
            "adjacency": adjacency,
            "node_mask": node_mask,
        }
        result, _, _ = predictor.apply(data, {}, None)

        # Classification predictions should be in [0, 1]
        task_preds = result["task_predictions"]

        # BBB_Martins is classification (index based on ADMET_TASK_NAMES)
        from diffbio.operators.drug_discovery.admet_predictor import ADMET_TASK_NAMES

        bbb_idx = ADMET_TASK_NAMES.index("BBB_Martins")
        bbb_pred = task_preds[bbb_idx]
        assert 0.0 <= float(bbb_pred) <= 1.0
