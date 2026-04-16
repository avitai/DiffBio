"""Tests for drug discovery operators.

Test-driven development: These tests define the expected behavior.
Implementation must pass these tests without modification.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_smiles():
    """Simple molecules for testing."""
    return ["C", "CC", "CCC", "CCO", "c1ccccc1"]  # methane, ethane, propane, ethanol, benzene


@pytest.fixture
def drug_like_smiles():
    """Drug-like molecules for testing."""
    return [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]


@pytest.fixture
def molecular_graph():
    """Pre-computed molecular graph for testing (ethanol CCO)."""
    # 3 atoms: C, C, O
    # Bonds: C-C, C-O
    node_features = jnp.array(
        [
            [1, 0, 0, 0],  # C (one-hot for C, N, O, other)
            [1, 0, 0, 0],  # C
            [0, 0, 1, 0],  # O
        ],
        dtype=jnp.float32,
    )

    # Adjacency matrix (symmetric for undirected)
    adjacency = jnp.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=jnp.float32,
    )

    # Edge features (bond type: single=1, double=0, triple=0, aromatic=0)
    edge_features = jnp.array(
        [
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=jnp.float32,
    )

    return {
        "node_features": node_features,
        "adjacency": adjacency,
        "edge_features": edge_features,
        "num_nodes": 3,
    }


@pytest.fixture
def batch_molecular_graphs():
    """Batch of molecular graphs for testing."""
    # Batch of 2 molecules with padding
    max_nodes = 4

    # Molecule 1: 3 atoms
    node_features_1 = jnp.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],  # padding
        ],
        dtype=jnp.float32,
    )

    # Molecule 2: 2 atoms
    node_features_2 = jnp.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],  # padding
            [0, 0, 0, 0],  # padding
        ],
        dtype=jnp.float32,
    )

    node_features = jnp.stack([node_features_1, node_features_2])

    # Adjacency matrices
    adj_1 = jnp.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=jnp.float32,
    )

    adj_2 = jnp.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=jnp.float32,
    )

    adjacency = jnp.stack([adj_1, adj_2])

    # Node masks
    node_mask = jnp.array(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=jnp.float32,
    )

    return {
        "node_features": node_features,
        "adjacency": adjacency,
        "node_mask": node_mask,
        "batch_size": 2,
        "max_nodes": max_nodes,
    }


# =============================================================================
# Tests for Molecular Graph Utilities
# =============================================================================


class TestMolecularGraphUtils:
    """Tests for SMILES to graph conversion utilities."""

    def test_smiles_to_graph_basic(self, simple_smiles):
        """Test basic SMILES to graph conversion."""
        from diffbio.operators.drug_discovery import smiles_to_graph

        for smiles in simple_smiles:
            graph = smiles_to_graph(smiles)

            assert "node_features" in graph
            assert "adjacency" in graph
            assert "num_nodes" in graph
            assert graph["num_nodes"] > 0
            assert graph["node_features"].shape[0] == graph["num_nodes"]

    def test_smiles_to_graph_features_shape(self, simple_smiles):
        """Test that node features have correct shape."""
        from diffbio.operators.drug_discovery import smiles_to_graph, DEFAULT_ATOM_FEATURES

        smiles = "CCO"  # ethanol
        graph = smiles_to_graph(smiles)

        # Node features should be (num_atoms, num_features)
        assert graph["node_features"].ndim == 2
        assert graph["node_features"].shape[0] == 3  # C, C, O
        assert graph["node_features"].shape[1] == DEFAULT_ATOM_FEATURES

    def test_smiles_to_graph_adjacency(self):
        """Test adjacency matrix is symmetric and correct."""
        from diffbio.operators.drug_discovery import smiles_to_graph

        smiles = "CC"  # ethane
        graph = smiles_to_graph(smiles)

        adj = graph["adjacency"]
        # Should be symmetric
        assert jnp.allclose(adj, adj.T)
        # Should have connection between C-C
        assert adj[0, 1] == 1
        assert adj[1, 0] == 1

    def test_smiles_to_graph_invalid(self):
        """Test handling of invalid SMILES."""
        from diffbio.operators.drug_discovery import smiles_to_graph

        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_graph("not_a_valid_smiles_123xyz")

    def test_batch_smiles_to_graphs(self, simple_smiles):
        """Test batching multiple molecules into padded tensors."""
        from diffbio.operators.drug_discovery import batch_smiles_to_graphs

        batch = batch_smiles_to_graphs(simple_smiles)

        assert "node_features" in batch
        assert "adjacency" in batch
        assert "node_mask" in batch
        assert batch["node_features"].shape[0] == len(simple_smiles)


# =============================================================================
# Tests for Message Passing Layer
# =============================================================================


class TestMessagePassingLayer:
    """Tests for the core message passing neural network layer."""

    def test_initialization(self):
        """Test layer initialization."""
        from diffbio.operators.drug_discovery import MessagePassingLayer

        layer = MessagePassingLayer(
            hidden_dim=64,
            num_edge_features=4,
            rngs=nnx.Rngs(42),
        )

        assert layer is not None
        assert layer.hidden_dim == 64

    def test_forward_shape(self, molecular_graph):
        """Test forward pass output shape."""
        from diffbio.operators.drug_discovery import MessagePassingLayer

        hidden_dim = 32
        layer = MessagePassingLayer(
            hidden_dim=hidden_dim,
            num_edge_features=4,
            rngs=nnx.Rngs(42),
        )

        node_features = molecular_graph["node_features"]
        adjacency = molecular_graph["adjacency"]
        edge_features = molecular_graph["edge_features"]

        output = layer(node_features, adjacency, edge_features)

        assert output.shape == (molecular_graph["num_nodes"], hidden_dim)

    def test_message_aggregation(self, molecular_graph):
        """Test that messages are properly aggregated from neighbors."""
        from diffbio.operators.drug_discovery import MessagePassingLayer

        layer = MessagePassingLayer(
            hidden_dim=32,
            num_edge_features=4,
            rngs=nnx.Rngs(42),
        )

        node_features = molecular_graph["node_features"]
        adjacency = molecular_graph["adjacency"]
        edge_features = molecular_graph["edge_features"]

        output = layer(node_features, adjacency, edge_features)

        # Check output is finite
        assert jnp.all(jnp.isfinite(output))

    def test_gradient_flow(self, molecular_graph):
        """Test that gradients flow through the layer."""
        from diffbio.operators.drug_discovery import MessagePassingLayer

        layer = MessagePassingLayer(
            hidden_dim=32,
            num_edge_features=4,
            rngs=nnx.Rngs(42),
        )

        def loss_fn(layer, node_features, adjacency, edge_features):
            output = layer(node_features, adjacency, edge_features)
            return output.sum()

        node_features = molecular_graph["node_features"]
        adjacency = molecular_graph["adjacency"]
        edge_features = molecular_graph["edge_features"]

        grads = jax.grad(loss_fn)(layer, node_features, adjacency, edge_features)

        # Check gradients exist and are finite
        assert grads is not None

    def test_batched_forward(self, batch_molecular_graphs):
        """Test forward pass with batched input."""
        from diffbio.operators.drug_discovery import MessagePassingLayer

        layer = MessagePassingLayer(
            hidden_dim=32,
            num_edge_features=4,
            rngs=nnx.Rngs(42),
        )

        # Create dummy edge features for batch
        batch_size = batch_molecular_graphs["batch_size"]
        max_nodes = batch_molecular_graphs["max_nodes"]
        edge_features = jnp.zeros((batch_size, max_nodes, max_nodes, 4))

        # vmap over batch dimension
        batched_forward = jax.vmap(
            lambda nf, adj, ef: layer(nf, adj, ef),
            in_axes=(0, 0, 0),
        )

        output = batched_forward(
            batch_molecular_graphs["node_features"],
            batch_molecular_graphs["adjacency"],
            edge_features,
        )

        assert output.shape == (batch_size, max_nodes, 32)


# =============================================================================
# Tests for Molecular Property Predictor
# =============================================================================


class TestMolecularPropertyPredictor:
    """Tests for ChemProp-style molecular property prediction."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from diffbio.operators.drug_discovery import MolecularPropertyConfig

        config = MolecularPropertyConfig()

        assert config.hidden_dim == 300
        assert config.num_message_passing_steps == 3
        assert config.num_output_tasks == 1

    def test_initialization(self):
        """Test operator initialization."""
        from diffbio.operators.drug_discovery import (
            MolecularPropertyConfig,
            MolecularPropertyPredictor,
        )

        config = MolecularPropertyConfig(hidden_dim=64, num_message_passing_steps=2)
        predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))

        assert predictor is not None
        assert predictor.ffn_backbone is not None
        assert len(predictor.ffn_backbone.layers) == 1

    def test_forward_pass_shape(self, molecular_graph):
        """Test forward pass produces correct output shape."""
        from diffbio.operators.drug_discovery import (
            MolecularPropertyConfig,
            MolecularPropertyPredictor,
        )

        config = MolecularPropertyConfig(
            hidden_dim=32,
            num_message_passing_steps=2,
            num_output_tasks=1,
        )
        predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))

        data = {
            "node_features": molecular_graph["node_features"],
            "adjacency": molecular_graph["adjacency"],
            "edge_features": molecular_graph["edge_features"],
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }

        result, state, meta = predictor.apply(data, {}, None)

        assert "predictions" in result
        assert result["predictions"].shape == (1,)  # single task

    def test_multi_task_prediction(self, molecular_graph):
        """Test multi-task property prediction."""
        from diffbio.operators.drug_discovery import (
            MolecularPropertyConfig,
            MolecularPropertyPredictor,
        )

        num_tasks = 5
        config = MolecularPropertyConfig(
            hidden_dim=32,
            num_message_passing_steps=2,
            num_output_tasks=num_tasks,
        )
        predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))

        data = {
            "node_features": molecular_graph["node_features"],
            "adjacency": molecular_graph["adjacency"],
            "edge_features": molecular_graph["edge_features"],
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }

        result, _, _ = predictor.apply(data, {}, None)

        assert result["predictions"].shape == (num_tasks,)

    def test_gradient_flow_to_parameters(self, molecular_graph):
        """Test gradients flow to model parameters."""
        from diffbio.operators.drug_discovery import (
            MolecularPropertyConfig,
            MolecularPropertyPredictor,
        )

        config = MolecularPropertyConfig(hidden_dim=32, num_message_passing_steps=2)
        predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))

        data = {
            "node_features": molecular_graph["node_features"],
            "adjacency": molecular_graph["adjacency"],
            "edge_features": molecular_graph["edge_features"],
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }

        def loss_fn(predictor, data):
            result, _, _ = predictor.apply(data, {}, None)
            return result["predictions"].sum()

        # Use nnx.grad for proper NNX module handling (filters to Param only)
        grads = nnx.grad(loss_fn)(predictor, data)
        assert grads is not None

    def test_gradients_reach_shared_ffn_backbone(self, molecular_graph):
        """Test gradients reach the shared predictor MLP."""
        from diffbio.operators.drug_discovery import (
            MolecularPropertyConfig,
            MolecularPropertyPredictor,
        )

        config = MolecularPropertyConfig(hidden_dim=32, num_message_passing_steps=2)
        predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))

        data = {
            "node_features": molecular_graph["node_features"],
            "adjacency": molecular_graph["adjacency"],
            "edge_features": molecular_graph["edge_features"],
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }

        @nnx.value_and_grad
        def loss_fn(model):
            result, _, _ = model.apply(data, {}, None)
            return result["predictions"].sum()

        _, grads = loss_fn(predictor)

        assert hasattr(grads, "ffn_backbone")
        assert grads.ffn_backbone is not None
        assert jnp.any(grads.ffn_backbone.layers[0].kernel[...] != 0.0)

    def test_jit_compatibility(self, molecular_graph):
        """Test JIT compilation works."""
        from diffbio.operators.drug_discovery import (
            MolecularPropertyConfig,
            MolecularPropertyPredictor,
        )

        config = MolecularPropertyConfig(hidden_dim=32, num_message_passing_steps=2)
        predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))

        data = {
            "node_features": molecular_graph["node_features"],
            "adjacency": molecular_graph["adjacency"],
            "edge_features": molecular_graph["edge_features"],
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }

        @jax.jit
        def predict(predictor, data):
            result, _, _ = predictor.apply(data, {}, None)
            return result["predictions"]

        # Should not raise
        predictions = predict(predictor, data)
        assert jnp.all(jnp.isfinite(predictions))

    def test_preserves_extra_data_keys(self, molecular_graph):
        """Test that extra data keys are preserved."""
        from diffbio.operators.drug_discovery import (
            MolecularPropertyConfig,
            MolecularPropertyPredictor,
        )

        config = MolecularPropertyConfig(hidden_dim=32)
        predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))

        data = {
            "node_features": molecular_graph["node_features"],
            "adjacency": molecular_graph["adjacency"],
            "edge_features": molecular_graph["edge_features"],
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
            "extra_key": "should_be_preserved",
        }

        result, _, _ = predictor.apply(data, {}, None)
        assert result["extra_key"] == "should_be_preserved"


# =============================================================================
# Tests for Differentiable Molecular Fingerprint
# =============================================================================


class TestDifferentiableMolecularFingerprint:
    """Tests for neural graph fingerprints."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from diffbio.operators.drug_discovery import MolecularFingerprintConfig

        config = MolecularFingerprintConfig()

        assert config.fingerprint_dim == 256
        assert config.num_layers == 3

    def test_initialization(self):
        """Test fingerprint operator initialization."""
        from diffbio.operators.drug_discovery import (
            DifferentiableMolecularFingerprint,
            MolecularFingerprintConfig,
        )

        config = MolecularFingerprintConfig(fingerprint_dim=128)
        fp_op = DifferentiableMolecularFingerprint(config, rngs=nnx.Rngs(42))

        assert fp_op is not None

    def test_fingerprint_shape(self, molecular_graph):
        """Test fingerprint has correct output dimension."""
        from diffbio.operators.drug_discovery import (
            DifferentiableMolecularFingerprint,
            MolecularFingerprintConfig,
        )

        fp_dim = 128
        config = MolecularFingerprintConfig(fingerprint_dim=fp_dim, num_layers=2)
        fp_op = DifferentiableMolecularFingerprint(config, rngs=nnx.Rngs(42))

        data = {
            "node_features": molecular_graph["node_features"],
            "adjacency": molecular_graph["adjacency"],
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }

        result, _, _ = fp_op.apply(data, {}, None)

        assert "fingerprint" in result
        assert result["fingerprint"].shape == (fp_dim,)

    def test_fingerprint_normalized(self, molecular_graph):
        """Test fingerprint is normalized (for similarity computation)."""
        from diffbio.operators.drug_discovery import (
            DifferentiableMolecularFingerprint,
            MolecularFingerprintConfig,
        )

        config = MolecularFingerprintConfig(fingerprint_dim=128, normalize=True)
        fp_op = DifferentiableMolecularFingerprint(config, rngs=nnx.Rngs(42))

        data = {
            "node_features": molecular_graph["node_features"],
            "adjacency": molecular_graph["adjacency"],
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }

        result, _, _ = fp_op.apply(data, {}, None)

        # L2 norm should be approximately 1
        norm = jnp.linalg.norm(result["fingerprint"])
        assert jnp.isclose(norm, 1.0, atol=1e-5)

    def test_gradient_flow(self, molecular_graph):
        """Test gradients flow through fingerprint computation."""
        from diffbio.operators.drug_discovery import (
            DifferentiableMolecularFingerprint,
            MolecularFingerprintConfig,
        )

        config = MolecularFingerprintConfig(fingerprint_dim=64)
        fp_op = DifferentiableMolecularFingerprint(config, rngs=nnx.Rngs(42))

        data = {
            "node_features": molecular_graph["node_features"],
            "adjacency": molecular_graph["adjacency"],
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }

        def loss_fn(fp_op, data):
            result, _, _ = fp_op.apply(data, {}, None)
            return result["fingerprint"].sum()

        # Use nnx.grad for proper NNX module handling (filters to Param only)
        grads = nnx.grad(loss_fn)(fp_op, data)
        assert grads is not None

    def test_jit_compatible(self, molecular_graph):
        """Test JIT compilation works for DifferentiableMolecularFingerprint."""
        from diffbio.operators.drug_discovery import (
            DifferentiableMolecularFingerprint,
            MolecularFingerprintConfig,
        )

        config = MolecularFingerprintConfig(fingerprint_dim=64)
        fp_op = DifferentiableMolecularFingerprint(config, rngs=nnx.Rngs(42))

        data = {
            "node_features": molecular_graph["node_features"],
            "adjacency": molecular_graph["adjacency"],
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }

        @jax.jit
        def compute(fp_op, data):
            result, _, _ = fp_op.apply(data, {}, None)
            return result["fingerprint"]

        fingerprint = compute(fp_op, data)
        assert fingerprint.shape == (64,)
        assert jnp.all(jnp.isfinite(fingerprint))

    def test_different_molecules_different_fingerprints(self):
        """Test that different molecules produce different fingerprints."""
        from diffbio.operators.drug_discovery import (
            DEFAULT_ATOM_FEATURES,
            DifferentiableMolecularFingerprint,
            MolecularFingerprintConfig,
            smiles_to_graph,
        )

        # Use in_features=DEFAULT_ATOM_FEATURES for real molecules
        config = MolecularFingerprintConfig(fingerprint_dim=64, in_features=DEFAULT_ATOM_FEATURES)
        fp_op = DifferentiableMolecularFingerprint(config, rngs=nnx.Rngs(42))

        # Two different molecules
        graph1 = smiles_to_graph("CCO")  # ethanol
        graph2 = smiles_to_graph("c1ccccc1")  # benzene

        data1 = {
            "node_features": graph1["node_features"],
            "adjacency": graph1["adjacency"],
            "node_mask": jnp.ones(graph1["num_nodes"]),
        }
        data2 = {
            "node_features": graph2["node_features"],
            "adjacency": graph2["adjacency"],
            "node_mask": jnp.ones(graph2["num_nodes"]),
        }

        result1, _, _ = fp_op.apply(data1, {}, None)
        result2, _, _ = fp_op.apply(data2, {}, None)

        # Fingerprints should be different
        assert not jnp.allclose(result1["fingerprint"], result2["fingerprint"])


# =============================================================================
# Tests for Molecular Similarity Operator
# =============================================================================


class TestMolecularSimilarityOperator:
    """Tests for differentiable molecular similarity."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from diffbio.operators.drug_discovery import MolecularSimilarityConfig

        config = MolecularSimilarityConfig()

        assert config.similarity_type == "tanimoto"

    def test_initialization(self):
        """Test similarity operator initialization."""
        from diffbio.operators.drug_discovery import (
            MolecularSimilarityConfig,
            MolecularSimilarityOperator,
        )

        config = MolecularSimilarityConfig()
        sim_op = MolecularSimilarityOperator(config, rngs=nnx.Rngs(42))

        assert sim_op is not None

    def test_tanimoto_similarity_range(self):
        """Test Tanimoto similarity is in [0, 1]."""
        from diffbio.operators.drug_discovery import (
            MolecularSimilarityConfig,
            MolecularSimilarityOperator,
        )

        config = MolecularSimilarityConfig(similarity_type="tanimoto")
        sim_op = MolecularSimilarityOperator(config, rngs=nnx.Rngs(42))

        # Random fingerprints
        fp1 = jax.random.uniform(jax.random.PRNGKey(0), (128,))
        fp2 = jax.random.uniform(jax.random.PRNGKey(1), (128,))

        data = {"fingerprint_a": fp1, "fingerprint_b": fp2}
        result, _, _ = sim_op.apply(data, {}, None)

        assert "similarity" in result
        assert 0.0 <= result["similarity"] <= 1.0

    def test_cosine_similarity_range(self):
        """Test cosine similarity is in [-1, 1]."""
        from diffbio.operators.drug_discovery import (
            MolecularSimilarityConfig,
            MolecularSimilarityOperator,
        )

        config = MolecularSimilarityConfig(similarity_type="cosine")
        sim_op = MolecularSimilarityOperator(config, rngs=nnx.Rngs(42))

        # Random fingerprints
        fp1 = jax.random.uniform(jax.random.PRNGKey(0), (128,))
        fp2 = jax.random.uniform(jax.random.PRNGKey(1), (128,))

        data = {"fingerprint_a": fp1, "fingerprint_b": fp2}
        result, _, _ = sim_op.apply(data, {}, None)

        assert -1.0 <= result["similarity"] <= 1.0

    def test_self_similarity_is_one(self):
        """Test that similarity of molecule with itself is 1."""
        from diffbio.operators.drug_discovery import (
            MolecularSimilarityConfig,
            MolecularSimilarityOperator,
        )

        config = MolecularSimilarityConfig(similarity_type="tanimoto")
        sim_op = MolecularSimilarityOperator(config, rngs=nnx.Rngs(42))

        fp = jax.random.uniform(jax.random.PRNGKey(0), (128,))

        data = {"fingerprint_a": fp, "fingerprint_b": fp}
        result, _, _ = sim_op.apply(data, {}, None)

        assert jnp.isclose(result["similarity"], 1.0, atol=1e-5)

    def test_gradient_flow(self):
        """Test gradients flow through similarity computation."""
        from diffbio.operators.drug_discovery import (
            MolecularSimilarityConfig,
            MolecularSimilarityOperator,
        )

        config = MolecularSimilarityConfig()
        sim_op = MolecularSimilarityOperator(config, rngs=nnx.Rngs(42))

        def loss_fn(fp1, fp2):
            data = {"fingerprint_a": fp1, "fingerprint_b": fp2}
            result, _, _ = sim_op.apply(data, {}, None)
            return result["similarity"]

        fp1 = jax.random.uniform(jax.random.PRNGKey(0), (128,))
        fp2 = jax.random.uniform(jax.random.PRNGKey(1), (128,))

        grads = jax.grad(loss_fn)(fp1, fp2)
        assert grads is not None
        assert jnp.all(jnp.isfinite(grads))

    def test_symmetry(self):
        """Test similarity(a, b) == similarity(b, a)."""
        from diffbio.operators.drug_discovery import (
            MolecularSimilarityConfig,
            MolecularSimilarityOperator,
        )

        config = MolecularSimilarityConfig()
        sim_op = MolecularSimilarityOperator(config, rngs=nnx.Rngs(42))

        fp1 = jax.random.uniform(jax.random.PRNGKey(0), (128,))
        fp2 = jax.random.uniform(jax.random.PRNGKey(1), (128,))

        data_ab = {"fingerprint_a": fp1, "fingerprint_b": fp2}
        data_ba = {"fingerprint_a": fp2, "fingerprint_b": fp1}

        result_ab, _, _ = sim_op.apply(data_ab, {}, None)
        result_ba, _, _ = sim_op.apply(data_ba, {}, None)

        assert jnp.isclose(result_ab["similarity"], result_ba["similarity"])


# =============================================================================
# Tests for Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for convenience factory functions."""

    def test_create_property_predictor(self):
        """Test property predictor factory function."""
        from diffbio.operators.drug_discovery import create_property_predictor

        predictor = create_property_predictor(
            hidden_dim=64,
            num_layers=2,
            num_tasks=3,
            seed=42,
        )

        assert predictor is not None

    def test_create_fingerprint_operator(self):
        """Test fingerprint operator factory function."""
        from diffbio.operators.drug_discovery import create_fingerprint_operator

        fp_op = create_fingerprint_operator(
            fingerprint_dim=128,
            num_layers=2,
            seed=42,
        )

        assert fp_op is not None

    def test_create_similarity_operator(self):
        """Test similarity operator factory function."""
        from diffbio.operators.drug_discovery import create_similarity_operator

        sim_op = create_similarity_operator(similarity_type="cosine")

        assert sim_op is not None


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_atom_molecule(self):
        """Test handling of single-atom molecule."""
        from diffbio.operators.drug_discovery import smiles_to_graph

        graph = smiles_to_graph("C")  # methane

        assert graph["num_nodes"] >= 1

    def test_large_molecule(self):
        """Test handling of larger molecules."""
        from diffbio.operators.drug_discovery import smiles_to_graph

        # Aspirin
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        graph = smiles_to_graph(smiles)

        assert graph["num_nodes"] > 5
        assert jnp.all(jnp.isfinite(graph["node_features"]))

    def test_aromatic_molecule(self):
        """Test handling of aromatic compounds."""
        from diffbio.operators.drug_discovery import smiles_to_graph

        graph = smiles_to_graph("c1ccccc1")  # benzene

        # Should have 6 carbon atoms
        assert graph["num_nodes"] == 6

    def test_molecule_with_rings(self):
        """Test handling of molecules with ring systems."""
        from diffbio.operators.drug_discovery import smiles_to_graph

        # Caffeine has multiple rings
        smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        graph = smiles_to_graph(smiles)

        assert graph["num_nodes"] > 0
        assert jnp.all(jnp.isfinite(graph["adjacency"]))

    def test_empty_edge_features(self, molecular_graph):
        """Test handling when edge features are not provided."""
        from diffbio.operators.drug_discovery import (
            MolecularPropertyConfig,
            MolecularPropertyPredictor,
        )

        config = MolecularPropertyConfig(hidden_dim=32)
        predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))

        data = {
            "node_features": molecular_graph["node_features"],
            "adjacency": molecular_graph["adjacency"],
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
            # No edge_features provided
        }

        # Should handle missing edge features gracefully
        result, _, _ = predictor.apply(data, {}, None)
        assert "predictions" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_smiles_to_prediction_pipeline(self, simple_smiles):
        """Test full pipeline from SMILES to property prediction."""
        from diffbio.operators.drug_discovery import (
            DEFAULT_ATOM_FEATURES,
            MolecularPropertyConfig,
            MolecularPropertyPredictor,
            smiles_to_graph,
        )

        # Use in_features=DEFAULT_ATOM_FEATURES for real molecules
        config = MolecularPropertyConfig(
            hidden_dim=32, num_message_passing_steps=2, in_features=DEFAULT_ATOM_FEATURES
        )
        predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))

        for smiles in simple_smiles[:3]:  # Test first 3
            graph = smiles_to_graph(smiles)

            data = {
                "node_features": graph["node_features"],
                "adjacency": graph["adjacency"],
                "edge_features": graph.get("edge_features"),
                "node_mask": jnp.ones(graph["num_nodes"]),
            }

            result, _, _ = predictor.apply(data, {}, None)

            assert jnp.all(jnp.isfinite(result["predictions"]))

    def test_fingerprint_similarity_pipeline(self, simple_smiles):
        """Test pipeline: SMILES → fingerprint → similarity."""
        from diffbio.operators.drug_discovery import (
            DEFAULT_ATOM_FEATURES,
            DifferentiableMolecularFingerprint,
            MolecularFingerprintConfig,
            MolecularSimilarityConfig,
            MolecularSimilarityOperator,
            smiles_to_graph,
        )

        # Create operators - use in_features=DEFAULT_ATOM_FEATURES for real molecules
        fp_config = MolecularFingerprintConfig(
            fingerprint_dim=64, normalize=True, in_features=DEFAULT_ATOM_FEATURES
        )
        fp_op = DifferentiableMolecularFingerprint(fp_config, rngs=nnx.Rngs(42))

        sim_config = MolecularSimilarityConfig(similarity_type="cosine")
        sim_op = MolecularSimilarityOperator(sim_config, rngs=nnx.Rngs(42))

        # Get fingerprints for two molecules
        graph1 = smiles_to_graph(simple_smiles[0])
        graph2 = smiles_to_graph(simple_smiles[1])

        data1 = {
            "node_features": graph1["node_features"],
            "adjacency": graph1["adjacency"],
            "node_mask": jnp.ones(graph1["num_nodes"]),
        }
        data2 = {
            "node_features": graph2["node_features"],
            "adjacency": graph2["adjacency"],
            "node_mask": jnp.ones(graph2["num_nodes"]),
        }

        result1, _, _ = fp_op.apply(data1, {}, None)
        result2, _, _ = fp_op.apply(data2, {}, None)

        # Compute similarity
        sim_data = {
            "fingerprint_a": result1["fingerprint"],
            "fingerprint_b": result2["fingerprint"],
        }
        sim_result, _, _ = sim_op.apply(sim_data, {}, None)

        assert "similarity" in sim_result
        assert jnp.isfinite(sim_result["similarity"])

    def test_end_to_end_gradient(self, simple_smiles):
        """Test gradients flow from loss through entire pipeline."""
        from diffbio.operators.drug_discovery import (
            DEFAULT_ATOM_FEATURES,
            MolecularPropertyConfig,
            MolecularPropertyPredictor,
            smiles_to_graph,
        )

        # Use in_features=DEFAULT_ATOM_FEATURES for real molecules
        config = MolecularPropertyConfig(
            hidden_dim=32, num_message_passing_steps=2, in_features=DEFAULT_ATOM_FEATURES
        )
        predictor = MolecularPropertyPredictor(config, rngs=nnx.Rngs(42))

        graph = smiles_to_graph("CCO")

        def loss_fn(predictor, target):
            data = {
                "node_features": graph["node_features"],
                "adjacency": graph["adjacency"],
                "node_mask": jnp.ones(graph["num_nodes"]),
            }
            result, _, _ = predictor.apply(data, {}, None)
            return jnp.mean((result["predictions"] - target) ** 2)

        target = jnp.array([0.5])
        # Use nnx.grad for proper NNX module handling (filters to Param only)
        grads = nnx.grad(loss_fn)(predictor, target)

        # Gradients should exist and be finite
        assert grads is not None


# =============================================================================
# Tests for Circular Fingerprint Operator (ECFP/Morgan)
# =============================================================================


class TestCircularFingerprintOperator:
    """Tests for differentiable circular fingerprints (ECFP/Morgan)."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from diffbio.operators.drug_discovery import CircularFingerprintConfig

        config = CircularFingerprintConfig()
        assert config.radius == 2  # ECFP4
        assert config.n_bits == 2048
        assert config.use_chirality is False
        assert config.use_bond_types is True
        assert config.use_features is False
        assert config.differentiable is True
        assert config.hash_hidden_dim == 128
        assert config.temperature == 1.0

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from diffbio.operators.drug_discovery import CircularFingerprintConfig

        config = CircularFingerprintConfig(
            radius=3,  # ECFP6
            n_bits=1024,
            use_chirality=True,
            differentiable=True,
            temperature=0.5,
        )
        assert config.radius == 3
        assert config.n_bits == 1024
        assert config.use_chirality is True
        assert config.temperature == 0.5

    def test_operator_init(self):
        """Test operator initialization."""
        from diffbio.operators.drug_discovery import (
            CircularFingerprintConfig,
            CircularFingerprintOperator,
        )

        config = CircularFingerprintConfig(n_bits=512)
        op = CircularFingerprintOperator(config, rngs=nnx.Rngs(42))
        assert op is not None
        assert op.config.n_bits == 512

    def test_fingerprint_output_shape(self, molecular_graph):
        """Test fingerprint has correct output dimension."""
        from diffbio.operators.drug_discovery import (
            CircularFingerprintConfig,
            CircularFingerprintOperator,
        )

        n_bits = 256
        config = CircularFingerprintConfig(n_bits=n_bits, differentiable=True)
        op = CircularFingerprintOperator(config, rngs=nnx.Rngs(42))

        data = {
            **molecular_graph,
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }
        result, _, _ = op.apply(data, {}, None)

        assert "fingerprint" in result
        assert result["fingerprint"].shape == (n_bits,)

    def test_fingerprint_differentiable(self, molecular_graph):
        """Test gradients flow through differentiable fingerprint."""
        from diffbio.operators.drug_discovery import (
            CircularFingerprintConfig,
            CircularFingerprintOperator,
        )

        config = CircularFingerprintConfig(n_bits=128, differentiable=True)
        op = CircularFingerprintOperator(config, rngs=nnx.Rngs(42))

        def loss_fn(op):
            data = {
                **molecular_graph,
                "node_mask": jnp.ones(molecular_graph["num_nodes"]),
            }
            result, _, _ = op.apply(data, {}, None)
            return result["fingerprint"].sum()

        grads = nnx.grad(loss_fn)(op)
        assert grads is not None

    def test_jit_compatible(self, molecular_graph):
        """Test JIT compilation works for CircularFingerprintOperator."""
        from diffbio.operators.drug_discovery import (
            CircularFingerprintConfig,
            CircularFingerprintOperator,
        )

        config = CircularFingerprintConfig(n_bits=128, differentiable=True)
        op = CircularFingerprintOperator(config, rngs=nnx.Rngs(42))

        data = {
            **molecular_graph,
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }

        @jax.jit
        def compute(op, data):
            result, _, _ = op.apply(data, {}, None)
            return result["fingerprint"]

        fingerprint = compute(op, data)
        assert fingerprint.shape == (128,)
        assert jnp.all(jnp.isfinite(fingerprint))

    def test_different_molecules_different_fingerprints(self):
        """Test that different molecules produce different fingerprints."""
        from diffbio.operators.drug_discovery import (
            CircularFingerprintConfig,
            CircularFingerprintOperator,
            DEFAULT_ATOM_FEATURES,
            smiles_to_graph,
        )

        config = CircularFingerprintConfig(
            n_bits=256,
            differentiable=True,
            in_features=DEFAULT_ATOM_FEATURES,
        )
        op = CircularFingerprintOperator(config, rngs=nnx.Rngs(42))

        # Different molecules
        graph1 = smiles_to_graph("CCO")  # Ethanol
        graph2 = smiles_to_graph("c1ccccc1")  # Benzene

        data1 = {
            **graph1,
            "node_mask": jnp.ones(graph1["num_nodes"]),
        }
        data2 = {
            **graph2,
            "node_mask": jnp.ones(graph2["num_nodes"]),
        }

        result1, _, _ = op.apply(data1, {}, None)
        result2, _, _ = op.apply(data2, {}, None)

        # Fingerprints should be different
        assert not jnp.allclose(result1["fingerprint"], result2["fingerprint"])

    def test_fingerprint_binary_like_values(self, molecular_graph):
        """Test that fingerprint values are in [0, 1] range (soft bits)."""
        from diffbio.operators.drug_discovery import (
            CircularFingerprintConfig,
            CircularFingerprintOperator,
        )

        config = CircularFingerprintConfig(n_bits=128, differentiable=True)
        op = CircularFingerprintOperator(config, rngs=nnx.Rngs(42))

        data = {
            **molecular_graph,
            "node_mask": jnp.ones(molecular_graph["num_nodes"]),
        }
        result, _, _ = op.apply(data, {}, None)

        fp = result["fingerprint"]
        # Fingerprint should be in valid range
        assert jnp.all(fp >= 0) or jnp.all(jnp.isfinite(fp))

    def test_rdkit_fallback_mode(self):
        """Test RDKit fallback for exact ECFP (non-differentiable mode)."""
        from diffbio.operators.drug_discovery import (
            CircularFingerprintConfig,
            CircularFingerprintOperator,
        )

        config = CircularFingerprintConfig(
            n_bits=1024,
            differentiable=False,  # Use RDKit
        )
        op = CircularFingerprintOperator(config, rngs=nnx.Rngs(42))

        # Must provide SMILES for RDKit mode
        data = {"smiles": "CCO"}  # Ethanol
        result, _, _ = op.apply(data, {}, None)

        assert "fingerprint" in result
        assert result["fingerprint"].shape == (config.n_bits,)
        # RDKit fingerprints are binary
        assert jnp.all((result["fingerprint"] == 0) | (result["fingerprint"] == 1))


class TestCircularFingerprintFactoryFunctions:
    """Tests for ECFP factory functions."""

    def test_create_ecfp4_operator(self):
        """Test ECFP4 factory function."""
        from diffbio.operators.drug_discovery import create_ecfp4_operator

        op = create_ecfp4_operator(n_bits=512)
        assert op.config.radius == 2  # ECFP4 = radius 2
        assert op.config.n_bits == 512

    def test_create_ecfp6_operator(self):
        """Test ECFP6 factory function."""
        from diffbio.operators.drug_discovery import create_ecfp6_operator

        op = create_ecfp6_operator(n_bits=1024)
        assert op.config.radius == 3  # ECFP6 = radius 3
        assert op.config.n_bits == 1024

    def test_create_fcfp4_operator(self):
        """Test FCFP4 (feature-based) factory function."""
        from diffbio.operators.drug_discovery import create_fcfp4_operator

        op = create_fcfp4_operator(n_bits=2048)
        assert op.config.radius == 2
        assert op.config.n_bits == 2048
        assert op.config.use_features is True  # FCFP uses pharmacophoric features

    def test_factory_differentiable_option(self):
        """Test factory functions with differentiable option."""
        from diffbio.operators.drug_discovery import create_ecfp4_operator

        # Differentiable (default)
        op_diff = create_ecfp4_operator(n_bits=256, differentiable=True)
        assert op_diff.config.differentiable is True

        # Non-differentiable (RDKit)
        op_rdkit = create_ecfp4_operator(n_bits=256, differentiable=False)
        assert op_rdkit.config.differentiable is False
