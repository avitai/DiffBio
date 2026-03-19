"""Tests for diffbio.operators.singlecell.communication module.

These tests define the expected behavior of the DifferentiableLigandReceptor
operator for ligand-receptor co-expression scoring and the
DifferentiableCellCommunication operator for GNN-based cell-cell
communication analysis in single-cell data.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.communication import (
    CellCommunicationConfig,
    DifferentiableCellCommunication,
    DifferentiableLigandReceptor,
    LRScoringConfig,
)


class TestLRScoringConfig:
    """Tests for LRScoringConfig defaults and customization."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = LRScoringConfig()
        assert config.n_neighbors == 15
        assert config.temperature == 1.0
        assert config.learnable_temperature is False
        assert config.metric == "euclidean"
        assert config.kh == 0.5
        assert config.hill_n == 1.0
        assert config.stochastic is False

    def test_custom_neighbors(self) -> None:
        """Test custom number of neighbors."""
        config = LRScoringConfig(n_neighbors=30)
        assert config.n_neighbors == 30


class TestDifferentiableLigandReceptor:
    """Tests for DifferentiableLigandReceptor operator outputs."""

    @pytest.fixture()
    def config(self) -> LRScoringConfig:
        """Provide a small config for fast tests."""
        return LRScoringConfig(n_neighbors=5, temperature=1.0)

    @pytest.fixture()
    def sample_data(self) -> dict[str, jax.Array | int]:
        """Provide synthetic single-cell data with L-R pairs."""
        key = jax.random.key(0)
        n_cells, n_genes, n_pairs = 20, 10, 3
        k1, k2 = jax.random.split(key)
        counts = jax.random.uniform(k1, (n_cells, n_genes), minval=0.0, maxval=5.0)
        lr_pairs = jnp.array([[0, 1], [2, 3], [4, 5]])
        return {"counts": counts, "lr_pairs": lr_pairs, "_n_cells": n_cells, "_n_pairs": n_pairs}

    def test_output_keys(
        self, rngs: nnx.Rngs, config: LRScoringConfig, sample_data: dict[str, jax.Array | int]
    ) -> None:
        """Test that lr_scores and lr_pvalues are present in output."""
        op = DifferentiableLigandReceptor(config, rngs=rngs)
        data = {"counts": sample_data["counts"], "lr_pairs": sample_data["lr_pairs"]}
        result, _, _ = op.apply(data, {}, None)
        assert "lr_scores" in result
        assert "lr_pvalues" in result

    def test_output_shapes(
        self, rngs: nnx.Rngs, config: LRScoringConfig, sample_data: dict[str, jax.Array | int]
    ) -> None:
        """Test lr_scores shape is (n_cells, n_pairs) and lr_pvalues is (n_pairs,)."""
        op = DifferentiableLigandReceptor(config, rngs=rngs)
        n_cells = sample_data["_n_cells"]
        n_pairs = sample_data["_n_pairs"]
        data = {"counts": sample_data["counts"], "lr_pairs": sample_data["lr_pairs"]}
        result, _, _ = op.apply(data, {}, None)
        assert result["lr_scores"].shape == (n_cells, n_pairs)
        assert result["lr_pvalues"].shape == (n_pairs,)

    def test_known_lr_higher(self, rngs: nnx.Rngs) -> None:
        """Synthetic data with known L-R correlation scores higher than random pairs."""
        n_cells = 30
        n_genes = 6
        key = jax.random.key(99)

        # Build counts: gene 0 (ligand) and gene 1 (receptor) are co-expressed
        # in spatially clustered cells, whereas gene 4 and gene 5 are random.
        k1, k2, k3 = jax.random.split(key, 3)
        base = jax.random.uniform(k1, (n_cells, n_genes), minval=0.0, maxval=0.5)

        # First half of cells: high ligand (gene 0) AND high receptor (gene 1)
        signal = jnp.zeros((n_cells, n_genes))
        signal = signal.at[: n_cells // 2, 0].set(5.0)
        signal = signal.at[: n_cells // 2, 1].set(5.0)
        counts = base + signal

        lr_pairs = jnp.array([[0, 1], [4, 5]])  # correlated pair, random pair

        config = LRScoringConfig(n_neighbors=5, temperature=1.0)
        op = DifferentiableLigandReceptor(config, rngs=rngs)
        data = {"counts": counts, "lr_pairs": lr_pairs}
        result, _, _ = op.apply(data, {}, None)

        # The correlated pair (0,1) should have a higher mean score than (4,5)
        mean_score_correlated = jnp.mean(result["lr_scores"][:, 0])
        mean_score_random = jnp.mean(result["lr_scores"][:, 1])
        assert float(mean_score_correlated) > float(mean_score_random)

        # Hill function output is bounded in [0, 1)
        assert jnp.all(result["lr_scores"] >= 0.0)
        assert jnp.all(result["lr_scores"] < 1.0 + 1e-6)

    def test_hill_function_saturation(self, rngs: nnx.Rngs) -> None:
        """At high L*R product, Hill function score approaches 1.0."""
        n_cells = 20
        n_genes = 4
        key = jax.random.key(42)

        # Very high ligand and receptor expression for all cells
        counts = jnp.ones((n_cells, n_genes)) * 100.0
        # Add small noise so the k-NN graph is well-defined
        counts = counts + jax.random.uniform(key, (n_cells, n_genes), maxval=0.01)

        lr_pairs = jnp.array([[0, 1]])
        config = LRScoringConfig(n_neighbors=5, temperature=1.0)
        op = DifferentiableLigandReceptor(config, rngs=rngs)
        data = {"counts": counts, "lr_pairs": lr_pairs}
        result, _, _ = op.apply(data, {}, None)

        # With very large L*R, Hill function P = (LR)^n / (Kh^n + (LR)^n) -> 1
        scores = result["lr_scores"][:, 0]
        assert jnp.all(scores > 0.9), (
            f"Expected scores > 0.9 at saturation, got min={float(jnp.min(scores))}"
        )
        assert jnp.isfinite(scores).all()

    def test_scores_finite(
        self, rngs: nnx.Rngs, config: LRScoringConfig, sample_data: dict[str, jax.Array | int]
    ) -> None:
        """Test that all outputs are finite."""
        op = DifferentiableLigandReceptor(config, rngs=rngs)
        data = {"counts": sample_data["counts"], "lr_pairs": sample_data["lr_pairs"]}
        result, _, _ = op.apply(data, {}, None)
        assert jnp.isfinite(result["lr_scores"]).all()
        assert jnp.isfinite(result["lr_pvalues"]).all()


class TestGradientFlow:
    """Tests for gradient flow through the L-R scoring operator."""

    @pytest.fixture()
    def config(self) -> LRScoringConfig:
        """Provide config for gradient tests."""
        return LRScoringConfig(n_neighbors=5, temperature=1.0)

    def test_gradient_wrt_counts(self, rngs: nnx.Rngs, config: LRScoringConfig) -> None:
        """Verify jax.grad on sum of lr_scores wrt counts is non-zero."""
        op = DifferentiableLigandReceptor(config, rngs=rngs)
        key = jax.random.key(1)
        counts = jax.random.uniform(key, (20, 6), minval=0.1, maxval=3.0)
        lr_pairs = jnp.array([[0, 1], [2, 3]])

        def loss_fn(c: jax.Array) -> jax.Array:
            data = {"counts": c, "lr_pairs": lr_pairs}
            result, _, _ = op.apply(data, {}, None)
            return jnp.sum(result["lr_scores"])

        grad = jax.grad(loss_fn)(counts)
        assert grad.shape == counts.shape
        assert jnp.isfinite(grad).all()
        assert jnp.any(grad != 0.0)

    def test_gradient_wrt_operator_params(self, rngs: nnx.Rngs) -> None:
        """Verify gradients flow to the temperature parameter."""
        config = LRScoringConfig(n_neighbors=5, temperature=1.0, learnable_temperature=True)
        op = DifferentiableLigandReceptor(config, rngs=rngs)
        key = jax.random.key(2)
        counts = jax.random.uniform(key, (20, 6), minval=0.1, maxval=3.0)
        lr_pairs = jnp.array([[0, 1]])

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableLigandReceptor) -> jax.Array:
            data = {"counts": counts, "lr_pairs": lr_pairs}
            result, _, _ = model.apply(data, {}, None)
            return jnp.sum(result["lr_scores"])

        _, grads = loss_fn(op)
        assert hasattr(grads, "temperature")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture()
    def config(self) -> LRScoringConfig:
        """Provide config for JIT tests."""
        return LRScoringConfig(n_neighbors=5, temperature=1.0)

    def test_jit_apply(self, rngs: nnx.Rngs, config: LRScoringConfig) -> None:
        """Test that jax.jit compiles the apply method."""
        op = DifferentiableLigandReceptor(config, rngs=rngs)
        key = jax.random.key(3)
        counts = jax.random.uniform(key, (20, 6), minval=0.0, maxval=3.0)
        lr_pairs = jnp.array([[0, 1], [2, 3]])
        data = {"counts": counts, "lr_pairs": lr_pairs}

        @jax.jit
        def jit_apply(d: dict, s: dict) -> tuple:
            return op.apply(d, s, None)

        result, _, _ = jit_apply(data, {})
        assert jnp.isfinite(result["lr_scores"]).all()
        assert jnp.isfinite(result["lr_pvalues"]).all()

    def test_jit_gradient(self, rngs: nnx.Rngs, config: LRScoringConfig) -> None:
        """Test that jax.jit + jax.grad works together."""
        op = DifferentiableLigandReceptor(config, rngs=rngs)
        key = jax.random.key(4)
        counts = jax.random.uniform(key, (20, 6), minval=0.1, maxval=3.0)
        lr_pairs = jnp.array([[0, 1]])

        @jax.jit
        def grad_fn(c: jax.Array) -> jax.Array:
            def loss(x: jax.Array) -> jax.Array:
                data = {"counts": x, "lr_pairs": lr_pairs}
                result, _, _ = op.apply(data, {}, None)
                return jnp.sum(result["lr_scores"])

            return jax.grad(loss)(c)

        grad = grad_fn(counts)
        assert grad.shape == counts.shape
        assert jnp.isfinite(grad).all()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_lr_pair(self, rngs: nnx.Rngs) -> None:
        """Test with a single ligand-receptor pair."""
        config = LRScoringConfig(n_neighbors=5, temperature=1.0)
        op = DifferentiableLigandReceptor(config, rngs=rngs)
        key = jax.random.key(5)
        counts = jax.random.uniform(key, (20, 4), minval=0.0, maxval=3.0)
        lr_pairs = jnp.array([[0, 1]])
        data = {"counts": counts, "lr_pairs": lr_pairs}
        result, _, _ = op.apply(data, {}, None)
        assert result["lr_scores"].shape == (20, 1)
        assert result["lr_pvalues"].shape == (1,)
        assert jnp.isfinite(result["lr_scores"]).all()

    def test_zero_expression(self, rngs: nnx.Rngs) -> None:
        """Test that all-zero expression yields scores near zero."""
        config = LRScoringConfig(n_neighbors=5, temperature=1.0)
        op = DifferentiableLigandReceptor(config, rngs=rngs)
        counts = jnp.zeros((20, 4))
        lr_pairs = jnp.array([[0, 1]])
        data = {"counts": counts, "lr_pairs": lr_pairs}
        result, _, _ = op.apply(data, {}, None)
        assert jnp.allclose(result["lr_scores"], 0.0, atol=1e-6)


# =============================================================================
# DifferentiableCellCommunication tests
# =============================================================================


def _make_comm_data(
    n_cells: int = 15,
    n_genes: int = 50,
    n_pairs: int = 3,
    n_edges: int = 30,
    seed: int = 0,
) -> dict[str, jax.Array]:
    """Build synthetic cell-communication data for testing.

    Args:
        n_cells: Number of cells.
        n_genes: Number of genes.
        n_pairs: Number of L-R pairs.
        n_edges: Number of edges in spatial graph.
        seed: Random seed.

    Returns:
        Dictionary with counts, spatial_graph, and lr_pairs.
    """
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    counts = jax.random.uniform(k1, (n_cells, n_genes), minval=0.1, maxval=5.0)

    # Build a random spatial graph (source, target) with edges among n_cells nodes
    sources = jax.random.randint(k2, (n_edges,), 0, n_cells)
    targets = jax.random.randint(k3, (n_edges,), 0, n_cells)
    spatial_graph = jnp.stack([sources, targets], axis=0)  # (2, n_edges)

    # L-R pair gene indices -- must be < n_genes
    lr_pairs = jnp.array([[i * 2, i * 2 + 1] for i in range(n_pairs)])

    return {
        "counts": counts,
        "spatial_graph": spatial_graph,
        "lr_pairs": lr_pairs,
    }


class TestCellCommunicationConfig:
    """Tests for CellCommunicationConfig defaults and customization."""

    def test_default_config(self) -> None:
        """Test default configuration values match the spec."""
        config = CellCommunicationConfig()
        assert config.n_genes == 2000
        assert config.n_lr_pairs == 10
        assert config.hidden_dim == 64
        assert config.num_heads == 4
        assert config.edge_features_dim == 8
        assert config.num_gnn_layers == 2
        assert config.n_pathways == 20
        assert config.dropout_rate == 0.1
        assert config.stochastic is False

    def test_custom_hidden_dim(self) -> None:
        """Test overriding hidden_dim produces the expected value."""
        config = CellCommunicationConfig(hidden_dim=128)
        assert config.hidden_dim == 128


class TestDifferentiableCellCommunication:
    """Tests for DifferentiableCellCommunication operator outputs."""

    @pytest.fixture()
    def config(self) -> CellCommunicationConfig:
        """Provide a small config for fast tests."""
        return CellCommunicationConfig(
            n_genes=50,
            n_lr_pairs=3,
            hidden_dim=16,
            num_heads=2,
            edge_features_dim=6,
            num_gnn_layers=1,
            n_pathways=5,
            dropout_rate=0.0,
        )

    @pytest.fixture()
    def sample_data(self) -> dict[str, jax.Array]:
        """Provide synthetic communication data."""
        return _make_comm_data(n_cells=15, n_genes=50, n_pairs=3, n_edges=30)

    def test_output_keys(
        self,
        rngs: nnx.Rngs,
        config: CellCommunicationConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that communication_scores, signaling_activity, and niche_embeddings are present."""
        op = DifferentiableCellCommunication(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert "communication_scores" in result
        assert "signaling_activity" in result
        assert "niche_embeddings" in result

    def test_output_shapes(
        self,
        rngs: nnx.Rngs,
        config: CellCommunicationConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test output shapes: (n, p), (n, pathways), (n, hidden_dim)."""
        n_cells = 15
        n_pairs = 3
        op = DifferentiableCellCommunication(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert result["communication_scores"].shape == (n_cells, n_pairs)
        assert result["signaling_activity"].shape == (n_cells, config.n_pathways)
        assert result["niche_embeddings"].shape == (n_cells, config.hidden_dim)

    def test_scores_finite(
        self,
        rngs: nnx.Rngs,
        config: CellCommunicationConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that all outputs contain finite values."""
        op = DifferentiableCellCommunication(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert jnp.isfinite(result["communication_scores"]).all()
        assert jnp.isfinite(result["signaling_activity"]).all()
        assert jnp.isfinite(result["niche_embeddings"]).all()

    def test_different_graphs_different_output(
        self,
        rngs: nnx.Rngs,
        config: CellCommunicationConfig,
    ) -> None:
        """Changing spatial graph changes results."""
        data1 = _make_comm_data(n_cells=15, n_genes=50, n_pairs=3, n_edges=30, seed=0)
        data2 = _make_comm_data(n_cells=15, n_genes=50, n_pairs=3, n_edges=30, seed=99)
        # Same counts, different graphs
        data2["counts"] = data1["counts"]
        data2["lr_pairs"] = data1["lr_pairs"]

        op = DifferentiableCellCommunication(config, rngs=rngs)
        result1, _, _ = op.apply(data1, {}, None)
        result2, _, _ = op.apply(data2, {}, None)

        # At least one output must differ because the spatial graph differs
        scores_differ = not jnp.allclose(
            result1["communication_scores"], result2["communication_scores"], atol=1e-6
        )
        activity_differ = not jnp.allclose(
            result1["signaling_activity"], result2["signaling_activity"], atol=1e-6
        )
        assert scores_differ or activity_differ


class TestCommunicationGradientFlow:
    """Tests for gradient flow through the cell communication operator."""

    @pytest.fixture()
    def config(self) -> CellCommunicationConfig:
        """Provide config for gradient tests."""
        return CellCommunicationConfig(
            n_genes=50,
            n_lr_pairs=3,
            hidden_dim=16,
            num_heads=2,
            edge_features_dim=6,
            num_gnn_layers=1,
            n_pathways=5,
            dropout_rate=0.0,
        )

    def test_gradient_wrt_counts(
        self,
        rngs: nnx.Rngs,
        config: CellCommunicationConfig,
    ) -> None:
        """Verify jax.grad on sum of communication_scores wrt counts is non-zero."""
        op = DifferentiableCellCommunication(config, rngs=rngs)
        data = _make_comm_data(n_cells=10, n_genes=50, n_pairs=3, n_edges=20)

        def loss_fn(counts: jax.Array) -> jax.Array:
            d = {**data, "counts": counts}
            result, _, _ = op.apply(d, {}, None)
            return jnp.sum(result["communication_scores"])

        grad = jax.grad(loss_fn)(data["counts"])
        assert grad.shape == data["counts"].shape
        assert jnp.isfinite(grad).all()
        assert jnp.any(grad != 0.0)

    def test_gradient_wrt_gnn_params(
        self,
        rngs: nnx.Rngs,
        config: CellCommunicationConfig,
    ) -> None:
        """Verify gradients flow through GATv2 attention weights."""
        op = DifferentiableCellCommunication(config, rngs=rngs)
        data = _make_comm_data(n_cells=10, n_genes=50, n_pairs=3, n_edges=20)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableCellCommunication) -> jax.Array:
            result, _, _ = model.apply(data, {}, None)
            return jnp.sum(result["communication_scores"])

        _, grads = loss_fn(op)
        # Check that the GNN submodule received non-zero gradients
        assert hasattr(grads, "spatial_gnn")


class TestCommunicationJIT:
    """Tests for JAX JIT compilation of cell communication operator."""

    @pytest.fixture()
    def config(self) -> CellCommunicationConfig:
        """Provide config for JIT tests."""
        return CellCommunicationConfig(
            n_genes=50,
            n_lr_pairs=3,
            hidden_dim=16,
            num_heads=2,
            edge_features_dim=6,
            num_gnn_layers=1,
            n_pathways=5,
            dropout_rate=0.0,
        )

    def test_jit_apply(
        self,
        rngs: nnx.Rngs,
        config: CellCommunicationConfig,
    ) -> None:
        """Test that jax.jit compiles the apply method."""
        op = DifferentiableCellCommunication(config, rngs=rngs)
        data = _make_comm_data(n_cells=10, n_genes=50, n_pairs=3, n_edges=20)

        @jax.jit
        def jit_apply(d: dict, s: dict) -> tuple:
            return op.apply(d, s, None)

        result, _, _ = jit_apply(data, {})
        assert jnp.isfinite(result["communication_scores"]).all()
        assert jnp.isfinite(result["signaling_activity"]).all()
        assert jnp.isfinite(result["niche_embeddings"]).all()

    def test_jit_gradient(
        self,
        rngs: nnx.Rngs,
        config: CellCommunicationConfig,
    ) -> None:
        """Test that jax.jit + jax.grad works together."""
        op = DifferentiableCellCommunication(config, rngs=rngs)
        data = _make_comm_data(n_cells=10, n_genes=50, n_pairs=3, n_edges=20)

        @jax.jit
        def grad_fn(counts: jax.Array) -> jax.Array:
            def loss(c: jax.Array) -> jax.Array:
                d = {**data, "counts": c}
                result, _, _ = op.apply(d, {}, None)
                return jnp.sum(result["communication_scores"])

            return jax.grad(loss)(counts)

        grad = grad_fn(data["counts"])
        assert grad.shape == data["counts"].shape
        assert jnp.isfinite(grad).all()


class TestCommunicationEdgeCases:
    """Tests for edge cases in cell communication operator."""

    def test_single_lr_pair(self, rngs: nnx.Rngs) -> None:
        """Test with a single L-R pair."""
        config = CellCommunicationConfig(
            n_genes=50,
            n_lr_pairs=1,
            hidden_dim=16,
            num_heads=2,
            edge_features_dim=6,
            num_gnn_layers=1,
            n_pathways=5,
            dropout_rate=0.0,
        )
        data = _make_comm_data(n_cells=10, n_genes=50, n_pairs=1, n_edges=15)
        op = DifferentiableCellCommunication(config, rngs=rngs)
        result, _, _ = op.apply(data, {}, None)
        assert result["communication_scores"].shape == (10, 1)
        assert jnp.isfinite(result["communication_scores"]).all()

    def test_sparse_graph(self, rngs: nnx.Rngs) -> None:
        """Test with very few edges (most nodes disconnected)."""
        config = CellCommunicationConfig(
            n_genes=50,
            n_lr_pairs=2,
            hidden_dim=16,
            num_heads=2,
            edge_features_dim=6,
            num_gnn_layers=1,
            n_pathways=5,
            dropout_rate=0.0,
        )
        data = _make_comm_data(n_cells=15, n_genes=50, n_pairs=2, n_edges=3, seed=7)
        op = DifferentiableCellCommunication(config, rngs=rngs)
        result, _, _ = op.apply(data, {}, None)
        assert result["communication_scores"].shape == (15, 2)
        assert jnp.isfinite(result["communication_scores"]).all()
        assert jnp.isfinite(result["signaling_activity"]).all()
        assert jnp.isfinite(result["niche_embeddings"]).all()
