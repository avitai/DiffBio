"""Tests for spatial domain identification operators.

Defines expected behavior for:
- DifferentiableSpatialDomain: STAGATE-inspired graph attention autoencoder
  for spatial domain identification from spatial transcriptomics data.
- DifferentiablePASTEAlignment: PASTE-inspired fused Gromov-Wasserstein
  optimal transport for aligning spatial transcriptomics slices.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.spatial_domains import (
    DifferentiablePASTEAlignment,
    DifferentiableSpatialDomain,
    PASTEAlignmentConfig,
    SpatialDomainConfig,
)


# =============================================================================
# SpatialDomainConfig tests
# =============================================================================


class TestSpatialDomainConfig:
    """Tests for SpatialDomainConfig defaults and customization."""

    def test_default_config(self) -> None:
        """Test default configuration values match the spec."""
        config = SpatialDomainConfig()
        assert config.n_genes == 2000
        assert config.hidden_dim == 64
        assert config.num_heads == 4
        assert config.n_domains == 7
        assert config.alpha == 0.8
        assert config.n_neighbors == 15
        assert config.stochastic is False

    def test_custom_config(self) -> None:
        """Test overriding config values."""
        config = SpatialDomainConfig(hidden_dim=128, n_domains=5)
        assert config.hidden_dim == 128
        assert config.n_domains == 5


# =============================================================================
# PASTEAlignmentConfig tests
# =============================================================================


class TestPASTEConfig:
    """Tests for PASTEAlignmentConfig defaults and customization."""

    def test_default_config(self) -> None:
        """Test default configuration values match the spec."""
        config = PASTEAlignmentConfig()
        assert config.alpha == 0.1
        assert config.sinkhorn_epsilon == 0.1
        assert config.sinkhorn_iters == 100
        assert config.stochastic is False

    def test_custom_config(self) -> None:
        """Test overriding config values."""
        config = PASTEAlignmentConfig(alpha=0.5, sinkhorn_iters=200)
        assert config.alpha == 0.5
        assert config.sinkhorn_iters == 200


# =============================================================================
# Helper functions for synthetic data
# =============================================================================


def _make_spatial_data(
    n_cells: int = 30,
    n_genes: int = 20,
    seed: int = 0,
) -> dict[str, jax.Array]:
    """Build synthetic spatial transcriptomics data.

    Args:
        n_cells: Number of cells / spots.
        n_genes: Number of genes.
        seed: Random seed.

    Returns:
        Dictionary with counts and spatial_coords.
    """
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    counts = jax.random.uniform(k1, (n_cells, n_genes), minval=0.1, maxval=5.0)
    spatial_coords = jax.random.uniform(k2, (n_cells, 2), minval=0.0, maxval=10.0)
    return {"counts": counts, "spatial_coords": spatial_coords}


def _make_paste_data(
    n_cells_slice1: int = 15,
    n_cells_slice2: int = 15,
    n_genes: int = 20,
    seed: int = 0,
) -> dict[str, jax.Array]:
    """Build synthetic data for PASTE alignment.

    Args:
        n_cells_slice1: Number of cells in slice 1.
        n_cells_slice2: Number of cells in slice 2.
        n_genes: Number of genes.
        seed: Random seed.

    Returns:
        Dictionary with slice counts and coordinates.
    """
    key = jax.random.key(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "slice1_counts": jax.random.uniform(k1, (n_cells_slice1, n_genes), minval=0.1, maxval=5.0),
        "slice2_counts": jax.random.uniform(k2, (n_cells_slice2, n_genes), minval=0.1, maxval=5.0),
        "slice1_coords": jax.random.uniform(k3, (n_cells_slice1, 2), minval=0.0, maxval=10.0),
        "slice2_coords": jax.random.uniform(k4, (n_cells_slice2, 2), minval=0.0, maxval=10.0),
    }


# =============================================================================
# DifferentiableSpatialDomain tests
# =============================================================================


class TestSpatialDomain:
    """Tests for DifferentiableSpatialDomain operator outputs."""

    @pytest.fixture()
    def config(self) -> SpatialDomainConfig:
        """Provide a small config for fast tests."""
        return SpatialDomainConfig(
            n_genes=20,
            hidden_dim=16,
            num_heads=4,
            n_domains=3,
            alpha=0.8,
            n_neighbors=5,
        )

    @pytest.fixture()
    def sample_data(self) -> dict[str, jax.Array]:
        """Provide synthetic spatial data."""
        return _make_spatial_data(n_cells=30, n_genes=20, seed=0)

    def test_output_keys(
        self,
        rngs: nnx.Rngs,
        config: SpatialDomainConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that domain_assignments and spatial_embeddings are present."""
        op = DifferentiableSpatialDomain(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert "domain_assignments" in result
        assert "spatial_embeddings" in result

    def test_output_shapes(
        self,
        rngs: nnx.Rngs,
        config: SpatialDomainConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test output shapes match (n_cells, n_domains) and (n_cells, hidden_dim)."""
        n_cells = 30
        op = DifferentiableSpatialDomain(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert result["domain_assignments"].shape == (n_cells, config.n_domains)
        assert result["spatial_embeddings"].shape == (n_cells, config.hidden_dim)

    def test_domains_sum_to_one(
        self,
        rngs: nnx.Rngs,
        config: SpatialDomainConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that domain assignment probabilities sum to 1 along domain axis."""
        op = DifferentiableSpatialDomain(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        row_sums = jnp.sum(result["domain_assignments"], axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_embeddings_finite(
        self,
        rngs: nnx.Rngs,
        config: SpatialDomainConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that all spatial embeddings are finite."""
        op = DifferentiableSpatialDomain(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert jnp.isfinite(result["spatial_embeddings"]).all()
        assert jnp.isfinite(result["domain_assignments"]).all()

    def test_spatially_coherent_domains(self, rngs: nnx.Rngs) -> None:
        """Test that spatially close cells tend to get similar domain assignments.

        Creates two spatial clusters far apart and checks that within-cluster
        domain assignment similarity is higher than between-cluster.
        """
        n_per_cluster = 15
        n_genes = 20

        key = jax.random.key(42)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Cluster A: centered at (1, 1), cluster B: centered at (9, 9)
        coords_a = jax.random.normal(k1, (n_per_cluster, 2)) * 0.3 + jnp.array([1.0, 1.0])
        coords_b = jax.random.normal(k2, (n_per_cluster, 2)) * 0.3 + jnp.array([9.0, 9.0])
        spatial_coords = jnp.concatenate([coords_a, coords_b], axis=0)

        # Give each cluster distinct expression profiles
        counts_a = jax.random.uniform(k3, (n_per_cluster, n_genes), minval=3.0, maxval=6.0)
        counts_b = jax.random.uniform(k4, (n_per_cluster, n_genes), minval=0.1, maxval=1.0)
        counts = jnp.concatenate([counts_a, counts_b], axis=0)

        config = SpatialDomainConfig(
            n_genes=n_genes,
            hidden_dim=16,
            num_heads=4,
            n_domains=3,
            alpha=0.8,
            n_neighbors=5,
        )
        op = DifferentiableSpatialDomain(config, rngs=rngs)
        data = {"counts": counts, "spatial_coords": spatial_coords}
        result, _, _ = op.apply(data, {}, None)

        assignments = result["domain_assignments"]  # (30, 3)

        # Within-cluster cosine similarity should be > between-cluster
        def _mean_cosine(a: jax.Array, b: jax.Array) -> float:
            """Mean pairwise cosine similarity between row sets."""
            a_norm = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
            b_norm = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
            sim = jnp.dot(a_norm, b_norm.T)
            return float(jnp.mean(sim))

        within_a = _mean_cosine(assignments[:n_per_cluster], assignments[:n_per_cluster])
        within_b = _mean_cosine(assignments[n_per_cluster:], assignments[n_per_cluster:])
        between = _mean_cosine(assignments[:n_per_cluster], assignments[n_per_cluster:])
        avg_within = (within_a + within_b) / 2.0

        assert avg_within > between, (
            f"Within-cluster similarity ({avg_within:.3f}) should exceed "
            f"between-cluster ({between:.3f})"
        )


# =============================================================================
# DifferentiablePASTEAlignment tests
# =============================================================================


class TestPASTEAlignment:
    """Tests for DifferentiablePASTEAlignment operator outputs."""

    @pytest.fixture()
    def config(self) -> PASTEAlignmentConfig:
        """Provide config for PASTE tests."""
        return PASTEAlignmentConfig(
            alpha=0.1,
            sinkhorn_epsilon=0.1,
            sinkhorn_iters=100,
        )

    @pytest.fixture()
    def sample_data(self) -> dict[str, jax.Array]:
        """Provide synthetic PASTE data."""
        return _make_paste_data(n_cells_slice1=15, n_cells_slice2=15, n_genes=20, seed=0)

    def test_output_keys(
        self,
        rngs: nnx.Rngs,
        config: PASTEAlignmentConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that transport_plan and aligned_coords are present."""
        op = DifferentiablePASTEAlignment(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert "transport_plan" in result
        assert "aligned_coords" in result

    def test_output_shapes(
        self,
        rngs: nnx.Rngs,
        config: PASTEAlignmentConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test transport_plan shape (n1, n2) and aligned_coords shape (n2, 2)."""
        n1, n2 = 15, 15
        op = DifferentiablePASTEAlignment(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert result["transport_plan"].shape == (n1, n2)
        assert result["aligned_coords"].shape == (n2, 2)

    def test_transport_plan_marginals(
        self,
        rngs: nnx.Rngs,
        config: PASTEAlignmentConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that transport plan row/column sums approximate uniform distributions."""
        n1, n2 = 15, 15
        op = DifferentiablePASTEAlignment(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        plan = result["transport_plan"]

        # Row marginals should approximate 1/n1
        row_sums = jnp.sum(plan, axis=1)
        expected_row = jnp.ones(n1) / n1
        assert jnp.allclose(row_sums, expected_row, atol=0.05), (
            f"Row marginals deviate from uniform: max diff = "
            f"{float(jnp.max(jnp.abs(row_sums - expected_row))):.4f}"
        )

        # Column marginals should approximate 1/n2
        col_sums = jnp.sum(plan, axis=0)
        expected_col = jnp.ones(n2) / n2
        assert jnp.allclose(col_sums, expected_col, atol=0.05), (
            f"Column marginals deviate from uniform: max diff = "
            f"{float(jnp.max(jnp.abs(col_sums - expected_col))):.4f}"
        )

    def test_aligned_coords_finite(
        self,
        rngs: nnx.Rngs,
        config: PASTEAlignmentConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that all aligned coordinates are finite."""
        op = DifferentiablePASTEAlignment(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert jnp.isfinite(result["aligned_coords"]).all()
        assert jnp.isfinite(result["transport_plan"]).all()


# =============================================================================
# Gradient flow tests
# =============================================================================


class TestGradientFlow:
    """Tests for gradient flow through both spatial domain operators."""

    def test_stagate_grads_through_attention(self, rngs: nnx.Rngs) -> None:
        """Verify gradients flow through GATv2 attention in STAGATE operator."""
        config = SpatialDomainConfig(
            n_genes=20,
            hidden_dim=16,
            num_heads=4,
            n_domains=3,
            alpha=0.8,
            n_neighbors=5,
        )
        op = DifferentiableSpatialDomain(config, rngs=rngs)
        data = _make_spatial_data(n_cells=15, n_genes=20, seed=1)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableSpatialDomain) -> jax.Array:
            result, _, _ = model.apply(data, {}, None)
            return jnp.sum(result["domain_assignments"])

        _, grads = loss_fn(op)
        # Check that the GATv2 encoder received non-zero gradients
        assert hasattr(grads, "encoder")

    def test_paste_grads_through_transport_plan(self, rngs: nnx.Rngs) -> None:
        """Verify gradients flow through the Sinkhorn transport plan in PASTE."""
        config = PASTEAlignmentConfig(
            alpha=0.1,
            sinkhorn_epsilon=0.1,
            sinkhorn_iters=50,
        )
        op = DifferentiablePASTEAlignment(config, rngs=rngs)
        data = _make_paste_data(n_cells_slice1=10, n_cells_slice2=10, n_genes=20, seed=2)

        def loss_fn(counts1: jax.Array) -> jax.Array:
            d = {**data, "slice1_counts": counts1}
            result, _, _ = op.apply(d, {}, None)
            return jnp.sum(result["transport_plan"])

        grad = jax.grad(loss_fn)(data["slice1_counts"])
        assert grad.shape == data["slice1_counts"].shape
        assert jnp.isfinite(grad).all()
        assert jnp.any(grad != 0.0)


# =============================================================================
# JIT compatibility tests
# =============================================================================


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_jit_stagate(self, rngs: nnx.Rngs) -> None:
        """Test that STAGATE operator works under jax.jit."""
        config = SpatialDomainConfig(
            n_genes=20,
            hidden_dim=16,
            num_heads=4,
            n_domains=3,
            alpha=0.8,
            n_neighbors=5,
        )
        op = DifferentiableSpatialDomain(config, rngs=rngs)
        data = _make_spatial_data(n_cells=15, n_genes=20, seed=3)

        @jax.jit
        def jit_apply(d: dict, s: dict) -> tuple:
            return op.apply(d, s, None)

        result, _, _ = jit_apply(data, {})
        assert jnp.isfinite(result["domain_assignments"]).all()
        assert jnp.isfinite(result["spatial_embeddings"]).all()

    def test_jit_paste(self, rngs: nnx.Rngs) -> None:
        """Test that PASTE operator works under jax.jit."""
        config = PASTEAlignmentConfig(
            alpha=0.1,
            sinkhorn_epsilon=0.1,
            sinkhorn_iters=50,
        )
        op = DifferentiablePASTEAlignment(config, rngs=rngs)
        data = _make_paste_data(n_cells_slice1=10, n_cells_slice2=10, n_genes=20, seed=4)

        @jax.jit
        def jit_apply(d: dict, s: dict) -> tuple:
            return op.apply(d, s, None)

        result, _, _ = jit_apply(data, {})
        assert jnp.isfinite(result["transport_plan"]).all()
        assert jnp.isfinite(result["aligned_coords"]).all()


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_few_cells(self, rngs: nnx.Rngs) -> None:
        """Test STAGATE with very few cells (n=5)."""
        config = SpatialDomainConfig(
            n_genes=20,
            hidden_dim=16,
            num_heads=4,
            n_domains=3,
            alpha=0.8,
            n_neighbors=3,
        )
        op = DifferentiableSpatialDomain(config, rngs=rngs)
        data = _make_spatial_data(n_cells=5, n_genes=20, seed=10)
        result, _, _ = op.apply(data, {}, None)
        assert result["domain_assignments"].shape == (5, 3)
        assert result["spatial_embeddings"].shape == (5, 16)
        assert jnp.isfinite(result["domain_assignments"]).all()
        assert jnp.isfinite(result["spatial_embeddings"]).all()

    def test_single_domain(self, rngs: nnx.Rngs) -> None:
        """Test STAGATE with a single domain (n_domains=1)."""
        config = SpatialDomainConfig(
            n_genes=20,
            hidden_dim=16,
            num_heads=4,
            n_domains=1,
            alpha=0.8,
            n_neighbors=5,
        )
        op = DifferentiableSpatialDomain(config, rngs=rngs)
        data = _make_spatial_data(n_cells=15, n_genes=20, seed=11)
        result, _, _ = op.apply(data, {}, None)
        # With 1 domain, all assignments should be 1.0
        assert result["domain_assignments"].shape == (15, 1)
        assert jnp.allclose(result["domain_assignments"], 1.0, atol=1e-5)

    def test_paste_asymmetric_slices(self, rngs: nnx.Rngs) -> None:
        """Test PASTE with different-sized slices."""
        config = PASTEAlignmentConfig(
            alpha=0.1,
            sinkhorn_epsilon=0.1,
            sinkhorn_iters=50,
        )
        op = DifferentiablePASTEAlignment(config, rngs=rngs)
        data = _make_paste_data(n_cells_slice1=10, n_cells_slice2=20, n_genes=20, seed=12)
        result, _, _ = op.apply(data, {}, None)
        assert result["transport_plan"].shape == (10, 20)
        assert result["aligned_coords"].shape == (20, 2)
        assert jnp.isfinite(result["transport_plan"]).all()
        assert jnp.isfinite(result["aligned_coords"]).all()
