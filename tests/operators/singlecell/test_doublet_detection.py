"""Tests for diffbio.operators.singlecell.doublet_detection module.

These tests define the expected behavior of the DifferentiableDoubletScorer
operator for Scrublet-style doublet detection in single-cell data.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.doublet_detection import (
    DifferentiableDoubletScorer,
    DoubletScorerConfig,
)


class TestDoubletScorerConfig:
    """Tests for DoubletScorerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values including sim_doublet_ratio."""
        config = DoubletScorerConfig()
        assert config.n_neighbors == 30
        assert config.expected_doublet_rate == 0.06
        assert config.sim_doublet_ratio == 2.0
        assert config.n_pca_components == 30
        assert config.n_genes == 2000
        assert config.stochastic is True
        assert config.stream_name == "sample"

    def test_custom_neighbors(self) -> None:
        """Test custom number of neighbors configuration."""
        config = DoubletScorerConfig(n_neighbors=50, expected_doublet_rate=0.1)
        assert config.n_neighbors == 50
        assert config.expected_doublet_rate == 0.1

    def test_custom_sim_doublet_ratio(self) -> None:
        """Test custom sim_doublet_ratio configuration."""
        config = DoubletScorerConfig(sim_doublet_ratio=3.0)
        assert config.sim_doublet_ratio == 3.0


class TestDifferentiableDoubletScorer:
    """Tests for DifferentiableDoubletScorer operator."""

    @pytest.fixture()
    def default_config(self) -> DoubletScorerConfig:
        """Provide default config with smaller sizes for fast tests."""
        return DoubletScorerConfig(
            n_neighbors=5,
            n_pca_components=5,
            n_genes=20,
        )

    @pytest.fixture()
    def count_data(self) -> dict[str, jax.Array]:
        """Provide count data for testing."""
        key = jax.random.key(0)
        n_cells, n_genes = 30, 20
        counts = jnp.abs(jax.random.normal(key, (n_cells, n_genes))) * 5.0 + 0.1
        return {"counts": counts}

    def test_output_keys(
        self,
        rngs: nnx.Rngs,
        default_config: DoubletScorerConfig,
        count_data: dict[str, jax.Array],
    ) -> None:
        """Test that apply returns doublet_scores and predicted_doublets."""
        op = DifferentiableDoubletScorer(default_config, rngs=rngs)
        rng_key = jax.random.key(99)
        random_params = op.generate_random_params(
            rng_key, {"counts": count_data["counts"].shape}
        )
        result, state, metadata = op.apply(
            count_data, {}, None, random_params=random_params
        )

        assert "doublet_scores" in result
        assert "predicted_doublets" in result
        assert "counts" in result

    def test_output_shapes(
        self,
        rngs: nnx.Rngs,
        default_config: DoubletScorerConfig,
        count_data: dict[str, jax.Array],
    ) -> None:
        """Test that doublet_scores and predicted_doublets have shape (n_cells,)."""
        op = DifferentiableDoubletScorer(default_config, rngs=rngs)
        n_cells = count_data["counts"].shape[0]
        rng_key = jax.random.key(99)
        random_params = op.generate_random_params(
            rng_key, {"counts": count_data["counts"].shape}
        )
        result, _, _ = op.apply(count_data, {}, None, random_params=random_params)

        assert result["doublet_scores"].shape == (n_cells,)
        assert result["predicted_doublets"].shape == (n_cells,)

    def test_scores_non_negative(
        self,
        rngs: nnx.Rngs,
        default_config: DoubletScorerConfig,
        count_data: dict[str, jax.Array],
    ) -> None:
        """Test that Bayesian doublet scores are non-negative."""
        op = DifferentiableDoubletScorer(default_config, rngs=rngs)
        rng_key = jax.random.key(99)
        random_params = op.generate_random_params(
            rng_key, {"counts": count_data["counts"].shape}
        )
        result, _, _ = op.apply(count_data, {}, None, random_params=random_params)

        scores = result["doublet_scores"]
        assert jnp.all(scores >= 0.0)

    def test_scores_finite(
        self,
        rngs: nnx.Rngs,
        default_config: DoubletScorerConfig,
        count_data: dict[str, jax.Array],
    ) -> None:
        """Test that all doublet scores are finite (no NaN or Inf)."""
        op = DifferentiableDoubletScorer(default_config, rngs=rngs)
        rng_key = jax.random.key(99)
        random_params = op.generate_random_params(
            rng_key, {"counts": count_data["counts"].shape}
        )
        result, _, _ = op.apply(count_data, {}, None, random_params=random_params)

        assert jnp.isfinite(result["doublet_scores"]).all()
        assert jnp.isfinite(result["predicted_doublets"]).all()

    def test_known_doublets_score_higher(self, rngs: nnx.Rngs) -> None:
        """Test that synthetic known doublets score higher than singlets.

        Creates two distinct clusters of singlets, then adds doublets that are
        sums of cells from different clusters.
        """
        key = jax.random.key(7)
        n_genes = 15
        n_singlets = 20

        # Create two distinct clusters
        k1, k2, k3 = jax.random.split(key, 3)
        cluster_a = jnp.abs(jax.random.normal(k1, (n_singlets // 2, n_genes))) + 0.1
        cluster_b = (
            jnp.abs(jax.random.normal(k2, (n_singlets // 2, n_genes))) + 10.0
        )

        # Create doublets by summing pairs from each cluster
        n_doublets = 5
        doublets = cluster_a[:n_doublets] + cluster_b[:n_doublets]

        # Combined dataset: singlets first, then doublets
        counts = jnp.concatenate([cluster_a, cluster_b, doublets], axis=0)

        config = DoubletScorerConfig(
            n_neighbors=5,
            n_pca_components=5,
            n_genes=n_genes,
        )
        op = DifferentiableDoubletScorer(config, rngs=rngs)
        rng_key = jax.random.key(42)
        random_params = op.generate_random_params(
            rng_key, {"counts": counts.shape}
        )
        result, _, _ = op.apply(
            {"counts": counts}, {}, None, random_params=random_params
        )

        scores = result["doublet_scores"]
        singlet_scores = scores[:n_singlets]
        doublet_scores = scores[n_singlets:]

        # Doublets should have higher mean score than singlets
        assert jnp.mean(doublet_scores) > jnp.mean(singlet_scores)

    def test_higher_rho_increases_scores(self, rngs: nnx.Rngs) -> None:
        """Test that higher expected_doublet_rate (rho) produces higher scores.

        The Bayesian likelihood ratio is proportional to rho in the numerator,
        so increasing the prior doublet rate should increase all scores.
        """
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (20, 15))) + 0.1
        data = {"counts": counts}
        rng_key = jax.random.key(99)

        config_low = DoubletScorerConfig(
            n_neighbors=5,
            n_pca_components=5,
            n_genes=15,
            expected_doublet_rate=0.02,
        )
        config_high = DoubletScorerConfig(
            n_neighbors=5,
            n_pca_components=5,
            n_genes=15,
            expected_doublet_rate=0.20,
        )

        op_low = DifferentiableDoubletScorer(config_low, rngs=rngs)
        op_high = DifferentiableDoubletScorer(config_high, rngs=rngs)

        rp_low = op_low.generate_random_params(
            rng_key, {"counts": counts.shape}
        )
        rp_high = op_high.generate_random_params(
            rng_key, {"counts": counts.shape}
        )

        result_low, _, _ = op_low.apply(data, {}, None, random_params=rp_low)
        result_high, _, _ = op_high.apply(
            data, {}, None, random_params=rp_high
        )

        assert jnp.mean(result_high["doublet_scores"]) > jnp.mean(
            result_low["doublet_scores"]
        )

    def test_sim_doublet_ratio_affects_synthetic_count(
        self, rngs: nnx.Rngs
    ) -> None:
        """Test that sim_doublet_ratio controls the number of synthetics."""
        key = jax.random.key(0)
        n_cells, n_genes = 20, 15
        counts = jnp.abs(jax.random.normal(key, (n_cells, n_genes))) + 0.1
        rng_key = jax.random.key(42)

        config_1x = DoubletScorerConfig(
            n_neighbors=5,
            n_pca_components=5,
            n_genes=n_genes,
            sim_doublet_ratio=1.0,
        )
        config_3x = DoubletScorerConfig(
            n_neighbors=5,
            n_pca_components=5,
            n_genes=n_genes,
            sim_doublet_ratio=3.0,
        )

        op_1x = DifferentiableDoubletScorer(config_1x, rngs=rngs)
        op_3x = DifferentiableDoubletScorer(config_3x, rngs=rngs)

        syn_1x = op_1x._generate_synthetic_doublets(counts, rng_key, 1.0)
        syn_3x = op_3x._generate_synthetic_doublets(counts, rng_key, 3.0)

        assert syn_1x.shape[0] == n_cells
        assert syn_3x.shape[0] == n_cells * 3


class TestGradientFlow:
    """Tests for gradient flow through doublet detection."""

    @pytest.fixture()
    def config(self) -> DoubletScorerConfig:
        """Provide config for gradient tests."""
        return DoubletScorerConfig(
            n_neighbors=5,
            n_pca_components=5,
            n_genes=15,
        )

    def test_gradient_wrt_input(
        self, rngs: nnx.Rngs, config: DoubletScorerConfig
    ) -> None:
        """Test that gradients flow from doublet_scores back to input counts."""
        op = DifferentiableDoubletScorer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (20, 15))) + 0.1

        rng_key = jax.random.key(99)
        random_params = op.generate_random_params(
            rng_key, {"counts": counts.shape}
        )

        def loss_fn(c: jax.Array) -> jax.Array:
            result, _, _ = op.apply(
                {"counts": c}, {}, None, random_params=random_params
            )
            return jnp.sum(result["doublet_scores"])

        grad = jax.grad(loss_fn)(counts)
        assert grad is not None
        assert grad.shape == counts.shape
        assert jnp.any(grad != 0.0)

    def test_gradient_finite(
        self, rngs: nnx.Rngs, config: DoubletScorerConfig
    ) -> None:
        """Test that all gradients are finite (no NaN or Inf)."""
        op = DifferentiableDoubletScorer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (20, 15))) + 0.1

        rng_key = jax.random.key(99)
        random_params = op.generate_random_params(
            rng_key, {"counts": counts.shape}
        )

        def loss_fn(c: jax.Array) -> jax.Array:
            result, _, _ = op.apply(
                {"counts": c}, {}, None, random_params=random_params
            )
            return jnp.sum(result["doublet_scores"])

        grad = jax.grad(loss_fn)(counts)
        assert jnp.isfinite(grad).all()


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    @pytest.fixture()
    def config(self) -> DoubletScorerConfig:
        """Provide config for JIT tests."""
        return DoubletScorerConfig(
            n_neighbors=5,
            n_pca_components=5,
            n_genes=15,
        )

    def test_jit_apply(
        self, rngs: nnx.Rngs, config: DoubletScorerConfig
    ) -> None:
        """Test that jax.jit compiles and runs apply."""
        op = DifferentiableDoubletScorer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (20, 15))) + 0.1

        rng_key = jax.random.key(99)
        random_params = op.generate_random_params(
            rng_key, {"counts": counts.shape}
        )

        @jax.jit
        def jit_apply(
            d: dict[str, jax.Array],
            rp: jax.Array,
        ) -> tuple:
            return op.apply(d, {}, None, random_params=rp)

        result, _, _ = jit_apply({"counts": counts}, random_params)
        assert jnp.isfinite(result["doublet_scores"]).all()

    def test_jit_gradient(
        self, rngs: nnx.Rngs, config: DoubletScorerConfig
    ) -> None:
        """Test that jax.jit + jax.grad works together."""
        op = DifferentiableDoubletScorer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (20, 15))) + 0.1

        rng_key = jax.random.key(99)
        random_params = op.generate_random_params(
            rng_key, {"counts": counts.shape}
        )

        @jax.jit
        def grad_fn(c: jax.Array) -> jax.Array:
            def loss(x: jax.Array) -> jax.Array:
                result, _, _ = op.apply(
                    {"counts": x}, {}, None, random_params=random_params
                )
                return jnp.sum(result["doublet_scores"])

            return jax.grad(loss)(c)

        grad = grad_fn(counts)
        assert jnp.isfinite(grad).all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_dataset(self, rngs: nnx.Rngs) -> None:
        """Test that doublet detection works with only 10 cells."""
        config = DoubletScorerConfig(
            n_neighbors=3,
            n_pca_components=3,
            n_genes=8,
        )
        op = DifferentiableDoubletScorer(config, rngs=rngs)
        key = jax.random.key(0)
        counts = jnp.abs(jax.random.normal(key, (10, 8))) + 0.1

        rng_key = jax.random.key(99)
        random_params = op.generate_random_params(
            rng_key, {"counts": counts.shape}
        )
        result, _, _ = op.apply(
            {"counts": counts}, {}, None, random_params=random_params
        )

        assert result["doublet_scores"].shape == (10,)
        assert jnp.isfinite(result["doublet_scores"]).all()

    def test_all_identical_cells(self, rngs: nnx.Rngs) -> None:
        """Test that identical cells produce uniform scores."""
        n_cells, n_genes = 20, 10
        counts = jnp.ones((n_cells, n_genes)) * 3.0

        config = DoubletScorerConfig(
            n_neighbors=5,
            n_pca_components=5,
            n_genes=n_genes,
        )
        op = DifferentiableDoubletScorer(config, rngs=rngs)
        rng_key = jax.random.key(99)
        random_params = op.generate_random_params(
            rng_key, {"counts": counts.shape}
        )
        result, _, _ = op.apply(
            {"counts": counts}, {}, None, random_params=random_params
        )

        scores = result["doublet_scores"]
        # All identical cells -> scores should have very low variance
        assert jnp.std(scores) < 0.1
