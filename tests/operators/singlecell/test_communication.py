"""Tests for diffbio.operators.singlecell.communication module.

These tests define the expected behavior of the DifferentiableLigandReceptor
operator for ligand-receptor co-expression scoring in single-cell data.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.communication import (
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
