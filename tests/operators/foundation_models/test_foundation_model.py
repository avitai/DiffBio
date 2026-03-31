"""Tests for diffbio.operators.foundation_models.foundation_model module.

These tests define the expected behavior of GeneTokenizer, FoundationModelConfig,
and DifferentiableFoundationModel for Geneformer/scGPT-style single-cell
foundation model infrastructure.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.foundation_models.foundation_model import (
    DifferentiableFoundationModel,
    FoundationModelConfig,
    GeneTokenizer,
)

# ---------------------------------------------------------------------------
# Small test dimensions
# ---------------------------------------------------------------------------
N_GENES = 50
HIDDEN_DIM = 32
NUM_LAYERS = 1
NUM_HEADS = 2
N_CELLS = 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_config() -> FoundationModelConfig:
    """Provide a small foundation model config for fast testing."""
    return FoundationModelConfig(
        n_genes=N_GENES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mask_ratio=0.15,
        dropout_rate=0.0,
        stochastic=True,
        stream_name="sample",
    )


@pytest.fixture()
def model(small_config: FoundationModelConfig) -> DifferentiableFoundationModel:
    """Provide an initialized foundation model."""
    rngs = nnx.Rngs(params=0, sample=1, dropout=2)
    return DifferentiableFoundationModel(small_config, rngs=rngs)


@pytest.fixture()
def sample_data() -> dict[str, jax.Array]:
    """Provide sample single-cell expression data."""
    key = jax.random.key(42)
    counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=0.0, maxval=10.0)
    gene_ids = jnp.arange(N_GENES, dtype=jnp.int32)
    return {"counts": counts, "gene_ids": gene_ids}


# ---------------------------------------------------------------------------
# TestFoundationModelConfig
# ---------------------------------------------------------------------------


class TestFoundationModelConfig:
    """Tests for FoundationModelConfig defaults and custom values."""

    def test_default_values(self) -> None:
        """Test that default config values match specification."""
        config = FoundationModelConfig()
        assert config.n_genes == 2000
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.num_heads == 4
        assert config.mask_ratio == 0.15
        assert config.dropout_rate == 0.1
        assert config.stochastic is True
        assert config.stream_name == "sample"

    def test_custom_values(self) -> None:
        """Test overriding config values."""
        config = FoundationModelConfig(
            n_genes=500,
            hidden_dim=64,
            num_layers=3,
            num_heads=8,
            mask_ratio=0.3,
        )
        assert config.n_genes == 500
        assert config.hidden_dim == 64
        assert config.num_layers == 3
        assert config.num_heads == 8
        assert config.mask_ratio == 0.3


# ---------------------------------------------------------------------------
# TestGeneTokenizer
# ---------------------------------------------------------------------------


class TestGeneTokenizer:
    """Tests for Geneformer-style rank-value gene tokenization."""

    def test_rank_ordering(self) -> None:
        """Test that genes are ranked by expression value in descending order."""
        rngs = nnx.Rngs(0)
        tokenizer = GeneTokenizer(n_genes=5, rngs=rngs)

        # Expression: gene 3 has highest, gene 0 has lowest
        expression = jnp.array([1.0, 3.0, 2.0, 5.0, 4.0])

        # Soft ranks should approximate the true descending order
        soft_ranks = tokenizer(expression, temperature=0.01)
        # With very low temperature, argmax of each row should give rank order
        hard_ranks = jnp.argmax(soft_ranks, axis=-1)

        # Gene 3 (value 5.0) should be ranked first (position 0)
        # Gene 4 (value 4.0) should be ranked second (position 1), etc.
        expected_order = jnp.array([3, 4, 1, 2, 0])
        assert jnp.allclose(hard_ranks, expected_order), (
            f"Expected order {expected_order}, got {hard_ranks}"
        )

    def test_differentiable_through_soft_ranking(self) -> None:
        """Test that gradients flow through the soft ranking operation."""
        rngs = nnx.Rngs(0)
        tokenizer = GeneTokenizer(n_genes=5, rngs=rngs)

        def loss_fn(expression: jax.Array) -> jax.Array:
            soft_ranks = tokenizer(expression, temperature=1.0)
            return jnp.sum(soft_ranks)

        expression = jnp.array([1.0, 3.0, 2.0, 5.0, 4.0])
        grads = jax.grad(loss_fn)(expression)

        assert grads is not None
        assert grads.shape == expression.shape
        assert jnp.all(jnp.isfinite(grads))


# ---------------------------------------------------------------------------
# TestDifferentiableFoundationModel
# ---------------------------------------------------------------------------


class TestDifferentiableFoundationModel:
    """Tests for the foundation model operator."""

    def test_output_keys(
        self,
        model: DifferentiableFoundationModel,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that apply returns all expected output keys."""
        rp = model.generate_random_params(
            jax.random.key(0), {"counts": sample_data["counts"].shape}
        )
        result, state, metadata = model.apply(sample_data, {}, None, random_params=rp)

        assert "gene_embeddings" in result
        assert "cell_embeddings" in result
        assert "predicted_expression" in result
        # Original data preserved
        assert "counts" in result
        assert "gene_ids" in result

    def test_output_shapes(
        self,
        model: DifferentiableFoundationModel,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test correct output tensor shapes."""
        rp = model.generate_random_params(
            jax.random.key(0), {"counts": sample_data["counts"].shape}
        )
        result, _, _ = model.apply(sample_data, {}, None, random_params=rp)

        assert result["gene_embeddings"].shape == (N_GENES, HIDDEN_DIM)
        assert result["cell_embeddings"].shape == (N_CELLS, HIDDEN_DIM)
        assert result["predicted_expression"].shape == (N_CELLS, N_GENES)

    def test_masked_genes_predicted(
        self,
        model: DifferentiableFoundationModel,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that masked gene expressions are predicted (not NaN/Inf)."""
        rp = model.generate_random_params(
            jax.random.key(0), {"counts": sample_data["counts"].shape}
        )
        result, _, _ = model.apply(sample_data, {}, None, random_params=rp)

        predicted = result["predicted_expression"]
        assert jnp.all(jnp.isfinite(predicted))

    def test_cell_embeddings_finite(
        self,
        model: DifferentiableFoundationModel,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that cell embeddings are finite."""
        rp = model.generate_random_params(
            jax.random.key(0), {"counts": sample_data["counts"].shape}
        )
        result, _, _ = model.apply(sample_data, {}, None, random_params=rp)

        assert jnp.all(jnp.isfinite(result["cell_embeddings"]))
        assert jnp.all(jnp.isfinite(result["gene_embeddings"]))


# ---------------------------------------------------------------------------
# TestGradientFlow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Tests for end-to-end gradient flow through the model."""

    def test_gradient_through_masked_prediction(
        self,
        model: DifferentiableFoundationModel,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test gradient flows from predicted expression back to input counts."""
        gene_ids = sample_data["gene_ids"]
        counts = sample_data["counts"]
        rp = model.generate_random_params(jax.random.key(0), {"counts": counts.shape})

        def loss_fn(c: jax.Array) -> jax.Array:
            data = {"counts": c, "gene_ids": gene_ids}
            result, _, _ = model.apply(data, {}, None, random_params=rp)
            return jnp.mean(result["predicted_expression"] ** 2)

        grad = jax.grad(loss_fn)(counts)
        assert grad is not None
        assert grad.shape == counts.shape
        assert jnp.any(grad != 0), "All gradients are zero"
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_through_gene_embeddings(
        self,
        model: DifferentiableFoundationModel,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test gradient flows from gene embeddings back to input counts."""
        gene_ids = sample_data["gene_ids"]
        counts = sample_data["counts"]
        rp = model.generate_random_params(jax.random.key(0), {"counts": counts.shape})

        def loss_fn(c: jax.Array) -> jax.Array:
            data = {"counts": c, "gene_ids": gene_ids}
            result, _, _ = model.apply(data, {}, None, random_params=rp)
            return jnp.mean(result["gene_embeddings"] ** 2)

        grad = jax.grad(loss_fn)(counts)
        assert grad is not None
        assert grad.shape == counts.shape
        assert jnp.any(grad != 0), "All gradients are zero for gene embeddings loss"
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# TestJITCompatibility
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """Tests for JIT compilation compatibility."""

    def test_jit_apply(
        self,
        model: DifferentiableFoundationModel,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that apply runs under jax.jit."""
        gene_ids = sample_data["gene_ids"]
        counts = sample_data["counts"]
        rp = model.generate_random_params(jax.random.key(0), {"counts": counts.shape})

        @jax.jit
        def run(c: jax.Array) -> dict[str, jax.Array]:
            data = {"counts": c, "gene_ids": gene_ids}
            result, _, _ = model.apply(data, {}, None, random_params=rp)
            return result

        result = run(counts)
        assert result["cell_embeddings"].shape == (N_CELLS, HIDDEN_DIM)
        assert jnp.all(jnp.isfinite(result["cell_embeddings"]))

    def test_jit_gradient(
        self,
        model: DifferentiableFoundationModel,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that gradient computation runs under jax.jit."""
        gene_ids = sample_data["gene_ids"]
        counts = sample_data["counts"]
        rp = model.generate_random_params(jax.random.key(0), {"counts": counts.shape})

        @jax.jit
        def grad_fn(c: jax.Array) -> jax.Array:
            def loss(x: jax.Array) -> jax.Array:
                data = {"counts": x, "gene_ids": gene_ids}
                result, _, _ = model.apply(data, {}, None, random_params=rp)
                return jnp.mean(result["predicted_expression"])

            return jax.grad(loss)(c)

        grad = grad_fn(counts)
        assert grad.shape == counts.shape
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_single_cell(self) -> None:
        """Test with a single cell (n_cells=1)."""
        config = FoundationModelConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            mask_ratio=0.15,
            dropout_rate=0.0,
            stochastic=True,
            stream_name="sample",
        )
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        mdl = DifferentiableFoundationModel(config, rngs=rngs)

        key = jax.random.key(99)
        counts = jax.random.uniform(key, (1, N_GENES), minval=0.0, maxval=10.0)
        gene_ids = jnp.arange(N_GENES, dtype=jnp.int32)
        data = {"counts": counts, "gene_ids": gene_ids}

        rp = mdl.generate_random_params(jax.random.key(0), {"counts": counts.shape})
        result, _, _ = mdl.apply(data, {}, None, random_params=rp)

        assert result["cell_embeddings"].shape == (1, HIDDEN_DIM)
        assert result["predicted_expression"].shape == (1, N_GENES)
        assert jnp.all(jnp.isfinite(result["cell_embeddings"]))

    def test_no_masking(self) -> None:
        """Test with mask_ratio=0 (no masking)."""
        config = FoundationModelConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            mask_ratio=0.0,
            dropout_rate=0.0,
            stochastic=True,
            stream_name="sample",
        )
        rngs = nnx.Rngs(params=0, sample=1, dropout=2)
        mdl = DifferentiableFoundationModel(config, rngs=rngs)

        key = jax.random.key(42)
        counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=0.0, maxval=10.0)
        gene_ids = jnp.arange(N_GENES, dtype=jnp.int32)
        data = {"counts": counts, "gene_ids": gene_ids}

        rp = mdl.generate_random_params(jax.random.key(0), {"counts": counts.shape})
        result, _, _ = mdl.apply(data, {}, None, random_params=rp)

        assert result["predicted_expression"].shape == (N_CELLS, N_GENES)
        assert jnp.all(jnp.isfinite(result["predicted_expression"]))
        assert jnp.all(jnp.isfinite(result["cell_embeddings"]))
