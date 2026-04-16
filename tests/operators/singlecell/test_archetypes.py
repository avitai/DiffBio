"""Tests for diffbio.operators.singlecell.archetypes module.

These tests define the expected behavior of the DifferentiableArchetypalAnalysis
operator implementing PCHA-style archetypal analysis with softmax simplex
constraints.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.archetypes import (
    ArchetypalAnalysisConfig,
    DifferentiableArchetypalAnalysis,
)

# ---------------------------------------------------------------------------
# Small dimensions for fast tests
# ---------------------------------------------------------------------------
N_GENES = 20
N_ARCHETYPES = 3
HIDDEN_DIM = 16
N_CELLS = 30


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config() -> ArchetypalAnalysisConfig:
    """Small config for fast tests."""
    return ArchetypalAnalysisConfig(
        n_genes=N_GENES,
        n_archetypes=N_ARCHETYPES,
        hidden_dim=HIDDEN_DIM,
    )


@pytest.fixture
def sample_counts() -> dict[str, jax.Array]:
    """Sample count matrix (n_cells, n_genes)."""
    key = jax.random.key(0)
    counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=0.0, maxval=10.0)
    return {"counts": counts}


# ---------------------------------------------------------------------------
# TestArchetypalConfig
# ---------------------------------------------------------------------------


class TestArchetypalConfig:
    """Tests for ArchetypalAnalysisConfig defaults and custom values."""

    def test_defaults(self) -> None:
        """Default config matches specification."""
        config = ArchetypalAnalysisConfig()
        assert config.n_genes == 2000
        assert config.n_archetypes == 5
        assert config.hidden_dim == 64
        assert config.temperature == 1.0
        assert config.learnable_temperature is False
        assert config.stochastic is False

    def test_custom(self) -> None:
        """Custom values are stored correctly."""
        config = ArchetypalAnalysisConfig(
            n_genes=100,
            n_archetypes=8,
            hidden_dim=32,
            temperature=0.5,
            learnable_temperature=True,
        )
        assert config.n_genes == 100
        assert config.n_archetypes == 8
        assert config.hidden_dim == 32
        assert config.temperature == 0.5
        assert config.learnable_temperature is True


# ---------------------------------------------------------------------------
# TestArchetypalAnalysis
# ---------------------------------------------------------------------------


class TestArchetypalAnalysis:
    """Tests for DifferentiableArchetypalAnalysis output keys and shapes."""

    def test_output_keys(
        self, rngs: nnx.Rngs, small_config: ArchetypalAnalysisConfig, sample_counts: dict
    ) -> None:
        """Output dictionary contains all expected keys."""
        op = DifferentiableArchetypalAnalysis(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_counts, {}, None)

        assert "archetype_weights" in result
        assert "archetypes" in result
        assert "reconstructed" in result
        # Original data is preserved
        assert "counts" in result

    def test_output_shapes(
        self, rngs: nnx.Rngs, small_config: ArchetypalAnalysisConfig, sample_counts: dict
    ) -> None:
        """Output tensors have correct shapes."""
        op = DifferentiableArchetypalAnalysis(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_counts, {}, None)

        assert result["archetype_weights"].shape == (N_CELLS, N_ARCHETYPES)
        assert result["archetypes"].shape == (N_ARCHETYPES, N_GENES)
        assert result["reconstructed"].shape == (N_CELLS, N_GENES)

    def test_weights_sum_to_one(
        self, rngs: nnx.Rngs, small_config: ArchetypalAnalysisConfig, sample_counts: dict
    ) -> None:
        """Archetype weights form a valid simplex (sum to 1 per cell)."""
        op = DifferentiableArchetypalAnalysis(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_counts, {}, None)

        row_sums = jnp.sum(result["archetype_weights"], axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_archetypes_finite(
        self, rngs: nnx.Rngs, small_config: ArchetypalAnalysisConfig, sample_counts: dict
    ) -> None:
        """Archetype prototypes are all finite."""
        op = DifferentiableArchetypalAnalysis(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_counts, {}, None)

        assert jnp.isfinite(result["archetypes"]).all()

    def test_reconstructed_finite(
        self, rngs: nnx.Rngs, small_config: ArchetypalAnalysisConfig, sample_counts: dict
    ) -> None:
        """Reconstructed counts are all finite."""
        op = DifferentiableArchetypalAnalysis(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_counts, {}, None)

        assert jnp.isfinite(result["reconstructed"]).all()


# ---------------------------------------------------------------------------
# TestSimplexConstraint
# ---------------------------------------------------------------------------


class TestSimplexConstraint:
    """Verify simplex properties: non-negative and sums to 1."""

    def test_weights_non_negative(
        self, rngs: nnx.Rngs, small_config: ArchetypalAnalysisConfig, sample_counts: dict
    ) -> None:
        """All archetype weights are non-negative (softmax guarantee)."""
        op = DifferentiableArchetypalAnalysis(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_counts, {}, None)

        assert jnp.all(result["archetype_weights"] >= 0.0)

    def test_weights_sum_to_one_per_cell(
        self, rngs: nnx.Rngs, small_config: ArchetypalAnalysisConfig, sample_counts: dict
    ) -> None:
        """Each cell's weights sum to 1."""
        op = DifferentiableArchetypalAnalysis(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_counts, {}, None)

        row_sums = jnp.sum(result["archetype_weights"], axis=-1)
        assert jnp.allclose(row_sums, jnp.ones(N_CELLS), atol=1e-5)


# ---------------------------------------------------------------------------
# TestGradientFlow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Gradients flow through archetype positions and encoder weights."""

    def test_grads_through_archetype_positions(
        self, rngs: nnx.Rngs, small_config: ArchetypalAnalysisConfig, sample_counts: dict
    ) -> None:
        """Gradients reach the archetype prototype parameters."""
        op = DifferentiableArchetypalAnalysis(small_config, rngs=rngs)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableArchetypalAnalysis) -> jax.Array:
            result, _, _ = model.apply(sample_counts, {}, None)
            return jnp.mean(result["reconstructed"])

        _, grads = loss_fn(op)
        assert hasattr(grads, "archetypes")
        assert jnp.isfinite(grads.archetypes[...]).all()
        assert jnp.any(grads.archetypes[...] != 0.0)

    def test_grads_through_encoder_weights(
        self, rngs: nnx.Rngs, small_config: ArchetypalAnalysisConfig, sample_counts: dict
    ) -> None:
        """Gradients reach the encoder MLP parameters."""
        op = DifferentiableArchetypalAnalysis(small_config, rngs=rngs)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableArchetypalAnalysis) -> jax.Array:
            result, _, _ = model.apply(sample_counts, {}, None)
            return jnp.mean(result["reconstructed"])

        _, grads = loss_fn(op)
        assert hasattr(grads, "encoder_layers")
        # First encoder layer must have non-zero gradients
        assert jnp.any(grads.encoder_layers.layers[0].kernel[...] != 0.0)


# ---------------------------------------------------------------------------
# TestJITCompatibility
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """JIT compilation works for both forward and gradient passes."""

    def test_jit_apply(
        self, rngs: nnx.Rngs, small_config: ArchetypalAnalysisConfig, sample_counts: dict
    ) -> None:
        """apply() runs under jax.jit without error."""
        op = DifferentiableArchetypalAnalysis(small_config, rngs=rngs)

        @jax.jit
        def jit_apply(data: dict, state: dict) -> tuple:
            return op.apply(data, state, None)

        result, _, _ = jit_apply(sample_counts, {})
        assert jnp.isfinite(result["archetype_weights"]).all()
        assert jnp.isfinite(result["reconstructed"]).all()

    def test_jit_gradient(
        self, rngs: nnx.Rngs, small_config: ArchetypalAnalysisConfig, sample_counts: dict
    ) -> None:
        """Gradient computation runs under jax.jit."""
        op = DifferentiableArchetypalAnalysis(small_config, rngs=rngs)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableArchetypalAnalysis) -> jax.Array:
            result, _, _ = model.apply(sample_counts, {}, None)
            return jnp.mean((sample_counts["counts"] - result["reconstructed"]) ** 2)

        jit_loss_fn = jax.jit(loss_fn)
        loss, grads = jit_loss_fn(op)
        assert jnp.isfinite(loss)
        assert hasattr(grads, "archetypes")


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: single archetype and n_archetypes == n_cells."""

    def test_single_archetype(self, rngs: nnx.Rngs, sample_counts: dict) -> None:
        """With one archetype, all weights must be 1.0."""
        config = ArchetypalAnalysisConfig(n_genes=N_GENES, n_archetypes=1, hidden_dim=HIDDEN_DIM)
        op = DifferentiableArchetypalAnalysis(config, rngs=rngs)
        result, _, _ = op.apply(sample_counts, {}, None)

        assert result["archetype_weights"].shape == (N_CELLS, 1)
        assert jnp.allclose(result["archetype_weights"], 1.0, atol=1e-5)

    def test_archetypes_equal_cells(self, rngs: nnx.Rngs) -> None:
        """n_archetypes == n_cells is valid and produces correct shapes."""
        n_cells = 10
        config = ArchetypalAnalysisConfig(
            n_genes=N_GENES, n_archetypes=n_cells, hidden_dim=HIDDEN_DIM
        )
        op = DifferentiableArchetypalAnalysis(config, rngs=rngs)

        key = jax.random.key(0)
        counts = jax.random.uniform(key, (n_cells, N_GENES))
        data = {"counts": counts}
        result, _, _ = op.apply(data, {}, None)

        assert result["archetype_weights"].shape == (n_cells, n_cells)
        row_sums = jnp.sum(result["archetype_weights"], axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)
