"""Tests for diffbio.operators.variant.classifier CellTypeAwareVariantClassifier.

These tests define the expected behavior of the CellTypeAwareVariantClassifier
operator, which performs cell-type-weighted variant calling using per-type
classification heads.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.variant.classifier import (
    CellTypeAwareVariantClassifier,
    CellTypeAwareVariantClassifierConfig,
)

# Small dims for fast tests
PILEUP_CHANNELS = 4
PILEUP_WIDTH = 20
HIDDEN_DIM = 16
N_CELL_TYPES = 3
N_CLASSES = 2
N_CELLS = 15


@pytest.fixture
def small_config() -> CellTypeAwareVariantClassifierConfig:
    """Provide small config for fast tests."""
    return CellTypeAwareVariantClassifierConfig(
        n_classes=N_CLASSES,
        hidden_dim=HIDDEN_DIM,
        n_cell_types=N_CELL_TYPES,
        pileup_channels=PILEUP_CHANNELS,
        pileup_width=PILEUP_WIDTH,
    )


@pytest.fixture
def sample_data() -> dict[str, jax.Array]:
    """Provide sample input data for cell-type-aware variant calling."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    pileup = jax.random.uniform(k1, (N_CELLS, PILEUP_CHANNELS, PILEUP_WIDTH))
    # Soft cell-type assignments that sum to 1 per cell
    raw_assignments = jax.random.uniform(k2, (N_CELLS, N_CELL_TYPES))
    cell_type_assignments = raw_assignments / raw_assignments.sum(axis=-1, keepdims=True)
    return {"pileup": pileup, "cell_type_assignments": cell_type_assignments}


class TestCellTypeConfig:
    """Tests for CellTypeAwareVariantClassifierConfig defaults."""

    def test_default_n_classes(self) -> None:
        """Test default n_classes is 3."""
        config = CellTypeAwareVariantClassifierConfig()
        assert config.n_classes == 3

    def test_default_hidden_dim(self) -> None:
        """Test default hidden_dim is 64."""
        config = CellTypeAwareVariantClassifierConfig()
        assert config.hidden_dim == 64

    def test_default_n_cell_types(self) -> None:
        """Test default n_cell_types is 5."""
        config = CellTypeAwareVariantClassifierConfig()
        assert config.n_cell_types == 5

    def test_default_pileup_channels(self) -> None:
        """Test default pileup_channels is 6."""
        config = CellTypeAwareVariantClassifierConfig()
        assert config.pileup_channels == 6

    def test_default_pileup_width(self) -> None:
        """Test default pileup_width is 100."""
        config = CellTypeAwareVariantClassifierConfig()
        assert config.pileup_width == 100

    def test_custom_values(self) -> None:
        """Test that custom values are applied correctly."""
        config = CellTypeAwareVariantClassifierConfig(
            n_classes=5,
            hidden_dim=128,
            n_cell_types=10,
            pileup_channels=8,
            pileup_width=50,
        )
        assert config.n_classes == 5
        assert config.hidden_dim == 128
        assert config.n_cell_types == 10
        assert config.pileup_channels == 8
        assert config.pileup_width == 50

    def test_not_stochastic_by_default(self) -> None:
        """Test that the operator is non-stochastic by default."""
        config = CellTypeAwareVariantClassifierConfig()
        assert config.stochastic is False


class TestCellTypeVariantClassifier:
    """Tests for CellTypeAwareVariantClassifier operator."""

    def test_output_keys(self, rngs, small_config, sample_data) -> None:
        """Test that output contains expected keys."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)
        transformed, _, _ = op.apply(sample_data, {}, None)

        assert "variant_probabilities" in transformed
        assert "per_type_probabilities" in transformed

    def test_output_shapes(self, rngs, small_config, sample_data) -> None:
        """Test that output arrays have correct shapes."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)
        transformed, _, _ = op.apply(sample_data, {}, None)

        assert transformed["variant_probabilities"].shape == (N_CELLS, N_CLASSES)
        assert transformed["per_type_probabilities"].shape == (
            N_CELLS,
            N_CELL_TYPES,
            N_CLASSES,
        )

    def test_probabilities_sum_to_one(self, rngs, small_config, sample_data) -> None:
        """Test that variant probabilities sum to 1 along class axis."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)
        transformed, _, _ = op.apply(sample_data, {}, None)

        prob_sum = jnp.sum(transformed["variant_probabilities"], axis=-1)
        assert jnp.allclose(prob_sum, 1.0, atol=1e-5)

    def test_per_type_probabilities_sum_to_one(self, rngs, small_config, sample_data) -> None:
        """Test that per-type probabilities sum to 1 along class axis."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)
        transformed, _, _ = op.apply(sample_data, {}, None)

        # Each cell type head should produce valid probabilities
        per_type = transformed["per_type_probabilities"]
        prob_sum = jnp.sum(per_type, axis=-1)  # (n_cells, n_cell_types)
        assert jnp.allclose(prob_sum, 1.0, atol=1e-5)

    def test_preserves_input_keys(self, rngs, small_config, sample_data) -> None:
        """Test that input keys are preserved in output."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)
        transformed, _, _ = op.apply(sample_data, {}, None)

        assert "pileup" in transformed
        assert "cell_type_assignments" in transformed

    def test_state_and_metadata_pass_through(self, rngs, small_config, sample_data) -> None:
        """Test that state and metadata are passed through unchanged."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)
        state_in = {"some_state": jnp.array(1.0)}
        meta_in = {"info": "test"}
        _, state_out, meta_out = op.apply(sample_data, state_in, meta_in)

        assert state_out is state_in
        assert meta_out is meta_in

    def test_output_values_are_finite(self, rngs, small_config, sample_data) -> None:
        """Test that all output values are finite (no NaN or Inf)."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)
        transformed, _, _ = op.apply(sample_data, {}, None)

        assert jnp.isfinite(transformed["variant_probabilities"]).all()
        assert jnp.isfinite(transformed["per_type_probabilities"]).all()


class TestGradientFlow:
    """Tests for gradient flow through cell-type-aware classifier."""

    def test_gradient_through_cell_type_weights(self, rngs, small_config) -> None:
        """Test that gradients flow through cell-type assignment weights."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(1)
        k1, k2 = jax.random.split(key)
        pileup = jax.random.uniform(k1, (N_CELLS, PILEUP_CHANNELS, PILEUP_WIDTH))
        raw = jax.random.uniform(k2, (N_CELLS, N_CELL_TYPES))
        assignments = raw / raw.sum(axis=-1, keepdims=True)

        def loss_fn(cell_type_assignments: jax.Array) -> jax.Array:
            data = {"pileup": pileup, "cell_type_assignments": cell_type_assignments}
            transformed, _, _ = op.apply(data, {}, None)
            return transformed["variant_probabilities"][:, 0].sum()

        grad = jax.grad(loss_fn)(assignments)
        assert grad is not None
        assert grad.shape == assignments.shape
        assert jnp.isfinite(grad).all()
        # Gradients should be non-trivial (not all zeros)
        assert jnp.any(grad != 0.0)

    def test_gradient_through_pileup_input(self, rngs, small_config) -> None:
        """Test that gradients flow through pileup input."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(2)
        k1, k2 = jax.random.split(key)
        pileup = jax.random.uniform(k1, (N_CELLS, PILEUP_CHANNELS, PILEUP_WIDTH))
        raw = jax.random.uniform(k2, (N_CELLS, N_CELL_TYPES))
        assignments = raw / raw.sum(axis=-1, keepdims=True)

        def loss_fn(pileup_input: jax.Array) -> jax.Array:
            data = {"pileup": pileup_input, "cell_type_assignments": assignments}
            transformed, _, _ = op.apply(data, {}, None)
            return transformed["variant_probabilities"][:, 0].sum()

        grad = jax.grad(loss_fn)(pileup)
        assert grad is not None
        assert grad.shape == pileup.shape
        assert jnp.isfinite(grad).all()
        assert jnp.any(grad != 0.0)

    def test_gradient_through_model_params(self, rngs, small_config) -> None:
        """Test that gradients flow to model parameters."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(3)
        k1, k2 = jax.random.split(key)
        pileup = jax.random.uniform(k1, (N_CELLS, PILEUP_CHANNELS, PILEUP_WIDTH))
        raw = jax.random.uniform(k2, (N_CELLS, N_CELL_TYPES))
        assignments = raw / raw.sum(axis=-1, keepdims=True)
        data = {"pileup": pileup, "cell_type_assignments": assignments}

        @nnx.value_and_grad
        def loss_fn(model: CellTypeAwareVariantClassifier) -> jax.Array:
            transformed, _, _ = model.apply(data, {}, None)
            return transformed["variant_probabilities"][:, 0].sum()

        _, grads = loss_fn(op)
        # Verify the model has classification heads with gradients
        assert hasattr(grads, "classification_heads")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_jit_apply(self, rngs, small_config, sample_data) -> None:
        """Test that apply works under JIT compilation."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)

        @jax.jit
        def jit_apply(data: dict[str, jax.Array], state: dict) -> tuple[dict, dict, None]:
            return op.apply(data, state, None)

        transformed, _, _ = jit_apply(sample_data, {})
        assert jnp.isfinite(transformed["variant_probabilities"]).all()
        assert transformed["variant_probabilities"].shape == (N_CELLS, N_CLASSES)

    def test_jit_gradient(self, rngs, small_config) -> None:
        """Test that gradient computation works under JIT."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(4)
        k1, k2 = jax.random.split(key)
        pileup = jax.random.uniform(k1, (N_CELLS, PILEUP_CHANNELS, PILEUP_WIDTH))
        raw = jax.random.uniform(k2, (N_CELLS, N_CELL_TYPES))
        assignments = raw / raw.sum(axis=-1, keepdims=True)

        @jax.jit
        def jit_grad(pileup_input: jax.Array) -> jax.Array:
            def loss_fn(p: jax.Array) -> jax.Array:
                data = {"pileup": p, "cell_type_assignments": assignments}
                transformed, _, _ = op.apply(data, {}, None)
                return transformed["variant_probabilities"][:, 0].sum()

            return jax.grad(loss_fn)(pileup_input)

        grad = jit_grad(pileup)
        assert grad.shape == pileup.shape
        assert jnp.isfinite(grad).all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_cell_type(self, rngs) -> None:
        """Test with single cell type degenerates to standard classifier.

        With one cell type and assignment weight 1.0, the result should
        equal the output of that single classification head.
        """
        config = CellTypeAwareVariantClassifierConfig(
            n_classes=N_CLASSES,
            hidden_dim=HIDDEN_DIM,
            n_cell_types=1,
            pileup_channels=PILEUP_CHANNELS,
            pileup_width=PILEUP_WIDTH,
        )
        op = CellTypeAwareVariantClassifier(config, rngs=rngs)

        key = jax.random.key(5)
        pileup = jax.random.uniform(key, (N_CELLS, PILEUP_CHANNELS, PILEUP_WIDTH))
        # Only one cell type, all weight on it
        assignments = jnp.ones((N_CELLS, 1))
        data = {"pileup": pileup, "cell_type_assignments": assignments}

        transformed, _, _ = op.apply(data, {}, None)

        # variant_probabilities should equal per_type_probabilities[:, 0, :]
        per_type = transformed["per_type_probabilities"]
        assert jnp.allclose(
            transformed["variant_probabilities"],
            per_type[:, 0, :],
            atol=1e-5,
        )

    def test_uniform_assignments(self, rngs, small_config) -> None:
        """Test with uniform cell-type assignments.

        All types get equal weight, so final probs should be the mean
        of per-type probabilities.
        """
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(6)
        pileup = jax.random.uniform(key, (N_CELLS, PILEUP_CHANNELS, PILEUP_WIDTH))
        # Uniform assignments: each type gets 1/n_cell_types
        assignments = jnp.ones((N_CELLS, N_CELL_TYPES)) / N_CELL_TYPES
        data = {"pileup": pileup, "cell_type_assignments": assignments}

        transformed, _, _ = op.apply(data, {}, None)

        # With uniform weights, result = mean of per-type probs
        per_type = transformed["per_type_probabilities"]
        expected = jnp.mean(per_type, axis=1)
        assert jnp.allclose(
            transformed["variant_probabilities"],
            expected,
            atol=1e-5,
        )

    def test_one_hot_assignments(self, rngs, small_config) -> None:
        """Test with one-hot cell-type assignments.

        Each cell is assigned to exactly one type, so the final probs
        should equal that type's head output.
        """
        op = CellTypeAwareVariantClassifier(small_config, rngs=rngs)

        key = jax.random.key(7)
        pileup = jax.random.uniform(key, (N_CELLS, PILEUP_CHANNELS, PILEUP_WIDTH))
        # One-hot: each cell assigned to cell type 1
        assignments = jnp.zeros((N_CELLS, N_CELL_TYPES))
        assignments = assignments.at[:, 1].set(1.0)
        data = {"pileup": pileup, "cell_type_assignments": assignments}

        transformed, _, _ = op.apply(data, {}, None)

        per_type = transformed["per_type_probabilities"]
        assert jnp.allclose(
            transformed["variant_probabilities"],
            per_type[:, 1, :],
            atol=1e-5,
        )

    def test_initialization_without_rngs(self, small_config) -> None:
        """Test that operator can be initialized without rngs."""
        op = CellTypeAwareVariantClassifier(small_config, rngs=None)
        assert op is not None
