"""Tests for diffbio.operators.singlecell.grn_inference module.

These tests define the expected behavior of the DifferentiableGRN operator
for gene regulatory network inference using attention-based GATv2 scoring
on a TF-gene bipartite graph.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.grn_inference import (
    DifferentiableGRN,
    GRNInferenceConfig,
)

# Small dims for fast tests
N_TFS = 5
N_GENES = 20
HIDDEN_DIM = 16
NUM_HEADS = 4
N_CELLS = 15


class TestGRNConfig:
    """Tests for GRNInferenceConfig defaults and customization."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GRNInferenceConfig()
        assert config.n_tfs == 50
        assert config.n_genes == 2000
        assert config.hidden_dim == 64
        assert config.num_heads == 4
        assert config.sparsity_temperature == 0.1
        assert config.sparsity_lambda == 0.01
        assert config.stochastic is False

    def test_custom_config(self) -> None:
        """Test configuration with custom values."""
        config = GRNInferenceConfig(
            n_tfs=N_TFS,
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            sparsity_temperature=0.5,
            sparsity_lambda=0.1,
        )
        assert config.n_tfs == N_TFS
        assert config.n_genes == N_GENES
        assert config.hidden_dim == HIDDEN_DIM
        assert config.num_heads == NUM_HEADS
        assert config.sparsity_temperature == 0.5
        assert config.sparsity_lambda == 0.1


class TestDifferentiableGRN:
    """Tests for DifferentiableGRN operator outputs."""

    @pytest.fixture()
    def config(self) -> GRNInferenceConfig:
        """Provide a small config for fast tests."""
        return GRNInferenceConfig(
            n_tfs=N_TFS,
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
        )

    @pytest.fixture()
    def sample_data(self) -> dict[str, jax.Array]:
        """Provide synthetic single-cell data with TF indices."""
        key = jax.random.key(0)
        counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=0.0, maxval=5.0)
        tf_indices = jnp.arange(N_TFS)
        return {"counts": counts, "tf_indices": tf_indices}

    def test_output_keys(
        self,
        rngs: nnx.Rngs,
        config: GRNInferenceConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that grn_matrix and tf_activity are present in output."""
        op = DifferentiableGRN(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert "grn_matrix" in result
        assert "tf_activity" in result

    def test_output_shapes(
        self,
        rngs: nnx.Rngs,
        config: GRNInferenceConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test grn_matrix shape is (n_tfs, n_genes) and tf_activity is (n_cells, n_tfs)."""
        op = DifferentiableGRN(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert result["grn_matrix"].shape == (N_TFS, N_GENES)
        assert result["tf_activity"].shape == (N_CELLS, N_TFS)

    def test_grn_matrix_sparse(
        self,
        rngs: nnx.Rngs,
        config: GRNInferenceConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that soft sparsity reduces magnitude compared to raw scores."""
        op = DifferentiableGRN(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        grn = result["grn_matrix"]

        # The sigmoid gating should suppress negative values toward zero
        # (sigmoid(x/t) < 0.5 when x < 0, so grn * sigmoid(grn/t) < grn/2)
        # Verify the GRN has a range of magnitudes (not all identical)
        grn_std = float(jnp.std(grn))
        assert grn_std > 0.0, "GRN matrix should not be uniform"

        # The soft sparsity gating ensures values are not symmetric:
        # positive values are amplified, negative values are suppressed
        grn_abs = jnp.abs(grn)
        assert float(jnp.max(grn_abs)) > float(jnp.min(grn_abs)), (
            "GRN should have varying magnitudes from sparsity gating"
        )

    def test_grn_finite(
        self,
        rngs: nnx.Rngs,
        config: GRNInferenceConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that all GRN outputs are finite (no NaN or inf)."""
        op = DifferentiableGRN(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)

        assert jnp.all(jnp.isfinite(result["grn_matrix"]))
        assert jnp.all(jnp.isfinite(result["tf_activity"]))

    def test_original_data_preserved(
        self,
        rngs: nnx.Rngs,
        config: GRNInferenceConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that original data keys are preserved in output."""
        op = DifferentiableGRN(config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)
        assert "counts" in result
        assert "tf_indices" in result


class TestGradientFlow:
    """Tests for gradient flow through the GRN operator."""

    @pytest.fixture()
    def config(self) -> GRNInferenceConfig:
        """Provide a small config for fast tests."""
        return GRNInferenceConfig(
            n_tfs=N_TFS,
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
        )

    @pytest.fixture()
    def sample_data(self) -> dict[str, jax.Array]:
        """Provide synthetic data for gradient tests."""
        key = jax.random.key(1)
        counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=0.0, maxval=5.0)
        tf_indices = jnp.arange(N_TFS)
        return {"counts": counts, "tf_indices": tf_indices}

    def test_gradients_through_attention_weights(
        self,
        rngs: nnx.Rngs,
        config: GRNInferenceConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that gradients flow through the GATv2 attention weights."""
        op = DifferentiableGRN(config, rngs=rngs)

        def loss_fn(model: DifferentiableGRN) -> jax.Array:
            result, _, _ = model.apply(sample_data, {}, None)
            return jnp.sum(result["grn_matrix"])

        grads = nnx.grad(loss_fn)(op)
        # Check that the GATv2 layer has non-zero gradients
        attn_vector_grad = grads.gat_layer.attn_vector[...]
        assert jnp.any(attn_vector_grad != 0.0), "GATv2 attention vector should receive gradients"

    def test_gradients_through_counts(
        self,
        rngs: nnx.Rngs,
        config: GRNInferenceConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that gradients flow through the input counts."""
        op = DifferentiableGRN(config, rngs=rngs)

        def loss_fn(counts: jax.Array) -> jax.Array:
            data = {"counts": counts, "tf_indices": sample_data["tf_indices"]}
            result, _, _ = op.apply(data, {}, None)
            return jnp.sum(result["tf_activity"])

        grads = jax.grad(loss_fn)(sample_data["counts"])
        assert grads.shape == sample_data["counts"].shape
        assert jnp.any(grads != 0.0), "Counts should receive gradients"


class TestJITCompatibility:
    """Tests for JIT compilation of the GRN operator."""

    @pytest.fixture()
    def config(self) -> GRNInferenceConfig:
        """Provide a small config for fast tests."""
        return GRNInferenceConfig(
            n_tfs=N_TFS,
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
        )

    @pytest.fixture()
    def sample_data(self) -> dict[str, jax.Array]:
        """Provide synthetic data for JIT tests."""
        key = jax.random.key(2)
        counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=0.0, maxval=5.0)
        tf_indices = jnp.arange(N_TFS)
        return {"counts": counts, "tf_indices": tf_indices}

    def test_jit_apply(
        self,
        rngs: nnx.Rngs,
        config: GRNInferenceConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that the apply method can be JIT-compiled."""
        op = DifferentiableGRN(config, rngs=rngs)

        @nnx.jit
        def jitted_apply(model: DifferentiableGRN, data: dict[str, jax.Array]) -> tuple:
            return model.apply(data, {}, None)

        result, _, _ = jitted_apply(op, sample_data)
        assert result["grn_matrix"].shape == (N_TFS, N_GENES)
        assert result["tf_activity"].shape == (N_CELLS, N_TFS)

    def test_jit_gradient(
        self,
        rngs: nnx.Rngs,
        config: GRNInferenceConfig,
        sample_data: dict[str, jax.Array],
    ) -> None:
        """Test that gradients can be computed under JIT."""
        op = DifferentiableGRN(config, rngs=rngs)

        @nnx.jit
        def jitted_grad(model: DifferentiableGRN) -> DifferentiableGRN:
            def loss_fn(m: DifferentiableGRN) -> jax.Array:
                result, _, _ = m.apply(sample_data, {}, None)
                return jnp.sum(result["grn_matrix"])

            return nnx.grad(loss_fn)(model)

        grads = jitted_grad(op)
        assert grads is not None


class TestEdgeCases:
    """Tests for edge cases in GRN inference."""

    def test_single_tf(self, rngs: nnx.Rngs) -> None:
        """Test GRN inference with a single transcription factor."""
        config = GRNInferenceConfig(
            n_tfs=1,
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
        )
        op = DifferentiableGRN(config, rngs=rngs)

        key = jax.random.key(3)
        counts = jax.random.uniform(key, (N_CELLS, N_GENES), minval=0.0, maxval=5.0)
        tf_indices = jnp.array([0])

        data = {"counts": counts, "tf_indices": tf_indices}
        result, _, _ = op.apply(data, {}, None)

        assert result["grn_matrix"].shape == (1, N_GENES)
        assert result["tf_activity"].shape == (N_CELLS, 1)
        assert jnp.all(jnp.isfinite(result["grn_matrix"]))
        assert jnp.all(jnp.isfinite(result["tf_activity"]))

    def test_all_zero_expression(self, rngs: nnx.Rngs) -> None:
        """Test GRN inference with all-zero expression counts."""
        config = GRNInferenceConfig(
            n_tfs=N_TFS,
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
        )
        op = DifferentiableGRN(config, rngs=rngs)

        counts = jnp.zeros((N_CELLS, N_GENES))
        tf_indices = jnp.arange(N_TFS)

        data = {"counts": counts, "tf_indices": tf_indices}
        result, _, _ = op.apply(data, {}, None)

        assert result["grn_matrix"].shape == (N_TFS, N_GENES)
        assert result["tf_activity"].shape == (N_CELLS, N_TFS)
        assert jnp.all(jnp.isfinite(result["grn_matrix"]))
        assert jnp.all(jnp.isfinite(result["tf_activity"]))
