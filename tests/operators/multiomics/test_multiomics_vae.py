"""Tests for diffbio.operators.multiomics.multiomics_vae module.

These tests define the expected behavior of the DifferentiableMultiOmicsVAE
operator for joint multi-omics integration using Product-of-Experts fusion.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.multiomics.multiomics_vae import (
    DifferentiableMultiOmicsVAE,
    MultiOmicsVAEConfig,
)

# ---------------------------------------------------------------------------
# Small test dimensions
# ---------------------------------------------------------------------------
N_CELLS = 15
RNA_DIM = 20
ATAC_DIM = 10
LATENT_DIM = 5
HIDDEN_DIM = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_config() -> MultiOmicsVAEConfig:
    """Two-modality config with small dimensions for fast tests."""
    return MultiOmicsVAEConfig(
        modality_dims=[RNA_DIM, ATAC_DIM],
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        modality_weight_mode="equal",
        stochastic=True,
        stream_name="sample",
    )


@pytest.fixture
def learnable_config() -> MultiOmicsVAEConfig:
    """Config with learnable modality weights."""
    return MultiOmicsVAEConfig(
        modality_dims=[RNA_DIM, ATAC_DIM],
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        modality_weight_mode="learnable",
        stochastic=True,
        stream_name="sample",
    )


@pytest.fixture
def sample_data() -> dict[str, jax.Array]:
    """Two-modality count data (RNA + ATAC)."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    return {
        "rna_counts": jax.random.uniform(k1, (N_CELLS, RNA_DIM)),
        "atac_counts": jax.random.uniform(k2, (N_CELLS, ATAC_DIM)),
    }


@pytest.fixture
def three_mod_config() -> MultiOmicsVAEConfig:
    """Three-modality config for edge-case tests."""
    return MultiOmicsVAEConfig(
        modality_dims=[RNA_DIM, ATAC_DIM, 8],
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        stochastic=True,
        stream_name="sample",
    )


@pytest.fixture
def three_mod_data() -> dict[str, jax.Array]:
    """Three-modality count data."""
    key = jax.random.key(1)
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "modality_0_counts": jax.random.uniform(k1, (N_CELLS, RNA_DIM)),
        "modality_1_counts": jax.random.uniform(k2, (N_CELLS, ATAC_DIM)),
        "modality_2_counts": jax.random.uniform(k3, (N_CELLS, 8)),
    }


# ===================================================================
# TestMultiOmicsConfig
# ===================================================================
class TestMultiOmicsConfig:
    """Tests for MultiOmicsVAEConfig defaults and overrides."""

    def test_defaults(self) -> None:
        """Default config has two modalities with standard dims."""
        config = MultiOmicsVAEConfig()
        assert config.modality_dims == [2000, 500]
        assert config.latent_dim == 10
        assert config.hidden_dim == 64
        assert config.modality_weight_mode == "equal"
        assert config.stochastic is True
        assert config.stream_name == "sample"

    def test_custom_modalities(self) -> None:
        """Custom modality dimensions are stored correctly."""
        config = MultiOmicsVAEConfig(
            modality_dims=[100, 200, 300],
            latent_dim=20,
            hidden_dim=32,
            modality_weight_mode="learnable",
        )
        assert config.modality_dims == [100, 200, 300]
        assert config.latent_dim == 20
        assert config.hidden_dim == 32
        assert config.modality_weight_mode == "learnable"


# ===================================================================
# TestMultiOmicsVAE
# ===================================================================
class TestMultiOmicsVAE:
    """Core operator tests: output keys, shapes, finiteness."""

    def test_output_keys(self, rngs, small_config, sample_data) -> None:
        """apply() output dict contains all expected keys."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)
        result, state, meta = op.apply(sample_data, {}, None)

        assert "joint_latent" in result
        assert "rna_reconstructed" in result
        assert "atac_reconstructed" in result
        assert "elbo_loss" in result

    def test_output_shapes(self, rngs, small_config, sample_data) -> None:
        """Output arrays have correct shapes."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)

        assert result["joint_latent"].shape == (N_CELLS, LATENT_DIM)
        assert result["rna_reconstructed"].shape == (N_CELLS, RNA_DIM)
        assert result["atac_reconstructed"].shape == (N_CELLS, ATAC_DIM)

    def test_latent_finite(self, rngs, small_config, sample_data) -> None:
        """Joint latent representation contains only finite values."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)

        assert jnp.all(jnp.isfinite(result["joint_latent"]))

    def test_reconstructions_finite(self, rngs, small_config, sample_data) -> None:
        """Reconstructed outputs contain only finite values."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)

        assert jnp.all(jnp.isfinite(result["rna_reconstructed"]))
        assert jnp.all(jnp.isfinite(result["atac_reconstructed"]))

    def test_elbo_finite_and_scalar(self, rngs, small_config, sample_data) -> None:
        """ELBO loss is a finite scalar."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)

        loss = result["elbo_loss"]
        assert loss.ndim == 0
        assert jnp.isfinite(loss)

    def test_original_data_preserved(self, rngs, small_config, sample_data) -> None:
        """Original input keys are preserved in output."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)
        result, _, _ = op.apply(sample_data, {}, None)

        assert "rna_counts" in result
        assert "atac_counts" in result


# ===================================================================
# TestPoEFusion
# ===================================================================
class TestPoEFusion:
    """Tests for Product-of-Experts fusion correctness."""

    def test_precision_is_sum_of_precisions(self, rngs, small_config) -> None:
        """PoE precision_joint = sum(1 / sigma_m^2) for each modality."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)

        # Create per-modality parameters
        mu_list = [jnp.ones((N_CELLS, LATENT_DIM)), jnp.zeros((N_CELLS, LATENT_DIM))]
        logvar_list = [
            jnp.full((N_CELLS, LATENT_DIM), 0.0),  # var = 1
            jnp.full((N_CELLS, LATENT_DIM), jnp.log(2.0)),  # var = 2
        ]

        mu_joint, logvar_joint = op.product_of_experts(mu_list, logvar_list)

        # precision = 1/1 + 1/2 = 1.5
        expected_precision = 1.0 + 0.5
        actual_precision = 1.0 / jnp.exp(logvar_joint)

        assert jnp.allclose(actual_precision, expected_precision, atol=1e-5)

    def test_mu_joint_is_precision_weighted_mean(self, rngs, small_config) -> None:
        """PoE mu_joint = (sum mu_m / sigma_m^2) / precision_joint."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)

        mu_list = [
            jnp.full((N_CELLS, LATENT_DIM), 2.0),
            jnp.full((N_CELLS, LATENT_DIM), 4.0),
        ]
        logvar_list = [
            jnp.full((N_CELLS, LATENT_DIM), 0.0),  # var = 1
            jnp.full((N_CELLS, LATENT_DIM), jnp.log(2.0)),  # var = 2
        ]

        mu_joint, _ = op.product_of_experts(mu_list, logvar_list)

        # weighted mean = (2/1 + 4/2) / (1/1 + 1/2) = 4 / 1.5 = 2.667
        precision = 1.0 + 0.5
        weighted_sum = 2.0 / 1.0 + 4.0 / 2.0
        expected_mu = weighted_sum / precision

        assert jnp.allclose(mu_joint, expected_mu, atol=1e-5)

    def test_single_modality_passes_through(self, rngs, small_config) -> None:
        """With one modality, PoE returns that modality's parameters."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)

        mu = jnp.full((N_CELLS, LATENT_DIM), 3.0)
        logvar = jnp.full((N_CELLS, LATENT_DIM), -1.0)

        mu_joint, logvar_joint = op.product_of_experts([mu], [logvar])

        assert jnp.allclose(mu_joint, mu, atol=1e-5)
        assert jnp.allclose(logvar_joint, logvar, atol=1e-5)


# ===================================================================
# TestLearnableWeights
# ===================================================================
class TestLearnableWeights:
    """Tests for learnable modality weight mode."""

    def test_weights_are_learnable_params(self, rngs, learnable_config) -> None:
        """In 'learnable' mode, modality log-weights are nnx.Param."""
        op = DifferentiableMultiOmicsVAE(learnable_config, rngs=rngs)
        assert isinstance(op.log_modality_weights, nnx.Param)

    def test_weights_sum_to_one(self, rngs, learnable_config) -> None:
        """Softmax of log_weights sums to 1."""
        op = DifferentiableMultiOmicsVAE(learnable_config, rngs=rngs)
        weights = jax.nn.softmax(op.log_modality_weights[...])
        assert jnp.allclose(jnp.sum(weights), 1.0, atol=1e-5)

    def test_gradients_flow_through_weights(self, rngs, learnable_config, sample_data) -> None:
        """Gradients reach the learnable weight parameter."""
        op = DifferentiableMultiOmicsVAE(learnable_config, rngs=rngs)

        def loss_fn(model: DifferentiableMultiOmicsVAE) -> jax.Array:
            result, _, _ = model.apply(sample_data, {}, None)
            return result["elbo_loss"]

        grads = nnx.grad(loss_fn)(op)
        weight_grad = grads.log_modality_weights[...]
        assert jnp.any(weight_grad != 0.0)


# ===================================================================
# TestGradientFlow
# ===================================================================
class TestGradientFlow:
    """Verify gradients flow through all encoder/decoder parameters."""

    def test_grads_through_all_params(self, rngs, small_config, sample_data) -> None:
        """All trainable parameters receive non-zero gradients."""
        n_modalities = len(small_config.modality_dims)
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)

        def loss_fn(model: DifferentiableMultiOmicsVAE) -> jax.Array:
            result, _, _ = model.apply(sample_data, {}, None)
            return result["elbo_loss"]

        grads = nnx.grad(loss_fn)(op)

        # Check encoder params for each modality (2 layers each)
        for m in range(n_modalities):
            for layer_idx in range(2):
                kernel_grad = grads.encoders[m].layers[layer_idx].kernel[...]
                assert jnp.any(kernel_grad != 0.0)

        # Check decoder params for each modality (2 layers each)
        for m in range(n_modalities):
            for layer_idx in range(2):
                kernel_grad = grads.decoders[m].layers[layer_idx].kernel[...]
                assert jnp.any(kernel_grad != 0.0)

    def test_grads_through_latent_projections(self, rngs, small_config, sample_data) -> None:
        """Gradients flow through mu and logvar projection layers."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)

        def loss_fn(model: DifferentiableMultiOmicsVAE) -> jax.Array:
            result, _, _ = model.apply(sample_data, {}, None)
            return result["elbo_loss"]

        grads = nnx.grad(loss_fn)(op)

        n_modalities = len(small_config.modality_dims)
        for i in range(n_modalities):
            assert jnp.any(grads.mu_heads[i].kernel[...] != 0.0)
            assert jnp.any(grads.logvar_heads[i].kernel[...] != 0.0)


# ===================================================================
# TestJITCompatibility
# ===================================================================
class TestJITCompatibility:
    """JIT compilation tests for apply and gradient computation."""

    def test_jit_apply(self, rngs, small_config, sample_data) -> None:
        """apply() works under jit without error."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)

        @nnx.jit
        def run(model: DifferentiableMultiOmicsVAE) -> dict:
            result, _, _ = model.apply(sample_data, {}, None)
            return result

        result = run(op)
        assert result["joint_latent"].shape == (N_CELLS, LATENT_DIM)
        assert jnp.all(jnp.isfinite(result["joint_latent"]))

    def test_jit_gradient(self, rngs, small_config, sample_data) -> None:
        """Gradient computation works under jit without error."""
        op = DifferentiableMultiOmicsVAE(small_config, rngs=rngs)

        @nnx.jit
        def grad_fn(model: DifferentiableMultiOmicsVAE) -> jax.Array:
            def loss_fn(m: DifferentiableMultiOmicsVAE) -> jax.Array:
                result, _, _ = m.apply(sample_data, {}, None)
                return result["elbo_loss"]

            grads = nnx.grad(loss_fn)(model)
            # Return a scalar so jit has a concrete output
            return grads.encoders[0].layers[0].kernel[...].sum()

        val = grad_fn(op)
        assert jnp.isfinite(val)


# ===================================================================
# TestEdgeCases
# ===================================================================
class TestEdgeCases:
    """Edge cases: single modality, three modalities."""

    def test_single_modality_degenerates_to_standard_vae(self, rngs) -> None:
        """With one modality, behaves like a standard VAE."""
        config = MultiOmicsVAEConfig(
            modality_dims=[RNA_DIM],
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            stochastic=True,
            stream_name="sample",
        )
        op = DifferentiableMultiOmicsVAE(config, rngs=rngs)

        key = jax.random.key(99)
        data = {"modality_0_counts": jax.random.uniform(key, (N_CELLS, RNA_DIM))}

        result, _, _ = op.apply(data, {}, None)
        assert result["joint_latent"].shape == (N_CELLS, LATENT_DIM)
        assert result["modality_0_reconstructed"].shape == (N_CELLS, RNA_DIM)
        assert jnp.all(jnp.isfinite(result["joint_latent"]))
        assert jnp.isfinite(result["elbo_loss"])

    def test_three_modalities(self, rngs, three_mod_config, three_mod_data) -> None:
        """Three-modality fusion produces correct shapes."""
        op = DifferentiableMultiOmicsVAE(three_mod_config, rngs=rngs)
        result, _, _ = op.apply(three_mod_data, {}, None)

        assert result["joint_latent"].shape == (N_CELLS, LATENT_DIM)
        assert result["modality_0_reconstructed"].shape == (N_CELLS, RNA_DIM)
        assert result["modality_1_reconstructed"].shape == (N_CELLS, ATAC_DIM)
        assert result["modality_2_reconstructed"].shape == (N_CELLS, 8)
        assert jnp.all(jnp.isfinite(result["joint_latent"]))
        assert jnp.isfinite(result["elbo_loss"])

    def test_three_modalities_gradient(self, rngs, three_mod_config, three_mod_data) -> None:
        """Gradients flow through all three modalities."""
        op = DifferentiableMultiOmicsVAE(three_mod_config, rngs=rngs)

        def loss_fn(model: DifferentiableMultiOmicsVAE) -> jax.Array:
            result, _, _ = model.apply(three_mod_data, {}, None)
            return result["elbo_loss"]

        grads = nnx.grad(loss_fn)(op)

        n_modalities = len(three_mod_config.modality_dims)
        for m in range(n_modalities):
            for layer_idx in range(2):
                kernel_grad = grads.encoders[m].layers[layer_idx].kernel[...]
                assert jnp.any(kernel_grad != 0.0)
