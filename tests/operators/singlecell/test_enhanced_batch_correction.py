"""Tests for enhanced batch correction operators (MMD and WGAN).

Tests define expected behavior for DifferentiableMMDBatchCorrection and
DifferentiableWGANBatchCorrection operators, which use MMD loss and
Wasserstein adversarial training respectively to remove batch effects.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffbio.operators.singlecell.enhanced_batch_correction import (
    DifferentiableMMDBatchCorrection,
    DifferentiableWGANBatchCorrection,
    MMDBatchCorrectionConfig,
    WGANBatchCorrectionConfig,
)

# Small dims for fast tests
N_GENES = 30
HIDDEN_DIM = 16
LATENT_DIM = 8
N_CELLS = 20
N_BATCHES = 2


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestMMDConfig:
    """Tests for MMDBatchCorrectionConfig defaults and overrides."""

    def test_defaults(self) -> None:
        """Default values match specification."""
        config = MMDBatchCorrectionConfig()
        assert config.n_genes == 2000
        assert config.hidden_dim == 128
        assert config.latent_dim == 64
        assert config.kernel_bandwidth == 1.0
        assert config.stochastic is False

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        config = MMDBatchCorrectionConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,
            kernel_bandwidth=2.0,
        )
        assert config.n_genes == N_GENES
        assert config.hidden_dim == HIDDEN_DIM
        assert config.latent_dim == LATENT_DIM
        assert config.kernel_bandwidth == 2.0


class TestWGANConfig:
    """Tests for WGANBatchCorrectionConfig defaults and overrides."""

    def test_defaults(self) -> None:
        """Default values match specification."""
        config = WGANBatchCorrectionConfig()
        assert config.n_genes == 2000
        assert config.hidden_dim == 128
        assert config.latent_dim == 64
        assert config.discriminator_hidden_dim == 64
        assert config.stochastic is False

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        config = WGANBatchCorrectionConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,
            discriminator_hidden_dim=32,
        )
        assert config.n_genes == N_GENES
        assert config.discriminator_hidden_dim == 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mmd_config() -> MMDBatchCorrectionConfig:
    """Small MMD config for tests."""
    return MMDBatchCorrectionConfig(
        n_genes=N_GENES,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    )


@pytest.fixture
def wgan_config() -> WGANBatchCorrectionConfig:
    """Small WGAN config for tests."""
    return WGANBatchCorrectionConfig(
        n_genes=N_GENES,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        discriminator_hidden_dim=HIDDEN_DIM,
    )


@pytest.fixture
def two_batch_data() -> dict[str, jax.Array]:
    """Expression matrix with two distinct batches that have a clear offset."""
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    half = N_CELLS // 2

    # Batch 0: centered at 0, Batch 1: shifted by +3
    batch0 = jax.random.normal(k1, (half, N_GENES))
    batch1 = jax.random.normal(k2, (half, N_GENES)) + 3.0

    expression = jnp.concatenate([batch0, batch1], axis=0)
    batch_labels = jnp.concatenate(
        [jnp.zeros(half, dtype=jnp.int32), jnp.ones(half, dtype=jnp.int32)]
    )
    return {"expression": expression, "batch_labels": batch_labels}


# ---------------------------------------------------------------------------
# MMD operator tests
# ---------------------------------------------------------------------------


class TestMMDBatchCorrection:
    """Tests for DifferentiableMMDBatchCorrection operator."""

    def test_output_keys(
        self, rngs: nnx.Rngs, mmd_config: MMDBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Output dictionary contains required keys."""
        op = DifferentiableMMDBatchCorrection(mmd_config, rngs=rngs)
        result, state, meta = op.apply(two_batch_data, {}, None)

        assert "corrected_expression" in result
        assert "latent" in result
        assert "mmd_loss" in result
        assert "reconstruction_loss" in result

    def test_output_shapes(
        self, rngs: nnx.Rngs, mmd_config: MMDBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Output shapes match expected dimensions."""
        op = DifferentiableMMDBatchCorrection(mmd_config, rngs=rngs)
        result, _, _ = op.apply(two_batch_data, {}, None)

        assert result["corrected_expression"].shape == (N_CELLS, N_GENES)
        assert result["latent"].shape == (N_CELLS, LATENT_DIM)
        assert result["mmd_loss"].ndim == 0
        assert result["reconstruction_loss"].ndim == 0

    def test_corrected_is_finite(
        self, rngs: nnx.Rngs, mmd_config: MMDBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Corrected expression values are all finite."""
        op = DifferentiableMMDBatchCorrection(mmd_config, rngs=rngs)
        result, _, _ = op.apply(two_batch_data, {}, None)

        assert jnp.isfinite(result["corrected_expression"]).all()
        assert jnp.isfinite(result["latent"]).all()

    def test_mmd_loss_nonnegative(
        self, rngs: nnx.Rngs, mmd_config: MMDBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """MMD loss is non-negative (it is a squared distance)."""
        op = DifferentiableMMDBatchCorrection(mmd_config, rngs=rngs)
        result, _, _ = op.apply(two_batch_data, {}, None)

        assert result["mmd_loss"] >= 0.0

    def test_preserves_input_data(
        self, rngs: nnx.Rngs, mmd_config: MMDBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Original expression and batch labels are preserved in output."""
        op = DifferentiableMMDBatchCorrection(mmd_config, rngs=rngs)
        result, _, _ = op.apply(two_batch_data, {}, None)

        assert "expression" in result
        assert "batch_labels" in result


# ---------------------------------------------------------------------------
# WGAN operator tests
# ---------------------------------------------------------------------------


class TestWGANBatchCorrection:
    """Tests for DifferentiableWGANBatchCorrection operator."""

    def test_output_keys(
        self, rngs: nnx.Rngs, wgan_config: WGANBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Output dictionary contains required keys."""
        op = DifferentiableWGANBatchCorrection(wgan_config, rngs=rngs)
        result, _, _ = op.apply(two_batch_data, {}, None)

        assert "corrected_expression" in result
        assert "latent" in result
        assert "discriminator_scores" in result
        assert "generator_loss" in result
        assert "discriminator_loss" in result

    def test_output_shapes(
        self, rngs: nnx.Rngs, wgan_config: WGANBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Output shapes match expected dimensions."""
        op = DifferentiableWGANBatchCorrection(wgan_config, rngs=rngs)
        result, _, _ = op.apply(two_batch_data, {}, None)

        assert result["corrected_expression"].shape == (N_CELLS, N_GENES)
        assert result["latent"].shape == (N_CELLS, LATENT_DIM)
        assert result["discriminator_scores"].shape == (N_CELLS,)
        assert result["generator_loss"].ndim == 0
        assert result["discriminator_loss"].ndim == 0

    def test_corrected_is_finite(
        self, rngs: nnx.Rngs, wgan_config: WGANBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Corrected expression values are all finite."""
        op = DifferentiableWGANBatchCorrection(wgan_config, rngs=rngs)
        result, _, _ = op.apply(two_batch_data, {}, None)

        assert jnp.isfinite(result["corrected_expression"]).all()
        assert jnp.isfinite(result["latent"]).all()

    def test_discriminator_output_for_both_batches(
        self, rngs: nnx.Rngs, wgan_config: WGANBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Discriminator produces scores for all cells across both batches."""
        op = DifferentiableWGANBatchCorrection(wgan_config, rngs=rngs)
        result, _, _ = op.apply(two_batch_data, {}, None)

        scores = result["discriminator_scores"]
        batch_labels = two_batch_data["batch_labels"]

        # Both batches should have discriminator scores
        batch0_scores = scores[batch_labels == 0]
        batch1_scores = scores[batch_labels == 1]

        assert batch0_scores.shape[0] == N_CELLS // 2
        assert batch1_scores.shape[0] == N_CELLS // 2
        assert jnp.isfinite(batch0_scores).all()
        assert jnp.isfinite(batch1_scores).all()

    def test_preserves_input_data(
        self, rngs: nnx.Rngs, wgan_config: WGANBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Original expression and batch labels are preserved in output."""
        op = DifferentiableWGANBatchCorrection(wgan_config, rngs=rngs)
        result, _, _ = op.apply(two_batch_data, {}, None)

        assert "expression" in result
        assert "batch_labels" in result


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Tests for gradient flow through both operators."""

    def test_mmd_gradients_through_encoder(
        self, rngs: nnx.Rngs, mmd_config: MMDBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Gradients flow through the MMD encoder parameters."""
        op = DifferentiableMMDBatchCorrection(mmd_config, rngs=rngs)

        @nnx.value_and_grad
        def loss_fn(model: DifferentiableMMDBatchCorrection) -> jax.Array:
            result, _, _ = model.apply(two_batch_data, {}, None)
            return result["mmd_loss"] + result["reconstruction_loss"]

        loss, grads = loss_fn(op)

        assert jnp.isfinite(loss)
        # Encoder should have gradients
        assert hasattr(grads, "encoder")
        encoder_grads = grads.encoder
        has_nonzero = False
        for leaf in jax.tree.leaves(encoder_grads):
            if jnp.any(leaf != 0):
                has_nonzero = True
                break
        assert has_nonzero, "Encoder gradients are all zero"

    def test_wgan_gradients_through_generator(
        self, rngs: nnx.Rngs, wgan_config: WGANBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Gradients flow through the WGAN generator (encoder+decoder)."""
        op = DifferentiableWGANBatchCorrection(wgan_config, rngs=rngs)

        @nnx.value_and_grad
        def gen_loss_fn(model: DifferentiableWGANBatchCorrection) -> jax.Array:
            result, _, _ = model.apply(two_batch_data, {}, None)
            return result["generator_loss"]

        loss, grads = gen_loss_fn(op)

        assert jnp.isfinite(loss)
        assert hasattr(grads, "encoder")
        encoder_grads = grads.encoder
        has_nonzero = False
        for leaf in jax.tree.leaves(encoder_grads):
            if jnp.any(leaf != 0):
                has_nonzero = True
                break
        assert has_nonzero, "Generator encoder gradients are all zero"

    def test_wgan_gradients_through_discriminator(
        self, rngs: nnx.Rngs, wgan_config: WGANBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """Gradients flow through the WGAN discriminator."""
        op = DifferentiableWGANBatchCorrection(wgan_config, rngs=rngs)

        @nnx.value_and_grad
        def disc_loss_fn(model: DifferentiableWGANBatchCorrection) -> jax.Array:
            result, _, _ = model.apply(two_batch_data, {}, None)
            return result["discriminator_loss"]

        loss, grads = disc_loss_fn(op)

        assert jnp.isfinite(loss)
        assert hasattr(grads, "discriminator")
        disc_grads = grads.discriminator
        has_nonzero = False
        for leaf in jax.tree.leaves(disc_grads):
            if jnp.any(leaf != 0):
                has_nonzero = True
                break
        assert has_nonzero, "Discriminator gradients are all zero"


# ---------------------------------------------------------------------------
# JIT compatibility tests
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_mmd_jit(
        self, rngs: nnx.Rngs, mmd_config: MMDBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """MMD operator works under JIT compilation."""
        op = DifferentiableMMDBatchCorrection(mmd_config, rngs=rngs)

        @jax.jit
        def jit_apply(data: dict, state: dict) -> tuple:
            return op.apply(data, state, None)

        result, _, _ = jit_apply(two_batch_data, {})
        assert jnp.isfinite(result["corrected_expression"]).all()

    def test_wgan_jit(
        self, rngs: nnx.Rngs, wgan_config: WGANBatchCorrectionConfig, two_batch_data: dict
    ) -> None:
        """WGAN operator works under JIT compilation."""
        op = DifferentiableWGANBatchCorrection(wgan_config, rngs=rngs)

        @jax.jit
        def jit_apply(data: dict, state: dict) -> tuple:
            return op.apply(data, state, None)

        result, _, _ = jit_apply(two_batch_data, {})
        assert jnp.isfinite(result["corrected_expression"]).all()


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_batch_mmd(self, rngs: nnx.Rngs) -> None:
        """MMD with a single batch should produce near-zero MMD loss."""
        config = MMDBatchCorrectionConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,
        )
        op = DifferentiableMMDBatchCorrection(config, rngs=rngs)

        key = jax.random.key(7)
        expression = jax.random.normal(key, (N_CELLS, N_GENES))
        batch_labels = jnp.zeros(N_CELLS, dtype=jnp.int32)
        data = {"expression": expression, "batch_labels": batch_labels}

        result, _, _ = op.apply(data, {}, None)
        assert jnp.isfinite(result["corrected_expression"]).all()
        # Single batch means no inter-batch discrepancy to measure;
        # MMD loss should still be finite and non-negative
        assert jnp.isfinite(result["mmd_loss"])
        assert result["mmd_loss"] >= 0.0

    def test_single_batch_wgan(self, rngs: nnx.Rngs) -> None:
        """WGAN with a single batch should still produce finite outputs."""
        config = WGANBatchCorrectionConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,
            discriminator_hidden_dim=HIDDEN_DIM,
        )
        op = DifferentiableWGANBatchCorrection(config, rngs=rngs)

        key = jax.random.key(7)
        expression = jax.random.normal(key, (N_CELLS, N_GENES))
        batch_labels = jnp.zeros(N_CELLS, dtype=jnp.int32)
        data = {"expression": expression, "batch_labels": batch_labels}

        result, _, _ = op.apply(data, {}, None)
        assert jnp.isfinite(result["corrected_expression"]).all()
        assert jnp.isfinite(result["discriminator_scores"]).all()

    def test_three_batches(self, rngs: nnx.Rngs) -> None:
        """MMD operator handles more than two batches."""
        config = MMDBatchCorrectionConfig(
            n_genes=N_GENES,
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,
        )
        op = DifferentiableMMDBatchCorrection(config, rngs=rngs)

        key = jax.random.key(9)
        expression = jax.random.normal(key, (30, N_GENES))
        batch_labels = jnp.array([0] * 10 + [1] * 10 + [2] * 10, dtype=jnp.int32)
        data = {"expression": expression, "batch_labels": batch_labels}

        result, _, _ = op.apply(data, {}, None)
        assert result["corrected_expression"].shape == (30, N_GENES)
        assert jnp.isfinite(result["corrected_expression"]).all()
