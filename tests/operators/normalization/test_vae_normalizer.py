"""Tests for diffbio.operators.normalization.vae_normalizer module.

These tests define the expected behavior of the VAENormalizer
operator. Implementation should be written to pass these tests.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jaxtyping import PyTree

from diffbio.operators.normalization.vae_normalizer import (
    VAENormalizer,
    VAENormalizerConfig,
)


class TestVAENormalizerConfig:
    """Tests for VAENormalizerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VAENormalizerConfig()
        assert config.latent_dim == 10
        assert config.n_genes == 2000
        assert config.stochastic is True
        assert config.stream_name == "sample"

    def test_custom_latent_dim(self) -> None:
        """Test custom latent dimension."""
        config = VAENormalizerConfig(latent_dim=32)
        assert config.latent_dim == 32

    def test_custom_architecture(self) -> None:
        """Test custom architecture parameters."""
        config = VAENormalizerConfig(hidden_dims=[256, 128, 64], n_genes=5000)
        assert config.hidden_dims == [256, 128, 64]
        assert config.n_genes == 5000


class TestVAENormalizer:
    """Tests for VAENormalizer operator."""

    @pytest.fixture
    def sample_counts(self) -> dict[str, jax.Array]:
        """Provide sample count data."""
        # Simulate gene expression counts for a single cell
        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        return {"counts": counts, "library_size": jnp.sum(counts)}

    @pytest.fixture
    def batch_counts(self) -> dict[str, jax.Array]:
        """Provide batch of count data."""
        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(8, 100)).astype(jnp.float32)
        library_sizes = jnp.sum(counts, axis=1)
        return {"counts": counts, "library_size": library_sizes}

    def test_initialization(self, rngs: nnx.Rngs) -> None:
        """Test operator initialization."""
        config = VAENormalizerConfig(n_genes=100)
        op = VAENormalizer(config, rngs=rngs)
        assert op is not None
        assert op.latent_dim == 10

    def test_initialization_custom_architecture(self, rngs: nnx.Rngs) -> None:
        """Test initialization with custom architecture."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=20, hidden_dims=[64, 32])
        op = VAENormalizer(config, rngs=rngs)
        assert op.latent_dim == 20

    def test_encode_output_shape(self, rngs: nnx.Rngs, sample_counts: dict[str, jax.Array]) -> None:
        """Test that encoder produces correct latent shape."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        mean, logvar = op.encode(sample_counts["counts"])

        assert mean.shape == (10,)
        assert logvar.shape == (10,)

    def test_decode_output_shape(self, rngs: nnx.Rngs, sample_counts: dict[str, jax.Array]) -> None:
        """Test that decoder produces correct output shape (dict with log_rate)."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        z = jnp.ones(10)
        library_size = sample_counts["library_size"]
        decode_output = op.decode(z, library_size)

        assert isinstance(decode_output, dict)
        assert "log_rate" in decode_output
        assert decode_output["log_rate"].shape == (100,)

    def test_reparameterize(self, rngs: nnx.Rngs) -> None:
        """Test reparameterization produces valid samples."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        mean = jnp.zeros(10)
        logvar = jnp.zeros(10)

        # Uses inherited reparameterize() from EncoderDecoderOperator
        # (uses self.rngs internally, no key argument)
        z = op.reparameterize(mean, logvar)

        assert z.shape == (10,)
        # z should be different from mean due to sampling
        # (unless logvar is very negative)

    def test_apply_returns_normalized(
        self, rngs: nnx.Rngs, sample_counts: dict[str, jax.Array]
    ) -> None:
        """Test that apply returns normalized expression."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_counts, {}, None, None)

        assert "normalized" in transformed_data
        assert transformed_data["normalized"].shape == sample_counts["counts"].shape

    def test_apply_returns_latent(
        self, rngs: nnx.Rngs, sample_counts: dict[str, jax.Array]
    ) -> None:
        """Test that apply returns latent representation."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_counts, {}, None, None)

        assert "latent_z" in transformed_data
        assert transformed_data["latent_z"].shape == (10,)

    def test_apply_returns_reconstruction_params(
        self, rngs: nnx.Rngs, sample_counts: dict[str, jax.Array]
    ) -> None:
        """Test that apply returns reconstruction parameters."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        transformed_data, state, metadata = op.apply(sample_counts, {}, None, None)

        assert "log_rate" in transformed_data
        assert "latent_mean" in transformed_data
        assert "latent_logvar" in transformed_data

    def test_apply_preserves_counts(
        self, rngs: nnx.Rngs, sample_counts: dict[str, jax.Array]
    ) -> None:
        """Test that apply preserves original counts."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        transformed_data, _, _ = op.apply(sample_counts, {}, None, None)

        assert "counts" in transformed_data
        assert jnp.allclose(transformed_data["counts"], sample_counts["counts"])


class TestVAELoss:
    """Tests for VAE loss computation."""

    def test_elbo_loss_computable(self, rngs: nnx.Rngs) -> None:
        """Test that ELBO loss can be computed."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)

        # Uses inherited reparameterize() internally (no key argument needed)
        loss = op.compute_elbo_loss(counts, library_size)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_reconstruction_loss_non_negative(self, rngs: nnx.Rngs) -> None:
        """Test that reconstruction loss is non-negative."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)

        # Get reconstruction (decode now returns dict)
        mean, logvar = op.encode(counts)
        z = op.reparameterize(mean, logvar)
        decode_output = op.decode(z, library_size)

        recon_loss = op.reconstruction_loss(counts, decode_output)

        assert recon_loss >= 0


class TestGradientFlow:
    """Tests for gradient flow through VAE normalizer."""

    def test_gradient_flows_through_apply(self, rngs: nnx.Rngs) -> None:
        """Test that gradients flow through the apply method."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        state = {}

        def loss_fn(c: jax.Array) -> jax.Array:
            data = {"counts": c, "library_size": library_size}
            transformed, _, _ = op.apply(data, state, None, None)
            return jnp.sum(transformed["normalized"])

        grad = jax.grad(loss_fn)(counts)
        assert grad is not None
        assert grad.shape == counts.shape

    def test_gradient_flows_through_elbo(self, rngs: nnx.Rngs) -> None:
        """Test that gradients flow through ELBO loss."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)

        def loss_fn(c: jax.Array) -> jax.Array:
            return op.compute_elbo_loss(c, library_size)

        grad = jax.grad(loss_fn)(counts)
        assert grad is not None
        assert grad.shape == counts.shape

    def test_encoder_is_learnable(self, rngs: nnx.Rngs) -> None:
        """Test that encoder parameters are learnable."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model: VAENormalizer) -> jax.Array:
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["normalized"])

        loss, grads = loss_fn(op)

        assert hasattr(grads, "encoder_backbone")

    def test_decoder_is_learnable(self, rngs: nnx.Rngs) -> None:
        """Test that decoder parameters are learnable."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}
        state = {}

        @nnx.value_and_grad
        def loss_fn(model: VAENormalizer) -> jax.Array:
            transformed, _, _ = model.apply(data, state, None, None)
            return jnp.sum(transformed["normalized"])

        loss, grads = loss_fn(op)

        assert hasattr(grads, "decoder_backbone")


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_apply_is_jit_compatible(self, rngs: nnx.Rngs) -> None:
        """Test that apply method works with JIT."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}
        state = {}

        @jax.jit
        def jit_apply(
            data: dict[str, jax.Array], state: dict
        ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
            return op.apply(data, state, None, None)

        transformed, new_state, metadata = jit_apply(data, state)
        assert transformed["normalized"].shape == counts.shape

    def test_encode_is_jit_compatible(self, rngs: nnx.Rngs) -> None:
        """Test that encode method works with JIT."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)

        @jax.jit
        def jit_encode(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            return op.encode(x)

        mean, logvar = jit_encode(counts)
        assert mean.shape == (10,)
        assert logvar.shape == (10,)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_counts(self, rngs: nnx.Rngs) -> None:
        """Test with all zero counts."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jnp.zeros(100)
        library_size = jnp.array(1.0)  # Avoid division by zero
        data = {"counts": counts, "library_size": library_size}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["normalized"].shape == (100,)
        assert jnp.all(jnp.isfinite(transformed["normalized"]))

    def test_sparse_counts(self, rngs: nnx.Rngs) -> None:
        """Test with sparse (mostly zero) counts."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jnp.zeros(100).at[0].set(100.0).at[50].set(50.0)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["normalized"].shape == (100,)
        assert jnp.all(jnp.isfinite(transformed["normalized"]))

    def test_high_counts(self, rngs: nnx.Rngs) -> None:
        """Test with high count values."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=10)
        op = VAENormalizer(config, rngs=rngs)

        counts = jnp.ones(100) * 10000.0
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["normalized"].shape == (100,)
        assert jnp.all(jnp.isfinite(transformed["normalized"]))

    def test_small_latent_dim(self, rngs: nnx.Rngs) -> None:
        """Test with very small latent dimension."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=2)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["latent_z"].shape == (2,)

    def test_large_latent_dim(self, rngs: nnx.Rngs) -> None:
        """Test with large latent dimension."""
        config = VAENormalizerConfig(n_genes=100, latent_dim=50)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(100,)).astype(jnp.float32)
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}

        transformed, _, _ = op.apply(data, {}, None, None)
        assert transformed["latent_z"].shape == (50,)


# ---------------------------------------------------------------------------
# ZINB likelihood tests
# ---------------------------------------------------------------------------

N_GENES = 100
LATENT_DIM = 10


class TestZINBConfig:
    """Tests for ZINB likelihood configuration."""

    def test_default_likelihood_is_poisson(self) -> None:
        """Test that the default likelihood is 'poisson'."""
        config = VAENormalizerConfig()
        assert config.likelihood == "poisson"

    def test_zinb_likelihood_config(self) -> None:
        """Test that ZINB likelihood can be configured."""
        config = VAENormalizerConfig(likelihood="zinb")
        assert config.likelihood == "zinb"


class TestZINBVAENormalizer:
    """Tests for ZINB-mode VAENormalizer."""

    @pytest.fixture
    def zinb_op(self, rngs: nnx.Rngs) -> VAENormalizer:
        """Provide a ZINB-mode VAENormalizer."""
        config = VAENormalizerConfig(n_genes=N_GENES, latent_dim=LATENT_DIM, likelihood="zinb")
        return VAENormalizer(config, rngs=rngs)

    @pytest.fixture
    def sample_counts(self) -> jax.Array:
        """Provide sample count data array."""
        return jax.random.poisson(jax.random.key(0), lam=10.0, shape=(N_GENES,)).astype(jnp.float32)

    def test_zinb_decode_returns_dict_with_extra_keys(self, zinb_op: VAENormalizer) -> None:
        """Test that ZINB decode returns dict with log_theta and pi_logit."""
        z = jnp.ones(LATENT_DIM)
        library_size = jnp.array(1000.0)
        decode_output = zinb_op.decode(z, library_size)

        assert isinstance(decode_output, dict)
        assert "log_rate" in decode_output
        assert "log_theta" in decode_output
        assert "pi_logit" in decode_output

    def test_zinb_output_shapes(self, zinb_op: VAENormalizer) -> None:
        """Test that ZINB output arrays have correct shapes."""
        z = jnp.ones(LATENT_DIM)
        library_size = jnp.array(1000.0)
        decode_output = zinb_op.decode(z, library_size)

        assert decode_output["log_rate"].shape == (N_GENES,)
        assert decode_output["log_theta"].shape == (N_GENES,)
        assert decode_output["pi_logit"].shape == (N_GENES,)

    def test_zinb_nll_finite(self, zinb_op: VAENormalizer, sample_counts: jax.Array) -> None:
        """Test that ZINB NLL produces finite values (no NaN/inf)."""
        library_size = jnp.sum(sample_counts)
        mean, logvar = zinb_op.encode(sample_counts)
        z = zinb_op.reparameterize(mean, logvar)
        decode_output = zinb_op.decode(z, library_size)

        loss = zinb_op.reconstruction_loss(sample_counts, decode_output)

        assert jnp.isfinite(loss)

    def test_zinb_nll_for_zero_counts(self, zinb_op: VAENormalizer) -> None:
        """Test that zero counts produce valid ZINB likelihood."""
        zero_counts = jnp.zeros(N_GENES)
        library_size = jnp.array(1.0)

        mean, logvar = zinb_op.encode(zero_counts)
        z = zinb_op.reparameterize(mean, logvar)
        decode_output = zinb_op.decode(z, library_size)

        loss = zinb_op.reconstruction_loss(zero_counts, decode_output)

        assert jnp.isfinite(loss)

    def test_zinb_elbo_finite(self, zinb_op: VAENormalizer, sample_counts: jax.Array) -> None:
        """Test that ZINB ELBO loss is finite."""
        library_size = jnp.sum(sample_counts)
        loss = zinb_op.compute_elbo_loss(sample_counts, library_size)

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_zinb_nll_stable_extreme_pi_logit(self, zinb_op: VAENormalizer) -> None:
        """ZINB NLL should be finite even with extreme dropout logits.

        Large positive pi_logit means sigmoid(pi_logit) near 1 (heavy dropout).
        The old formulation materialized sigmoid(pi) then computed log(1-pi),
        which is -inf when pi near 1. The softplus formulation avoids this.
        """
        counts = jax.random.poisson(jax.random.key(1), lam=5.0, shape=(N_GENES,)).astype(
            jnp.float32
        )
        log_rate = jnp.zeros(N_GENES)
        log_theta = jnp.zeros(N_GENES)

        # Extreme positive pi_logit: sigmoid(50) ~ 1.0, so 1-pi ~ 0
        extreme_pi_logit = jnp.full((N_GENES,), 50.0)
        loss_pos = zinb_op._zinb_nll(counts, log_rate, log_theta, extreme_pi_logit)
        assert jnp.isfinite(loss_pos), f"NaN/inf with pi_logit=+50: {loss_pos}"

        # Extreme negative pi_logit: sigmoid(-50) ~ 0, so pi ~ 0
        extreme_pi_logit_neg = jnp.full((N_GENES,), -50.0)
        loss_neg = zinb_op._zinb_nll(counts, log_rate, log_theta, extreme_pi_logit_neg)
        assert jnp.isfinite(loss_neg), f"NaN/inf with pi_logit=-50: {loss_neg}"

    def test_zinb_nll_gradient_stable_extreme_pi_logit(self, zinb_op: VAENormalizer) -> None:
        """ZINB NLL gradients should be finite with extreme dropout logits."""
        counts = jax.random.poisson(jax.random.key(2), lam=5.0, shape=(N_GENES,)).astype(
            jnp.float32
        )
        log_rate = jnp.zeros(N_GENES)
        log_theta = jnp.zeros(N_GENES)

        def nll_fn(pi_logit: jax.Array) -> jax.Array:
            return zinb_op._zinb_nll(counts, log_rate, log_theta, pi_logit)

        # Gradient at extreme positive pi_logit
        grad_pos = jax.grad(nll_fn)(jnp.full((N_GENES,), 50.0))
        assert jnp.all(jnp.isfinite(grad_pos)), "NaN/inf gradient at pi_logit=+50"

        # Gradient at extreme negative pi_logit
        grad_neg = jax.grad(nll_fn)(jnp.full((N_GENES,), -50.0))
        assert jnp.all(jnp.isfinite(grad_neg)), "NaN/inf gradient at pi_logit=-50"


class TestZINBGradientFlow:
    """Tests for gradient flow through ZINB components."""

    @pytest.fixture
    def zinb_op(self, rngs: nnx.Rngs) -> VAENormalizer:
        """Provide a ZINB-mode VAENormalizer."""
        config = VAENormalizerConfig(n_genes=N_GENES, latent_dim=LATENT_DIM, likelihood="zinb")
        return VAENormalizer(config, rngs=rngs)

    def test_gradient_through_zinb_nll(self, zinb_op: VAENormalizer) -> None:
        """Test that gradients for log_theta and pi_logit heads are non-zero."""
        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(N_GENES,)).astype(
            jnp.float32
        )
        library_size = jnp.sum(counts)

        @nnx.value_and_grad
        def loss_fn(model: VAENormalizer) -> jax.Array:
            mean, logvar = model.encode(counts)
            z = model.reparameterize(mean, logvar)
            decode_output = model.decode(z, library_size)
            return model.reconstruction_loss(counts, decode_output)

        _loss, grads = loss_fn(zinb_op)

        # ZINB heads should receive gradients
        assert hasattr(grads, "fc_log_theta")
        assert hasattr(grads, "fc_pi_logit")
        # At least some gradients should be non-zero
        theta_grad_norm = jnp.sum(jnp.abs(grads.fc_log_theta.kernel[...]))
        pi_grad_norm = jnp.sum(jnp.abs(grads.fc_pi_logit.kernel[...]))
        assert theta_grad_norm > 0
        assert pi_grad_norm > 0

    def test_gradient_through_zinb_elbo(self, zinb_op: VAENormalizer) -> None:
        """Test that full ELBO gradients are non-zero with ZINB."""
        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(N_GENES,)).astype(
            jnp.float32
        )
        library_size = jnp.sum(counts)

        @nnx.value_and_grad
        def loss_fn(model: VAENormalizer) -> jax.Array:
            return model.compute_elbo_loss(counts, library_size)

        _loss, grads = loss_fn(zinb_op)

        # Encoder, decoder, and ZINB heads all get gradients
        assert hasattr(grads, "encoder_backbone")
        assert hasattr(grads, "decoder_backbone")
        assert hasattr(grads, "fc_log_theta")
        assert hasattr(grads, "fc_pi_logit")


class TestZINBJIT:
    """Tests for JIT compilation with ZINB mode."""

    @pytest.fixture
    def zinb_op(self, rngs: nnx.Rngs) -> VAENormalizer:
        """Provide a ZINB-mode VAENormalizer."""
        config = VAENormalizerConfig(n_genes=N_GENES, latent_dim=LATENT_DIM, likelihood="zinb")
        return VAENormalizer(config, rngs=rngs)

    def test_jit_zinb_apply(self, zinb_op: VAENormalizer) -> None:
        """Test that JIT compiles with ZINB mode."""
        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(N_GENES,)).astype(
            jnp.float32
        )
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}

        @jax.jit
        def jit_apply(
            data: dict[str, jax.Array], state: dict
        ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
            return zinb_op.apply(data, state, None, None)

        transformed, _, _ = jit_apply(data, {})
        assert transformed["normalized"].shape == (N_GENES,)
        assert jnp.all(jnp.isfinite(transformed["normalized"]))

    def test_jit_zinb_elbo(self, zinb_op: VAENormalizer) -> None:
        """Test that JIT compiles ELBO computation with ZINB."""
        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(N_GENES,)).astype(
            jnp.float32
        )
        library_size = jnp.sum(counts)

        @jax.jit
        def jit_elbo(counts: jax.Array, library_size: jax.Array) -> jax.Array:
            return zinb_op.compute_elbo_loss(counts, library_size)

        loss = jit_elbo(counts, library_size)
        assert loss.shape == ()
        assert jnp.isfinite(loss)


class TestPoissonUnchanged:
    """Tests verifying that default Poisson behavior is unchanged."""

    def test_existing_poisson_tests_still_pass(self, rngs: nnx.Rngs) -> None:
        """Verify default (Poisson) behavior is unchanged after ZINB addition."""
        config = VAENormalizerConfig(n_genes=N_GENES, latent_dim=LATENT_DIM)
        op = VAENormalizer(config, rngs=rngs)

        # Config defaults
        assert config.likelihood == "poisson"

        # Decode returns dict but only with log_rate
        z = jnp.ones(LATENT_DIM)
        library_size = jnp.array(1000.0)
        decode_output = op.decode(z, library_size)
        assert isinstance(decode_output, dict)
        assert "log_rate" in decode_output
        assert "log_theta" not in decode_output
        assert "pi_logit" not in decode_output

        # Poisson should not have ZINB-specific layers
        assert not hasattr(op, "fc_log_theta")
        assert not hasattr(op, "fc_pi_logit")

    def test_poisson_reconstruction_loss_unchanged(self, rngs: nnx.Rngs) -> None:
        """Verify Poisson reconstruction loss works with new dict interface."""
        config = VAENormalizerConfig(n_genes=N_GENES, latent_dim=LATENT_DIM)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(N_GENES,)).astype(
            jnp.float32
        )
        library_size = jnp.sum(counts)

        mean, logvar = op.encode(counts)
        z = op.reparameterize(mean, logvar)
        decode_output = op.decode(z, library_size)
        loss = op.reconstruction_loss(counts, decode_output)

        assert jnp.isfinite(loss)
        assert loss >= 0

    def test_poisson_apply_output_keys(self, rngs: nnx.Rngs) -> None:
        """Verify apply output contains expected keys for Poisson mode."""
        config = VAENormalizerConfig(n_genes=N_GENES, latent_dim=LATENT_DIM)
        op = VAENormalizer(config, rngs=rngs)

        counts = jax.random.poisson(jax.random.key(0), lam=10.0, shape=(N_GENES,)).astype(
            jnp.float32
        )
        library_size = jnp.sum(counts)
        data = {"counts": counts, "library_size": library_size}

        transformed, _, _ = op.apply(data, {}, None, None)

        expected_keys = {
            "counts",
            "library_size",
            "normalized",
            "latent_z",
            "latent_mean",
            "latent_logvar",
            "log_rate",
        }
        assert set(transformed.keys()) == expected_keys
