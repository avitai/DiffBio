"""Tests for diffbio.losses.statistical_losses module.

These tests define the expected behavior of statistical loss functions
for differentiable bioinformatics pipelines.
"""

import jax
import jax.numpy as jnp
from flax import nnx

import diffbio.losses
from diffbio.losses import nb_log_prob
from diffbio.losses.statistical_losses import (
    HMMLikelihoodLoss,
    NegativeBinomialLoss,
    VAELoss,
)


class TestNegativeBinomialLoss:
    """Tests for NegativeBinomialLoss."""

    def test_initialization(self, rngs):
        """Test loss initialization."""
        loss_fn = NegativeBinomialLoss(rngs=rngs)
        assert loss_fn is not None

    def test_loss_shape(self, rngs):
        """Test that loss returns scalar."""
        loss_fn = NegativeBinomialLoss(rngs=rngs)

        key = jax.random.key(0)
        counts = jax.random.poisson(key, 10.0, (50, 100)).astype(jnp.float32)
        mu = jnp.abs(jax.random.normal(key, (50, 100))) + 1.0
        theta = jnp.ones((100,)) * 10.0

        loss = loss_fn(counts, mu, theta)
        assert loss.shape == ()

    def test_loss_finite(self, rngs):
        """Test that loss is finite."""
        loss_fn = NegativeBinomialLoss(rngs=rngs)

        key = jax.random.key(0)
        counts = jax.random.poisson(key, 10.0, (50, 100)).astype(jnp.float32)
        mu = jnp.abs(jax.random.normal(key, (50, 100))) + 1.0
        theta = jnp.ones((100,)) * 10.0

        loss = loss_fn(counts, mu, theta)
        assert jnp.isfinite(loss)

    def test_loss_positive(self, rngs):
        """Test that NLL is positive (or can be negative for well-fit data)."""
        loss_fn = NegativeBinomialLoss(rngs=rngs)

        key = jax.random.key(0)
        counts = jax.random.poisson(key, 10.0, (50, 100)).astype(jnp.float32)
        mu = counts + 0.1  # Close to true values
        theta = jnp.ones((100,)) * 100.0  # High dispersion

        loss = loss_fn(counts, mu, theta)
        assert jnp.isfinite(loss)

    def test_gradient_flows(self, rngs):
        """Test that gradients flow through loss."""
        loss_fn = NegativeBinomialLoss(rngs=rngs)

        key = jax.random.key(0)
        counts = jax.random.poisson(key, 10.0, (50, 100)).astype(jnp.float32)
        theta = jnp.ones((100,)) * 10.0

        def compute_loss(mu):
            return loss_fn(counts, mu, theta)

        mu = jnp.abs(jax.random.normal(key, (50, 100))) + 1.0
        grad = jax.grad(compute_loss)(mu)

        assert grad is not None
        assert grad.shape == mu.shape
        assert jnp.isfinite(grad).all()


class TestVAELoss:
    """Tests for VAELoss."""

    def test_initialization(self, rngs):
        """Test loss initialization."""
        loss_fn = VAELoss(rngs=rngs)
        assert loss_fn is not None

    def test_loss_shape(self, rngs):
        """Test that loss returns scalar."""
        loss_fn = VAELoss(rngs=rngs)

        key = jax.random.key(0)
        x = jax.random.normal(key, (50, 100))
        x_recon = jax.random.normal(key, (50, 100))
        mean = jax.random.normal(key, (50, 32))
        logvar = jax.random.normal(key, (50, 32)) * 0.1

        loss = loss_fn(x, x_recon, mean, logvar)
        assert loss.shape == ()

    def test_loss_finite(self, rngs):
        """Test that loss is finite."""
        loss_fn = VAELoss(rngs=rngs)

        key = jax.random.key(0)
        x = jax.random.normal(key, (50, 100))
        x_recon = x + jax.random.normal(key, (50, 100)) * 0.1
        mean = jax.random.normal(key, (50, 32)) * 0.1
        logvar = jnp.zeros((50, 32))

        loss = loss_fn(x, x_recon, mean, logvar)
        assert jnp.isfinite(loss)

    def test_kl_weight(self, rngs):
        """Test that KL weight affects loss."""
        loss_fn_low = VAELoss(kl_weight=0.1, rngs=rngs)
        loss_fn_high = VAELoss(kl_weight=10.0, rngs=rngs)

        key = jax.random.key(0)
        x = jax.random.normal(key, (50, 100))
        x_recon = x
        mean = jax.random.normal(key, (50, 32))
        logvar = jax.random.normal(key, (50, 32))

        loss_low = loss_fn_low(x, x_recon, mean, logvar)
        loss_high = loss_fn_high(x, x_recon, mean, logvar)

        # Higher KL weight should give higher loss when KL > 0
        assert loss_high > loss_low or jnp.isclose(loss_high, loss_low)

    def test_gradient_flows(self, rngs):
        """Test that gradients flow through loss."""
        loss_fn = VAELoss(rngs=rngs)

        key = jax.random.key(0)
        x = jax.random.normal(key, (50, 100))
        mean = jax.random.normal(key, (50, 32))
        logvar = jax.random.normal(key, (50, 32)) * 0.1

        def compute_loss(x_recon):
            return loss_fn(x, x_recon, mean, logvar)

        x_recon = jax.random.normal(key, (50, 100))
        grad = jax.grad(compute_loss)(x_recon)

        assert grad is not None
        assert grad.shape == x_recon.shape
        assert jnp.isfinite(grad).all()


class TestHMMLikelihoodLoss:
    """Tests for HMMLikelihoodLoss."""

    def test_initialization(self, rngs):
        """Test loss initialization."""
        loss_fn = HMMLikelihoodLoss(n_states=3, n_emissions=4, rngs=rngs)
        assert loss_fn is not None

    def test_loss_shape(self, rngs):
        """Test that loss returns scalar."""
        loss_fn = HMMLikelihoodLoss(n_states=3, n_emissions=4, rngs=rngs)

        key = jax.random.key(0)
        # Batch of sequences
        observations = jax.random.randint(key, (10, 50), 0, 4)
        log_probs = loss_fn(observations)

        assert log_probs.shape == ()

    def test_loss_finite(self, rngs):
        """Test that loss is finite."""
        loss_fn = HMMLikelihoodLoss(n_states=3, n_emissions=4, rngs=rngs)

        key = jax.random.key(0)
        observations = jax.random.randint(key, (10, 50), 0, 4)
        log_probs = loss_fn(observations)

        assert jnp.isfinite(log_probs)

    def test_gradient_flows(self, rngs):
        """Test that gradients flow through loss."""
        loss_fn = HMMLikelihoodLoss(n_states=3, n_emissions=4, rngs=rngs)

        key = jax.random.key(0)
        observations = jax.random.randint(key, (10, 50), 0, 4)

        @nnx.value_and_grad
        def compute_loss(model):
            return model(observations)

        loss, grads = compute_loss(loss_fn)

        assert jnp.isfinite(loss)
        assert hasattr(grads, "log_transitions")


class TestNBLogProb:
    """``nb_log_prob``, the elementwise NB log-probability the NB losses are built on."""

    COUNTS = jnp.array([[0.0, 1.0, 5.0], [2.0, 0.0, 30.0]])
    MU = jnp.array([[1.0, 2.0, 4.0], [3.0, 0.5, 25.0]])
    THETA = jnp.array([0.5, 2.0, 10.0])

    def test_is_exported_from_diffbio_losses(self) -> None:
        assert "nb_log_prob" in diffbio.losses.__all__

    def test_is_elementwise_with_a_per_gene_theta(self) -> None:
        log_prob = nb_log_prob(self.COUNTS, self.MU, self.THETA[None, :])

        assert log_prob.shape == self.COUNTS.shape
        assert bool(jnp.all(log_prob <= 0.0))

    def test_negative_binomial_loss_is_its_negative_mean(self) -> None:
        """The loss is exactly the negative mean, so the loss's precision bounds hold for it."""
        loss = NegativeBinomialLoss()(self.COUNTS, self.MU, self.THETA)

        assert loss == -jnp.mean(nb_log_prob(self.COUNTS, self.MU, self.THETA[None, :]))

    def test_its_gradient_in_mu_is_the_closed_form_partial(self) -> None:
        theta = self.THETA[None, :]
        gradient = jax.grad(lambda mu: jnp.sum(nb_log_prob(self.COUNTS, mu, theta)))(self.MU)

        partial = theta * (self.COUNTS - self.MU) / (self.MU * (theta + self.MU))
        assert bool(jnp.array_equal(gradient, partial))

    def test_runs_under_jit_vmap_and_grad(self) -> None:
        theta = self.THETA[None, :]
        jitted = jax.jit(nb_log_prob)(self.COUNTS, self.MU, theta)
        mapped = jax.vmap(nb_log_prob, in_axes=(0, 0, None))(self.COUNTS, self.MU, self.THETA)
        gradient = jax.grad(lambda t: jnp.sum(nb_log_prob(self.COUNTS, self.MU, t)))(theta)

        assert jitted.shape == mapped.shape == self.COUNTS.shape
        assert gradient.shape == theta.shape
        assert bool(jnp.all(jnp.isfinite(jitted)))
        assert bool(jnp.all(jnp.isfinite(gradient)))


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_nb_loss_jit(self, rngs):
        """Test NB loss with JIT."""
        loss_fn = NegativeBinomialLoss(rngs=rngs)

        @jax.jit
        def jit_loss(counts, mu, theta):
            return loss_fn(counts, mu, theta)

        key = jax.random.key(0)
        counts = jax.random.poisson(key, 10.0, (50, 100)).astype(jnp.float32)
        mu = jnp.abs(jax.random.normal(key, (50, 100))) + 1.0
        theta = jnp.ones((100,)) * 10.0

        loss = jit_loss(counts, mu, theta)
        assert jnp.isfinite(loss)

    def test_vae_loss_jit(self, rngs):
        """Test VAE loss with JIT."""
        loss_fn = VAELoss(rngs=rngs)

        @jax.jit
        def jit_loss(x, x_recon, mean, logvar):
            return loss_fn(x, x_recon, mean, logvar)

        key = jax.random.key(0)
        x = jax.random.normal(key, (50, 100))
        x_recon = jax.random.normal(key, (50, 100))
        mean = jax.random.normal(key, (50, 32))
        logvar = jax.random.normal(key, (50, 32))

        loss = jit_loss(x, x_recon, mean, logvar)
        assert jnp.isfinite(loss)

    def test_hmm_loss_jit(self, rngs):
        """Test HMM likelihood loss with JIT."""
        loss_fn = HMMLikelihoodLoss(n_states=3, n_emissions=4, rngs=rngs)

        @jax.jit
        def jit_loss(observations):
            return loss_fn(observations)

        key = jax.random.key(0)
        observations = jax.random.randint(key, (10, 50), 0, 4)

        loss = jit_loss(observations)
        assert jnp.isfinite(loss)
