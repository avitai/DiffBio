"""Statistical loss functions for differentiable bioinformatics.

This module provides differentiable implementations of statistical loss
functions commonly used in bioinformatics applications.

Includes:
- zinb_negative_log_likelihood: Zero-Inflated Negative Binomial NLL
- NegativeBinomialLoss: For count data modeling (scRNA-seq, RNA-seq)
- VAELoss: ELBO loss for variational autoencoders
- HMMLikelihoodLoss: Negative log-likelihood for HMM sequence models
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from diffbio.constants import EPSILON


def zinb_negative_log_likelihood(
    counts: Float[Array, "... n_genes"],
    log_rate: Float[Array, "... n_genes"],
    log_theta: Float[Array, "... n_genes"],
    pi_logit: Float[Array, "... n_genes"],
) -> Float[Array, ""]:
    """Zero-Inflated Negative Binomial negative log-likelihood.

    Uses the scVI-style logit-space formulation for numerical stability.
    All log-sigmoid terms are computed via softplus, avoiding explicit
    materialisation of ``sigmoid(pi_logit)`` which is unstable when
    ``pi_logit`` is large positive (pi near 1, so 1-pi near 0).

    Key identities::

        log(sigmoid(pi))   = -softplus(-pi)
        log(1-sigmoid(pi)) = -softplus(pi)

    ZINB: ``P(x) = sigmoid(pi) * delta_0(x) + (1 - sigmoid(pi)) * NB(x; mu, theta)``
    where ``mu = exp(log_rate)``, ``theta = exp(log_theta)``.

    Args:
        counts: Observed counts.
        log_rate: Log mean parameter from decoder.
        log_theta: Log dispersion parameter.
        pi_logit: Logit of zero-inflation probability.

    Returns:
        Negative log-likelihood (scalar, summed over all elements).
    """
    mu = jnp.exp(log_rate)
    theta = jnp.exp(jnp.clip(log_theta, -10.0, 10.0))
    eps = EPSILON

    # Log-space sigmoid computations (numerically stable)
    softplus_pi = jax.nn.softplus(-pi_logit)  # = -log(sigmoid(pi_logit))
    log_theta_mu = jnp.log(theta + mu + eps)

    # NB(0) in log-space combined with dropout logit
    pi_theta_log = -pi_logit + theta * (jnp.log(theta + eps) - log_theta_mu)

    # Case x == 0: log[sigmoid(pi) + (1 - sigmoid(pi)) * NB(0)]
    case_zero = jax.nn.softplus(pi_theta_log) - softplus_pi

    # Case x > 0: log[(1 - sigmoid(pi)) * NB(x)]
    case_nonzero = (
        -softplus_pi
        + pi_theta_log
        + counts * (jnp.log(mu + eps) - log_theta_mu)
        + jax.scipy.special.gammaln(counts + theta)
        - jax.scipy.special.gammaln(theta)
        - jax.scipy.special.gammaln(counts + 1.0)
    )

    is_zero = (counts < eps).astype(jnp.float32)
    log_prob = is_zero * case_zero + (1.0 - is_zero) * case_nonzero

    return -jnp.sum(log_prob)


class NegativeBinomialLoss(nnx.Module):
    """Negative binomial log-likelihood loss for count data.

    The negative binomial distribution is parameterized by mean (mu) and
    dispersion (theta), suitable for overdispersed count data like RNA-seq.

    NB(x | mu, theta) = Gamma(x + theta) / (Gamma(theta) * Gamma(x + 1))
                        * (theta / (theta + mu))^theta
                        * (mu / (theta + mu))^x

    Args:
        eps: Small constant for numerical stability.
        rngs: Flax NNX random number generators.

    Example:
        ```python
        loss_fn = NegativeBinomialLoss(rngs=nnx.Rngs(42))
        loss = loss_fn(counts, mu, theta)
        ```
    """

    def __init__(
        self,
        eps: float = 1e-8,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the NB loss.

        Args:
            eps: Numerical stability constant.
            rngs: Random number generators (not used, for API consistency).
        """
        super().__init__()
        self.eps = eps

    def __call__(
        self,
        counts: Float[Array, "batch genes"],
        mu: Float[Array, "batch genes"],
        theta: Float[Array, "genes"],
    ) -> Float[Array, ""]:
        """Compute negative binomial negative log-likelihood.

        Args:
            counts: Observed counts.
            mu: Predicted mean.
            theta: Dispersion parameter (per gene).

        Returns:
            Mean negative log-likelihood (scalar).
        """
        # Ensure positive values
        mu = jnp.maximum(mu, self.eps)
        theta = jnp.maximum(theta, self.eps)

        # Log-likelihood of NB distribution
        # log P(x | mu, theta) = log Gamma(x + theta) - log Gamma(theta) - log Gamma(x + 1)
        #                       + theta * log(theta / (theta + mu))
        #                       + x * log(mu / (theta + mu))

        log_theta_mu = jnp.log(theta[None, :] + mu + self.eps)

        ll = (
            jax.scipy.special.gammaln(counts + theta[None, :])
            - jax.scipy.special.gammaln(theta[None, :])
            - jax.scipy.special.gammaln(counts + 1)
            + theta[None, :] * (jnp.log(theta[None, :] + self.eps) - log_theta_mu)
            + counts * (jnp.log(mu + self.eps) - log_theta_mu)
        )

        # Return mean negative log-likelihood
        return -jnp.mean(ll)


class VAELoss(nnx.Module):
    """Variational autoencoder ELBO loss.

    Combines reconstruction loss with KL divergence regularization:
    ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

    For Gaussian encoder and prior:
    KL = -0.5 * sum(1 + log(var) - mean^2 - var)

    Args:
        kl_weight: Weight for KL divergence term (beta-VAE).
        reconstruction_type: Type of reconstruction loss ("mse" or "bce").
        rngs: Flax NNX random number generators.

    Example:
        ```python
        loss_fn = VAELoss(kl_weight=1.0, rngs=nnx.Rngs(42))
        loss = loss_fn(x, x_recon, mean, logvar)
        ```
    """

    def __init__(
        self,
        kl_weight: float = 1.0,
        reconstruction_type: str = "mse",
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the VAE loss.

        Args:
            kl_weight: Weight for KL term.
            reconstruction_type: "mse" or "bce".
            rngs: Random number generators (not used, for API consistency).
        """
        super().__init__()
        self.kl_weight = kl_weight
        self.reconstruction_type = reconstruction_type

    def __call__(
        self,
        x: Float[Array, "batch features"],
        x_recon: Float[Array, "batch features"],
        mean: Float[Array, "batch latent"],
        logvar: Float[Array, "batch latent"],
    ) -> Float[Array, ""]:
        """Compute VAE ELBO loss.

        Args:
            x: Original input.
            x_recon: Reconstructed input.
            mean: Encoder mean.
            logvar: Encoder log-variance.

        Returns:
            Negative ELBO (scalar).
        """
        # Reconstruction loss
        if self.reconstruction_type == "mse":
            recon_loss = jnp.mean((x - x_recon) ** 2)
        else:  # bce
            x_recon = jax.nn.sigmoid(x_recon)
            recon_loss = -jnp.mean(
                x * jnp.log(x_recon + 1e-8) + (1 - x) * jnp.log(1 - x_recon + 1e-8)
            )

        # KL divergence: -0.5 * sum(1 + log(var) - mean^2 - var)
        kl_loss = -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))

        # Total loss
        return recon_loss + self.kl_weight * kl_loss


class HMMLikelihoodLoss(nnx.Module):
    """HMM negative log-likelihood loss.

    Computes the negative log-likelihood of sequences under a Hidden
    Markov Model using the forward algorithm with logsumexp for stability.

    Args:
        n_states: Number of hidden states.
        n_emissions: Number of emission symbols.
        rngs: Flax NNX random number generators.

    Example:
        ```python
        loss_fn = HMMLikelihoodLoss(n_states=3, n_emissions=4, rngs=rnx.Rngs(42))
        nll = loss_fn(observations)
        ```
    """

    def __init__(
        self,
        n_states: int,
        n_emissions: int,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the HMM loss.

        Args:
            n_states: Number of hidden states.
            n_emissions: Number of emission symbols.
            rngs: Random number generators.
        """
        super().__init__()

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.n_states = n_states
        self.n_emissions = n_emissions

        key = rngs.params()
        k1, k2, k3 = jax.random.split(key, 3)

        # Learnable parameters (log-space for stability)
        # Initial state distribution
        self.log_initial = nnx.Param(jax.random.normal(k1, (n_states,)) * 0.1)

        # Transition matrix (log probabilities)
        self.log_transitions = nnx.Param(jax.random.normal(k2, (n_states, n_states)) * 0.1)

        # Emission matrix (log probabilities)
        self.log_emissions = nnx.Param(jax.random.normal(k3, (n_states, n_emissions)) * 0.1)

    def _get_log_initial(self) -> Float[Array, "n_states"]:
        """Get normalized log initial distribution."""
        return jax.nn.log_softmax(self.log_initial[...])

    def _get_log_transitions(self) -> Float[Array, "n_states n_states"]:
        """Get normalized log transition matrix."""
        return jax.nn.log_softmax(self.log_transitions[...], axis=-1)

    def _get_log_emissions(self) -> Float[Array, "n_states n_emissions"]:
        """Get normalized log emission matrix."""
        return jax.nn.log_softmax(self.log_emissions[...], axis=-1)

    def _forward_single(
        self,
        observations: Int[Array, "seq_len"],
    ) -> Float[Array, ""]:
        """Forward algorithm for a single sequence.

        Args:
            observations: Integer-encoded observations.

        Returns:
            Log-likelihood of the sequence.
        """
        log_init = self._get_log_initial()
        log_trans = self._get_log_transitions()
        log_emit = self._get_log_emissions()

        # Initialize with first observation
        log_alpha = log_init + log_emit[:, observations[0]]

        # Forward pass
        def step(log_alpha, obs):
            # log_alpha: (n_states,)
            # Transition: log_alpha[i] + log_trans[i, j] for all j
            log_alpha_expanded = log_alpha[:, None] + log_trans  # (n_states, n_states)
            log_alpha_new = jax.scipy.special.logsumexp(log_alpha_expanded, axis=0)
            # Add emission
            log_alpha_new = log_alpha_new + log_emit[:, obs]
            return log_alpha_new, None

        log_alpha, _ = jax.lax.scan(step, log_alpha, observations[1:])

        # Final log-likelihood
        return jax.scipy.special.logsumexp(log_alpha)

    def __call__(
        self,
        observations: Int[Array, "batch seq_len"],
    ) -> Float[Array, ""]:
        """Compute mean negative log-likelihood over batch.

        Args:
            observations: Batch of integer-encoded sequences.

        Returns:
            Mean negative log-likelihood (scalar).
        """
        # Compute log-likelihood for each sequence
        log_probs = jax.vmap(self._forward_single)(observations)

        # Return mean negative log-likelihood
        return -jnp.mean(log_probs)
