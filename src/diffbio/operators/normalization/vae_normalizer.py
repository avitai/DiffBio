"""VAE-based count normalization operator.

This module provides a variational autoencoder for normalizing gene
expression count data, inspired by scVI (Lopez et al., 2018).

Key technique: Learn a latent representation of cell state while
modeling count data with a configurable likelihood (Poisson or ZINB).

The ZINB (Zero-Inflated Negative Binomial) likelihood is particularly
suited for single-cell RNA-seq data which exhibits both overdispersion
and excess zeros (dropout events).
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
from artifex.generative_models.core.losses.divergence import gaussian_kl_divergence
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.base_operators import EncoderDecoderOperator


@dataclass
class VAENormalizerConfig(OperatorConfig):
    """Configuration for VAENormalizer.

    Attributes:
        latent_dim: Dimension of latent space.
        hidden_dims: Hidden layer dimensions for encoder/decoder.
        n_genes: Number of genes (input/output dimension).
        use_batch_correction: Whether to include batch effects.
        likelihood: Likelihood model for reconstruction loss.
            'poisson' for standard Poisson NLL, 'zinb' for
            Zero-Inflated Negative Binomial.
        stochastic: Whether the operator uses randomness (True for VAE).
        stream_name: RNG stream name for sampling.
    """

    latent_dim: int = 10
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    n_genes: int = 2000
    use_batch_correction: bool = False
    likelihood: Literal["poisson", "zinb"] = "poisson"
    stochastic: bool = True
    stream_name: str = "sample"


class VAENormalizer(EncoderDecoderOperator):
    """Variational autoencoder for count normalization.

    This operator learns a low-dimensional latent representation of
    single-cell gene expression data while accounting for technical
    factors like library size.

    The model:
    - Encoder: counts -> latent (mean, logvar)
    - Reparameterization: z = mean + exp(0.5 * logvar) * epsilon
    - Decoder: z -> gene expression rates (and optionally dispersion/dropout)

    Supports two likelihood models:
    - Poisson: Simple count model (default)
    - ZINB: Zero-Inflated Negative Binomial for overdispersed data
      with excess zeros, as used in scVI

    Inherits from EncoderDecoderOperator to get:

    - reparameterize() for sampling with reparameterization trick
    - kl_divergence() for KL from standard normal
    - elbo_loss() for combining reconstruction and KL losses

    Args:
        config: VAENormalizerConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = VAENormalizerConfig(n_genes=2000, latent_dim=10)
        normalizer = VAENormalizer(config, rngs=nnx.Rngs(42))
        data = {"counts": counts, "library_size": lib_size}
        result, state, meta = normalizer.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: VAENormalizerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the VAE normalizer.

        Args:
            config: VAE configuration.
            rngs: Random number generators for initialization and sampling.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.n_genes = config.n_genes
        self.stream_name = config.stream_name

        # Build encoder layers
        encoder_layers: list[nnx.Linear] = []
        prev_dim = config.n_genes

        for hidden_dim in config.hidden_dims:
            encoder_layers.append(
                nnx.Linear(in_features=prev_dim, out_features=hidden_dim, rngs=rngs)
            )
            prev_dim = hidden_dim

        self.encoder_layers = nnx.List(encoder_layers)

        # Latent space projection (mean and logvar)
        self.fc_mean = nnx.Linear(in_features=prev_dim, out_features=config.latent_dim, rngs=rngs)
        self.fc_logvar = nnx.Linear(in_features=prev_dim, out_features=config.latent_dim, rngs=rngs)

        # Build decoder layers
        decoder_layers: list[nnx.Linear] = []
        decoder_prev_dim = config.latent_dim

        for hidden_dim in reversed(config.hidden_dims):
            decoder_layers.append(
                nnx.Linear(in_features=decoder_prev_dim, out_features=hidden_dim, rngs=rngs)
            )
            decoder_prev_dim = hidden_dim

        self.decoder_layers = nnx.List(decoder_layers)

        # Output layer (log rates)
        self.fc_output = nnx.Linear(
            in_features=decoder_prev_dim, out_features=config.n_genes, rngs=rngs
        )

        # ZINB-specific decoder heads
        if config.likelihood == "zinb":
            self.fc_log_theta = nnx.Linear(
                in_features=decoder_prev_dim, out_features=config.n_genes, rngs=rngs
            )
            self.fc_pi_logit = nnx.Linear(
                in_features=decoder_prev_dim, out_features=config.n_genes, rngs=rngs
            )

    def encode(
        self,
        counts: Float[Array, "n_genes"],
    ) -> tuple[Float[Array, "latent_dim"], Float[Array, "latent_dim"]]:
        """Encode counts to latent distribution parameters.

        Args:
            counts: Gene expression counts.

        Returns:
            Tuple of (mean, logvar) for latent distribution.
        """
        # Log-transform counts for stability
        x = jnp.log1p(counts)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
            x = jax.nn.relu(x)

        # Project to latent space
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        # Clamp logvar for numerical stability
        logvar = jnp.clip(logvar, -10.0, 10.0)

        return mean, logvar

    # reparameterize() is inherited from EncoderDecoderOperator

    def decode(
        self,
        z: Float[Array, "latent_dim"],
        library_size: Float[Array, ""],
    ) -> dict[str, jax.Array]:
        """Decode latent representation to gene expression parameters.

        Args:
            z: Latent representation.
            library_size: Total counts (library size) for normalization.

        Returns:
            Dictionary with keys:
                - 'log_rate': Log rates for each gene (always present).
                - 'log_theta': Log dispersion parameter (ZINB only).
                - 'pi_logit': Dropout logit for zero inflation (ZINB only).
        """
        x = z

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
            x = jax.nn.relu(x)

        # Output layer (log rates, normalized by library size)
        log_rate = self.fc_output(x)

        result: dict[str, jax.Array] = {}

        # ZINB heads are computed before adding library size
        if self.config.likelihood == "zinb":
            result["log_theta"] = self.fc_log_theta(x)
            result["pi_logit"] = self.fc_pi_logit(x)

        # Add library size effect (log scale)
        log_rate = log_rate + jnp.log(library_size + 1e-8)
        result["log_rate"] = log_rate

        return result

    def _poisson_nll(
        self,
        counts: Float[Array, "n_genes"],
        log_rate: Float[Array, "n_genes"],
    ) -> Float[Array, ""]:
        """Compute Poisson negative log-likelihood.

        Args:
            counts: Original counts.
            log_rate: Log rates from decoder.

        Returns:
            Negative log-likelihood (scalar).
        """
        rate = jnp.exp(log_rate)
        return jnp.sum(rate - counts * log_rate)

    def _zinb_nll(
        self,
        counts: Float[Array, "n_genes"],
        log_rate: Float[Array, "n_genes"],
        log_theta: Float[Array, "n_genes"],
        pi_logit: Float[Array, "n_genes"],
    ) -> Float[Array, ""]:
        """Compute Zero-Inflated Negative Binomial negative log-likelihood.

        Uses the scVI-style logit-space formulation for numerical stability.
        All log-sigmoid terms are computed via softplus, avoiding explicit
        materialization of sigmoid(pi_logit) which is unstable when pi_logit
        is large positive (pi near 1, so 1-pi near 0).

        Key identities:
            log(sigmoid(pi))   = -softplus(-pi)
            log(1-sigmoid(pi)) = -softplus(pi)

        ZINB: P(x) = sigmoid(pi) * delta_0(x) + (1 - sigmoid(pi)) * NB(x; mu, theta)
        where mu = exp(log_rate), theta = exp(log_theta).

        Args:
            counts: Original counts.
            log_rate: Log mean parameter from decoder.
            log_theta: Log dispersion parameter.
            pi_logit: Logit of zero-inflation probability.

        Returns:
            Negative log-likelihood (scalar).
        """
        mu = jnp.exp(log_rate)
        theta = jnp.exp(jnp.clip(log_theta, -10.0, 10.0))
        eps = 1e-8

        # Log-space sigmoid computations (numerically stable)
        softplus_pi = jax.nn.softplus(-pi_logit)  # = -log(sigmoid(pi_logit))
        log_theta_mu = jnp.log(theta + mu + eps)

        # NB(0) in log-space combined with dropout logit:
        # -pi_logit + theta * log(theta / (theta + mu))
        pi_theta_log = -pi_logit + theta * (jnp.log(theta + eps) - log_theta_mu)

        # Case x == 0: log[sigmoid(pi) + (1-sigmoid(pi)) * NB(0)]
        case_zero = jax.nn.softplus(pi_theta_log) - softplus_pi

        # Case x > 0: log[(1-sigmoid(pi)) * NB(x)]
        case_nonzero = (
            -softplus_pi
            + pi_theta_log
            + counts * (jnp.log(mu + eps) - log_theta_mu)
            + jax.scipy.special.gammaln(counts + theta)
            - jax.scipy.special.gammaln(theta)
            - jax.scipy.special.gammaln(counts + 1.0)
        )

        # Select case based on count value
        is_zero = (counts < eps).astype(jnp.float32)
        log_prob = is_zero * case_zero + (1.0 - is_zero) * case_nonzero

        return -jnp.sum(log_prob)

    def reconstruction_loss(
        self,
        counts: Float[Array, "n_genes"],
        decode_output: dict[str, jax.Array],
    ) -> Float[Array, ""]:
        """Compute reconstruction loss based on configured likelihood.

        Args:
            counts: Original counts.
            decode_output: Dictionary from decode() containing 'log_rate'
                and optionally 'log_theta', 'pi_logit' for ZINB.

        Returns:
            Negative log-likelihood (scalar).
        """
        log_rate = decode_output["log_rate"]
        if self.config.likelihood == "zinb":
            return self._zinb_nll(
                counts,
                log_rate,
                decode_output["log_theta"],
                decode_output["pi_logit"],
            )
        return self._poisson_nll(counts, log_rate)

    # kl_divergence() is inherited from EncoderDecoderOperator

    def compute_elbo_loss(
        self,
        counts: Float[Array, "n_genes"],
        library_size: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Compute negative ELBO loss.

        Uses gaussian_kl_divergence from artifex for the KL term.

        Args:
            counts: Gene expression counts.
            library_size: Total counts.

        Returns:
            Negative ELBO (reconstruction loss + KL divergence).
        """
        # Encode
        mean, logvar = self.encode(counts)

        # Sample latent using inherited reparameterize (uses self.rngs)
        z = self.reparameterize(mean, logvar)

        # Decode (returns dict)
        decode_output = self.decode(z, library_size)

        # Reconstruction loss
        recon_loss = self.reconstruction_loss(counts, decode_output)

        # KL divergence via artifex (handles unbatched 1-D inputs with sum)
        kl = gaussian_kl_divergence(mean, logvar, reduction="sum")

        return recon_loss + kl

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply VAE normalization to count data.

        This method encodes the counts to latent space, samples a
        latent representation, and decodes to normalized expression.

        Args:
            data: Dictionary containing:
                - "counts": Gene expression counts (n_genes,)
                - "library_size": Total counts for the cell
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Optional random parameters (not used)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "counts": Original counts
                    - "normalized": Normalized expression
                    - "latent_z": Sampled latent representation
                    - "latent_mean": Mean of latent distribution
                    - "latent_logvar": Log variance of latent distribution
                    - "log_rate": Decoded log rates
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts = data["counts"]
        library_size = data["library_size"]

        # Encode to latent distribution
        mean, logvar = self.encode(counts)

        # Sample from latent distribution using inherited reparameterize
        # (uses self.rngs from EncoderDecoderOperator)
        z = self.reparameterize(mean, logvar)

        # Decode to gene expression rates (returns dict)
        decode_output = self.decode(z, library_size)
        log_rate = decode_output["log_rate"]

        # Compute normalized expression (rate normalized by library size)
        # This is the "denoised" expression
        normalized = jnp.exp(log_rate - jnp.log(library_size + 1e-8))

        # Build output data
        transformed_data = {
            "counts": counts,
            "library_size": library_size,
            "normalized": normalized,
            "latent_z": z,
            "latent_mean": mean,
            "latent_logvar": logvar,
            "log_rate": log_rate,
        }

        return transformed_data, state, metadata
