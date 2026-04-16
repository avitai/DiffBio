"""Shared count-VAE building blocks for DiffBio operators."""

from __future__ import annotations

from typing import Any, cast

from artifex.generative_models.core.base import MLP
from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from diffbio.losses.statistical_losses import zinb_negative_log_likelihood
from diffbio.utils.nn_utils import ensure_rngs


class CountVAEBackboneMixin:
    """Shared encoder/decoder backbone for count-based VAEs."""

    encoder_backbone: MLP | None
    fc_mean: nnx.Linear
    fc_logvar: nnx.Linear
    decoder_backbone: MLP | None
    fc_output: nnx.Linear
    n_genes: int
    stream_name: Any

    def _init_count_vae_backbone(
        self,
        *,
        n_inputs: int,
        latent_dim: int,
        hidden_dims: list[int],
        n_outputs: int,
        rngs: nnx.Rngs | None,
    ) -> None:
        """Initialise the shared count-VAE encoder and decoder layers."""
        safe_rngs = ensure_rngs(rngs)

        encoder_hidden_dims = list(hidden_dims)
        decoder_hidden_dims = list(reversed(hidden_dims))

        if encoder_hidden_dims:
            self.encoder_backbone = MLP(
                hidden_dims=encoder_hidden_dims,
                in_features=n_inputs,
                activation="relu",
                output_activation="relu",
                use_batch_norm=False,
                rngs=safe_rngs,
            )
            encoder_out_dim = encoder_hidden_dims[-1]
        else:
            self.encoder_backbone = None
            encoder_out_dim = n_inputs

        self.fc_mean = nnx.Linear(
            in_features=encoder_out_dim,
            out_features=latent_dim,
            rngs=safe_rngs,
        )
        self.fc_logvar = nnx.Linear(
            in_features=encoder_out_dim,
            out_features=latent_dim,
            rngs=safe_rngs,
        )

        if decoder_hidden_dims:
            self.decoder_backbone = MLP(
                hidden_dims=decoder_hidden_dims,
                in_features=latent_dim,
                activation="relu",
                output_activation="relu",
                use_batch_norm=False,
                rngs=safe_rngs,
            )
            decoder_out_dim = decoder_hidden_dims[-1]
        else:
            self.decoder_backbone = None
            decoder_out_dim = latent_dim

        self.fc_output = nnx.Linear(
            in_features=decoder_out_dim,
            out_features=n_outputs,
            rngs=safe_rngs,
        )

    def _init_count_vae_operator(
        self,
        *,
        config: Any,
        rngs: nnx.Rngs | None,
    ) -> nnx.Rngs:
        """Initialise shared count-VAE operator state and return safe RNGs."""
        safe_rngs = ensure_rngs(rngs)
        self.n_genes = config.n_genes
        self.stream_name = nnx.static(config.stream_name)
        self._init_count_vae_backbone(
            n_inputs=config.n_genes,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            n_outputs=config.n_genes,
            rngs=safe_rngs,
        )
        return safe_rngs

    def encode(
        self,
        counts: Float[Array, "... n_genes"],
    ) -> tuple[Float[Array, "... latent_dim"], Float[Array, "... latent_dim"]]:
        """Encode count vectors to latent Gaussian parameters."""
        x = jnp.log1p(counts)
        if self.encoder_backbone is not None:
            x = cast(jax.Array, self.encoder_backbone(x))
        mean = self.fc_mean(x)
        logvar = jnp.clip(self.fc_logvar(x), -10.0, 10.0)
        return mean, logvar

    def decode_hidden(
        self,
        z: Float[Array, "... latent_dim"],
    ) -> Float[Array, "... hidden_dim"]:
        """Decode latent vectors to the shared decoder hidden representation."""
        if self.decoder_backbone is None:
            return z
        return cast(jax.Array, self.decoder_backbone(z))

    def decode_rates(
        self,
        z: Float[Array, "... latent_dim"],
    ) -> Float[Array, "... n_outputs"]:
        """Decode latent vectors directly to output log-rates."""
        return self.fc_output(self.decode_hidden(z))


class CountVAEBackbone(CountVAEBackboneMixin, nnx.Module):
    """Concrete test harness for the shared count-VAE backbone."""

    def __init__(
        self,
        *,
        n_inputs: int,
        latent_dim: int,
        hidden_dims: list[int],
        n_outputs: int,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialise a standalone shared count-VAE backbone."""
        super().__init__()
        self._init_count_vae_backbone(
            n_inputs=n_inputs,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            n_outputs=n_outputs,
            rngs=rngs,
        )


class CountReconstructionMixin:
    """Shared reconstruction losses for count-based VAE operators."""

    @staticmethod
    def _poisson_nll(
        counts: Float[Array, "... n_genes"],
        log_rate: Float[Array, "... n_genes"],
    ) -> Float[Array, ""]:
        """Compute Poisson negative log-likelihood."""
        rate = jnp.exp(log_rate)
        return jnp.sum(rate - counts * log_rate)

    @staticmethod
    def _zinb_nll(
        counts: Float[Array, "... n_genes"],
        log_rate: Float[Array, "... n_genes"],
        log_theta: Float[Array, "... n_genes"],
        pi_logit: Float[Array, "... n_genes"],
    ) -> Float[Array, ""]:
        """Compute Zero-Inflated Negative Binomial negative log-likelihood."""
        return zinb_negative_log_likelihood(counts, log_rate, log_theta, pi_logit)

    def reconstruction_loss(
        self,
        counts: Float[Array, "... n_genes"],
        decode_output: dict[str, jax.Array],
    ) -> Float[Array, ""]:
        """Compute reconstruction loss for Poisson or ZINB decoder outputs."""
        log_rate = decode_output["log_rate"]
        if "log_theta" in decode_output:
            return self._zinb_nll(
                counts,
                log_rate,
                decode_output["log_theta"],
                decode_output["pi_logit"],
            )
        return self._poisson_nll(counts, log_rate)
