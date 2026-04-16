"""Differentiable metagenomic binning operators.

This module provides VAE-based approaches to metagenomic binning
inspired by VAMB (Variational Autoencoders for Metagenomic Binning).

The approach encodes tetranucleotide frequencies (TNF) and abundance
profiles into a latent space where contigs from the same genome cluster together.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.base import MLP
from flax import nnx
from jaxtyping import Array, Float

from diffbio.configs import TemperatureConfig
from diffbio.core.base_operators import EncoderDecoderOperator, TemperatureOperator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetagenomicBinnerConfig(TemperatureConfig):
    """Configuration for metagenomic binning VAE.

    Attributes:
        n_tnf_features: Number of tetranucleotide frequency features (default 136).
        n_abundance_features: Number of sample abundance features.
        latent_dim: Dimension of the latent space.
        hidden_dims: tuple of hidden layer dimensions for encoder/decoder.
        dropout_rate: Dropout rate for regularization.
        beta: KL divergence weight (beta-VAE).
        n_clusters: Number of clusters for soft binning.
    """

    n_tnf_features: int = 136  # 4^4 / 2 for canonical k-mers
    n_abundance_features: int = 10
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout_rate: float = 0.2
    beta: float = 1.0
    n_clusters: int = 100

    def __post_init__(self) -> None:
        """Set stochastic config and validate."""
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "sample")
        super().__post_init__()
        if not self.hidden_dims:
            raise ValueError(
                "MetagenomicBinnerConfig.hidden_dims must contain at least one hidden layer."
            )


class DifferentiableMetagenomicBinner(TemperatureOperator, EncoderDecoderOperator):
    """VAMB-style differentiable metagenomic binning.

    This operator implements a Variational Autoencoder for metagenomic binning,
    encoding tetranucleotide frequencies (TNF) and abundance profiles into a
    shared latent space where contigs from the same genome cluster together.

    The approach is fully differentiable, enabling:
    - End-to-end optimization with downstream tasks
    - Soft cluster assignments via temperature-controlled softmax
    - Integration with neural abundance estimation

    Input data structure:
        - tnf: Float[Array, "n_contigs n_tnf"] - Tetranucleotide frequencies
        - abundance: Float[Array, "n_contigs n_samples"] - Sample abundances

    Output data structure (adds):
        - latent_z: Float[Array, "n_contigs latent_dim"] - Latent representations
        - latent_mu: Float[Array, "n_contigs latent_dim"] - Latent means
        - latent_logvar: Float[Array, "n_contigs latent_dim"] - Latent log variance
        - cluster_assignments: Float[Array, "n_contigs n_clusters"] - Soft bins
        - reconstructed_tnf: Float[Array, "n_contigs n_tnf"] - Reconstructed TNF
        - reconstructed_abundance: Float[Array, "n_contigs n_samples"] - Recon. abundance

    Example:
        ```python
        config = MetagenomicBinnerConfig(n_abundance_features=5, n_clusters=50)
        binner = DifferentiableMetagenomicBinner(config, rngs=nnx.Rngs(42))
        result, state, meta = binner.apply(data, {}, None)
        bins = result["cluster_assignments"].argmax(axis=-1)
        ```
    """

    def __init__(
        self,
        config: MetagenomicBinnerConfig,
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ):
        """Initialize the metagenomic binner.

        Args:
            config: Binner configuration.
            rngs: Random number generators.
            name: Optional name for the operator.
        """
        super().__init__(config, rngs=rngs, name=name)

        input_dim = config.n_tnf_features + config.n_abundance_features

        self.encoder_backbone = MLP(
            hidden_dims=list(config.hidden_dims),
            in_features=input_dim,
            activation="relu",
            dropout_rate=config.dropout_rate,
            output_activation="relu",
            use_batch_norm=True,
            rngs=rngs,
        )
        self.fc_latent = nnx.List(
            [
                nnx.Linear(config.hidden_dims[-1], config.latent_dim, rngs=rngs),
                nnx.Linear(config.hidden_dims[-1], config.latent_dim, rngs=rngs),
            ]
        )

        decoder_hidden_dims = list(reversed(config.hidden_dims))
        self.decoder_backbone = MLP(
            hidden_dims=decoder_hidden_dims,
            in_features=config.latent_dim,
            activation="relu",
            dropout_rate=config.dropout_rate,
            output_activation="relu",
            use_batch_norm=True,
            rngs=rngs,
        )
        self.decoder_heads = nnx.List(
            [
                nnx.Linear(decoder_hidden_dims[-1], config.n_tnf_features, rngs=rngs),
                nnx.Linear(decoder_hidden_dims[-1], config.n_abundance_features, rngs=rngs),
            ]
        )

        # Tracks train/eval mode for latent sampling even when no stochastic
        # submodule on the encoder is active for a given configuration.
        self.latent_sampling_mode = nnx.Dropout(rate=0.0, rngs=rngs)

        # Learnable cluster centroids
        self.centroids = nnx.Param(
            jax.random.normal(rngs.params(), (config.n_clusters, config.latent_dim)) * 0.1
        )

    def encode(
        self, x: Float[Array, "batch input_dim"]
    ) -> tuple[Float[Array, "batch latent"], Float[Array, "batch latent"]]:
        """Encode input to latent distribution.

        Args:
            x: Concatenated TNF and abundance features.

        Returns:
            Tuple of (mu, logvar).
        """
        encoded = self.encoder_backbone(x)
        if isinstance(encoded, tuple):
            raise TypeError("Metagenomic binner encoder backbone must return a single tensor.")

        fc_mu, fc_logvar = self.fc_latent
        mu = fc_mu(encoded)
        logvar = fc_logvar(encoded)
        return mu, logvar

    def decode(
        self, z: Float[Array, "batch latent"]
    ) -> tuple[Float[Array, "batch n_tnf"], Float[Array, "batch n_abundance"]]:
        """Decode latent to reconstructed features.

        Args:
            z: Latent representation.

        Returns:
            Tuple of (tnf, abundance).
        """
        decoded = self.decoder_backbone(z)
        if isinstance(decoded, tuple):
            raise TypeError("Metagenomic binner decoder backbone must return a single tensor.")

        fc_tnf, fc_abundance = self.decoder_heads
        # TNF uses softmax (frequencies sum to 1)
        tnf_recon = nnx.softmax(fc_tnf(decoded), axis=-1)
        # Abundance uses softplus (positive values)
        abundance_recon = nnx.softplus(fc_abundance(decoded))

        return tnf_recon, abundance_recon

    def soft_cluster(self, z: Float[Array, "batch latent"]) -> Float[Array, "batch n_clusters"]:
        """Compute soft cluster assignments.

        Args:
            z: Latent representations.

        Returns:
            Soft cluster assignment probabilities.
        """
        # Compute squared distances to centroids
        z_expanded = z[:, None, :]  # (batch, 1, latent)
        centroids_expanded = self.centroids[...][None, :, :]  # (1, n_clusters, latent)
        sq_distances = jnp.sum((z_expanded - centroids_expanded) ** 2, axis=-1)

        # Soft assignment via softmax
        temperature = jnp.maximum(self._temperature, 1e-6)
        assignments = nnx.softmax(-sq_distances / temperature, axis=-1)
        return assignments

    def apply(
        self,
        data: dict[str, Array],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,  # noqa: ARG002
        stats: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Array], dict[str, Any], dict[str, Any] | None]:
        """Apply metagenomic binning.

        Args:
            data: Input data containing:
                - tnf: Float[Array, "n_contigs n_tnf"]
                - abundance: Float[Array, "n_contigs n_samples"]
            state: Element state (passed through).
            metadata: Element metadata (passed through).
            random_params: Random parameters.
            stats: Optional statistics dict.

        Returns:
            Tuple of (output_data, state, metadata).
        """
        tnf = data["tnf"]
        abundance = data["abundance"]

        # Concatenate features
        x = jnp.concatenate([tnf, abundance], axis=-1)

        # Encode
        mu, logvar = self.encode(x)

        # Sample latent (use mu during eval for determinism)
        if self.latent_sampling_mode.deterministic:
            z = mu  # Eval mode: deterministic
        else:
            z = self.reparameterize(mu, logvar)  # Train mode: stochastic

        # Decode
        tnf_recon, abundance_recon = self.decode(z)

        # Soft cluster assignments
        cluster_assignments = self.soft_cluster(z)

        # Build output
        output_data = {
            **data,
            "latent_z": z,
            "latent_mu": mu,
            "latent_logvar": logvar,
            "cluster_assignments": cluster_assignments,
            "reconstructed_tnf": tnf_recon,
            "reconstructed_abundance": abundance_recon,
        }

        return output_data, state, metadata


def create_metagenomic_binner(
    n_abundance_features: int = 10,
    n_clusters: int = 100,
    latent_dim: int = 32,
    hidden_dims: tuple[int, ...] | None = None,
    seed: int = 42,
) -> DifferentiableMetagenomicBinner:
    """Factory function to create a metagenomic binner.

    Args:
        n_abundance_features: Number of sample abundance features.
        n_clusters: Number of clusters/bins.
        latent_dim: Dimension of latent space.
        hidden_dims: Hidden layer dimensions.
        seed: Random seed.

    Returns:
        Configured DifferentiableMetagenomicBinner instance.
    """
    if hidden_dims is None:
        hidden_dims = (512, 256)

    config = MetagenomicBinnerConfig(
        n_abundance_features=n_abundance_features,
        n_clusters=n_clusters,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
    rngs = nnx.Rngs(seed)
    return DifferentiableMetagenomicBinner(config, rngs=rngs)
