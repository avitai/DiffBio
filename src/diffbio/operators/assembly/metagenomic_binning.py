"""Differentiable metagenomic binning operators.

This module provides VAE-based approaches to metagenomic binning
inspired by VAMB (Variational Autoencoders for Metagenomic Binning).

The approach encodes tetranucleotide frequencies (TNF) and abundance
profiles into a latent space where contigs from the same genome cluster together.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float

from diffbio.core.base_operators import EncoderDecoderOperator


@dataclass
class MetagenomicBinnerConfig(OperatorConfig):
    """Configuration for metagenomic binning VAE.

    Attributes:
        n_tnf_features: Number of tetranucleotide frequency features (default 136).
        n_abundance_features: Number of sample abundance features.
        latent_dim: Dimension of the latent space.
        hidden_dims: Hidden layer dimensions for encoder/decoder.
        dropout_rate: Dropout rate for regularization.
        beta: KL divergence weight (beta-VAE).
        temperature: Temperature for soft clustering.
        n_clusters: Number of clusters for soft binning.
        stochastic: Whether the operator uses stochastic operations.
        stream_name: RNG stream name for stochastic operations.
    """

    n_tnf_features: int = 136  # 4^4 / 2 for canonical k-mers
    n_abundance_features: int = 10
    latent_dim: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    dropout_rate: float = 0.2
    beta: float = 1.0
    temperature: float = 1.0
    n_clusters: int = 100
    stochastic: bool = True
    stream_name: str | None = field(default="sample", repr=False)


class DifferentiableMetagenomicBinner(EncoderDecoderOperator):
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
        >>> config = MetagenomicBinnerConfig(n_abundance_features=5, n_clusters=50)
        >>> binner = DifferentiableMetagenomicBinner(config, rngs=nnx.Rngs(42))
        >>> result, state, meta = binner.apply(data, {}, None)
        >>> bins = result["cluster_assignments"].argmax(axis=-1)
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

        self.config: MetagenomicBinnerConfig = config

        input_dim = config.n_tnf_features + config.n_abundance_features

        # Build encoder (using nnx.List for proper pytree handling)
        encoder_linear = []
        encoder_bn = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            encoder_linear.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            encoder_bn.append(nnx.BatchNorm(hidden_dim, rngs=rngs))
            prev_dim = hidden_dim
        self.encoder_linear = nnx.List(encoder_linear)
        self.encoder_bn = nnx.List(encoder_bn)
        self.encoder_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.fc_mu = nnx.Linear(config.hidden_dims[-1], config.latent_dim, rngs=rngs)
        self.fc_logvar = nnx.Linear(config.hidden_dims[-1], config.latent_dim, rngs=rngs)

        # Build decoder (using nnx.List for proper pytree handling)
        decoder_linear = []
        decoder_bn = []
        prev_dim = config.latent_dim
        for hidden_dim in reversed(config.hidden_dims):
            decoder_linear.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            decoder_bn.append(nnx.BatchNorm(hidden_dim, rngs=rngs))
            prev_dim = hidden_dim
        self.decoder_linear = nnx.List(decoder_linear)
        self.decoder_bn = nnx.List(decoder_bn)
        self.decoder_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.fc_tnf = nnx.Linear(config.hidden_dims[0], config.n_tnf_features, rngs=rngs)
        self.fc_abundance = nnx.Linear(
            config.hidden_dims[0], config.n_abundance_features, rngs=rngs
        )

        # Learnable cluster centroids
        self.centroids = nnx.Param(
            jax.random.normal(rngs.params(), (config.n_clusters, config.latent_dim)) * 0.1
        )

        # Temperature for soft clustering
        self.temperature = nnx.Param(jnp.array(config.temperature))

    def encode(
        self, x: Float[Array, "batch input_dim"]
    ) -> tuple[Float[Array, "batch latent"], Float[Array, "batch latent"]]:
        """Encode input to latent distribution.

        Args:
            x: Concatenated TNF and abundance features.

        Returns:
            Tuple of (mu, logvar).
        """
        h = x
        for linear, bn in zip(self.encoder_linear, self.encoder_bn):
            h = linear(h)
            h = bn(h)
            h = nnx.relu(h)
            h = self.encoder_dropout(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
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
        h = z
        for linear, bn in zip(self.decoder_linear, self.decoder_bn):
            h = linear(h)
            h = bn(h)
            h = nnx.relu(h)
            h = self.decoder_dropout(h)

        # TNF uses softmax (frequencies sum to 1)
        tnf_recon = nnx.softmax(self.fc_tnf(h), axis=-1)
        # Abundance uses softplus (positive values)
        abundance_recon = nnx.softplus(self.fc_abundance(h))

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
        centroids_expanded = self.centroids.value[None, :, :]  # (1, n_clusters, latent)
        sq_distances = jnp.sum((z_expanded - centroids_expanded) ** 2, axis=-1)

        # Soft assignment via softmax
        temperature = jnp.maximum(self.temperature.value, 1e-6)
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
        # Check if in eval mode via dropout's deterministic flag
        if self.encoder_dropout.deterministic:
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
    hidden_dims: list[int] | None = None,
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
        hidden_dims = [512, 256]

    config = MetagenomicBinnerConfig(
        n_abundance_features=n_abundance_features,
        n_clusters=n_clusters,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
    rngs = nnx.Rngs(seed)
    return DifferentiableMetagenomicBinner(config, rngs=rngs)
