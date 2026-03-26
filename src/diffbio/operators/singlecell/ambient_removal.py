"""Differentiable ambient RNA removal operator.

This module provides CellBender-style ambient RNA removal using a VAE
architecture that learns to separate cell-intrinsic from ambient signal.

Key technique: Uses variational autoencoder to model cell-specific expression
and ambient contamination fraction, enabling decontamination with uncertainty
quantification.

Applications: Removing ambient RNA contamination from single-cell RNA-seq data,
improving cell type identification and differential expression analysis.

Inherits from EncoderDecoderOperator to get:

- reparameterize() for sampling with reparameterization trick
- kl_divergence() for KL from standard normal
- elbo_loss() for combining reconstruction and KL losses
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.base_operators import EncoderDecoderOperator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AmbientRemovalConfig(OperatorConfig):
    """Configuration for DifferentiableAmbientRemoval.

    Attributes:
        n_genes: Number of genes in expression profiles.
        latent_dim: Dimension of latent space.
        hidden_dims: Hidden layer dimensions for encoder/decoder.
        ambient_prior: Prior probability of ambient contamination.
        temperature: Temperature for softmax operations.
    """

    n_genes: int = 2000
    latent_dim: int = 64
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    ambient_prior: float = 0.01
    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Set stochastic defaults and validate."""
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "sample")
        super().__post_init__()


class AmbientEncoder(nnx.Module):
    """Encoder network for ambient removal VAE."""

    def __init__(
        self,
        n_genes: int,
        hidden_dims: list[int],
        latent_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the encoder.

        Args:
            n_genes: Number of input genes.
            hidden_dims: Hidden layer dimensions.
            latent_dim: Latent space dimension.
            rngs: Random number generators.
        """
        super().__init__()

        # Build encoder layers
        layers = []
        norms = []
        in_dim = n_genes
        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_features=in_dim, out_features=hidden_dim, rngs=rngs))
            norms.append(nnx.LayerNorm(num_features=hidden_dim, rngs=rngs))
            in_dim = hidden_dim

        self.layers = nnx.List(layers)
        self.norms = nnx.List(norms)

        # Latent projections
        self.mean_proj = nnx.Linear(in_features=in_dim, out_features=latent_dim, rngs=rngs)
        self.logvar_proj = nnx.Linear(in_features=in_dim, out_features=latent_dim, rngs=rngs)

        # Contamination fraction projection
        self.contamination_proj = nnx.Linear(in_features=in_dim, out_features=1, rngs=rngs)

    def __call__(
        self,
        counts: Float[Array, "n_cells n_genes"],
    ) -> tuple[
        Float[Array, "n_cells latent_dim"],
        Float[Array, "n_cells latent_dim"],
        Float[Array, "n_cells"],
    ]:
        """Encode counts to latent space.

        Args:
            counts: Input count matrix.

        Returns:
            Tuple of (mean, logvar, contamination_fraction).
        """
        # Log-normalize for encoder input
        x = jnp.log1p(counts)

        # Forward through encoder layers
        for linear, norm in zip(self.layers, self.norms):
            x = norm(nnx.gelu(linear(x)))

        # Latent parameters
        mean = self.mean_proj(x)
        logvar = self.logvar_proj(x)

        # Contamination fraction (bounded 0-1)
        contamination = jax.nn.sigmoid(self.contamination_proj(x)).squeeze(-1)

        return mean, logvar, contamination


class AmbientDecoder(nnx.Module):
    """Decoder network for ambient removal VAE."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        n_genes: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the decoder.

        Args:
            latent_dim: Latent space dimension.
            hidden_dims: Hidden layer dimensions (reversed from encoder).
            n_genes: Number of output genes.
            rngs: Random number generators.
        """
        super().__init__()

        # Build decoder layers (reversed hidden dims)
        layers = []
        norms = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            layers.append(nnx.Linear(in_features=in_dim, out_features=hidden_dim, rngs=rngs))
            norms.append(nnx.LayerNorm(num_features=hidden_dim, rngs=rngs))
            in_dim = hidden_dim

        self.layers = nnx.List(layers)
        self.norms = nnx.List(norms)

        # Output projection (log-rate for Poisson/NB)
        self.output_proj = nnx.Linear(in_features=in_dim, out_features=n_genes, rngs=rngs)

    def __call__(
        self,
        z: Float[Array, "n_cells latent_dim"],
    ) -> Float[Array, "n_cells n_genes"]:
        """Decode latent to gene expression rates.

        Args:
            z: Latent representation.

        Returns:
            Log-rate parameters for gene expression.
        """
        x = z
        for linear, norm in zip(self.layers, self.norms):
            x = norm(nnx.gelu(linear(x)))

        # Output log-rates
        log_rate = self.output_proj(x)

        return log_rate


class DifferentiableAmbientRemoval(EncoderDecoderOperator):
    """Differentiable ambient RNA removal using VAE.

    This operator removes ambient RNA contamination from single-cell
    count data using a variational autoencoder that models both
    cell-intrinsic expression and ambient contamination.

    Algorithm:
    1. Encode counts to latent space + contamination fraction
    2. Sample latent (reparameterization trick)
    3. Decode to cell-intrinsic expression rate
    4. Compute decontaminated counts by subtracting ambient contribution

    Inherits from EncoderDecoderOperator to get:

    - reparameterize() for sampling with reparameterization trick
    - kl_divergence() for KL from standard normal
    - elbo_loss() for combining reconstruction and KL losses

    Args:
        config: AmbientRemovalConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = AmbientRemovalConfig(n_genes=2000)
        remover = DifferentiableAmbientRemoval(config, rngs=nnx.Rngs(42))
        data = {"counts": counts, "ambient_profile": ambient}
        result, state, meta = remover.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: AmbientRemovalConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the ambient removal operator.

        Args:
            config: Ambient removal configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.ambient_prior = config.ambient_prior
        self.stochastic = nnx.static(config.stochastic)

        # Encoder
        self.encoder = AmbientEncoder(
            n_genes=config.n_genes,
            hidden_dims=config.hidden_dims,
            latent_dim=config.latent_dim,
            rngs=rngs,
        )

        # Decoder
        self.decoder = AmbientDecoder(
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            n_genes=config.n_genes,
            rngs=rngs,
        )

    # reparameterize() is inherited from EncoderDecoderOperator
    # kl_divergence() is inherited from EncoderDecoderOperator
    # elbo_loss() is inherited from EncoderDecoderOperator

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply ambient RNA removal.

        Args:
            data: Dictionary containing:
                - "counts": Raw count matrix (n_cells, n_genes)
                - "ambient_profile": Ambient expression profile (n_genes,)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Random key for stochastic sampling
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "counts": Original counts
                    - "ambient_profile": Original ambient profile
                    - "decontaminated_counts": Decontaminated counts
                    - "contamination_fraction": Estimated contamination per cell
                    - "latent": Latent representation
                    - "latent_mean": Mean of latent distribution
                    - "latent_logvar": Log variance of latent distribution
                    - "reconstructed": Reconstructed expression
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts = data["counts"]
        ambient_profile = data["ambient_profile"]

        # Encode
        mean, logvar, contamination = self.encoder(counts)

        # Sample latent using inherited reparameterize (uses self.rngs)
        z = self.reparameterize(mean, logvar)

        # Decode to cell-intrinsic expression rate
        log_rate = self.decoder(z)
        cell_rate = jax.nn.softplus(log_rate)  # Non-negative

        # Compute total counts per cell for scaling
        total_counts = jnp.sum(counts, axis=-1, keepdims=True)

        # Model: observed = (1 - contamination) * cell + contamination * ambient * total
        # Decontaminated = observed - contamination * ambient * total
        ambient_contribution = contamination[:, None] * ambient_profile[None, :] * total_counts
        decontaminated = jnp.maximum(counts - ambient_contribution, 0.0)

        # Reconstructed expression (for loss computation)
        cell_contribution = (1.0 - contamination[:, None]) * cell_rate
        reconstructed = cell_contribution + ambient_contribution

        transformed_data = {
            "counts": counts,
            "ambient_profile": ambient_profile,
            "decontaminated_counts": decontaminated,
            "contamination_fraction": contamination,
            "latent": z,
            "latent_mean": mean,
            "latent_logvar": logvar,
            "reconstructed": reconstructed,
        }

        return transformed_data, state, metadata
