"""Differentiable Ancestry Estimation Operator.

This module implements a Neural ADMIXTURE-style differentiable ancestry estimator
using an autoencoder architecture. The model learns to decompose individual
genotypes into ancestry proportions from K ancestral populations.

Reference:
    Dias et al. (2022). "Neural ADMIXTURE: A Neural Network Approach for
    Fast and Accurate Estimation of Population Structure."
    https://github.com/AI-sandbox/neural-admixture
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx


@dataclass
class AncestryEstimatorConfig(OperatorConfig):
    """Configuration for DifferentiableAncestryEstimator.

    Attributes:
        n_snps: Number of SNP markers in genotype input.
        n_populations: Number of ancestral populations (K).
        hidden_dims: Hidden layer dimensions for encoder.
        temperature: Temperature for softmax ancestry proportions.
            Lower values produce sharper (more confident) estimates.
        dropout_rate: Dropout rate for regularization.
        stochastic: Whether the operator uses stochastic computations.
        stream_name: Optional stream name for data routing.
    """

    n_snps: int = 10000
    n_populations: int = 5
    hidden_dims: tuple[int, ...] = (128, 64)
    temperature: float = 1.0
    dropout_rate: float = 0.1
    stochastic: bool = False
    stream_name: str | None = None


class DifferentiableAncestryEstimator(OperatorModule):
    """Neural ADMIXTURE-style differentiable ancestry estimator.

    This operator uses an autoencoder architecture to estimate ancestry
    proportions from genotype data. The encoder maps genotypes to a latent
    representation, which is then transformed to ancestry proportions via
    temperature-controlled softmax. The decoder reconstructs genotypes
    from ancestry proportions, enabling unsupervised learning.

    The model follows the ADMIXTURE generative model:
        G_ij = sum_k Q_ik * P_kj
    Where:
        - G is the genotype matrix (individuals x SNPs)
        - Q is the ancestry proportion matrix (individuals x K populations)
        - P is the population allele frequency matrix (K x SNPs)

    Attributes:
        config: Operator configuration.
        encoder_layers: Encoder network layers.
        encoder_dropout: Dropout layer for encoder.
        ancestry_head: Linear layer for ancestry proportions.
        population_frequencies: Learnable population allele frequencies (P matrix).
        temperature: Temperature parameter for softmax.

    Example:
        >>> from diffbio.operators.population import (
        ...     DifferentiableAncestryEstimator,
        ...     AncestryEstimatorConfig,
        ... )
        >>> config = AncestryEstimatorConfig(n_snps=1000, n_populations=5)
        >>> estimator = DifferentiableAncestryEstimator(config, rngs=nnx.Rngs(42))
        >>> data = {"genotypes": genotype_matrix}  # (n_samples, n_snps)
        >>> result, _, _ = estimator.apply(data, {}, None)
        >>> ancestry = result["ancestry_proportions"]  # (n_samples, K)
    """

    def __init__(
        self,
        config: AncestryEstimatorConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the ancestry estimator.

        Args:
            config: Operator configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)

        self.config = config

        # Build encoder layers
        encoder_layers = []
        prev_dim = config.n_snps
        for hidden_dim in config.hidden_dims:
            encoder_layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim
        self.encoder_layers = nnx.List(encoder_layers)

        # Dropout for regularization
        self.encoder_dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        # Ancestry proportion head
        self.ancestry_head = nnx.Linear(prev_dim, config.n_populations, rngs=rngs)

        # Learnable population allele frequencies (P matrix)
        # Shape: (n_populations, n_snps)
        # Initialized with small random values
        self.population_frequencies = nnx.Param(
            jnp.abs(
                jnp.array(
                    nnx.initializers.normal(0.1)(
                        rngs.params(),
                        (config.n_populations, config.n_snps),
                    )
                )
            )
            + 0.01
        )

        # Temperature parameter
        self.temperature = nnx.Param(jnp.array(config.temperature))

    def encode(self, genotypes: jnp.ndarray) -> jnp.ndarray:
        """Encode genotypes to latent representation.

        Args:
            genotypes: Genotype matrix of shape (n_samples, n_snps).
                Values should be 0, 1, or 2 representing allele counts.

        Returns:
            Latent representation of shape (n_samples, hidden_dims[-1]).
        """
        x = genotypes

        # Pass through encoder layers with ReLU and dropout
        for layer in self.encoder_layers:
            x = layer(x)
            x = nnx.relu(x)
            x = self.encoder_dropout(x)

        return x

    def compute_ancestry(self, latent: jnp.ndarray) -> jnp.ndarray:
        """Compute ancestry proportions from latent representation.

        Args:
            latent: Latent representation of shape (n_samples, hidden_dims[-1]).

        Returns:
            Ancestry proportions of shape (n_samples, n_populations).
            Each row sums to 1 and all values are non-negative.
        """
        # Get raw ancestry logits
        logits = self.ancestry_head(latent)

        # Apply temperature-controlled softmax
        temperature = jnp.maximum(self.temperature[...], 1e-6)
        proportions = nnx.softmax(logits / temperature, axis=-1)

        return proportions

    def decode(self, ancestry: jnp.ndarray) -> jnp.ndarray:
        """Decode ancestry proportions to reconstructed genotypes.

        Following the ADMIXTURE model: G = Q @ P
        Where Q is ancestry proportions and P is population frequencies.

        Args:
            ancestry: Ancestry proportions of shape (n_samples, n_populations).

        Returns:
            Reconstructed genotypes of shape (n_samples, n_snps).
            Values represent expected allele counts (continuous 0-2).
        """
        # Get population frequencies normalized to [0, 1]
        pop_freqs = nnx.sigmoid(self.population_frequencies[...])

        # Reconstruct: G = Q @ P, scaled to 0-2 range
        # ancestry: (n_samples, n_populations)
        # pop_freqs: (n_populations, n_snps)
        reconstructed = 2.0 * jnp.matmul(ancestry, pop_freqs)

        return reconstructed

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Apply ancestry estimation to genotype data.

        Args:
            data: Dictionary containing:
                - "genotypes": Genotype matrix (n_samples, n_snps) with values 0/1/2.
            state: Per-element state (passed through).
            metadata: Optional metadata (passed through).
            random_params: Random parameters for stochastic operations.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of (transformed_data, state, metadata) where transformed_data
            contains:
                - "genotypes": Original genotype matrix.
                - "ancestry_proportions": Estimated ancestry (n_samples, K).
                - "reconstructed": Reconstructed genotypes (n_samples, n_snps).
                - "latent": Latent representation.
        """
        genotypes = data["genotypes"]

        # Encode genotypes to latent space
        latent = self.encode(genotypes)

        # Compute ancestry proportions
        ancestry = self.compute_ancestry(latent)

        # Decode to reconstructed genotypes
        reconstructed = self.decode(ancestry)

        # Build output
        output = {
            **data,
            "ancestry_proportions": ancestry,
            "reconstructed": reconstructed,
            "latent": latent,
        }

        return output, state, metadata


def create_ancestry_estimator(
    n_snps: int,
    n_populations: int,
    hidden_dims: tuple[int, ...] = (128, 64),
    temperature: float = 1.0,
    dropout_rate: float = 0.1,
    seed: int = 42,
) -> DifferentiableAncestryEstimator:
    """Factory function to create an ancestry estimator.

    Args:
        n_snps: Number of SNP markers.
        n_populations: Number of ancestral populations (K).
        hidden_dims: Hidden layer dimensions for encoder.
        temperature: Softmax temperature for ancestry proportions.
        dropout_rate: Dropout rate for regularization.
        seed: Random seed for initialization.

    Returns:
        Configured DifferentiableAncestryEstimator instance.

    Example:
        >>> estimator = create_ancestry_estimator(
        ...     n_snps=10000,
        ...     n_populations=5,
        ... )
        >>> result, _, _ = estimator.apply({"genotypes": data}, {}, None)
    """
    config = AncestryEstimatorConfig(
        n_snps=n_snps,
        n_populations=n_populations,
        hidden_dims=hidden_dims,
        temperature=temperature,
        dropout_rate=dropout_rate,
    )

    return DifferentiableAncestryEstimator(config, rngs=nnx.Rngs(seed))
