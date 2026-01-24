"""Differentiable spectral similarity operator for metabolomics (MS2DeepScore-style).

This module implements a Siamese neural network for predicting molecular
structural similarity from tandem mass spectra (MS/MS), based on the
MS2DeepScore architecture.

The approach uses a shared base network to generate spectral embeddings,
then computes cosine similarity between embeddings to predict structural
similarity (Tanimoto scores).

Architecture based on:
    Huber et al. (2021). "MS2DeepScore: a novel deep learning similarity
    measure to compare tandem mass spectra." Journal of Cheminformatics.

Key features:
- Siamese architecture with shared weights for spectrum encoding
- 200-dimensional spectral embeddings
- Cosine similarity for structure prediction
- Monte-Carlo dropout for uncertainty estimation
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx


@dataclass
class SpectralSimilarityConfig(OperatorConfig):
    """Configuration for DifferentiableSpectralSimilarity.

    Attributes:
        n_bins: Number of m/z bins for spectrum discretization.
            Default 1000 (10-1000 m/z at 1 m/z resolution).
            Original MS2DeepScore uses 10000 bins at 0.1 m/z resolution.
        embedding_dim: Dimension of spectral embeddings. Default 200.
        hidden_dims: Tuple of hidden layer dimensions. Default (512, 256).
            Original MS2DeepScore uses (500, 500).
        dropout_rate: Dropout rate for regularization. Default 0.2.
        min_mz: Minimum m/z value for binning. Default 0.0.
        max_mz: Maximum m/z value for binning. Default 1000.0.
        use_batch_norm: Whether to use batch normalization. Default True.
        stochastic: Whether to use stochastic operations. Default False.
        stream_name: Optional stream name for data routing.
    """

    n_bins: int = 1000
    embedding_dim: int = 200
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout_rate: float = 0.2
    min_mz: float = 0.0
    max_mz: float = 1000.0
    use_batch_norm: bool = True
    stochastic: bool = False
    stream_name: str | None = None


class DifferentiableSpectralSimilarity(OperatorModule):
    """Siamese neural network for spectral similarity prediction.

    This operator implements the MS2DeepScore architecture for predicting
    molecular structural similarity from tandem mass spectra. The network
    uses a shared encoder to generate spectral embeddings, then computes
    cosine similarity between pairs of embeddings.

    The operator supports two modes of operation:
    1. Single spectrum input: Generates embeddings for spectra
    2. Paired spectra input: Computes similarity between spectrum pairs

    Architecture:
        Input (n_bins) -> Dense -> [BatchNorm] -> ReLU -> Dropout
                      -> Dense -> [BatchNorm] -> ReLU -> Dropout
                      -> Dense -> Embedding (embedding_dim)

    Attributes:
        config: SpectralSimilarityConfig with hyperparameters.
        encoder_layers: Dense layers for encoding spectra.
        encoder_bn: Batch normalization layers.
        encoder_dropout: Dropout layers for regularization.
        embedding_layer: Final layer producing embeddings.

    Example:
        >>> config = SpectralSimilarityConfig(n_bins=1000, embedding_dim=200)
        >>> operator = DifferentiableSpectralSimilarity(config, rngs=nnx.Rngs(42))
        >>>
        >>> # Get embeddings for spectra
        >>> spectra = jax.random.uniform(jax.random.PRNGKey(0), (10, 1000))
        >>> result, _, _ = operator.apply({"spectra": spectra}, {}, None)
        >>> embeddings = result["embeddings"]  # (10, 200)
        >>>
        >>> # Compute pairwise similarity
        >>> spectra_a = jax.random.uniform(jax.random.PRNGKey(0), (5, 1000))
        >>> spectra_b = jax.random.uniform(jax.random.PRNGKey(1), (5, 1000))
        >>> result, _, _ = operator.apply(
        ...     {"spectra_a": spectra_a, "spectra_b": spectra_b}, {}, None
        ... )
        >>> similarity = result["similarity_scores"]  # (5,) in [-1, 1]
    """

    def __init__(self, config: SpectralSimilarityConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize the spectral similarity operator.

        Args:
            config: Configuration with network hyperparameters.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)
        self.config = config

        # Build encoder layers
        encoder_layers = []
        encoder_bn = []
        encoder_dropout = []

        in_features = config.n_bins
        for hidden_dim in config.hidden_dims:
            # Add L1/L2 regularization to first layer (as in MS2DeepScore)
            encoder_layers.append(nnx.Linear(in_features, hidden_dim, rngs=rngs))

            if config.use_batch_norm:
                encoder_bn.append(nnx.BatchNorm(hidden_dim, rngs=rngs))
            else:
                encoder_bn.append(None)

            if config.dropout_rate > 0:
                encoder_dropout.append(nnx.Dropout(rate=config.dropout_rate, rngs=rngs))
            else:
                encoder_dropout.append(None)

            in_features = hidden_dim

        self.encoder_layers = nnx.List(encoder_layers)
        self.encoder_bn = nnx.List(encoder_bn)
        self.encoder_dropout = nnx.List(encoder_dropout)

        # Final embedding layer (no batch norm or dropout after this)
        self.embedding_layer = nnx.Linear(in_features, config.embedding_dim, rngs=rngs)

    def encode(self, spectra: jnp.ndarray) -> jnp.ndarray:
        """Encode binned spectra into embeddings.

        BatchNorm and Dropout respect the model's train/eval mode:
        - Call model.train() before training to enable dropout and update batch stats
        - Call model.eval() before inference to disable dropout and use running stats

        Args:
            spectra: Binned spectra with shape (n_spectra, n_bins).

        Returns:
            Embeddings with shape (n_spectra, embedding_dim).
        """
        x = spectra

        # Apply encoder layers
        for idx in range(len(self.encoder_layers)):
            layer = self.encoder_layers[idx]
            bn = self.encoder_bn[idx]
            dropout = self.encoder_dropout[idx]

            x = layer(x)

            if bn is not None:
                x = bn(x)

            x = nnx.relu(x)

            if dropout is not None:
                x = dropout(x)

        # Final embedding layer (no activation for embedding output)
        embeddings = self.embedding_layer(x)

        return embeddings

    def cosine_similarity(
        self, embeddings_a: jnp.ndarray, embeddings_b: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute cosine similarity between embedding pairs.

        Args:
            embeddings_a: First set of embeddings (n, embedding_dim).
            embeddings_b: Second set of embeddings (n, embedding_dim).

        Returns:
            Cosine similarity scores with shape (n,).
        """
        # Normalize embeddings
        norm_a = jnp.linalg.norm(embeddings_a, axis=-1, keepdims=True)
        norm_b = jnp.linalg.norm(embeddings_b, axis=-1, keepdims=True)

        # Avoid division by zero
        norm_a = jnp.maximum(norm_a, 1e-8)
        norm_b = jnp.maximum(norm_b, 1e-8)

        embeddings_a_normalized = embeddings_a / norm_a
        embeddings_b_normalized = embeddings_b / norm_b

        # Compute cosine similarity
        similarity = jnp.sum(embeddings_a_normalized * embeddings_b_normalized, axis=-1)

        return similarity

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Apply the spectral similarity operator.

        The operator supports two input modes:

        1. Single spectra mode (embedding generation):
           Input: {"spectra": (n, n_bins)}
           Output: {"embeddings": (n, embedding_dim)}

        2. Paired spectra mode (similarity computation):
           Input: {"spectra_a": (n, n_bins), "spectra_b": (n, n_bins)}
           Output: {"similarity_scores": (n,), "embeddings_a": ..., "embeddings_b": ...}

        Args:
            data: Input data dictionary with spectra.
            state: Per-element state (passed through).
            metadata: Optional metadata (passed through).
            random_params: Random parameters (unused).
            stats: Optional statistics (unused).

        Returns:
            Tuple of (output_data, state, metadata).
        """
        if "spectra" in data:
            # Single spectra mode: compute embeddings
            spectra = data["spectra"]
            embeddings = self.encode(spectra)

            output = {**data, "embeddings": embeddings}

        elif "spectra_a" in data and "spectra_b" in data:
            # Paired spectra mode: compute similarity
            spectra_a = data["spectra_a"]
            spectra_b = data["spectra_b"]

            embeddings_a = self.encode(spectra_a)
            embeddings_b = self.encode(spectra_b)

            similarity_scores = self.cosine_similarity(embeddings_a, embeddings_b)

            output = {
                **data,
                "embeddings_a": embeddings_a,
                "embeddings_b": embeddings_b,
                "similarity_scores": similarity_scores,
            }

        else:
            raise ValueError(
                "Input must contain either 'spectra' or both 'spectra_a' and 'spectra_b'"
            )

        return output, state, metadata


def bin_spectrum(
    mz_values: jnp.ndarray,
    intensities: jnp.ndarray,
    n_bins: int = 1000,
    min_mz: float = 0.0,
    max_mz: float = 1000.0,
    normalize: bool = True,
) -> jnp.ndarray:
    """Bin a mass spectrum into fixed-width m/z bins.

    This function discretizes a continuous mass spectrum (m/z, intensity pairs)
    into a fixed-size vector suitable for neural network input.

    Args:
        mz_values: Array of m/z values with shape (n_peaks,).
        intensities: Array of intensity values with shape (n_peaks,).
        n_bins: Number of bins to use. Default 1000.
        min_mz: Minimum m/z value for binning. Default 0.0.
        max_mz: Maximum m/z value for binning. Default 1000.0.
        normalize: Whether to normalize intensities to max=1.0. Default True.

    Returns:
        Binned spectrum with shape (n_bins,).

    Example:
        >>> mz = jnp.array([100.0, 200.0, 300.0])
        >>> intensity = jnp.array([0.5, 1.0, 0.3])
        >>> binned = bin_spectrum(mz, intensity, n_bins=100)
        >>> binned.shape
        (100,)
    """
    # Compute bin edges
    bin_width = (max_mz - min_mz) / n_bins

    # Compute bin indices for each m/z value
    bin_indices = ((mz_values - min_mz) / bin_width).astype(jnp.int32)

    # Clip to valid range
    bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)

    # Create empty binned spectrum
    binned = jnp.zeros(n_bins)

    # Accumulate intensities in bins (use segment_sum for differentiability)
    binned = binned.at[bin_indices].add(intensities)

    # Normalize if requested
    if normalize:
        max_intensity = jnp.maximum(jnp.max(binned), 1e-8)
        binned = binned / max_intensity

    return binned


def create_spectral_similarity(
    n_bins: int = 1000,
    embedding_dim: int = 200,
    hidden_dims: tuple[int, ...] = (512, 256),
    dropout_rate: float = 0.2,
    seed: int = 42,
) -> DifferentiableSpectralSimilarity:
    """Factory function to create a spectral similarity operator.

    Args:
        n_bins: Number of m/z bins. Default 1000.
        embedding_dim: Embedding dimension. Default 200.
        hidden_dims: Hidden layer dimensions. Default (512, 256).
        dropout_rate: Dropout rate. Default 0.2.
        seed: Random seed. Default 42.

    Returns:
        Configured DifferentiableSpectralSimilarity operator.

    Example:
        >>> operator = create_spectral_similarity(n_bins=500, embedding_dim=128)
        >>> spectra = jax.random.uniform(jax.random.PRNGKey(0), (10, 500))
        >>> result, _, _ = operator.apply({"spectra": spectra}, {}, None)
    """
    config = SpectralSimilarityConfig(
        n_bins=n_bins,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
    )
    return DifferentiableSpectralSimilarity(config, rngs=nnx.Rngs(seed))
