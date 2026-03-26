"""Differentiable CRISPR Guide Scoring Operator.

This module implements a DeepCRISPR-inspired differentiable guide RNA scoring
operator using a CNN architecture to predict on-target efficiency.

The architecture is inspired by DeepCRISPR which uses:
1. A deep convolutional denoising neural network (DCDNN) autoencoder
   for unsupervised representation learning
2. A CNN classifier for efficiency prediction

This implementation provides a simplified but differentiable version that:
- Uses 1D convolutions over the one-hot encoded sequence
- Supports optional epigenetic feature channels
- Outputs efficiency scores in [0, 1]

For SpCas9, the standard input is 20nt guide + 3nt PAM = 23nt context.

References:
    Chuai et al. (2018). "DeepCRISPR: Optimized CRISPR guide RNA design
    by deep learning." Genome Biology.
    https://github.com/bm2-lab/DeepCRISPR

    Liu et al. (2021). "Enhancing CRISPR-Cas9 gRNA efficiency prediction
    by data integration and deep learning." Nature Communications.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CRISPRScorerConfig(OperatorConfig):
    """Configuration for DifferentiableCRISPRScorer.

    Attributes:
        guide_length: Length of guide RNA sequence (typically 20-23 nt).
        alphabet_size: Size of nucleotide alphabet (4 for A/C/G/T).
        hidden_channels: CNN hidden channel dimensions.
        fc_dims: Fully connected layer dimensions.
        dropout_rate: Dropout rate for regularization.
    """

    guide_length: int = 23
    alphabet_size: int = 4
    hidden_channels: tuple[int, ...] = (64, 128, 256)
    fc_dims: tuple[int, ...] = (256, 128)
    dropout_rate: float = 0.2


class DifferentiableCRISPRScorer(OperatorModule):
    """DeepCRISPR-style differentiable guide RNA scoring.

    This operator uses a 1D CNN architecture to predict CRISPR guide RNA
    on-target efficiency from sequence features. The model learns sequence
    patterns that correlate with efficient target cleavage.

    The architecture consists of:
    1. 1D convolutional layers for sequence feature extraction
    2. Batch normalization and ReLU activations
    3. Fully connected layers for efficiency prediction
    4. Sigmoid output for efficiency score in [0, 1]

    Attributes:
        config: Operator configuration.
        conv_layers: 1D convolutional layers.
        conv_bn: Batch normalization layers for conv.
        fc_layers: Fully connected layers.
        dropout: Dropout layer.
        output_head: Final output layer.

    Example:
        ```python
        from diffbio.operators.crispr import (
            DifferentiableCRISPRScorer,
            CRISPRScorerConfig,
        )
        config = CRISPRScorerConfig(guide_length=23)
        scorer = DifferentiableCRISPRScorer(config, rngs=nnx.Rngs(42))
        data = {"guides": guide_sequences}  # (n_guides, length, 4)
        result, _, _ = scorer.apply(data, {}, None)
        scores = result["efficiency_scores"]  # (n_guides,)
        ```
    """

    def __init__(
        self,
        config: CRISPRScorerConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the CRISPR scorer.

        Args:
            config: Operator configuration.
            rngs: Flax NNX random number generators.
        """
        super().__init__(config, rngs=rngs)

        channel_pairs = zip((config.alphabet_size, *config.hidden_channels), config.hidden_channels)
        self.conv_layers = nnx.List(
            [
                nnx.Conv(
                    in_features=in_channels,
                    out_features=out_channels,
                    kernel_size=(3,),
                    padding="SAME",
                    rngs=rngs,
                )
                for in_channels, out_channels in channel_pairs
            ]
        )
        self.conv_bn = nnx.List(
            [nnx.BatchNorm(out_channels, rngs=rngs) for out_channels in config.hidden_channels]
        )

        # Calculate flattened size after convolutions
        # With SAME padding, spatial size is preserved
        flat_size = config.guide_length * config.hidden_channels[-1]

        # Build fully connected layers
        fc_layers = []
        prev_dim = flat_size
        for fc_dim in config.fc_dims:
            fc_layers.append(nnx.Linear(prev_dim, fc_dim, rngs=rngs))
            prev_dim = fc_dim
        self.fc_layers = nnx.List(fc_layers)

        # Dropout for regularization
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

        # Output head for efficiency score
        self.output_head = nnx.Linear(prev_dim, 1, rngs=rngs)

    def extract_features(self, guides: jnp.ndarray) -> jnp.ndarray:
        """Extract features from guide sequences using CNN.

        Args:
            guides: One-hot encoded guides (n_guides, guide_length, 4).

        Returns:
            Feature vectors (n_guides, feature_dim).
        """
        # Input shape: (batch, length, channels)
        x = guides

        # Apply 1D convolutions with batch norm and ReLU
        for conv, bn in zip(self.conv_layers, self.conv_bn):
            x = conv(x)
            x = bn(x)
            x = nnx.relu(x)

        # Flatten: (batch, length, channels) -> (batch, length * channels)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        return x

    def predict_efficiency(self, features: jnp.ndarray) -> jnp.ndarray:
        """Predict efficiency score from features.

        Args:
            features: Feature vectors (n_guides, feature_dim).

        Returns:
            Efficiency scores (n_guides,) in range [0, 1].
        """
        x = features

        # Fully connected layers with ReLU and dropout
        for fc in self.fc_layers:
            x = fc(x)
            x = nnx.relu(x)
            x = self.dropout(x)

        # Output layer with sigmoid for [0, 1] output
        x = self.output_head(x)
        scores = nnx.sigmoid(x).squeeze(-1)

        return scores

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Apply CRISPR scoring to guide sequences.

        Args:
            data: Dictionary containing:
                - "guides": One-hot encoded guides (n_guides, guide_length, 4).
            state: Per-element state (passed through).
            metadata: Optional metadata (passed through).
            random_params: Random parameters for stochastic operations.
            stats: Optional statistics dictionary.

        Returns:
            Tuple of (transformed_data, state, metadata) where transformed_data
            contains:

                - "guides": Original guide sequences.
                - "efficiency_scores": Predicted efficiency (n_guides,).
                - "features": Extracted feature vectors.
        """
        guides = data["guides"]

        # Extract features using CNN
        features = self.extract_features(guides)

        # Predict efficiency scores
        efficiency_scores = self.predict_efficiency(features)

        # Build output
        output = {
            **data,
            "efficiency_scores": efficiency_scores,
            "features": features,
        }

        return output, state, metadata


def create_crispr_scorer(
    guide_length: int = 23,
    hidden_channels: tuple[int, ...] = (64, 128, 256),
    fc_dims: tuple[int, ...] = (256, 128),
    dropout_rate: float = 0.2,
    seed: int = 42,
) -> DifferentiableCRISPRScorer:
    """Factory function to create a CRISPR scorer.

    Args:
        guide_length: Length of guide RNA sequence.
        hidden_channels: CNN hidden channel dimensions.
        fc_dims: Fully connected layer dimensions.
        dropout_rate: Dropout rate for regularization.
        seed: Random seed for initialization.

    Returns:
        Configured DifferentiableCRISPRScorer instance.

    Example:
        ```python
        scorer = create_crispr_scorer(guide_length=23)
        result, _, _ = scorer.apply({"guides": data}, {}, None)
        ```
    """
    config = CRISPRScorerConfig(
        guide_length=guide_length,
        hidden_channels=hidden_channels,
        fc_dims=fc_dims,
        dropout_rate=dropout_rate,
    )

    return DifferentiableCRISPRScorer(config, rngs=nnx.Rngs(seed))
