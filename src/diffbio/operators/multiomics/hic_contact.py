"""Hi-C chromatin contact analysis operator.

This module provides differentiable analysis of Hi-C contact matrices
for chromatin structure inference.

Key technique: Uses neural network to learn bin embeddings from contact
patterns, then predicts compartments and TAD boundaries using attention
over neighboring bins.

Applications: Chromatin compartment identification, TAD boundary detection,
3D genome structure prediction.

Inherits from TemperatureOperator to get:

- _temperature property for temperature-controlled smoothing
- soft_max() for logsumexp-based smooth maximum
- soft_argmax() for soft position selection
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.base_operators import TemperatureOperator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HiCContactAnalysisConfig(OperatorConfig):
    """Configuration for HiCContactAnalysis.

    Attributes:
        n_bins: Number of genomic bins.
        hidden_dim: Hidden dimension for neural networks.
        num_layers: Number of encoder layers.
        num_heads: Number of attention heads.
        bin_features: Dimension of input bin features.
        dropout_rate: Dropout rate for regularization.
        temperature: Temperature for softmax operations.
    """

    n_bins: int = 1000
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    bin_features: int = 16
    dropout_rate: float = 0.1
    temperature: float = 1.0


class ContactEncoder(nnx.Module):
    """Encoder for Hi-C contact patterns."""

    def __init__(
        self,
        n_bins: int,
        hidden_dim: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the contact encoder.

        Args:
            n_bins: Number of bins.
            hidden_dim: Hidden dimension.
            num_layers: Number of layers.
            rngs: Random number generators.
        """
        super().__init__()

        # Project contact row to hidden dim
        self.input_proj = nnx.Linear(
            in_features=n_bins,
            out_features=hidden_dim,
            rngs=rngs,
        )

        # MLP layers
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs))

        self.layers = nnx.List(layers)
        self.layer_norms = nnx.List(
            [nnx.LayerNorm(num_features=hidden_dim, rngs=rngs) for _ in range(num_layers - 1)]
        )

    def __call__(
        self,
        contact_matrix: Float[Array, "n_bins n_bins"],
    ) -> Float[Array, "n_bins hidden_dim"]:
        """Encode contact patterns.

        Args:
            contact_matrix: Hi-C contact matrix.

        Returns:
            Bin embeddings from contact patterns.
        """
        # Each bin gets embedding from its contact row
        x = self.input_proj(contact_matrix)  # (n_bins, hidden_dim)

        for linear, norm in zip(self.layers, self.layer_norms):
            x = norm(nnx.gelu(linear(x)) + x)  # Residual

        return x


class BinFeatureEncoder(nnx.Module):
    """Encoder for genomic bin features."""

    def __init__(
        self,
        bin_features: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the bin feature encoder.

        Args:
            bin_features: Input feature dimension.
            hidden_dim: Hidden dimension.
            rngs: Random number generators.
        """
        super().__init__()

        self.linear1 = nnx.Linear(in_features=bin_features, out_features=hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)

    def __call__(
        self,
        bin_features: Float[Array, "n_bins bin_features"],
    ) -> Float[Array, "n_bins hidden_dim"]:
        """Encode bin features.

        Args:
            bin_features: Genomic bin features.

        Returns:
            Bin feature embeddings.
        """
        x = nnx.gelu(self.linear1(bin_features))
        x = self.norm(self.linear2(x))
        return x


class LocalAttention(nnx.Module):
    """Local attention for TAD boundary detection."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize local attention.

        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
            rngs: Random number generators.
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.query = nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs)
        self.key = nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs)
        self.value = nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs)
        self.output = nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "n_bins hidden_dim"],
    ) -> Float[Array, "n_bins hidden_dim"]:
        """Apply local attention.

        Args:
            x: Input embeddings.

        Returns:
            Attended embeddings.
        """
        n_bins = x.shape[0]

        Q = self.query(x).reshape(n_bins, self.num_heads, self.head_dim)
        K = self.key(x).reshape(n_bins, self.num_heads, self.head_dim)
        V = self.value(x).reshape(n_bins, self.num_heads, self.head_dim)

        # Full attention (could be windowed for efficiency)
        attn = jnp.einsum("qhd,khd->hqk", Q, K) * self.scale
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum("hqk,khd->qhd", attn, V)
        out = out.reshape(n_bins, -1)
        out = self.output(out)

        return out


class HiCContactAnalysis(TemperatureOperator):
    """Differentiable Hi-C contact analysis.

    This operator analyzes Hi-C contact matrices to identify
    chromatin compartments and TAD boundaries using neural networks.

    Algorithm:
    1. Encode contact patterns per bin
    2. Encode genomic bin features
    3. Combine contact and feature embeddings
    4. Apply attention for context
    5. Predict compartment scores
    6. Detect TAD boundaries
    7. Reconstruct contacts from embeddings

    Args:
        config: HiCContactAnalysisConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        config = HiCContactAnalysisConfig(n_bins=1000)
        analyzer = HiCContactAnalysis(config, rngs=nnx.Rngs(42))
        data = {"contact_matrix": contacts, "bin_features": features}
        result, state, meta = analyzer.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: HiCContactAnalysisConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the Hi-C contact analyzer.

        Args:
            config: Analysis configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.hidden_dim = config.hidden_dim
        # Temperature is managed by TemperatureOperator via self._temperature

        # Contact pattern encoder
        self.contact_encoder = ContactEncoder(
            n_bins=config.n_bins,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            rngs=rngs,
        )

        # Bin feature encoder
        self.feature_encoder = BinFeatureEncoder(
            bin_features=config.bin_features,
            hidden_dim=config.hidden_dim,
            rngs=rngs,
        )

        # Combine contact and feature embeddings
        self.combine = nnx.Linear(
            in_features=config.hidden_dim * 2,
            out_features=config.hidden_dim,
            rngs=rngs,
        )

        # Attention for context
        self.attention = LocalAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            rngs=rngs,
        )

        # Output heads
        self.compartment_head = nnx.Linear(
            in_features=config.hidden_dim,
            out_features=1,
            rngs=rngs,
        )

        self.boundary_head = nnx.Linear(
            in_features=config.hidden_dim,
            out_features=1,
            rngs=rngs,
        )

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply Hi-C contact analysis.

        Args:
            data: Dictionary containing:
                - "contact_matrix": Hi-C contact matrix (n_bins, n_bins)
                - "bin_features": Bin genomic features (n_bins, bin_features)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - "contact_matrix": Original contacts
                    - "bin_features": Original features
                    - "bin_embeddings": Learned bin embeddings
                    - "compartment_scores": A/B compartment scores
                    - "tad_boundary_scores": TAD boundary probabilities
                    - "predicted_contacts": Reconstructed contacts
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        contact_matrix = data["contact_matrix"]
        bin_features = data["bin_features"]

        # Encode contact patterns
        contact_emb = self.contact_encoder(contact_matrix)  # (n_bins, hidden_dim)

        # Encode bin features
        feature_emb = self.feature_encoder(bin_features)  # (n_bins, hidden_dim)

        # Combine embeddings
        combined = jnp.concatenate([contact_emb, feature_emb], axis=-1)
        combined = nnx.gelu(self.combine(combined))  # (n_bins, hidden_dim)

        # Apply attention for context
        attended = self.attention(combined) + combined  # Residual

        # Predict compartment scores (continuous A/B)
        compartment_scores = self.compartment_head(attended).squeeze(-1)  # (n_bins,)

        # Predict TAD boundary scores
        boundary_logits = self.boundary_head(attended).squeeze(-1)  # (n_bins,)
        tad_boundary_scores = jax.nn.sigmoid(boundary_logits)

        # Reconstruct contacts from embeddings (dot product)
        # (n_bins, hidden_dim) @ (hidden_dim, n_bins) -> (n_bins, n_bins)
        predicted_contacts = jnp.einsum("ih,jh->ij", attended, attended)
        predicted_contacts = jax.nn.softplus(predicted_contacts)  # Non-negative

        # Build output
        transformed_data = {
            "contact_matrix": contact_matrix,
            "bin_features": bin_features,
            "bin_embeddings": attended,
            "compartment_scores": compartment_scores,
            "tad_boundary_scores": tad_boundary_scores,
            "predicted_contacts": predicted_contacts,
        }

        return transformed_data, state, metadata
