"""Spatial transcriptomics deconvolution operator.

This module provides differentiable cell type deconvolution for
spatial transcriptomics data.

Key technique: Uses neural network to learn spot embeddings that account
for spatial context, then performs soft assignment to reference cell type
profiles using attention mechanisms.

Applications: Cell type mapping in spatial transcriptomics, tissue
composition analysis, spatial cell-cell interaction studies.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree


@dataclass
class SpatialDeconvolutionConfig(OperatorConfig):
    """Configuration for SpatialDeconvolution.

    Attributes:
        n_genes: Number of genes in expression profiles.
        n_cell_types: Number of reference cell types.
        hidden_dim: Hidden dimension for neural networks.
        num_layers: Number of encoder layers.
        spatial_hidden: Hidden dimension for spatial encoder.
        dropout_rate: Dropout rate for regularization.
        temperature: Temperature for softmax operations.
    """

    n_genes: int = 2000
    n_cell_types: int = 10
    hidden_dim: int = 128
    num_layers: int = 2
    spatial_hidden: int = 32
    dropout_rate: float = 0.1
    temperature: float = 1.0


class SpotEncoder(nnx.Module):
    """Encoder for spatial spot expression profiles."""

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the spot encoder.

        Args:
            n_genes: Number of genes.
            hidden_dim: Hidden dimension.
            num_layers: Number of layers.
            rngs: Random number generators.
        """
        super().__init__()

        layers = []
        in_dim = n_genes
        for _ in range(num_layers):
            layers.append(nnx.Linear(in_features=in_dim, out_features=hidden_dim, rngs=rngs))
            in_dim = hidden_dim

        self.layers = nnx.List(layers)
        self.layer_norms = nnx.List(
            [nnx.LayerNorm(num_features=hidden_dim, rngs=rngs) for _ in range(num_layers)]
        )

    def __call__(
        self,
        expression: Float[Array, "n_spots n_genes"],
    ) -> Float[Array, "n_spots hidden_dim"]:
        """Encode spot expression.

        Args:
            expression: Spot expression matrix.

        Returns:
            Spot embeddings.
        """
        x = expression
        for linear, norm in zip(self.layers, self.layer_norms):
            x = norm(nnx.gelu(linear(x)))
        return x


class SpatialEncoder(nnx.Module):
    """Encoder for spatial coordinates."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the spatial encoder.

        Args:
            hidden_dim: Hidden dimension.
            rngs: Random number generators.
        """
        super().__init__()

        self.linear1 = nnx.Linear(in_features=2, out_features=hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=hidden_dim, out_features=hidden_dim, rngs=rngs)

    def __call__(
        self,
        coordinates: Float[Array, "n_spots 2"],
    ) -> Float[Array, "n_spots hidden_dim"]:
        """Encode spatial coordinates.

        Args:
            coordinates: Spot coordinates (x, y).

        Returns:
            Spatial embeddings.
        """
        x = nnx.gelu(self.linear1(coordinates))
        x = self.linear2(x)
        return x


class SpatialDeconvolution(OperatorModule):
    """Differentiable spatial transcriptomics deconvolution.

    This operator performs cell type deconvolution of spatial
    transcriptomics spots using reference single-cell profiles.
    It incorporates spatial context through coordinate embeddings.

    Algorithm:
    1. Encode spot expression profiles
    2. Encode spatial coordinates
    3. Combine expression and spatial features
    4. Compute attention to reference cell type profiles
    5. Apply softmax for cell type proportions
    6. Reconstruct expression from proportions

    Args:
        config: SpatialDeconvolutionConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = SpatialDeconvolutionConfig(n_cell_types=10)
        >>> deconv = SpatialDeconvolution(config, rngs=nnx.Rngs(42))
        >>> data = {"spot_expression": spots, "reference_profiles": refs, "coordinates": coords}
        >>> result, state, meta = deconv.apply(data, {}, None)
    """

    def __init__(
        self,
        config: SpatialDeconvolutionConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the spatial deconvolution operator.

        Args:
            config: Deconvolution configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.hidden_dim = config.hidden_dim
        self.temperature = config.temperature

        # Expression encoder
        self.spot_encoder = SpotEncoder(
            n_genes=config.n_genes,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            rngs=rngs,
        )

        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            hidden_dim=config.spatial_hidden,
            rngs=rngs,
        )

        # Combine expression and spatial
        self.combine_linear = nnx.Linear(
            in_features=config.hidden_dim + config.spatial_hidden,
            out_features=config.hidden_dim,
            rngs=rngs,
        )

        # Reference profile encoder
        self.ref_encoder = nnx.Linear(
            in_features=config.n_genes,
            out_features=config.hidden_dim,
            rngs=rngs,
        )

        # Output projection for cell type scores
        self.output_linear = nnx.Linear(
            in_features=config.hidden_dim,
            out_features=config.n_cell_types,
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
        """Apply spatial deconvolution.

        Args:
            data: Dictionary containing:
                - "spot_expression": Spot expression (n_spots, n_genes)
                - "reference_profiles": Reference profiles (n_cell_types, n_genes)
                - "coordinates": Spot coordinates (n_spots, 2)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:
                    - "spot_expression": Original expression
                    - "reference_profiles": Original references
                    - "coordinates": Original coordinates
                    - "cell_proportions": Deconvolved proportions
                    - "reconstructed_expression": Reconstructed expression
                    - "spatial_embeddings": Spatial feature embeddings
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        spot_expression = data["spot_expression"]
        reference_profiles = data["reference_profiles"]
        coordinates = data["coordinates"]

        # Encode spot expression
        spot_emb = self.spot_encoder(spot_expression)  # (n_spots, hidden_dim)

        # Encode spatial coordinates
        spatial_emb = self.spatial_encoder(coordinates)  # (n_spots, spatial_hidden)

        # Combine expression and spatial features
        combined = jnp.concatenate([spot_emb, spatial_emb], axis=-1)
        combined = nnx.gelu(self.combine_linear(combined))  # (n_spots, hidden_dim)

        # Encode reference profiles
        ref_emb = self.ref_encoder(reference_profiles)  # (n_cell_types, hidden_dim)

        # Compute attention scores (dot product similarity)
        # (n_spots, hidden_dim) @ (hidden_dim, n_cell_types) -> (n_spots, n_cell_types)
        scores = jnp.einsum("sh,ch->sc", combined, ref_emb)

        # Cell type proportions via softmax
        cell_proportions = jax.nn.softmax(scores / self.temperature, axis=-1)

        # Reconstruct expression: proportions @ reference_profiles
        # (n_spots, n_cell_types) @ (n_cell_types, n_genes) -> (n_spots, n_genes)
        reconstructed = jnp.einsum("sc,cg->sg", cell_proportions, reference_profiles)

        # Build output
        transformed_data = {
            "spot_expression": spot_expression,
            "reference_profiles": reference_profiles,
            "coordinates": coordinates,
            "cell_proportions": cell_proportions,
            "reconstructed_expression": reconstructed,
            "spatial_embeddings": combined,
        }

        return transformed_data, state, metadata
