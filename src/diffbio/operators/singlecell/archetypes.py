"""Differentiable archetypal analysis for single-cell data.

Implements PCHA (Principal Convex Hull Analysis, Morup & Hansen 2012) as a
differentiable autoencoder with softmax bottleneck.  Each cell is represented
as a temperature-controlled convex combination of learnable archetype
prototypes.

Algorithm:
    1. Encode cells to archetype weight space via MLP.
    2. Apply temperature-scaled softmax to enforce simplex constraints.
    3. Reconstruct cells as the convex combination ``weights @ archetypes``.

Inherits from ``TemperatureOperator`` to get temperature-controlled smoothing.
"""

from dataclasses import dataclass
from typing import Any

import jax
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.base_operators import TemperatureOperator
from diffbio.utils.nn_utils import build_mlp_encoder, ensure_rngs, forward_mlp, get_rng_key

__all__ = [
    "ArchetypalAnalysisConfig",
    "DifferentiableArchetypalAnalysis",
]


@dataclass
class ArchetypalAnalysisConfig(OperatorConfig):
    """Configuration for DifferentiableArchetypalAnalysis.

    Attributes:
        n_genes: Number of input genes (features per cell).
        n_archetypes: Number of archetype prototypes to learn.
        hidden_dim: Hidden dimension for the encoder MLP.
        temperature: Softmax temperature (lower = sharper assignments).
        learnable_temperature: Whether temperature is a learnable parameter.
    """

    n_genes: int = 2000
    n_archetypes: int = 5
    hidden_dim: int = 64
    temperature: float = 1.0
    learnable_temperature: bool = False


class DifferentiableArchetypalAnalysis(TemperatureOperator):
    """Differentiable archetypal analysis with softmax simplex constraints.

    Each cell is encoded into archetype weight space via an MLP, then
    temperature-controlled softmax produces simplex weights.  The
    reconstruction is the convex combination of learnable archetype
    prototypes, enabling end-to-end gradient-based optimisation.

    Inherits from ``TemperatureOperator`` to get:

    - ``_temperature`` property for temperature-controlled smoothing
    - ``soft_max()`` for logsumexp-based smooth maximum

    Args:
        config: ArchetypalAnalysisConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        ```python
        import jax.numpy as jnp
        config = ArchetypalAnalysisConfig(n_genes=2000, n_archetypes=5)
        op = DifferentiableArchetypalAnalysis(config, rngs=nnx.Rngs(0))
        data = {"counts": jnp.ones((100, 2000))}
        result, state, meta = op.apply(data, {}, None)
        ```
    """

    def __init__(
        self,
        config: ArchetypalAnalysisConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the archetypal analysis operator.

        Args:
            config: Archetypal analysis configuration.
            rngs: Random number generators for weight initialisation.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        rngs = ensure_rngs(rngs)

        # Encoder MLP: n_genes -> hidden_dim -> n_archetypes (logits)
        self.encoder_layers = build_mlp_encoder(
            in_features=config.n_genes,
            hidden_dims=[config.hidden_dim],
            rngs=rngs,
        )
        self.projection = nnx.Linear(
            in_features=config.hidden_dim,
            out_features=config.n_archetypes,
            rngs=rngs,
        )

        # Learnable archetype prototypes (n_archetypes, n_genes)
        key = get_rng_key(rngs, "params", fallback_seed=1)
        init_archetypes = jax.random.normal(key, (config.n_archetypes, config.n_genes)) * 0.1
        self.archetypes = nnx.Param(init_archetypes)

    def encode(
        self,
        counts: Float[Array, "n_cells n_genes"],
    ) -> Float[Array, "n_cells n_archetypes"]:
        """Encode cells to simplex weights over archetypes.

        Args:
            counts: Cell-by-gene count matrix.

        Returns:
            Simplex weights of shape ``(n_cells, n_archetypes)``.
        """
        hidden = forward_mlp(self.encoder_layers, counts)
        logits = self.projection(hidden)
        weights = jax.nn.softmax(logits / self._temperature, axis=-1)
        return weights

    def reconstruct(
        self,
        weights: Float[Array, "n_cells n_archetypes"],
    ) -> Float[Array, "n_cells n_genes"]:
        """Reconstruct cells as convex combinations of archetypes.

        Args:
            weights: Simplex weights per cell.

        Returns:
            Reconstructed cell-by-gene matrix.
        """
        return weights @ self.archetypes[...]

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply archetypal analysis to a cell-by-gene count matrix.

        Args:
            data: Dictionary containing:
                - ``"counts"``: Cell-by-gene matrix ``(n_cells, n_genes)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used.
            stats: Not used.

        Returns:
            Tuple of ``(transformed_data, state, metadata)`` where
            ``transformed_data`` contains:

            - ``"counts"``: Original count matrix
            - ``"archetype_weights"``: Simplex weights ``(n_cells, n_archetypes)``
            - ``"archetypes"``: Archetype prototypes ``(n_archetypes, n_genes)``
            - ``"reconstructed"``: Reconstructed counts ``(n_cells, n_genes)``
        """
        counts = data["counts"]

        weights = self.encode(counts)
        reconstructed = self.reconstruct(weights)

        transformed_data = {
            **data,
            "archetype_weights": weights,
            "archetypes": self.archetypes[...],
            "reconstructed": reconstructed,
        }

        return transformed_data, state, metadata
