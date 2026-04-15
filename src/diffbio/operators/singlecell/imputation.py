"""Differentiable imputation operators for single-cell data.

This module provides two complementary imputation strategies:

1. **DifferentiableDiffusionImputer**: MAGIC-style diffusion imputation that
   constructs a cell-cell affinity graph using an alpha-decaying kernel,
   builds a row-stochastic Markov matrix ``M = D^{-1} A``, and computes
   ``M^t`` via repeated matrix multiplication for diffusion-based imputation.

2. **DifferentiableTransformerDenoiser**: Transformer-based gene denoiser that
   treats genes as tokens, randomly masks a fraction of them, and predicts
   the masked gene expression values from the unmasked context using a
   transformer encoder. Reuses ``TransformerSequenceEncoder`` from the
   foundation models module (DRY).

Applications: Denoising dropout events in scRNA-seq count matrices, recovering
gene-gene relationships masked by technical noise.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.constants import DISTANCE_MASK_SENTINEL
from diffbio.core import soft_ops
from diffbio.core.graph_utils import (
    compute_pairwise_distances,
    symmetrize_graph,
)
from diffbio.operators._masked_gene_transformer import (
    MaskedGeneTransformerConfigBase,
    MaskedGeneTransformerOperatorMixin,
    build_masked_gene_transformer_encoder,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiffusionImputerConfig(OperatorConfig):
    """Configuration for MAGIC-style diffusion imputation.

    Attributes:
        n_neighbors: Number of neighbors for local bandwidth estimation.
        diffusion_t: Number of diffusion time steps (matrix power).
        n_pca_components: Number of PCA components (reserved for future use).
        decay: Exponent for the alpha-decaying kernel (MAGIC default is 1).
        metric: Distance metric, either ``"euclidean"`` or ``"cosine"``.
    """

    n_neighbors: int = 5
    diffusion_t: int = 3
    n_pca_components: int = 100
    decay: float = 1.0
    metric: str = "euclidean"


class DifferentiableDiffusionImputer(OperatorModule):
    """Differentiable MAGIC-style diffusion imputation.

    Constructs a cell-cell affinity graph using an alpha-decaying kernel,
    symmetrizes it, builds a row-stochastic Markov matrix ``M = D^{-1} A``,
    and computes ``M^t`` via repeated matrix multiplication for imputation.
    This avoids eigendecomposition (whose backward pass produces NaN when
    eigenvalues are near-degenerate) while remaining fully differentiable.

    Algorithm:
        1. Compute pairwise distances between cells
        2. Build alpha-decay affinity: ``K(i,j) = exp(-(d/sigma_i)^decay)``
        3. Symmetrize the affinity via fuzzy set union
        4. Row-normalize to Markov matrix ``M = D^{-1} A``
        5. Compute ``M^t`` via repeated matrix multiplication (t iterations)
        6. Impute: ``imputed = M^t @ counts``

    Args:
        config: DiffusionImputerConfig with operator parameters.
        rngs: Flax NNX random number generators (not used, kept for API).
        name: Optional operator name.

    Example:
        >>> config = DiffusionImputerConfig(n_neighbors=5, diffusion_t=3)
        >>> imputer = DifferentiableDiffusionImputer(config, rngs=nnx.Rngs(0))
        >>> data = {"counts": jnp.ones((100, 2000))}
        >>> result, state, meta = imputer.apply(data, {}, None)
        >>> result["imputed_counts"].shape
        (100, 2000)
    """

    def __init__(
        self,
        config: DiffusionImputerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the diffusion imputer.

        Args:
            config: Imputation configuration.
            rngs: Random number generators (unused, present for API consistency).
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

    def _build_alpha_decay_affinity(
        self,
        distances: Float[Array, "n n"],
        k: int,
        decay: float,
    ) -> Float[Array, "n n"]:
        """Build alpha-decaying kernel following MAGIC.

        ``K(i,j) = exp(-(d(i,j) / sigma_i)^decay)`` where sigma_i is the
        distance to the k-th nearest neighbor.

        Args:
            distances: Pairwise distance matrix with diagonal masked to DISTANCE_MASK_SENTINEL.
            k: Number of neighbors for local bandwidth estimation.
            decay: Exponent for the alpha-decaying kernel.

        Returns:
            Affinity matrix of shape ``(n, n)`` with zero diagonal.
        """
        n = distances.shape[0]
        k_eff = min(k, n - 2)

        # Local bandwidth: k-th nearest neighbor distance
        sorted_dists = soft_ops.sort(distances, axis=-1, softness=0.1)
        sigma = jnp.maximum(sorted_dists[:, k_eff], 1e-8)

        # Alpha-decay kernel
        affinity = jnp.exp(-((distances / sigma[:, None]) ** decay))

        # Zero diagonal
        affinity = affinity * (1.0 - jnp.eye(n))

        return affinity

    def _build_symmetric_affinity(
        self,
        counts: Float[Array, "n_cells n_genes"],
    ) -> Float[Array, "n_cells n_cells"]:
        """Build the symmetric affinity matrix from counts.

        Args:
            counts: Gene expression matrix of shape ``(n_cells, n_genes)``.

        Returns:
            Symmetric affinity matrix of shape ``(n_cells, n_cells)``.
        """
        n_cells = counts.shape[0]

        # Pairwise distances
        distances = compute_pairwise_distances(counts, metric=self.config.metric)

        # Mask diagonal
        distances = distances + jnp.eye(n_cells) * DISTANCE_MASK_SENTINEL

        # Alpha-decay kernel
        affinity = self._build_alpha_decay_affinity(
            distances, self.config.n_neighbors, self.config.decay
        )

        # Symmetrize via fuzzy set union
        return symmetrize_graph(affinity)

    def _diffuse(
        self,
        affinity_sym: Float[Array, "n_cells n_cells"],
        counts: Float[Array, "n_cells n_genes"],
        t: int,
    ) -> tuple[Float[Array, "n_cells n_genes"], Float[Array, "n_cells n_cells"]]:
        """Compute M^t via repeated matrix multiplication of the Markov matrix.

        The Markov matrix is ``M = D^{-1} A`` where ``A`` is the symmetric
        affinity and ``D`` is the diagonal degree matrix.  We compute ``M^t``
        by repeatedly multiplying ``M`` by itself ``t`` times.  This avoids
        eigendecomposition (whose backward pass produces NaN gradients when
        eigenvalues are near-degenerate) while remaining fully differentiable.

        Args:
            affinity_sym: Symmetric affinity matrix.
            counts: Original gene expression counts.
            t: Diffusion time (exponent).

        Returns:
            Tuple of (imputed counts, diffusion operator M^t).
        """
        n_cells = affinity_sym.shape[0]

        if t == 0:
            identity = jnp.eye(n_cells)
            return counts, identity

        # Build row-stochastic Markov matrix M = D^{-1} A
        degree = jnp.sum(affinity_sym, axis=1, keepdims=True)
        markov = affinity_sym / jnp.maximum(degree, 1e-10)

        # Compute M^t via repeated matrix multiplication
        diffusion_op = markov
        for _ in range(t - 1):
            diffusion_op = diffusion_op @ markov

        # Ensure row-stochasticity after powering (numerical correction)
        row_sums = jnp.sum(diffusion_op, axis=1, keepdims=True)
        diffusion_op = diffusion_op / jnp.maximum(row_sums, 1e-10)

        # Impute
        imputed = diffusion_op @ counts

        return imputed, diffusion_op

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply diffusion imputation to single-cell count data.

        Args:
            data: Dictionary containing:
                - ``"counts"``: Gene expression matrix ``(n_cells, n_genes)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: Not used (deterministic operator).
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - ``"counts"``: Original counts
                    - ``"imputed_counts"``: Diffusion-imputed counts
                    - ``"diffusion_operator"``: The M^t matrix
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts = data["counts"]

        # Build symmetric affinity matrix
        affinity_sym = self._build_symmetric_affinity(counts)

        # Diffuse via repeated matrix multiplication of Markov matrix
        imputed, diffusion_op = self._diffuse(affinity_sym, counts, self.config.diffusion_t)

        transformed_data = {
            **data,
            "imputed_counts": imputed,
            "diffusion_operator": diffusion_op,
        }

        return transformed_data, state, metadata


@dataclass(frozen=True)
class TransformerDenoiserConfig(MaskedGeneTransformerConfigBase):
    """Configuration for transformer-based gene denoising.

    The denoiser treats genes as tokens: each gene has an expression value and
    a gene ID.  A random fraction of genes is masked (expression zeroed) and the
    transformer predicts the original expression from the unmasked context.
    """


class DifferentiableTransformerDenoiser(
    MaskedGeneTransformerOperatorMixin,
    OperatorModule,
):
    """Transformer-based gene denoiser for single-cell expression data.

    Genes are treated as tokens in a sequence.  For each cell the operator:

    1. Randomly masks ``mask_ratio`` fraction of genes (sets expression to 0).
    2. Projects gene IDs into embeddings via ``TransformerSequenceEncoder``
       (token_embedding mode) and adds a learned projection of the expression
       value so the transformer receives both identity and magnitude.
    3. Passes the sequence through a transformer encoder to obtain
       contextualised gene representations.
    4. Predicts masked gene expression from context via a linear output head.
    5. Returns imputed counts where masked positions are replaced with
       predictions and unmasked positions are kept from the original input.

    Each cell is processed independently via ``jax.vmap`` over the cell
    dimension.

    Args:
        config: TransformerDenoiserConfig with operator parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = TransformerDenoiserConfig(n_genes=2000, hidden_dim=128)
        >>> denoiser = DifferentiableTransformerDenoiser(
        ...     config, rngs=nnx.Rngs(params=0, sample=1, dropout=2))
        >>> rp = denoiser.generate_random_params(
        ...     jax.random.key(0), {"counts": (100, 2000)})
        >>> data = {"counts": counts, "gene_ids": jnp.arange(2000)}
        >>> result, state, meta = denoiser.apply(data, {}, None, random_params=rp)
        >>> result["imputed_counts"].shape
        (100, 2000)
    """

    def __init__(
        self,
        config: TransformerDenoiserConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the transformer denoiser.

        Args:
            config: Denoiser configuration.
            rngs: Random number generators for parameter initialisation.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(params=0, sample=1, dropout=2)

        # Reuse the shared masked-gene token encoder contract.
        self.encoder = build_masked_gene_transformer_encoder(config, rngs=rngs)

        # Learned projection from scalar expression value to hidden_dim
        self.expression_projection = nnx.Linear(1, config.hidden_dim, rngs=rngs)

        # Output head: hidden_dim -> 1 (predict scalar expression per gene)
        self.output_head = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

    def _impute_single_cell(
        self,
        expression: Float[Array, "n_genes"],
        gene_ids: Float[Array, "n_genes"],
        mask: Float[Array, "n_genes"],
    ) -> Float[Array, "n_genes"]:
        """Impute expression for a single cell.

        Args:
            expression: Gene expression values for one cell.
            gene_ids: Integer gene IDs.
            mask: Binary mask (1 = masked / to-predict, 0 = observed).

        Returns:
            Imputed expression values for all genes.
        """
        # Zero out masked gene expression (the transformer must predict these)
        masked_expression = expression * (1.0 - mask)

        # Embed gene IDs via the encoder's input projection (nnx.Embed)
        gene_embeddings = self.encoder.input_projection(gene_ids)

        # Add expression information: project scalar expression to hidden_dim
        expr_projected = self.expression_projection(
            masked_expression[:, None]
        )  # (n_genes, hidden_dim)
        hidden = gene_embeddings + expr_projected  # (n_genes, hidden_dim)

        # Add batch dimension for transformer (expects [batch, seq, hidden])
        hidden = hidden[None, :, :]  # (1, n_genes, hidden_dim)

        # Apply transformer encoder
        hidden = self.encoder.transformer(hidden, mask=None, deterministic=True)

        # Remove batch dimension
        hidden = hidden[0]  # (n_genes, hidden_dim)

        # Predict expression from contextualised representations
        predictions = self.output_head(hidden).squeeze(-1)  # (n_genes,)

        # Replace masked positions with predictions, keep originals for unmasked
        imputed = jnp.where(mask > 0.5, predictions, expression)

        return imputed

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply transformer denoising to single-cell count data.

        Args:
            data: Dictionary containing:
                - ``"counts"``: Gene expression matrix ``(n_cells, n_genes)``
                - ``"gene_ids"``: Integer gene IDs ``(n_genes,)``
            state: Element state (passed through unchanged).
            metadata: Element metadata (passed through unchanged).
            random_params: JAX random key for mask generation.
            stats: Not used.

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:

                    - All original keys from data
                    - ``"imputed_counts"``: Denoised expression ``(n_cells, n_genes)``
                    - ``"mask"``: Binary mask used ``(n_genes,)``
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts, gene_ids_int, mask = self.prepare_masked_gene_batch(data, random_params)

        # Process each cell independently via vmap
        imputed = jax.vmap(
            self._impute_single_cell,
            in_axes=(0, None, None),
        )(counts, gene_ids_int, mask)

        transformed_data = {
            **data,
            "imputed_counts": imputed,
            "mask": mask,
        }

        return transformed_data, state, metadata
