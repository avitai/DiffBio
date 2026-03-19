"""Differentiable imputation operators for single-cell data.

This module provides two complementary imputation strategies:

1. **DifferentiableDiffusionImputer**: MAGIC-style diffusion imputation that
   constructs a cell-cell affinity graph and diffuses information across
   neighboring cells via eigendecomposition of the Markov transition matrix.

2. **DifferentiableTransformerDenoiser**: Transformer-based gene denoiser that
   treats genes as tokens, randomly masks a fraction of them, and predicts
   the masked gene expression values from the unmasked context using a
   transformer encoder. Reuses ``TransformerSequenceEncoder`` from the
   language models module (DRY).

Applications: Denoising dropout events in scRNA-seq count matrices, recovering
gene-gene relationships masked by technical noise.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree

from diffbio.core.graph_utils import (
    compute_fuzzy_membership,
    compute_pairwise_distances,
    symmetrize_graph,
)
from diffbio.operators.language_models.transformer_encoder import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
)


@dataclass
class DiffusionImputerConfig(OperatorConfig):
    """Configuration for MAGIC-style diffusion imputation.

    Attributes:
        n_neighbors: Number of neighbors for local bandwidth estimation.
        diffusion_t: Number of diffusion time steps (matrix power).
        n_pca_components: Number of PCA components (reserved for future use).
        metric: Distance metric, either ``"euclidean"`` or ``"cosine"``.
    """

    n_neighbors: int = 15
    diffusion_t: int = 3
    n_pca_components: int = 100
    metric: str = "euclidean"


class DifferentiableDiffusionImputer(OperatorModule):
    """Differentiable MAGIC-style diffusion imputation.

    Constructs a cell-cell affinity graph from pairwise distances, computes
    a fuzzy membership matrix, symmetrizes it, row-normalizes to obtain a
    Markov transition matrix, and raises it to power *t* via eigendecomposition
    to perform diffusion-based imputation.

    Algorithm:
        1. Compute pairwise distances between cells
        2. Mask the diagonal with a large sentinel value
        3. Compute fuzzy membership using local bandwidth (k-th neighbor)
        4. Symmetrize the graph via fuzzy set union
        5. Row-normalize to obtain Markov transition matrix M
        6. Compute M^t via eigendecomposition
        7. Impute: ``imputed = M^t @ counts``

    Args:
        config: DiffusionImputerConfig with operator parameters.
        rngs: Flax NNX random number generators (not used, kept for API).
        name: Optional operator name.

    Example:
        >>> config = DiffusionImputerConfig(n_neighbors=15, diffusion_t=3)
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

    def _build_markov_matrix(
        self,
        counts: Float[Array, "n_cells n_genes"],
    ) -> Float[Array, "n_cells n_cells"]:
        """Build the row-stochastic Markov transition matrix from counts.

        Args:
            counts: Gene expression matrix of shape ``(n_cells, n_genes)``.

        Returns:
            Row-stochastic Markov transition matrix of shape ``(n_cells, n_cells)``.
        """
        n_cells = counts.shape[0]

        # Step 1: Pairwise distances
        distances = compute_pairwise_distances(counts, metric=self.config.metric)

        # Step 2: Mask diagonal
        distances = distances + jnp.eye(n_cells) * 1e10

        # Step 3: Fuzzy membership
        membership = compute_fuzzy_membership(distances, k=self.config.n_neighbors)

        # Step 4: Symmetrize
        symmetric = symmetrize_graph(membership)

        # Step 5: Row-normalize to get Markov transition matrix
        row_sums = jnp.sum(symmetric, axis=1, keepdims=True)
        markov = symmetric / (row_sums + 1e-10)

        return markov

    def _diffuse(
        self,
        markov: Float[Array, "n_cells n_cells"],
        counts: Float[Array, "n_cells n_genes"],
        t: int,
    ) -> tuple[Float[Array, "n_cells n_genes"], Float[Array, "n_cells n_cells"]]:
        """Apply diffusion by raising the Markov matrix to power t.

        Uses eigendecomposition for differentiable matrix powering:
        ``M^t = V @ diag(lambda^t) @ V^T``

        Args:
            markov: Row-stochastic Markov transition matrix.
            counts: Original gene expression counts.
            t: Diffusion time (exponent).

        Returns:
            Tuple of (imputed counts, diffusion operator M^t).
        """
        n_cells = markov.shape[0]

        if t == 0:
            identity = jnp.eye(n_cells)
            return counts, identity

        # Make the matrix symmetric for eigh (it should be nearly symmetric
        # after symmetrize_graph + row normalization, but enforce it)
        markov_sym = (markov + markov.T) / 2.0

        # Eigendecomposition of the symmetric matrix
        eigenvalues, eigenvectors = jnp.linalg.eigh(markov_sym)

        # Clamp eigenvalues to [0, 1] for numerical stability
        eigenvalues = jnp.clip(eigenvalues, 0.0, 1.0)

        # M^t = V @ diag(lambda^t) @ V^T
        eigenvalues_t = eigenvalues**t
        diffusion_op = eigenvectors @ jnp.diag(eigenvalues_t) @ eigenvectors.T

        # Ensure row-stochasticity after powering
        row_sums = jnp.sum(diffusion_op, axis=1, keepdims=True)
        diffusion_op = diffusion_op / (row_sums + 1e-10)

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

        # Build Markov transition matrix
        markov = self._build_markov_matrix(counts)

        # Diffuse
        imputed, diffusion_op = self._diffuse(markov, counts, self.config.diffusion_t)

        transformed_data = {
            **data,
            "imputed_counts": imputed,
            "diffusion_operator": diffusion_op,
        }

        return transformed_data, state, metadata


@dataclass
class TransformerDenoiserConfig(OperatorConfig):
    """Configuration for transformer-based gene denoising.

    The denoiser treats genes as tokens: each gene has an expression value and
    a gene ID.  A random fraction of genes is masked (expression zeroed) and the
    transformer predicts the original expression from the unmasked context.

    Attributes:
        n_genes: Number of genes in the input expression profile.
        hidden_dim: Dimension of hidden states and embeddings.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        mask_ratio: Fraction of genes to mask during training.
        dropout_rate: Dropout rate for regularisation.
        stochastic: Whether the operator uses randomness (always ``True``).
        stream_name: RNG stream name for mask generation.
    """

    n_genes: int = 2000
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mask_ratio: float = 0.15
    dropout_rate: float = 0.1
    stochastic: bool = True
    stream_name: str | None = "sample"


class DifferentiableTransformerDenoiser(OperatorModule):
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

        # Reuse TransformerSequenceEncoder in token_embedding mode (DRY)
        encoder_config = TransformerSequenceEncoderConfig(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_dim=4 * config.hidden_dim,
            max_length=config.n_genes,
            input_embedding_type="token_embedding",
            vocab_size=config.n_genes,
            dropout_rate=config.dropout_rate,
            pooling="mean",
        )
        self.encoder = TransformerSequenceEncoder(encoder_config, rngs=rngs)

        # Learned projection from scalar expression value to hidden_dim
        self.expression_projection = nnx.Linear(1, config.hidden_dim, rngs=rngs)

        # Output head: hidden_dim -> 1 (predict scalar expression per gene)
        self.output_head = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> jax.Array:
        """Generate random parameters for gene masking.

        Args:
            rng: JAX random key.
            data_shapes: PyTree with shapes (unused beyond API contract).

        Returns:
            A JAX random key for reproducible mask generation inside apply.
        """
        return rng

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
        counts = data["counts"]
        gene_ids = data["gene_ids"]
        n_genes = counts.shape[1]

        # Generate mask from random_params
        if random_params is not None and self.config.mask_ratio > 0:
            noise = jax.random.uniform(random_params, (n_genes,))
            mask = (noise < self.config.mask_ratio).astype(jnp.float32)
        else:
            mask = jnp.zeros(n_genes, dtype=jnp.float32)

        # Ensure gene_ids are integer type for embedding lookup
        gene_ids_int = gene_ids.astype(jnp.int32)

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
