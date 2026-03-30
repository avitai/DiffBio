"""Foundation model infrastructure for single-cell genomics.

This module provides differentiable implementations inspired by Geneformer and
scGPT architectures for single-cell gene expression foundation models.

Key components:

- **GeneTokenizer**: Geneformer-style rank-value encoding via differentiable
  soft sorting.  For each cell, genes are ranked by expression value in
  descending order using a temperature-controlled soft permutation matrix.
- **DifferentiableFoundationModel**: scGPT-inspired masked gene expression
  model that tokenizes gene expression, embeds gene identities and expression
  values, applies random masking, encodes with a transformer, and predicts
  masked expression values.

References:
    - Geneformer: Theodoris et al. (2023) Nature
    - scGPT: Cui et al. (2024) Nature Methods
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

from diffbio.core import soft_ops

from diffbio.operators.language_models.transformer_encoder import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoundationModelConfig(OperatorConfig):
    """Configuration for DifferentiableFoundationModel.

    Attributes:
        n_genes: Number of genes in the vocabulary.
        hidden_dim: Dimension of hidden states and embeddings.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        mask_ratio: Fraction of genes to mask during training (scGPT default 0.15).
        dropout_rate: Dropout rate for regularization.
    """

    n_genes: int = 2000
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mask_ratio: float = 0.15
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        """Set stochastic defaults for masking and validate."""
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "sample")
        super().__post_init__()


class GeneTokenizer(nnx.Module):
    """Geneformer-style rank-value gene tokenizer.

    Converts gene expression vectors into rank-ordered representations using
    a differentiable soft sort approximation.  For each cell, genes are ranked
    by expression value in descending order.  The output is a soft permutation
    matrix of shape ``(n_genes, n_genes)`` where row *i* is a soft one-hot
    indicating which gene occupies rank *i*.

    The key insight from Geneformer: token IDs are gene indices sorted by
    expression magnitude.  We approximate the discrete argsort with a
    temperature-controlled soft permutation to maintain differentiability.

    Args:
        n_genes: Number of genes.
        rngs: Flax NNX random number generators.
    """

    def __init__(self, n_genes: int, *, rngs: nnx.Rngs) -> None:
        """Initialize the gene tokenizer.

        Args:
            n_genes: Number of genes in the vocabulary.
            rngs: Random number generators (unused, kept for NNX API).
        """
        super().__init__()
        self.n_genes = n_genes

    def __call__(
        self,
        expression: Float[Array, "n_genes"],
        temperature: float = 1.0,
    ) -> Float[Array, "n_genes n_genes"]:
        """Compute soft permutation matrix from expression values.

        Genes are ranked in descending order of expression.  The returned
        matrix ``P`` has shape ``(n_genes, n_genes)`` where ``P[i, j]``
        approximates the probability that gene *j* occupies rank *i*.

        At low temperature this approaches a hard permutation matrix
        (the true argsort).

        Args:
            expression: Gene expression values for one cell, shape ``(n_genes,)``.
            temperature: Softmax temperature (lower is sharper).

        Returns:
            Soft permutation matrix of shape ``(n_genes, n_genes)``.
        """
        return _soft_sort_permutation(expression, temperature)


def _soft_sort_permutation(
    values: Float[Array, "n"],
    temperature: float,
) -> Float[Array, "n n"]:
    """Compute a differentiable soft permutation for descending sort.

    Uses the pairwise comparison approach: for each pair (i, j), compute
    a soft indicator of whether ``values[j] > values[i]``.  The soft rank
    of gene *j* is the sum of these indicators.  A softmax over negative
    squared distance between soft ranks and integer positions yields the
    permutation matrix.

    Args:
        values: 1-D array of values to sort.
        temperature: Temperature for softmax (lower is sharper).

    Returns:
        Soft permutation matrix ``P`` of shape ``(n, n)`` where ``P[i, j]``
        approximates the probability that element *j* is at position *i*
        in the descending sort.
    """
    return soft_ops.argsort(values, axis=0, descending=True, softness=temperature)


class DifferentiableFoundationModel(OperatorModule):
    """Differentiable single-cell foundation model operator.

    Implements a masked gene expression prediction model inspired by
    Geneformer (rank-value tokenization) and scGPT (masked expression
    prediction with gene + value embeddings).

    Algorithm:

    1. **Tokenize**: Rank genes by expression per cell via soft sort
       (Geneformer-style, used for gene embedding ordering context).
    2. **Embed gene IDs** via ``TransformerSequenceEncoder`` with
       ``input_embedding_type="token_embedding"``.
    3. **Add expression value projection**: scalar expression values are
       projected to ``hidden_dim`` and added to gene embeddings (scGPT-style).
    4. **Random mask**: ``mask_ratio`` fraction of genes have their expression
       embeddings replaced with a learned mask token.
    5. **Transformer encoder**: contextualizes gene representations.
    6. **Predict**: linear output head predicts masked gene expression.
    7. **Cell embedding**: mean pooling of non-masked gene representations.

    Args:
        config: FoundationModelConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = FoundationModelConfig(n_genes=2000, hidden_dim=128)
        >>> model = DifferentiableFoundationModel(
        ...     config, rngs=nnx.Rngs(params=0, sample=1, dropout=2))
        >>> rp = model.generate_random_params(
        ...     jax.random.key(0), {"counts": (100, 2000)})
        >>> data = {"counts": counts, "gene_ids": jnp.arange(2000)}
        >>> result, state, meta = model.apply(data, {}, None, random_params=rp)
    """

    def __init__(
        self,
        config: FoundationModelConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the foundation model.

        Args:
            config: Foundation model configuration.
            rngs: Random number generators for parameter initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(params=0, sample=1, dropout=2)

        # Gene tokenizer for rank-value encoding
        self.tokenizer = GeneTokenizer(config.n_genes, rngs=rngs)

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

        # Expression value projection: scalar -> hidden_dim (scGPT ContinuousValueEncoder)
        self.expression_projection = nnx.Sequential(
            nnx.Linear(1, config.hidden_dim, rngs=rngs),
            nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs),
        )

        # Learned mask token embedding (replaces expression embedding for masked genes)
        self.mask_token = nnx.Param(jax.random.normal(rngs.params(), (config.hidden_dim,)) * 0.02)

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

    def _process_single_cell(
        self,
        expression: Float[Array, "n_genes"],
        gene_ids: Float[Array, "n_genes"],
        mask: Float[Array, "n_genes"],
    ) -> tuple[
        Float[Array, "n_genes hidden_dim"],
        Float[Array, "hidden_dim"],
        Float[Array, "n_genes"],
    ]:
        """Process a single cell through the foundation model.

        Args:
            expression: Gene expression values for one cell.
            gene_ids: Integer gene IDs.
            mask: Binary mask (1 = masked, 0 = observed).

        Returns:
            Tuple of (gene_representations, cell_embedding, predicted_expression).
        """
        # Step 1: Embed gene IDs via the encoder's input projection (nnx.Embed)
        gene_embeddings = self.encoder.input_projection(gene_ids)  # (n_genes, hidden_dim)

        # Step 2: Project expression values to hidden_dim (scGPT-style value encoder)
        expr_projected = self.expression_projection(expression[:, None])  # (n_genes, hidden_dim)

        # Step 3: Apply mask -- replace masked expression embeddings with mask token
        mask_expanded = mask[:, None]  # (n_genes, 1)
        mask_token_broadcast = jnp.broadcast_to(self.mask_token[...][None, :], expr_projected.shape)
        expr_projected = jnp.where(mask_expanded > 0.5, mask_token_broadcast, expr_projected)

        # Step 4: Combine gene identity embeddings + expression value embeddings
        hidden = gene_embeddings + expr_projected  # (n_genes, hidden_dim)

        # Step 5: Apply transformer encoder
        hidden = hidden[None, :, :]  # (1, n_genes, hidden_dim)
        hidden = self.encoder.transformer(hidden, mask=None, deterministic=True)
        hidden = hidden[0]  # (n_genes, hidden_dim)

        # Step 6: Predict expression for all genes via output head
        predicted = self.output_head(hidden).squeeze(-1)  # (n_genes,)

        # Step 7: Cell embedding = mean pooling of non-masked gene representations
        # Use (1 - mask) to select non-masked genes
        non_masked_weight = (1.0 - mask)[:, None]  # (n_genes, 1)
        non_masked_count = jnp.maximum(jnp.sum(1.0 - mask), 1.0)
        cell_embedding = (
            jnp.sum(hidden * non_masked_weight, axis=0) / non_masked_count
        )  # (hidden_dim,)

        return hidden, cell_embedding, predicted

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply foundation model to single-cell count data.

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
                    - ``"gene_embeddings"``: Gene representations ``(n_genes, hidden_dim)``
                    - ``"cell_embeddings"``: Cell embeddings ``(n_cells, hidden_dim)``
                    - ``"predicted_expression"``: Predicted expression ``(n_cells, n_genes)``
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
        gene_reps, cell_embeddings, predicted = jax.vmap(
            self._process_single_cell,
            in_axes=(0, None, None),
        )(counts, gene_ids_int, mask)
        # gene_reps: (n_cells, n_genes, hidden_dim)
        # cell_embeddings: (n_cells, hidden_dim)
        # predicted: (n_cells, n_genes)

        # Gene embeddings: mean over cells for a single per-gene representation
        gene_embeddings = jnp.mean(gene_reps, axis=0)  # (n_genes, hidden_dim)

        transformed_data = {
            **data,
            "gene_embeddings": gene_embeddings,
            "cell_embeddings": cell_embeddings,
            "predicted_expression": predicted,
        }

        return transformed_data, state, metadata
