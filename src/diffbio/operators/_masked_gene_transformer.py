"""Shared scaffolding for masked-gene transformer operators.

These helpers centralize the common single-cell transformer setup used by the
foundation-model and imputation operators so both paths share one encoder
construction and one mask/input preparation flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from flax import nnx
from jaxtyping import Array, PyTree

from diffbio.operators.foundation_models.transformer_encoder import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
)


@dataclass(frozen=True)
class MaskedGeneTransformerConfigBase(OperatorConfig):
    """Shared config fields for masked-gene transformer operators."""

    n_genes: int = 2000
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    mask_ratio: float = 0.15
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        """Default masked-gene operators to sampled stochastic execution."""
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "sample")
        super().__post_init__()


def build_masked_gene_transformer_encoder(
    config: MaskedGeneTransformerConfigBase,
    *,
    rngs: nnx.Rngs,
) -> TransformerSequenceEncoder:
    """Build the shared token-embedding transformer encoder contract."""
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
    return TransformerSequenceEncoder(encoder_config, rngs=rngs)


def build_masked_gene_mask(
    *,
    key: jax.Array | None,
    mask_ratio: float,
    n_genes: int,
) -> Array:
    """Build a per-gene binary mask for masked-gene transformer operators.

    With ``mask_ratio == 0`` nothing is masked and no key is needed; otherwise the mask is
    drawn from ``key``, which must be given.
    """
    if mask_ratio <= 0:
        return jnp.zeros(n_genes, dtype=jnp.float32)
    if key is None:
        raise ValueError("masking genes needs the record's PRNG key")
    noise = jax.random.uniform(key, (n_genes,))
    return (noise < mask_ratio).astype(jnp.float32)


def prepare_masked_gene_batch(
    data: PyTree,
    *,
    key: jax.Array | None,
    mask_ratio: float,
) -> tuple[Array, Array, Array]:
    """Extract counts, int32 gene IDs, and the shared masking vector."""
    counts = data["counts"]
    gene_ids = jnp.asarray(data["gene_ids"], dtype=jnp.int32)
    mask = build_masked_gene_mask(
        key=key,
        mask_ratio=mask_ratio,
        n_genes=int(counts.shape[1]),
    )
    return counts, gene_ids, mask


class MaskedGeneTransformerOperatorMixin:
    """Mixin for operators built on the shared masked-gene transformer flow."""

    config: MaskedGeneTransformerConfigBase

    def prepare_masked_gene_batch(
        self,
        data: PyTree,
        key: jax.Array | None,
    ) -> tuple[Array, Array, Array]:
        """Prepare shared masked-gene inputs for per-cell `vmap` execution."""
        return prepare_masked_gene_batch(
            data,
            key=key,
            mask_ratio=self.config.mask_ratio,
        )
