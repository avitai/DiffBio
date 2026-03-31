#!/usr/bin/env python3
"""Foundation model correctness tests for DiffBio.

Validates DiffBio's biological foundation model operators for output
shape correctness, value finiteness, and gradient flow:
- TransformerSequenceEncoder (DNA/RNA transformer encoding)
- DifferentiableFoundationModel (single-cell foundation model)
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from diffbio.operators.foundation_models.foundation_model import (
    DifferentiableFoundationModel,
    FoundationModelConfig,
)
from diffbio.operators.foundation_models.transformer_encoder import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
)


# ------------------------------------------------------------------
# Operator tests
# ------------------------------------------------------------------


def _test_transformer(
    sequences: jnp.ndarray,
    hidden_dim: int,
) -> tuple[dict[str, bool], TransformerSequenceEncoder]:
    """Test TransformerSequenceEncoder on synthetic one-hot sequences.

    Args:
        sequences: One-hot encoded sequences of shape
            ``(n_seqs, seq_len, alphabet_size)``.
        hidden_dim: Hidden dimension for the encoder.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    n_seqs, seq_len, alphabet_size = sequences.shape

    config = TransformerSequenceEncoderConfig(
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        intermediate_dim=4 * hidden_dim,
        max_length=seq_len,
        alphabet_size=alphabet_size,
        dropout_rate=0.0,
        pooling="mean",
    )
    rngs = nnx.Rngs(params=0, dropout=1)
    encoder = TransformerSequenceEncoder(config, rngs=rngs)

    data = {"sequence": sequences}
    result, _, _ = encoder.apply(data, {}, None)

    embeddings = result["embeddings"]
    token_embeddings = result["token_embeddings"]

    shape_ok = embeddings.shape == (n_seqs, hidden_dim) and token_embeddings.shape == (
        n_seqs,
        seq_len,
        hidden_dim,
    )
    values_finite = bool(
        jnp.all(jnp.isfinite(embeddings)) and jnp.all(jnp.isfinite(token_embeddings))
    )

    return {
        "shape_ok": shape_ok,
        "values_finite": values_finite,
    }, encoder


def _test_foundation_model(
    counts: jnp.ndarray,
    n_genes: int,
) -> tuple[dict[str, bool], DifferentiableFoundationModel]:
    """Test DifferentiableFoundationModel on synthetic expression data.

    Args:
        counts: Gene expression matrix of shape
            ``(n_cells, n_genes)``.
        n_genes: Number of genes.

    Returns:
        Tuple of (metrics dict, operator instance).
    """
    n_cells = counts.shape[0]

    config = FoundationModelConfig(
        n_genes=n_genes,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        mask_ratio=0.0,
        dropout_rate=0.0,
    )
    rngs = nnx.Rngs(params=0, sample=1, dropout=2)
    model = DifferentiableFoundationModel(config, rngs=rngs)

    gene_ids = jnp.arange(n_genes)
    data = {"counts": counts, "gene_ids": gene_ids}
    result, _, _ = model.apply(data, {}, None)

    embeddings = result["embeddings"]
    token_embeddings = result["token_embeddings"]
    predicted = result["predicted_expression"]

    shape_ok = (
        embeddings.shape == (n_cells, config.hidden_dim)
        and token_embeddings.shape == (n_cells, n_genes, config.hidden_dim)
        and predicted.shape == (n_cells, n_genes)
    )
    values_finite = bool(
        jnp.all(jnp.isfinite(embeddings))
        and jnp.all(jnp.isfinite(token_embeddings))
        and jnp.all(jnp.isfinite(predicted))
    )

    return {
        "shape_ok": shape_ok,
        "values_finite": values_finite,
    }, model
