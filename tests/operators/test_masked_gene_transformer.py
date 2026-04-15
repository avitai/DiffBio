"""Tests for shared masked-gene transformer helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from diffbio.operators._masked_gene_transformer import (
    MaskedGeneTransformerConfigBase,
    build_masked_gene_transformer_encoder,
    prepare_masked_gene_batch,
)


class TestMaskedGeneTransformerConfigBase:
    """Tests for shared masked-gene transformer config defaults."""

    def test_defaults_enable_sampling_stream(self) -> None:
        """Shared masked-gene configs should default to sampled stochastic mode."""
        config = MaskedGeneTransformerConfigBase()

        assert config.n_genes == 2000
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.num_heads == 4
        assert config.mask_ratio == 0.15
        assert config.dropout_rate == 0.1
        assert config.stochastic is True
        assert config.stream_name == "sample"


class TestBuildMaskedGeneTransformerEncoder:
    """Tests for shared transformer encoder construction."""

    def test_builds_token_embedding_encoder_with_config_contract(self) -> None:
        """The shared builder should always create a token-embedding encoder."""
        config = MaskedGeneTransformerConfigBase(
            n_genes=32,
            hidden_dim=16,
            num_layers=3,
            num_heads=2,
            dropout_rate=0.0,
        )

        encoder = build_masked_gene_transformer_encoder(config, rngs=nnx.Rngs(0))

        assert encoder.config.hidden_dim == 16
        assert encoder.config.num_layers == 3
        assert encoder.config.num_heads == 2
        assert encoder.config.max_length == 32
        assert encoder.config.vocab_size == 32
        assert encoder.config.input_embedding_type == "token_embedding"
        assert encoder.config.pooling == "mean"


class TestPrepareMaskedGeneBatch:
    """Tests for shared masked-gene input preparation."""

    def test_coerces_gene_ids_to_int32_and_builds_mask(self) -> None:
        """Shared preparation should return int32 gene IDs and a reproducible mask."""
        data = {
            "counts": jnp.arange(12, dtype=jnp.float32).reshape(3, 4),
            "gene_ids": jnp.arange(4, dtype=jnp.float32),
        }

        counts, gene_ids, mask = prepare_masked_gene_batch(
            data,
            random_params=jax.random.key(0),
            mask_ratio=1.0,
        )

        assert counts.shape == (3, 4)
        assert gene_ids.dtype == jnp.int32
        assert jnp.array_equal(gene_ids, jnp.arange(4, dtype=jnp.int32))
        assert jnp.array_equal(mask, jnp.ones((4,), dtype=jnp.float32))

    def test_returns_zero_mask_without_random_params(self) -> None:
        """Shared preparation should emit an all-zero mask when masking is disabled."""
        data = {
            "counts": jnp.ones((2, 5), dtype=jnp.float32),
            "gene_ids": jnp.arange(5, dtype=jnp.int32),
        }

        _, gene_ids, mask = prepare_masked_gene_batch(
            data,
            random_params=None,
            mask_ratio=0.15,
        )

        assert gene_ids.dtype == jnp.int32
        assert jnp.array_equal(mask, jnp.zeros((5,), dtype=jnp.float32))
