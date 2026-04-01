"""Tests for frozen in-process foundation-model adapters."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from diffbio.operators.foundation_models import (
    AdapterMode,
    FrozenSequenceEncoderAdapter,
    FoundationArtifactSpec,
    FoundationModelKind,
    PoolingStrategy,
    SequencePrecomputedAdapter,
    TransformerSequenceEncoderConfig,
    decode_foundation_text,
)
from diffbio.sequences.dna import encode_dna_string


def _make_one_hot_sequences() -> jnp.ndarray:
    """Build deterministic one-hot DNA inputs for frozen-adapter tests."""
    sequences = ("AAAAAACCCC", "CCCCCCAAAA", "GGGGGGTTTT")
    encoded = [np.asarray(encode_dna_string(sequence), dtype=np.float32) for sequence in sequences]
    return jnp.asarray(
        np.stack(encoded),
        dtype=jnp.float32,
    )


class TestFrozenSequenceEncoderAdapter:
    """Tests for the frozen in-process sequence adapter."""

    def test_exposes_frozen_encoder_metadata(self) -> None:
        adapter = FrozenSequenceEncoderAdapter(
            config=TransformerSequenceEncoderConfig(
                hidden_dim=8,
                num_layers=1,
                num_heads=2,
                intermediate_dim=16,
                max_length=10,
                dropout_rate=0.0,
                pooling="mean",
                artifact_id="diffbio.sequence_frozen_encoder",
                preprocessing_version="one_hot_v1",
                adapter_mode=AdapterMode.FROZEN_ENCODER,
            ),
            rngs=nnx.Rngs(42),
        )

        metadata = adapter.result_data()["foundation_model"]

        assert decode_foundation_text(metadata["model_family"]) == "sequence_transformer"
        assert decode_foundation_text(metadata["adapter_mode"]) == "frozen_encoder"
        assert decode_foundation_text(metadata["artifact_id"]) == "diffbio.sequence_frozen_encoder"
        assert decode_foundation_text(metadata["preprocessing_version"]) == "one_hot_v1"
        assert decode_foundation_text(metadata["pooling_strategy"]) == "mean"

    def test_matches_precomputed_export_of_the_same_encoder(self, tmp_path: Path) -> None:
        reference_sequence_ids = ["seq_a", "seq_b", "seq_c"]
        one_hot_sequences = _make_one_hot_sequences()
        adapter = FrozenSequenceEncoderAdapter(
            config=TransformerSequenceEncoderConfig(
                hidden_dim=8,
                num_layers=1,
                num_heads=2,
                intermediate_dim=16,
                max_length=10,
                dropout_rate=0.0,
                pooling="mean",
                artifact_id="diffbio.sequence_frozen_encoder",
                preprocessing_version="one_hot_v1",
                adapter_mode=AdapterMode.FROZEN_ENCODER,
            ),
            rngs=nnx.Rngs(42),
        )

        frozen_embeddings = adapter.load_dataset_embeddings(
            reference_sequence_ids=reference_sequence_ids,
            one_hot_sequences=one_hot_sequences,
        )

        artifact_path = tmp_path / "frozen_export.npz"
        np.savez(
            artifact_path,
            embeddings=np.asarray(frozen_embeddings),
            sequence_ids=np.asarray(reference_sequence_ids),
        )
        precomputed_adapter = SequencePrecomputedAdapter(
            artifact_path=artifact_path,
            artifact_spec=FoundationArtifactSpec(
                model_family=FoundationModelKind.SEQUENCE_TRANSFORMER,
                artifact_id="diffbio.sequence_frozen_encoder.export",
                preprocessing_version="one_hot_v1",
                adapter_mode=AdapterMode.PRECOMPUTED,
                pooling_strategy=PoolingStrategy.MEAN,
            ),
            source_name="frozen_export",
        )

        precomputed_embeddings = precomputed_adapter.load_dataset_embeddings(
            reference_sequence_ids=reference_sequence_ids,
            one_hot_sequences=one_hot_sequences,
        )

        np.testing.assert_allclose(
            np.asarray(frozen_embeddings),
            np.asarray(precomputed_embeddings),
            atol=1e-6,
            rtol=1e-6,
        )
