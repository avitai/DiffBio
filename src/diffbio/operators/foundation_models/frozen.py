"""Frozen in-process adapters for benchmarked foundation-model integrations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from flax import nnx

from diffbio.operators.foundation_models.adapters import (
    FoundationBenchmarkAdapterBase,
    register_foundation_adapter,
)
from diffbio.operators.foundation_models.contracts import AdapterMode
from diffbio.operators.foundation_models.transformer_encoder import (
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
)


class FrozenSequenceEncoderAdapter(FoundationBenchmarkAdapterBase):
    """Benchmark adapter for a frozen in-process sequence encoder."""

    def __init__(
        self,
        *,
        config: TransformerSequenceEncoderConfig,
        rngs: nnx.Rngs | None = None,
        source_name: str = "diffbio_frozen_encoder",
    ) -> None:
        if config.adapter_mode is not AdapterMode.FROZEN_ENCODER:
            raise ValueError("FrozenSequenceEncoderAdapter requires adapter_mode='frozen_encoder'.")

        self.encoder = TransformerSequenceEncoder(config, rngs=rngs)
        super().__init__(
            artifact_spec=self.encoder.foundation_artifact_spec(),
            source_name=source_name,
            embedding_source="in_process_operator",
        )

    def load_dataset_embeddings(
        self,
        *,
        reference_sequence_ids: Sequence[str],
        one_hot_sequences: Any,
    ) -> jnp.ndarray:
        """Encode a benchmark dataset in-process without updating encoder weights."""
        sequences = jnp.asarray(one_hot_sequences, dtype=jnp.float32)
        if sequences.shape[0] != len(reference_sequence_ids):
            raise ValueError(
                "reference_sequence_ids and one_hot_sequences "
                "must share the same leading dimension."
            )
        result, _, _ = self.encoder.apply({"sequence": sequences}, {}, None)
        return jnp.asarray(result["embeddings"], dtype=jnp.float32)


register_foundation_adapter("diffbio_frozen_encoder", FrozenSequenceEncoderAdapter)
