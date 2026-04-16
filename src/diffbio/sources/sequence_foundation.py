"""Sequence embedding sources following the Datarax source model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from flax import nnx

from diffbio.sources.indexed_embeddings import IndexedEmbeddingSource, IndexedEmbeddingSourceConfig


@dataclass(frozen=True)
class SequenceEmbeddingSourceConfig(IndexedEmbeddingSourceConfig):
    """Configuration for sequence embedding artifacts."""

    row_id_key: str = "sequence_ids"


class SequenceEmbeddingSource(IndexedEmbeddingSource):
    """Indexed embedding source specialized for sequence artifacts."""

    config: SequenceEmbeddingSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def sequence_ids(self) -> tuple[str, ...] | None:
        """Tuple of persisted sequence identifiers, if present."""
        return self.row_ids

    def align_to_reference_sequence_ids(
        self,
        *,
        reference_sequence_ids: list[str] | tuple[str, ...],
        require_sequence_ids: bool = True,
    ) -> jnp.ndarray:
        """Align external embeddings to the benchmark sequence order."""
        return self.align_to_reference_ids(
            reference_ids=reference_sequence_ids,
            require_row_ids=require_sequence_ids,
            artifact_label="Sequence",
            id_display_name="Sequence ID",
        )


def load_sequence_embedding_source(
    path: Path | str,
    *,
    rngs: nnx.Rngs | None = None,
) -> SequenceEmbeddingSource:
    """Build the canonical sequence embedding source for an artifact."""
    return SequenceEmbeddingSource(
        SequenceEmbeddingSourceConfig(file_path=str(path)),
        rngs=rngs,
    )


def align_sequence_embeddings(
    *,
    reference_sequence_ids: list[str] | tuple[str, ...],
    artifact_path: Path | str,
    require_sequence_ids: bool = True,
) -> jnp.ndarray:
    """Align external embeddings to the benchmark sequence order."""
    return load_sequence_embedding_source(artifact_path).align_to_reference_sequence_ids(
        reference_sequence_ids=reference_sequence_ids,
        require_sequence_ids=require_sequence_ids,
    )
