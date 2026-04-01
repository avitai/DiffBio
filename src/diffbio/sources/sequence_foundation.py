"""Shared source utilities for sequence foundation-model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

from diffbio.sources.indexed_embeddings import (
    align_indexed_embeddings,
    load_indexed_embedding_artifact,
)


@dataclass(frozen=True, slots=True)
class SequenceEmbeddingArtifact:
    """External sequence embedding artifact plus optional row identities."""

    embeddings: jnp.ndarray
    sequence_ids: tuple[str, ...] | None = None


def load_sequence_embedding_artifact(path: Path | str) -> SequenceEmbeddingArtifact:
    """Load a sequence embedding artifact with optional ``sequence_ids``."""
    artifact = load_indexed_embedding_artifact(path, row_id_key="sequence_ids")
    return SequenceEmbeddingArtifact(
        embeddings=artifact.embeddings,
        sequence_ids=artifact.row_ids,
    )


def align_sequence_embeddings(
    *,
    reference_sequence_ids: list[str] | tuple[str, ...],
    artifact_path: Path | str,
    require_sequence_ids: bool = True,
) -> jnp.ndarray:
    """Align external sequence embeddings to the benchmark sequence order."""
    return align_indexed_embeddings(
        reference_ids=reference_sequence_ids,
        artifact_path=artifact_path,
        row_id_key="sequence_ids",
        require_row_ids=require_sequence_ids,
        artifact_label="Sequence",
        id_display_name="Sequence ID",
    )
