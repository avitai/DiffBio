"""Shared source utilities for single-cell foundation-model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from diffbio.sources.indexed_embeddings import (
    load_indexed_embedding_artifact,
    align_indexed_embeddings,
)


@dataclass(frozen=True, slots=True)
class SingleCellEmbeddingArtifact:
    """External single-cell embedding artifact plus optional row identities."""

    embeddings: jnp.ndarray
    cell_ids: tuple[str, ...] | None = None


def load_singlecell_embedding_artifact(path: Path | str) -> SingleCellEmbeddingArtifact:
    """Load a single-cell embedding artifact.

    For ``.npz`` artifacts, this loader reads the canonical ``embeddings`` array
    and an optional ``cell_ids`` array for row-level alignment. Other supported
    embedding formats reuse the generic embedding loader and carry no cell IDs.
    """
    artifact = load_indexed_embedding_artifact(path, row_id_key="cell_ids")
    return SingleCellEmbeddingArtifact(
        embeddings=artifact.embeddings,
        cell_ids=artifact.row_ids,
    )


def align_singlecell_embeddings(
    *,
    reference_cell_ids: list[str] | tuple[str, ...],
    artifact_path: Path | str,
    require_cell_ids: bool = True,
) -> jnp.ndarray:
    """Align external embeddings to the benchmark cell order."""
    return align_indexed_embeddings(
        reference_ids=reference_cell_ids,
        artifact_path=artifact_path,
        row_id_key="cell_ids",
        require_row_ids=require_cell_ids,
        artifact_label="Single-cell",
        id_display_name="Cell ID",
    )
