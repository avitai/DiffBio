"""Single-cell embedding sources following the Datarax source model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from flax import nnx

from diffbio.sources.indexed_embeddings import IndexedEmbeddingSource, IndexedEmbeddingSourceConfig


@dataclass(frozen=True)
class SingleCellEmbeddingSourceConfig(IndexedEmbeddingSourceConfig):
    """Configuration for single-cell embedding artifacts."""

    row_id_key: str = "cell_ids"


class SingleCellEmbeddingSource(IndexedEmbeddingSource):
    """Indexed embedding source specialized for single-cell artifacts."""

    config: SingleCellEmbeddingSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def cell_ids(self) -> tuple[str, ...] | None:
        """Tuple of persisted cell identifiers, if present."""
        return self.row_ids

    def align_to_reference_cell_ids(
        self,
        *,
        reference_cell_ids: list[str] | tuple[str, ...],
        require_cell_ids: bool = True,
    ) -> jnp.ndarray:
        """Align external embeddings to the benchmark cell order."""
        return self.align_to_reference_ids(
            reference_ids=reference_cell_ids,
            require_row_ids=require_cell_ids,
            artifact_label="Single-cell",
            id_display_name="Cell ID",
        )


def load_singlecell_embedding_source(
    path: Path | str,
    *,
    rngs: nnx.Rngs | None = None,
) -> SingleCellEmbeddingSource:
    """Build the canonical single-cell embedding source for an artifact."""
    return SingleCellEmbeddingSource(
        SingleCellEmbeddingSourceConfig(file_path=str(path)),
        rngs=rngs,
    )


def align_singlecell_embeddings(
    *,
    reference_cell_ids: list[str] | tuple[str, ...],
    artifact_path: Path | str,
    require_cell_ids: bool = True,
) -> jnp.ndarray:
    """Align external embeddings to the benchmark cell order."""
    return load_singlecell_embedding_source(artifact_path).align_to_reference_cell_ids(
        reference_cell_ids=reference_cell_ids,
        require_cell_ids=require_cell_ids,
    )
