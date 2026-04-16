"""Indexed embedding-artifact sources with strict row-identity alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
from flax import nnx

from diffbio.sources.embeddings import EmbeddingArtifactSource, EmbeddingArtifactSourceConfig


@dataclass(frozen=True)
class IndexedEmbeddingSourceConfig(EmbeddingArtifactSourceConfig):
    """Configuration for embedding sources that persist row identities."""

    row_id_key: str | None = None

    def __post_init__(self) -> None:
        """Validate the required row-identity field name."""
        super().__post_init__()
        if self.row_id_key is None:
            raise ValueError("row_id_key is required for IndexedEmbeddingSourceConfig")
        if not self.row_id_key.strip():
            raise ValueError("row_id_key must be a non-empty string")


class IndexedEmbeddingSource(EmbeddingArtifactSource):
    """Eager Datarax-style source for embeddings with optional row identities."""

    config: IndexedEmbeddingSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: IndexedEmbeddingSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load the embedding artifact and persist the configured row IDs."""
        super().__init__(config, rngs=rngs, name=name)

        row_id_key = config.row_id_key
        if row_id_key is None:
            raise ValueError("row_id_key is required for IndexedEmbeddingSource")

        row_ids_array = self.artifact_metadata.get(row_id_key)
        row_ids: tuple[str, ...] | None = None
        if row_ids_array is not None:
            if int(np.asarray(row_ids_array).shape[0]) != int(self.embeddings.shape[0]):
                raise ValueError(
                    f"{row_id_key} must contain one value per embedding row "
                    f"({np.asarray(row_ids_array).shape[0]} vs {self.embeddings.shape[0]})."
                )
            row_ids = tuple(str(item) for item in np.asarray(row_ids_array))
            cast(dict[str, Any], self.data)[row_id_key] = row_ids

        object.__setattr__(self, "_row_ids", row_ids)

    @property
    def row_id_key(self) -> str:
        """Configured metadata field name that identifies artifact rows."""
        return cast(str, self.config.row_id_key)

    @property
    def row_ids(self) -> tuple[str, ...] | None:
        """Tuple of persisted row identifiers, if present."""
        return self._row_ids

    def load(self) -> dict[str, Any]:
        """Return the eager in-memory payload exposed by the source."""
        payload = super().load()
        if self.row_ids is not None:
            payload[self.row_id_key] = self.row_ids
        return payload

    def align_to_reference_ids(
        self,
        *,
        reference_ids: list[str] | tuple[str, ...],
        require_row_ids: bool = True,
        artifact_label: str,
        id_display_name: str,
    ) -> jnp.ndarray:
        """Align embedding rows to a reference dataset order."""
        canonical_reference_ids = tuple(str(item) for item in reference_ids)

        if self.embeddings.ndim != 2:
            raise ValueError(
                f"{artifact_label} embedding artifacts must be rank-2 matrices "
                f"(received shape {self.embeddings.shape})."
            )

        if self.row_ids is None:
            if require_row_ids:
                raise ValueError(
                    f"{artifact_label} embedding artifacts must include {self.row_id_key} "
                    "for strict alignment."
                )
            if self.embeddings.shape[0] != len(canonical_reference_ids):
                raise ValueError(
                    "Positional embedding alignment requires the same number of rows "
                    f"as reference items ({self.embeddings.shape[0]} vs "
                    f"{len(canonical_reference_ids)})."
                )
            return self.embeddings

        if len(set(canonical_reference_ids)) != len(canonical_reference_ids):
            raise ValueError(f"Reference {self.row_id_key} values must be unique.")
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError(f"Embedding artifact {self.row_id_key} values must be unique.")

        reference_set = set(canonical_reference_ids)
        artifact_set = set(self.row_ids)
        if reference_set != artifact_set:
            missing = sorted(reference_set - artifact_set)
            extra = sorted(artifact_set - reference_set)
            raise ValueError(
                f"{id_display_name} mismatch between reference dataset and embedding artifact. "
                f"Missing: {missing}. Extra: {extra}."
            )

        index_by_row_id = {row_id: index for index, row_id in enumerate(self.row_ids)}
        row_indices = [index_by_row_id[row_id] for row_id in canonical_reference_ids]
        reordered = np.asarray(self.embeddings)[row_indices]
        return jnp.asarray(reordered, dtype=jnp.float32)
