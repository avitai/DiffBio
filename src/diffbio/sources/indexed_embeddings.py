"""Generic indexed embedding-artifact helpers.

These helpers load rank-2 embedding matrices plus optional row identities from
external artifacts, then align rows to a reference dataset order. They are used
across single-cell and sequence foundation-model integrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from diffbio.sources.embeddings import load_embedding_array


@dataclass(frozen=True, slots=True)
class IndexedEmbeddingArtifact:
    """External embedding artifact plus optional row identities."""

    embeddings: jnp.ndarray
    row_ids: tuple[str, ...] | None = None


def load_indexed_embedding_artifact(
    path: Path | str,
    *,
    row_id_key: str,
) -> IndexedEmbeddingArtifact:
    """Load an embedding artifact with an optional row-identity array."""
    resolved_path = Path(path)
    if resolved_path.suffix.lower() == ".npz":
        with np.load(resolved_path, allow_pickle=False) as archive:
            if "embeddings" in archive:
                embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            else:
                first_key = next(iter(archive.files), None)
                if first_key is None:
                    raise ValueError(f"Embedding archive is empty: {resolved_path}")
                embeddings = np.asarray(archive[first_key], dtype=np.float32)

            row_ids: tuple[str, ...] | None = None
            if row_id_key in archive:
                row_ids = tuple(str(item) for item in np.asarray(archive[row_id_key]))

        return IndexedEmbeddingArtifact(
            embeddings=jnp.asarray(embeddings, dtype=jnp.float32),
            row_ids=row_ids,
        )

    return IndexedEmbeddingArtifact(embeddings=load_embedding_array(resolved_path))


def align_indexed_embeddings(
    *,
    reference_ids: list[str] | tuple[str, ...],
    artifact_path: Path | str,
    row_id_key: str,
    require_row_ids: bool = True,
    artifact_label: str,
    id_display_name: str,
) -> jnp.ndarray:
    """Align external embeddings to a reference dataset order."""
    artifact = load_indexed_embedding_artifact(artifact_path, row_id_key=row_id_key)
    canonical_reference_ids = tuple(str(item) for item in reference_ids)

    if artifact.embeddings.ndim != 2:
        raise ValueError(
            f"{artifact_label} embedding artifacts must be rank-2 matrices "
            f"(received shape {artifact.embeddings.shape})."
        )

    if artifact.row_ids is None:
        if require_row_ids:
            raise ValueError(
                f"{artifact_label} embedding artifacts must include {row_id_key} "
                "for strict alignment."
            )
        if artifact.embeddings.shape[0] != len(canonical_reference_ids):
            raise ValueError(
                "Positional embedding alignment requires the same number of rows "
                f"as reference items ({artifact.embeddings.shape[0]} vs "
                f"{len(canonical_reference_ids)})."
            )
        return artifact.embeddings

    if len(set(canonical_reference_ids)) != len(canonical_reference_ids):
        raise ValueError(f"Reference {row_id_key} values must be unique.")
    if len(set(artifact.row_ids)) != len(artifact.row_ids):
        raise ValueError(f"Embedding artifact {row_id_key} values must be unique.")

    reference_set = set(canonical_reference_ids)
    artifact_set = set(artifact.row_ids)
    if reference_set != artifact_set:
        missing = sorted(reference_set - artifact_set)
        extra = sorted(artifact_set - reference_set)
        raise ValueError(
            f"{id_display_name} mismatch between reference dataset and embedding artifact. "
            f"Missing: {missing}. Extra: {extra}."
        )

    index_by_row_id = {row_id: index for index, row_id in enumerate(artifact.row_ids)}
    row_indices = [index_by_row_id[row_id] for row_id in canonical_reference_ids]
    reordered = np.asarray(artifact.embeddings)[row_indices]
    return jnp.asarray(reordered, dtype=jnp.float32)
