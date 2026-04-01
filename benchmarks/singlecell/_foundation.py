"""Shared helpers for single-cell foundation-model benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from diffbio.sources.embeddings import load_embedding_array


@dataclass(frozen=True)
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

            cell_ids: tuple[str, ...] | None = None
            if "cell_ids" in archive:
                cell_ids = tuple(str(item) for item in np.asarray(archive["cell_ids"]))

        return SingleCellEmbeddingArtifact(
            embeddings=jnp.asarray(embeddings, dtype=jnp.float32),
            cell_ids=cell_ids,
        )

    return SingleCellEmbeddingArtifact(embeddings=load_embedding_array(resolved_path))


def align_singlecell_embeddings(
    *,
    reference_cell_ids: list[str] | tuple[str, ...],
    artifact_path: Path | str,
    require_cell_ids: bool = True,
) -> jnp.ndarray:
    """Align external embeddings to the benchmark cell order.

    Args:
        reference_cell_ids: Canonical cell order from the benchmark dataset.
        artifact_path: Path to the external embedding artifact.
        require_cell_ids: Whether artifacts must carry explicit row identities.

    Returns:
        Embeddings reordered to match ``reference_cell_ids``.

    Raises:
        ValueError: If the artifact is missing ``cell_ids`` when required, if
            the row counts disagree for positional alignment, or if the cell-ID
            sets do not match exactly.
    """
    artifact = load_singlecell_embedding_artifact(artifact_path)
    reference_ids = tuple(str(cell_id) for cell_id in reference_cell_ids)

    if artifact.embeddings.ndim != 2:
        raise ValueError(
            "Single-cell embedding artifacts must be rank-2 matrices "
            f"(received shape {artifact.embeddings.shape})."
        )

    if artifact.cell_ids is None:
        if require_cell_ids:
            raise ValueError(
                "Single-cell embedding artifacts must include cell_ids for strict alignment."
            )
        if artifact.embeddings.shape[0] != len(reference_ids):
            raise ValueError(
                "Positional single-cell embedding alignment requires the same number of rows "
                f"as reference cells ({artifact.embeddings.shape[0]} vs {len(reference_ids)})."
            )
        return artifact.embeddings

    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("Reference cell IDs must be unique.")
    if len(set(artifact.cell_ids)) != len(artifact.cell_ids):
        raise ValueError("Embedding artifact cell_ids must be unique.")

    reference_set = set(reference_ids)
    artifact_set = set(artifact.cell_ids)
    if reference_set != artifact_set:
        missing = sorted(reference_set - artifact_set)
        extra = sorted(artifact_set - reference_set)
        raise ValueError(
            "Cell ID mismatch between reference dataset and embedding artifact. "
            f"Missing: {missing}. Extra: {extra}."
        )

    index_by_cell_id = {cell_id: index for index, cell_id in enumerate(artifact.cell_ids)}
    row_indices = [index_by_cell_id[cell_id] for cell_id in reference_ids]
    reordered = np.asarray(artifact.embeddings)[row_indices]
    return jnp.asarray(reordered, dtype=jnp.float32)
