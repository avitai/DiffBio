"""Shared multi-omics provenance and embedding-artifact sources."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax.numpy as jnp
from flax import nnx

from diffbio.sources.indexed_embeddings import IndexedEmbeddingSource, IndexedEmbeddingSourceConfig

MULTIOMICS_DATASET_PROVENANCE_KEYS = (
    "dataset_name",
    "source_type",
    "modalities",
    "curation_status",
    "biological_validation",
    "promotion_eligible",
    "source_path",
)
MULTIOMICS_ARTIFACT_METADATA_KEYS = (
    "artifact_id",
    "artifact_type",
    "modalities",
    "embedding_source",
    "foundation_source_name",
    "promotion_eligible",
)
_SUPPORTED_MULTIOMICS_MODALITIES = frozenset(
    {
        "rna",
        "atac",
        "protein",
        "spatial",
        "metabolomics",
        "mass_spectrometry",
    }
)


@dataclass(frozen=True)
class MultiOmicsEmbeddingSourceConfig(IndexedEmbeddingSourceConfig):
    """Configuration for sample-indexed multi-omics embedding artifacts."""

    row_id_key: str = "sample_ids"


class MultiOmicsEmbeddingSource(IndexedEmbeddingSource):
    """Indexed embedding source specialized for multi-omics sample artifacts."""

    config: MultiOmicsEmbeddingSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> MultiOmicsEmbeddingSource:
        """Build the canonical sample-indexed source for a multi-omics artifact."""
        return cls(MultiOmicsEmbeddingSourceConfig(file_path=str(path)), rngs=rngs)

    @property
    def sample_ids(self) -> tuple[str, ...] | None:
        """Tuple of persisted sample identifiers, if present."""
        return self.row_ids

    def align_to_reference_sample_ids(
        self,
        *,
        reference_sample_ids: Sequence[str],
        require_sample_ids: bool = True,
    ) -> jnp.ndarray:
        """Align imported embeddings to a reference multi-omics sample order."""
        return self.align_to_reference_ids(
            reference_ids=tuple(reference_sample_ids),
            require_row_ids=require_sample_ids,
            artifact_label="Multi-omics",
            id_display_name="Sample ID",
        )


@dataclass(frozen=True)
class MetabolomicsEmbeddingSourceConfig(IndexedEmbeddingSourceConfig):
    """Configuration for spectrum-indexed metabolomics embedding artifacts."""

    row_id_key: str = "spectrum_ids"


class MetabolomicsEmbeddingSource(IndexedEmbeddingSource):
    """Indexed embedding source specialized for metabolomics spectrum artifacts."""

    config: MetabolomicsEmbeddingSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> MetabolomicsEmbeddingSource:
        """Build the canonical spectrum-indexed source for a metabolomics artifact."""
        return cls(MetabolomicsEmbeddingSourceConfig(file_path=str(path)), rngs=rngs)

    @property
    def spectrum_ids(self) -> tuple[str, ...] | None:
        """Tuple of persisted spectrum identifiers, if present."""
        return self.row_ids

    def align_to_reference_spectrum_ids(
        self,
        *,
        reference_spectrum_ids: Sequence[str],
        require_spectrum_ids: bool = True,
    ) -> jnp.ndarray:
        """Align imported embeddings to a reference spectrum order."""
        return self.align_to_reference_ids(
            reference_ids=tuple(reference_spectrum_ids),
            require_row_ids=require_spectrum_ids,
            artifact_label="Metabolomics",
            id_display_name="Spectrum ID",
        )


def build_multiomics_dataset_provenance(
    *,
    dataset_name: str,
    source_type: str,
    modalities: Sequence[str],
    curation_status: str,
    biological_validation: str,
    promotion_eligible: bool,
    source_path: str | None = None,
) -> dict[str, Any]:
    """Build canonical provenance for benchmarked multi-omics datasets."""
    provenance: dict[str, Any] = {
        "dataset_name": dataset_name,
        "source_type": source_type,
        "modalities": list(modalities),
        "curation_status": curation_status,
        "biological_validation": biological_validation,
        "promotion_eligible": promotion_eligible,
        "source_path": source_path,
    }
    return validate_multiomics_dataset_provenance(provenance)


def validate_multiomics_dataset_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize a multi-omics dataset provenance payload."""
    missing = [key for key in MULTIOMICS_DATASET_PROVENANCE_KEYS if key not in provenance]
    if missing:
        raise ValueError(f"multi-omics dataset_provenance is missing required keys: {missing}")

    normalized = {key: provenance[key] for key in MULTIOMICS_DATASET_PROVENANCE_KEYS}
    for key in (
        "dataset_name",
        "source_type",
        "curation_status",
        "biological_validation",
    ):
        _require_non_empty_string(normalized[key], field_name=f"dataset_provenance.{key}")

    normalized["modalities"] = _normalize_modalities(normalized["modalities"])
    if not isinstance(normalized["promotion_eligible"], bool):
        raise TypeError("multi-omics dataset_provenance.promotion_eligible must be a bool.")
    if str(normalized["source_type"]).startswith("synthetic") and normalized["promotion_eligible"]:
        raise ValueError("Synthetic multi-omics provenance cannot be promotion_eligible.")
    if normalized["source_path"] is not None:
        _require_non_empty_string(
            normalized["source_path"],
            field_name="dataset_provenance.source_path",
        )
    return normalized


def build_multiomics_artifact_metadata(
    *,
    artifact_id: str,
    artifact_type: str,
    modalities: Sequence[str],
    embedding_source: str,
    foundation_source_name: str,
    promotion_eligible: bool,
) -> dict[str, Any]:
    """Build canonical metadata for imported or benchmark-produced omics artifacts."""
    metadata: dict[str, Any] = {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "modalities": list(modalities),
        "embedding_source": embedding_source,
        "foundation_source_name": foundation_source_name,
        "promotion_eligible": promotion_eligible,
    }
    return validate_multiomics_artifact_metadata(metadata)


def validate_multiomics_artifact_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize multi-omics artifact metadata."""
    missing = [key for key in MULTIOMICS_ARTIFACT_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"multi-omics artifact metadata is missing required keys: {missing}")

    normalized = {key: metadata[key] for key in MULTIOMICS_ARTIFACT_METADATA_KEYS}
    for key in ("artifact_id", "artifact_type", "embedding_source", "foundation_source_name"):
        _require_non_empty_string(normalized[key], field_name=f"artifact_metadata.{key}")
    normalized["modalities"] = _normalize_modalities(normalized["modalities"])
    if not isinstance(normalized["promotion_eligible"], bool):
        raise TypeError("multi-omics artifact metadata promotion_eligible must be a bool.")
    return normalized


def load_multiomics_embedding_source(
    path: Path | str,
    *,
    rngs: nnx.Rngs | None = None,
) -> MultiOmicsEmbeddingSource:
    """Build the canonical sample-indexed multi-omics embedding source."""
    return MultiOmicsEmbeddingSource.from_path(path, rngs=rngs)


def align_multiomics_embeddings(
    *,
    reference_sample_ids: Sequence[str],
    artifact_path: Path | str,
    require_sample_ids: bool = True,
) -> jnp.ndarray:
    """Align imported multi-omics embeddings to a reference sample order."""
    return load_multiomics_embedding_source(artifact_path).align_to_reference_sample_ids(
        reference_sample_ids=reference_sample_ids,
        require_sample_ids=require_sample_ids,
    )


def load_metabolomics_embedding_source(
    path: Path | str,
    *,
    rngs: nnx.Rngs | None = None,
) -> MetabolomicsEmbeddingSource:
    """Build the canonical spectrum-indexed metabolomics embedding source."""
    return MetabolomicsEmbeddingSource.from_path(path, rngs=rngs)


def align_metabolomics_embeddings(
    *,
    reference_spectrum_ids: Sequence[str],
    artifact_path: Path | str,
    require_spectrum_ids: bool = True,
) -> jnp.ndarray:
    """Align imported metabolomics embeddings to a reference spectrum order."""
    return load_metabolomics_embedding_source(artifact_path).align_to_reference_spectrum_ids(
        reference_spectrum_ids=reference_spectrum_ids,
        require_spectrum_ids=require_spectrum_ids,
    )


def _normalize_modalities(raw_modalities: Any) -> list[str]:
    """Normalize and validate modality identifiers."""
    if not isinstance(raw_modalities, Sequence) or isinstance(raw_modalities, str):
        raise TypeError("modalities must be a non-empty sequence of strings.")
    modalities = [str(modality) for modality in raw_modalities]
    if not modalities:
        raise ValueError("modalities must contain at least one modality.")
    invalid = sorted(set(modalities) - _SUPPORTED_MULTIOMICS_MODALITIES)
    if invalid:
        raise ValueError(f"Unsupported multi-omics modalities: {invalid}")
    return modalities


def _require_non_empty_string(value: Any, *, field_name: str) -> None:
    """Require one non-empty string field."""
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be a non-empty string.")
