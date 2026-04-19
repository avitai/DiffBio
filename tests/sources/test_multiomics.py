"""Tests for shared multi-omics source and artifact contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffbio.sources import (
    MULTIOMICS_ARTIFACT_METADATA_KEYS,
    MULTIOMICS_DATASET_PROVENANCE_KEYS,
    MetabolomicsEmbeddingSource,
    MultiOmicsEmbeddingSource,
    build_multiomics_artifact_metadata,
    build_multiomics_dataset_provenance,
    validate_multiomics_artifact_metadata,
    validate_multiomics_dataset_provenance,
)


def test_multiomics_dataset_provenance_contract_is_validated() -> None:
    provenance = build_multiomics_dataset_provenance(
        dataset_name="seqfish_cortex",
        source_type="curated_spatial_transcriptomics",
        modalities=("rna", "spatial"),
        curation_status="download_required_local_h5ad",
        biological_validation="published_benchmark_dataset",
        promotion_eligible=True,
        source_path="/tmp/seqfish_cortex.h5ad",
    )

    assert tuple(provenance) == MULTIOMICS_DATASET_PROVENANCE_KEYS
    assert provenance["modalities"] == ["rna", "spatial"]
    assert validate_multiomics_dataset_provenance(provenance) == provenance


def test_multiomics_dataset_provenance_rejects_scaffold_promotion() -> None:
    provenance = build_multiomics_dataset_provenance(
        dataset_name="synthetic_rna_atac",
        source_type="synthetic_scaffold",
        modalities=("rna", "atac"),
        curation_status="synthetic",
        biological_validation="interface_validation_only",
        promotion_eligible=False,
    )
    provenance["promotion_eligible"] = True

    with pytest.raises(ValueError, match="Synthetic multi-omics provenance"):
        validate_multiomics_dataset_provenance(provenance)


def test_multiomics_artifact_metadata_contract_is_validated() -> None:
    metadata = build_multiomics_artifact_metadata(
        artifact_id="multiomics.latent.v1",
        artifact_type="precomputed_embedding",
        modalities=("rna", "atac"),
        embedding_source="external_artifact",
        foundation_source_name="rna_atac_precomputed",
        promotion_eligible=False,
    )

    assert tuple(metadata) == MULTIOMICS_ARTIFACT_METADATA_KEYS
    assert metadata["modalities"] == ["rna", "atac"]
    assert validate_multiomics_artifact_metadata(metadata) == metadata


def test_multiomics_artifact_metadata_rejects_empty_modalities() -> None:
    with pytest.raises(ValueError, match="modalities"):
        build_multiomics_artifact_metadata(
            artifact_id="bad.v1",
            artifact_type="precomputed_embedding",
            modalities=(),
            embedding_source="external_artifact",
            foundation_source_name="bad_precomputed",
            promotion_eligible=False,
        )


def test_multiomics_embedding_source_aligns_sample_ids(tmp_path: Path) -> None:
    artifact_path = tmp_path / "rna_atac_embeddings.npz"
    np.savez(
        artifact_path,
        embeddings=np.array([[3.0, 3.1], [1.0, 1.1], [2.0, 2.1]], dtype=np.float32),
        sample_ids=np.array(["sample_c", "sample_a", "sample_b"]),
    )

    source = MultiOmicsEmbeddingSource.from_path(artifact_path)
    aligned = source.align_to_reference_sample_ids(
        reference_sample_ids=["sample_a", "sample_b", "sample_c"],
    )

    np.testing.assert_allclose(
        np.asarray(aligned),
        np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], dtype=np.float32),
    )


def test_metabolomics_embedding_source_aligns_spectrum_ids(tmp_path: Path) -> None:
    artifact_path = tmp_path / "spectral_embeddings.npz"
    np.savez(
        artifact_path,
        embeddings=np.array([[8.0, 8.1], [4.0, 4.1]], dtype=np.float32),
        spectrum_ids=np.array(["spectrum_b", "spectrum_a"]),
    )

    source = MetabolomicsEmbeddingSource.from_path(artifact_path)
    aligned = source.align_to_reference_spectrum_ids(
        reference_spectrum_ids=["spectrum_a", "spectrum_b"],
    )

    np.testing.assert_allclose(
        np.asarray(aligned),
        np.array([[4.0, 4.1], [8.0, 8.1]], dtype=np.float32),
    )
