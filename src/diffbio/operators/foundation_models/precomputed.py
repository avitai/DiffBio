"""Precomputed artifact adapters for imported foundation-model embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp

from diffbio.operators.foundation_models.contracts import (
    AdapterMode,
    FoundationArtifactSpec,
    FoundationModelKind,
    PoolingStrategy,
    build_foundation_model_metadata,
)
from diffbio.sources.sequence_foundation import align_sequence_embeddings
from diffbio.sources.singlecell_foundation import align_singlecell_embeddings


def _singlecell_precomputed_spec(
    *,
    artifact_id: str,
    preprocessing_version: str,
    pooling_strategy: PoolingStrategy,
) -> FoundationArtifactSpec:
    """Build the canonical artifact spec for imported single-cell embeddings."""
    return FoundationArtifactSpec(
        model_family=FoundationModelKind.SINGLE_CELL_TRANSFORMER,
        artifact_id=artifact_id,
        preprocessing_version=preprocessing_version,
        adapter_mode=AdapterMode.PRECOMPUTED,
        pooling_strategy=pooling_strategy,
    )


def _sequence_precomputed_spec(
    *,
    artifact_id: str,
    preprocessing_version: str,
    pooling_strategy: PoolingStrategy,
) -> FoundationArtifactSpec:
    """Build the canonical artifact spec for imported sequence embeddings."""
    return FoundationArtifactSpec(
        model_family=FoundationModelKind.SEQUENCE_TRANSFORMER,
        artifact_id=artifact_id,
        preprocessing_version=preprocessing_version,
        adapter_mode=AdapterMode.PRECOMPUTED,
        pooling_strategy=pooling_strategy,
    )


class _ArtifactBackedFoundationAdapter:
    """Shared base for artifact-backed imported foundation-model adapters."""

    def __init__(
        self,
        *,
        artifact_path: Path | str,
        artifact_spec: FoundationArtifactSpec,
        source_name: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.artifact_spec = artifact_spec
        self.source_name = source_name
        self.extra_metadata = {} if extra_metadata is None else dict(sorted(extra_metadata.items()))

    def load_aligned_embeddings(
        self,
        *,
        reference_cell_ids: list[str] | tuple[str, ...],
        require_cell_ids: bool = True,
    ) -> jnp.ndarray:
        """Load and align embeddings to the benchmark dataset order."""
        return align_singlecell_embeddings(
            reference_cell_ids=reference_cell_ids,
            artifact_path=self.artifact_path,
            require_cell_ids=require_cell_ids,
        )

    def result_data(self) -> dict[str, Any]:
        """Return benchmark-ready foundation-model metadata."""
        return {"foundation_model": build_foundation_model_metadata(self.artifact_spec)}

    def benchmark_metadata(self) -> dict[str, Any]:
        """Return benchmark metadata describing the artifact source."""
        metadata: dict[str, Any] = {
            "embedding_source": "external_artifact",
            "foundation_source_name": self.source_name,
        }
        metadata.update(self.extra_metadata)
        return metadata


class SingleCellPrecomputedAdapter(_ArtifactBackedFoundationAdapter):
    """Base adapter for precomputed single-cell embedding artifacts."""

    def load_aligned_embeddings(
        self,
        *,
        reference_cell_ids: list[str] | tuple[str, ...],
        require_cell_ids: bool = True,
    ) -> jnp.ndarray:
        """Load and align embeddings to the benchmark dataset order."""
        return align_singlecell_embeddings(
            reference_cell_ids=reference_cell_ids,
            artifact_path=self.artifact_path,
            require_cell_ids=require_cell_ids,
        )


class SequencePrecomputedAdapter(_ArtifactBackedFoundationAdapter):
    """Base adapter for precomputed sequence embedding artifacts."""

    def load_aligned_embeddings(
        self,
        *,
        reference_sequence_ids: list[str] | tuple[str, ...],
        require_sequence_ids: bool = True,
    ) -> jnp.ndarray:
        """Load and align embeddings to the benchmark sequence order."""
        return align_sequence_embeddings(
            reference_sequence_ids=reference_sequence_ids,
            artifact_path=self.artifact_path,
            require_sequence_ids=require_sequence_ids,
        )


class DNABERT2PrecomputedAdapter(SequencePrecomputedAdapter):
    """Precomputed embedding adapter for DNABERT-2 artifacts."""

    def __init__(
        self,
        *,
        artifact_path: Path | str,
        artifact_id: str = "dnabert2.v1",
        preprocessing_version: str = "kmer6_v1",
        pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN,
    ) -> None:
        super().__init__(
            artifact_path=artifact_path,
            artifact_spec=_sequence_precomputed_spec(
                artifact_id=artifact_id,
                preprocessing_version=preprocessing_version,
                pooling_strategy=pooling_strategy,
            ),
            source_name="dnabert2_precomputed",
        )


class NucleotideTransformerPrecomputedAdapter(SequencePrecomputedAdapter):
    """Precomputed embedding adapter for Nucleotide Transformer artifacts."""

    def __init__(
        self,
        *,
        artifact_path: Path | str,
        artifact_id: str = "nucleotide_transformer.v1",
        preprocessing_version: str = "bpe_v1",
        pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN,
    ) -> None:
        super().__init__(
            artifact_path=artifact_path,
            artifact_spec=_sequence_precomputed_spec(
                artifact_id=artifact_id,
                preprocessing_version=preprocessing_version,
                pooling_strategy=pooling_strategy,
            ),
            source_name="nucleotide_transformer_precomputed",
        )


class GeneformerPrecomputedAdapter(SingleCellPrecomputedAdapter):
    """Precomputed embedding adapter for Geneformer artifacts."""

    def __init__(
        self,
        *,
        artifact_path: Path | str,
        artifact_id: str = "geneformer.v1",
        preprocessing_version: str = "rank_value_v1",
        pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN,
    ) -> None:
        super().__init__(
            artifact_path=artifact_path,
            artifact_spec=_singlecell_precomputed_spec(
                artifact_id=artifact_id,
                preprocessing_version=preprocessing_version,
                pooling_strategy=pooling_strategy,
            ),
            source_name="geneformer_precomputed",
        )


class ScGPTPrecomputedAdapter(SingleCellPrecomputedAdapter):
    """Precomputed embedding adapter for scGPT artifacts."""

    def __init__(
        self,
        *,
        artifact_path: Path | str,
        artifact_id: str = "scgpt.v1",
        preprocessing_version: str = "gene_vocab_v1",
        pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN,
        batch_key: str | None = None,
        context_version: str | None = None,
    ) -> None:
        extra_metadata: dict[str, Any] = {
            "requires_batch_context": batch_key is not None or context_version is not None,
        }
        if batch_key is not None:
            extra_metadata["batch_key"] = batch_key
        if context_version is not None:
            extra_metadata["context_version"] = context_version

        super().__init__(
            artifact_path=artifact_path,
            artifact_spec=_singlecell_precomputed_spec(
                artifact_id=artifact_id,
                preprocessing_version=preprocessing_version,
                pooling_strategy=pooling_strategy,
            ),
            source_name="scgpt_precomputed",
            extra_metadata=extra_metadata,
        )
