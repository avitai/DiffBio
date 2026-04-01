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
from diffbio.sources.singlecell_foundation import align_singlecell_embeddings


class SingleCellPrecomputedAdapter:
    """Base adapter for precomputed single-cell embedding artifacts."""

    def __init__(
        self,
        *,
        artifact_path: Path | str,
        artifact_spec: FoundationArtifactSpec,
        source_name: str,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.artifact_spec = artifact_spec
        self.source_name = source_name

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
        return {
            "embedding_source": "external_artifact",
            "foundation_source_name": self.source_name,
        }


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
            artifact_spec=FoundationArtifactSpec(
                model_family=FoundationModelKind.SINGLE_CELL_TRANSFORMER,
                artifact_id=artifact_id,
                preprocessing_version=preprocessing_version,
                adapter_mode=AdapterMode.PRECOMPUTED,
                pooling_strategy=pooling_strategy,
            ),
            source_name="geneformer_precomputed",
        )
