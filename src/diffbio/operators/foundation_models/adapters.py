"""Shared adapter interfaces for benchmark-facing foundation-model integrations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import jax.numpy as jnp

from diffbio.operators.foundation_models.contracts import (
    FoundationArtifactSpec,
    build_foundation_model_metadata,
)


class FoundationBenchmarkAdapter(Protocol):
    """Common benchmark-facing contract for foundation-model adapters."""

    def result_data(self) -> dict[str, Any]:
        """Return canonical operator metadata for benchmark tagging."""
        ...

    def benchmark_metadata(self) -> dict[str, Any]:
        """Return benchmark metadata describing the adapter source."""
        ...


class SequenceFoundationAdapter(FoundationBenchmarkAdapter, Protocol):
    """Shared contract for sequence foundation-model adapters."""

    def load_dataset_embeddings(
        self,
        *,
        reference_sequence_ids: Sequence[str],
        one_hot_sequences: Any,
    ) -> jnp.ndarray:
        """Return embeddings aligned to a benchmark dataset order."""
        ...


class FoundationBenchmarkAdapterBase:
    """Base implementation for stable benchmark metadata handling."""

    def __init__(
        self,
        *,
        artifact_spec: FoundationArtifactSpec,
        source_name: str,
        embedding_source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.artifact_spec = artifact_spec
        self.source_name = source_name
        self.embedding_source = embedding_source
        self.extra_metadata = {} if extra_metadata is None else dict(sorted(extra_metadata.items()))

    def result_data(self) -> dict[str, Any]:
        """Return benchmark-ready foundation-model metadata."""
        return {"foundation_model": build_foundation_model_metadata(self.artifact_spec)}

    def benchmark_metadata(self) -> dict[str, Any]:
        """Return benchmark metadata describing the adapter source."""
        metadata: dict[str, Any] = {
            "embedding_source": self.embedding_source,
            "foundation_source_name": self.source_name,
        }
        metadata.update(self.extra_metadata)
        return metadata
