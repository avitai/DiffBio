"""Shared adapter interfaces for benchmark-facing foundation-model integrations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import jax.numpy as jnp

from diffbio.operators.foundation_models.contracts import (
    FoundationArtifactSpec,
    build_foundation_model_metadata,
)

_CANONICAL_BENCHMARK_METADATA_KEYS = (
    "embedding_source",
    "foundation_source_name",
)


def _validate_benchmark_text(value: str, *, field_name: str) -> None:
    """Require non-empty adapter metadata text for stable benchmark contracts."""
    if not value:
        raise ValueError(f"{field_name} must be non-empty.")


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
        _validate_benchmark_text(source_name, field_name="source_name")
        _validate_benchmark_text(embedding_source, field_name="embedding_source")
        normalized_extra_metadata = (
            {} if extra_metadata is None else dict(sorted(extra_metadata.items()))
        )
        conflicting_keys = [
            key for key in _CANONICAL_BENCHMARK_METADATA_KEYS if key in normalized_extra_metadata
        ]
        if conflicting_keys:
            keys = ", ".join(conflicting_keys)
            raise ValueError(f"extra_metadata cannot override canonical key(s): {keys}")

        self.artifact_spec = artifact_spec
        self.source_name = source_name
        self.embedding_source = embedding_source
        self.extra_metadata = normalized_extra_metadata

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


_FOUNDATION_ADAPTER_REGISTRY: dict[str, type[object]] = {}


def register_foundation_adapter(adapter_key: str, adapter_cls: type[object]) -> None:
    """Register a benchmark-facing foundation-model adapter class."""
    _validate_benchmark_text(adapter_key, field_name="adapter_key")
    _FOUNDATION_ADAPTER_REGISTRY[adapter_key] = adapter_cls


def get_foundation_adapter_cls(adapter_key: str) -> type[object]:
    """Return the registered adapter class for a canonical adapter key."""
    try:
        return _FOUNDATION_ADAPTER_REGISTRY[adapter_key]
    except KeyError as exc:
        raise KeyError(f"No adapter registered for {adapter_key!r}.") from exc


def create_foundation_adapter(adapter_key: str, *args: Any, **kwargs: Any) -> object:
    """Instantiate a registered foundation-model adapter."""
    adapter_cls = get_foundation_adapter_cls(adapter_key)
    return adapter_cls(*args, **kwargs)
