"""Tests for shared foundation benchmark adapter contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffbio.operators.foundation_models import (
    DNABERT2PrecomputedAdapter,
    create_foundation_adapter,
    get_foundation_adapter_cls,
)
from diffbio.operators.foundation_models.adapters import FoundationBenchmarkAdapterBase
from diffbio.operators.foundation_models.contracts import (
    AdapterMode,
    FoundationArtifactSpec,
    FoundationModelKind,
    PoolingStrategy,
)


def _sequence_spec() -> FoundationArtifactSpec:
    return FoundationArtifactSpec(
        model_family=FoundationModelKind.SEQUENCE_TRANSFORMER,
        artifact_id="diffbio.sequence.contract",
        preprocessing_version="one_hot_v1",
        adapter_mode=AdapterMode.PRECOMPUTED,
        pooling_strategy=PoolingStrategy.MEAN,
    )


class TestFoundationBenchmarkAdapterBase:
    """Tests for canonical benchmark adapter metadata behavior."""

    def test_benchmark_metadata_rejects_canonical_key_override(self) -> None:
        with pytest.raises(ValueError, match="embedding_source"):
            FoundationBenchmarkAdapterBase(
                artifact_spec=_sequence_spec(),
                source_name="sequence_precomputed",
                embedding_source="external_artifact",
                extra_metadata={"embedding_source": "override"},
            )

    def test_benchmark_metadata_requires_non_empty_source_fields(self) -> None:
        with pytest.raises(ValueError, match="source_name"):
            FoundationBenchmarkAdapterBase(
                artifact_spec=_sequence_spec(),
                source_name="",
                embedding_source="external_artifact",
            )

        with pytest.raises(ValueError, match="embedding_source"):
            FoundationBenchmarkAdapterBase(
                artifact_spec=_sequence_spec(),
                source_name="sequence_precomputed",
                embedding_source="",
            )

    def test_benchmark_metadata_keeps_canonical_keys_stable(self) -> None:
        adapter = FoundationBenchmarkAdapterBase(
            artifact_spec=_sequence_spec(),
            source_name="sequence_precomputed",
            embedding_source="external_artifact",
            extra_metadata={"context_version": "one_hot_v1"},
        )

        assert adapter.benchmark_metadata() == {
            "embedding_source": "external_artifact",
            "foundation_source_name": "sequence_precomputed",
            "context_version": "one_hot_v1",
        }


class TestFoundationBenchmarkAdapterRegistry:
    """Tests for the shared adapter registry and factory path."""

    def test_builtin_registry_entries(self) -> None:
        assert get_foundation_adapter_cls("dnabert2_precomputed") is DNABERT2PrecomputedAdapter

    def test_factory_instantiates_registered_adapter(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "dnabert2_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.ones((2, 4), dtype=np.float32),
            sequence_ids=np.array(["seq_a", "seq_b"]),
        )

        adapter = create_foundation_adapter(
            "dnabert2_precomputed",
            artifact_path=artifact_path,
        )

        assert isinstance(adapter, DNABERT2PrecomputedAdapter)

    def test_unknown_adapter_key_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown_adapter"):
            get_foundation_adapter_cls("unknown_adapter")
