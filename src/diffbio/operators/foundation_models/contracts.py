"""Shared contracts for DiffBio foundation-model operators.

This module centralizes the common metadata, output schema, and registry used
by DiffBio foundation-model operators. The contract is intentionally shared
across sequence, single-cell, and future imported biological foundation models
so downstream code can rely on one stable interface instead of task-specific
ad hoc keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from jaxtyping import Array, PyTree


class FoundationModelKind(StrEnum):
    """Supported high-level foundation-model families."""

    SEQUENCE_TRANSFORMER = "sequence_transformer"
    SINGLE_CELL_TRANSFORMER = "single_cell_transformer"


class AdapterMode(StrEnum):
    """How DiffBio integrates the underlying foundation model."""

    PRECOMPUTED = "precomputed"
    FROZEN_ENCODER = "frozen_encoder"
    NATIVE_TRAINABLE = "native_trainable"


class PoolingStrategy(StrEnum):
    """Canonical pooling strategies for foundation-model outputs."""

    NONE = "none"
    MEAN = "mean"
    CLS = "cls"


@dataclass(frozen=True)
class FoundationEmbeddingOperatorConfig(OperatorConfig):
    """Shared config fields for foundation-model operators."""

    adapter_mode: AdapterMode = AdapterMode.NATIVE_TRAINABLE
    artifact_id: str = "diffbio.builtin"
    preprocessing_version: str = "native_v1"

    def __post_init__(self) -> None:
        """Validate foundation-model metadata fields."""
        super().__post_init__()
        _validate_ascii_text(self.artifact_id, field_name="artifact_id")
        _validate_ascii_text(
            self.preprocessing_version,
            field_name="preprocessing_version",
        )


@dataclass(frozen=True)
class FoundationArtifactSpec:
    """Immutable spec describing a foundation-model artifact and interface."""

    model_family: FoundationModelKind
    artifact_id: str
    preprocessing_version: str
    adapter_mode: AdapterMode
    pooling_strategy: PoolingStrategy

    def __post_init__(self) -> None:
        """Validate artifact fields for JAX-safe metadata encoding."""
        _validate_ascii_text(self.artifact_id, field_name="artifact_id")
        _validate_ascii_text(
            self.preprocessing_version,
            field_name="preprocessing_version",
        )


def _validate_ascii_text(value: str, *, field_name: str) -> None:
    """Require non-empty ASCII metadata for JAX-safe output encoding."""
    if not value:
        raise ValueError(f"{field_name} must be non-empty.")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be ASCII-only.") from exc


def encode_foundation_text(value: str) -> Array:
    """Encode ASCII metadata text as a JAX array for jit-safe outputs."""
    _validate_ascii_text(value, field_name="metadata_text")
    encoded = value.encode("ascii")
    return jnp.asarray(list(encoded), dtype=jnp.uint8)


def decode_foundation_text(value: Array) -> str:
    """Decode a JAX uint8 text array back into an ASCII string."""
    data = bytes(int(item) for item in value.tolist())
    return data.decode("ascii")


def build_foundation_model_metadata(
    artifact_spec: FoundationArtifactSpec,
) -> dict[str, Array]:
    """Build a jit-safe metadata payload for a foundation-model result."""
    return {
        "model_family": encode_foundation_text(artifact_spec.model_family.value),
        "artifact_id": encode_foundation_text(artifact_spec.artifact_id),
        "preprocessing_version": encode_foundation_text(artifact_spec.preprocessing_version),
        "adapter_mode": encode_foundation_text(artifact_spec.adapter_mode.value),
        "pooling_strategy": encode_foundation_text(artifact_spec.pooling_strategy.value),
    }


class FoundationEmbeddingMixin:
    """Mixin providing canonical outputs for foundation-model operators."""

    config: FoundationEmbeddingOperatorConfig
    foundation_model_kind: FoundationModelKind

    def foundation_pooling_strategy(self) -> PoolingStrategy:
        """Return the pooling strategy used for the global embedding."""
        return PoolingStrategy.NONE

    def foundation_artifact_spec(self) -> FoundationArtifactSpec:
        """Build the artifact spec for the current operator."""
        return FoundationArtifactSpec(
            model_family=self.foundation_model_kind,
            artifact_id=self.config.artifact_id,
            preprocessing_version=self.config.preprocessing_version,
            adapter_mode=self.config.adapter_mode,
            pooling_strategy=self.foundation_pooling_strategy(),
        )

    def foundation_result(
        self,
        data: PyTree,
        embeddings: Array,
        *,
        token_embeddings: Array | None = None,
        extra_outputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the canonical operator result payload."""
        transformed_data = {
            **data,
            "embeddings": embeddings,
            "foundation_model": build_foundation_model_metadata(self.foundation_artifact_spec()),
        }

        if token_embeddings is not None:
            transformed_data["token_embeddings"] = token_embeddings

        if extra_outputs:
            for key in ("embeddings", "token_embeddings", "foundation_model"):
                if key in extra_outputs:
                    raise ValueError(f"extra_outputs cannot override canonical key {key!r}.")
            transformed_data.update(extra_outputs)

        return transformed_data


_FOUNDATION_MODEL_REGISTRY: dict[FoundationModelKind, type[OperatorModule]] = {}


def register_foundation_model(
    model_family: FoundationModelKind,
    operator_cls: type[OperatorModule],
) -> None:
    """Register an operator class for a foundation-model family."""
    _FOUNDATION_MODEL_REGISTRY[model_family] = operator_cls


def get_foundation_model_cls(
    model_family: FoundationModelKind,
) -> type[OperatorModule]:
    """Return the registered operator class for a foundation-model family."""
    try:
        return _FOUNDATION_MODEL_REGISTRY[model_family]
    except KeyError as exc:
        raise KeyError(f"No operator registered for {model_family.value!r}.") from exc


def create_foundation_model(
    model_family: FoundationModelKind,
    *args: Any,
    **kwargs: Any,
) -> OperatorModule:
    """Instantiate a registered foundation-model operator."""
    operator_cls = get_foundation_model_cls(model_family)
    return operator_cls(*args, **kwargs)
