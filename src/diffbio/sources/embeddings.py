"""Embedding-artifact sources built on Datarax source primitives.

This module keeps file-format parsing local to DiffBio because the supported
artifacts are biology-specific, but the actual source abstraction follows the
same Datarax source model used elsewhere in the repository. The canonical
runtime substrate is therefore:

1. file decoding in one place
2. source semantics via Datarax ``MemorySource``
3. biology-specific alignment handled by specialized source subclasses
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx


def _require_torch() -> Any:
    """Import torch, raising a clear error if it is not installed."""
    try:
        import torch  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        return torch
    except ImportError as err:
        raise ImportError(
            "PyTorch is required to load .pt embedding files. "
            "Install with: uv pip install 'diffbio[torch-io]'"
        ) from err


@dataclass(frozen=True, slots=True)
class EmbeddingArtifactPayload:
    """Canonical embedding matrix plus optional artifact metadata arrays."""

    embeddings: np.ndarray
    metadata: dict[str, np.ndarray]


def _coerce_pt_value(value: Any, *, field_name: str) -> np.ndarray:
    """Convert a supported PyTorch payload field to a NumPy array."""
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return np.asarray(value.detach().cpu().numpy())

    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, (list, tuple)):
        return np.asarray(value)

    raise TypeError(
        "Expected PyTorch artifact field "
        f"'{field_name}' to be a tensor, NumPy array, list, or tuple, "
        f"but received {type(value).__name__}."
    )


def _load_npz_payload(path: Path) -> EmbeddingArtifactPayload:
    """Load the canonical array and metadata arrays from a NumPy archive."""
    with np.load(path) as archive:
        embedding_key = "embeddings" if "embeddings" in archive else next(iter(archive.files), None)
        if embedding_key is None:
            raise ValueError(f"Embedding archive is empty: {path}")

        metadata = {key: np.asarray(archive[key]) for key in archive.files if key != embedding_key}
        return EmbeddingArtifactPayload(
            embeddings=np.asarray(archive[embedding_key], dtype=np.float32),
            metadata=metadata,
        )


def _load_pt_payload(path: Path) -> EmbeddingArtifactPayload:
    """Load a PyTorch embedding artifact plus optional metadata fields."""
    torch = _require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(payload, dict):
        if "embeddings" not in payload:
            raise ValueError(
                "PyTorch embedding artifacts stored as mappings must include an 'embeddings' entry."
            )
        metadata = {
            str(key): _coerce_pt_value(value, field_name=str(key))
            for key, value in payload.items()
            if key != "embeddings"
        }
        return EmbeddingArtifactPayload(
            embeddings=np.asarray(
                _coerce_pt_value(payload["embeddings"], field_name="embeddings"),
                dtype=np.float32,
            ),
            metadata=metadata,
        )

    if hasattr(payload, "numpy"):
        return EmbeddingArtifactPayload(
            embeddings=np.asarray(payload.numpy(), dtype=np.float32),
            metadata={},
        )

    raise TypeError(
        "Expected a PyTorch tensor or mapping in the embedding artifact, "
        f"but received {type(payload).__name__}."
    )


def load_embedding_artifact(path: Path | str) -> EmbeddingArtifactPayload:
    """Load a canonical embedding artifact plus optional metadata arrays."""
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    if suffix == ".npy":
        return EmbeddingArtifactPayload(
            embeddings=np.asarray(np.load(resolved_path), dtype=np.float32),
            metadata={},
        )
    if suffix == ".npz":
        return _load_npz_payload(resolved_path)
    if suffix == ".pt":
        return _load_pt_payload(resolved_path)

    raise ValueError(
        f"Unsupported embedding file extension '{suffix}'. Use .npy, .npz, or .pt format."
    )


@dataclass(frozen=True)
class EmbeddingArtifactSourceConfig(MemorySourceConfig):
    """Configuration for eager artifact-backed embedding sources."""

    file_path: str | None = None

    def __post_init__(self) -> None:
        """Validate the artifact path and delegate common source validation."""
        super().__post_init__()

        if self.file_path is None:
            raise ValueError("file_path is required for EmbeddingArtifactSourceConfig")

        resolved_path = Path(self.file_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {resolved_path}")

        if resolved_path.suffix.lower() not in {".npy", ".npz", ".pt"}:
            raise ValueError(
                "Embedding artifact sources only support .npy, .npz, or .pt files, "
                f"got '{resolved_path.suffix}'."
            )


class EmbeddingArtifactSource(MemorySource):
    """Eager Datarax-style source for external embedding artifacts."""

    config: EmbeddingArtifactSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]
    _artifact_metadata: dict[str, np.ndarray] = nnx.data()

    def __init__(
        self,
        config: EmbeddingArtifactSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load an embedding artifact into a Datarax ``MemorySource``."""
        file_path = config.file_path
        if file_path is None:
            raise ValueError("file_path is required for EmbeddingArtifactSource")

        payload = load_embedding_artifact(file_path)
        source_name = name or f"EmbeddingArtifactSource({file_path})"
        data = {"embeddings": jnp.asarray(payload.embeddings, dtype=jnp.float32)}

        super().__init__(config, data=data, rngs=rngs, name=source_name)

        self._artifact_metadata = payload.metadata
        object.__setattr__(self, "_artifact_path", Path(file_path))

    def load(self) -> dict[str, Any]:
        """Return the eager in-memory payload exposed by the source."""
        return dict(cast(dict[str, Any], self.data))

    @property
    def embeddings(self) -> jnp.ndarray:
        """Full embedding matrix loaded from the artifact."""
        return cast(jnp.ndarray, cast(dict[str, Any], self.data)["embeddings"])

    @property
    def artifact_metadata(self) -> dict[str, np.ndarray]:
        """Auxiliary metadata arrays loaded from the artifact."""
        return dict(self._artifact_metadata)

    @property
    def artifact_path(self) -> Path:
        """Resolved path to the backing embedding artifact."""
        return self._artifact_path
