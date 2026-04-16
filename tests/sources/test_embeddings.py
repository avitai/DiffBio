"""Tests for Datarax-style embedding artifact sources."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from datarax.core.data_source import DataSourceModule
from datarax.sources import MemorySource

from diffbio.sources.embeddings import EmbeddingArtifactSource, EmbeddingArtifactSourceConfig


class TestEmbeddingArtifactSource:
    """Tests for eager embedding artifact sources."""

    def test_is_a_datarax_memory_source(self, tmp_path: Path) -> None:
        """The artifact source should reuse Datarax's in-memory source substrate."""
        data = np.random.default_rng(42).standard_normal((50, 32)).astype(np.float32)
        path = tmp_path / "embeddings.npy"
        np.save(path, data)

        source = EmbeddingArtifactSource(EmbeddingArtifactSourceConfig(file_path=str(path)))

        assert isinstance(source, DataSourceModule)
        assert isinstance(source, MemorySource)
        assert isinstance(source.embeddings, jnp.ndarray)
        np.testing.assert_allclose(np.asarray(source.embeddings), data, atol=1e-6)

    def test_load_npz_embeddings_key(self, tmp_path: Path) -> None:
        """Prefer the canonical ``embeddings`` key when reading ``.npz`` archives."""
        data = np.random.default_rng(42).standard_normal((10, 8)).astype(np.float32)
        other = np.random.default_rng(0).standard_normal((2, 2)).astype(np.float32)
        path = tmp_path / "embeddings.npz"
        np.savez(path, embeddings=data, other=other)

        source = EmbeddingArtifactSource(EmbeddingArtifactSourceConfig(file_path=str(path)))

        np.testing.assert_allclose(np.asarray(source.embeddings), data, atol=1e-6)
        np.testing.assert_allclose(source.artifact_metadata["other"], other, atol=1e-6)

    def test_load_npz_first_key_fallback(self, tmp_path: Path) -> None:
        """Fall back to the first array when ``embeddings`` is absent."""
        data = np.random.default_rng(42).standard_normal((12, 4)).astype(np.float32)
        path = tmp_path / "alt_embeddings.npz"
        np.savez(path, latent=data)

        source = EmbeddingArtifactSource(EmbeddingArtifactSourceConfig(file_path=str(path)))

        np.testing.assert_allclose(np.asarray(source.embeddings), data, atol=1e-6)

    def test_load_pt_tensor_file(self, tmp_path: Path) -> None:
        """Load an embedding matrix from a PyTorch tensor artifact."""
        torch = pytest.importorskip("torch")
        data = np.random.default_rng(123).standard_normal((7, 5)).astype(np.float32)
        path = tmp_path / "embeddings.pt"
        torch.save(torch.from_numpy(data), path)

        source = EmbeddingArtifactSource(EmbeddingArtifactSourceConfig(file_path=str(path)))

        np.testing.assert_allclose(np.asarray(source.embeddings), data, atol=1e-6)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Raise ``FileNotFoundError`` for missing files."""
        with pytest.raises(FileNotFoundError):
            EmbeddingArtifactSourceConfig(file_path=str(tmp_path / "nonexistent.npy"))

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        """Raise ``ValueError`` for unsupported embedding file formats."""
        path = tmp_path / "embeddings.csv"
        path.write_text("1,2,3\n", encoding="utf-8")

        with pytest.raises(ValueError, match="only support .npy, .npz, or .pt files"):
            EmbeddingArtifactSourceConfig(file_path=str(path))
