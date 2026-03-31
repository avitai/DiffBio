"""Tests for generic embedding file loading utilities."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from diffbio.sources.embeddings import load_embedding_array


class TestLoadEmbeddingArray:
    """Tests for generic embedding array loading."""

    def test_load_npy_file(self, tmp_path: Path) -> None:
        """Load an embedding matrix from an ``.npy`` file."""
        data = np.random.default_rng(42).standard_normal((50, 32)).astype(np.float32)
        path = tmp_path / "embeddings.npy"
        np.save(path, data)

        result = load_embedding_array(path)

        assert isinstance(result, jnp.ndarray)
        assert result.shape == (50, 32)
        np.testing.assert_allclose(result, data, atol=1e-6)

    def test_load_npz_embeddings_key(self, tmp_path: Path) -> None:
        """Prefer the canonical ``embeddings`` key when reading ``.npz`` archives."""
        data = np.random.default_rng(42).standard_normal((10, 8)).astype(np.float32)
        other = np.random.default_rng(0).standard_normal((2, 2)).astype(np.float32)
        path = tmp_path / "embeddings.npz"
        np.savez(path, embeddings=data, other=other)

        result = load_embedding_array(path)

        assert result.shape == data.shape
        np.testing.assert_allclose(result, data, atol=1e-6)

    def test_load_npz_first_key_fallback(self, tmp_path: Path) -> None:
        """Fall back to the first array when ``embeddings`` is absent."""
        data = np.random.default_rng(42).standard_normal((12, 4)).astype(np.float32)
        path = tmp_path / "alt_embeddings.npz"
        np.savez(path, latent=data)

        result = load_embedding_array(path)

        assert result.shape == data.shape
        np.testing.assert_allclose(result, data, atol=1e-6)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Raise ``FileNotFoundError`` for missing files."""
        with pytest.raises(FileNotFoundError):
            load_embedding_array(tmp_path / "nonexistent.npy")

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        """Raise ``ValueError`` for unsupported embedding file formats."""
        path = tmp_path / "embeddings.csv"
        path.write_text("1,2,3\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported embedding file extension"):
            load_embedding_array(path)
