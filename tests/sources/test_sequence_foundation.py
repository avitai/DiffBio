"""Tests for shared sequence foundation embedding sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diffbio.sources.sequence_foundation import (
    align_sequence_embeddings,
    load_sequence_embedding_artifact,
)


class TestLoadSequenceEmbeddingArtifact:
    """Tests for loading sequence embedding artifacts."""

    def test_load_npz_with_sequence_ids(self, tmp_path: Path) -> None:
        embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
        sequence_ids = np.array(["seq_a", "seq_b", "seq_c"])
        path = tmp_path / "artifact.npz"
        np.savez(path, embeddings=embeddings, sequence_ids=sequence_ids)

        artifact = load_sequence_embedding_artifact(path)

        np.testing.assert_allclose(np.asarray(artifact.embeddings), embeddings)
        assert artifact.sequence_ids == ("seq_a", "seq_b", "seq_c")


class TestAlignSequenceEmbeddings:
    """Tests for strict sequence-level embedding alignment."""

    def test_reorders_embeddings_to_reference_sequence_order(self, tmp_path: Path) -> None:
        embeddings = np.array(
            [
                [30.0, 31.0],
                [10.0, 11.0],
                [20.0, 21.0],
            ],
            dtype=np.float32,
        )
        sequence_ids = np.array(["seq_c", "seq_a", "seq_b"])
        path = tmp_path / "shuffled_embeddings.npz"
        np.savez(path, embeddings=embeddings, sequence_ids=sequence_ids)

        aligned = align_sequence_embeddings(
            reference_sequence_ids=["seq_a", "seq_b", "seq_c"],
            artifact_path=path,
        )

        np.testing.assert_allclose(
            np.asarray(aligned),
            np.array(
                [
                    [10.0, 11.0],
                    [20.0, 21.0],
                    [30.0, 31.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_requires_sequence_ids_by_default(self, tmp_path: Path) -> None:
        embeddings = np.arange(6, dtype=np.float32).reshape(3, 2)
        path = tmp_path / "embeddings.npy"
        np.save(path, embeddings)

        with pytest.raises(ValueError, match="must include sequence_ids"):
            align_sequence_embeddings(
                reference_sequence_ids=["seq_a", "seq_b", "seq_c"],
                artifact_path=path,
            )

    def test_rejects_mismatched_sequence_id_sets(self, tmp_path: Path) -> None:
        embeddings = np.arange(6, dtype=np.float32).reshape(3, 2)
        sequence_ids = np.array(["seq_a", "seq_b", "seq_extra"])
        path = tmp_path / "mismatch.npz"
        np.savez(path, embeddings=embeddings, sequence_ids=sequence_ids)

        with pytest.raises(ValueError, match="Sequence ID mismatch"):
            align_sequence_embeddings(
                reference_sequence_ids=["seq_a", "seq_b", "seq_c"],
                artifact_path=path,
            )

    def test_allows_positional_alignment_only_when_explicit(self, tmp_path: Path) -> None:
        embeddings = np.arange(6, dtype=np.float32).reshape(3, 2)
        path = tmp_path / "embeddings.npy"
        np.save(path, embeddings)

        aligned = align_sequence_embeddings(
            reference_sequence_ids=["seq_a", "seq_b", "seq_c"],
            artifact_path=path,
            require_sequence_ids=False,
        )

        np.testing.assert_allclose(np.asarray(aligned), embeddings)
