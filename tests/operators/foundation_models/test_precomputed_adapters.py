"""Tests for precomputed single-cell foundation-model adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from diffbio.operators.foundation_models import (
    DNABERT2PrecomputedAdapter,
    FoundationArtifactSpec,
    FoundationModelKind,
    GeneformerPrecomputedAdapter,
    NucleotideTransformerPrecomputedAdapter,
    ProteinLMPrecomputedAdapter,
    PoolingStrategy,
    ScGPTPrecomputedAdapter,
    SequencePrecomputedAdapter,
    AdapterMode,
    decode_foundation_text,
)


class TestGeneformerPrecomputedAdapter:
    """Tests for strict Geneformer artifact handling."""

    def test_loads_aligned_embeddings_from_cell_ids(self, tmp_path: Path) -> None:
        embeddings = np.array(
            [
                [3.0, 3.1],
                [1.0, 1.1],
                [2.0, 2.1],
            ],
            dtype=np.float32,
        )
        cell_ids = np.array(["cell_c", "cell_a", "cell_b"])
        artifact_path = tmp_path / "geneformer_embeddings.npz"
        np.savez(artifact_path, embeddings=embeddings, cell_ids=cell_ids)

        adapter = GeneformerPrecomputedAdapter(
            artifact_path=artifact_path,
            artifact_id="geneformer.v1",
            preprocessing_version="rank_value_v1",
        )

        aligned = adapter.load_aligned_embeddings(
            reference_cell_ids=["cell_a", "cell_b", "cell_c"],
        )

        np.testing.assert_allclose(
            np.asarray(aligned),
            np.array(
                [
                    [1.0, 1.1],
                    [2.0, 2.1],
                    [3.0, 3.1],
                ],
                dtype=np.float32,
            ),
        )

    def test_exposes_foundation_model_metadata(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "geneformer_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.ones((2, 3), dtype=np.float32),
            cell_ids=np.array(["cell_a", "cell_b"]),
        )
        adapter = GeneformerPrecomputedAdapter(
            artifact_path=artifact_path,
            artifact_id="geneformer.v1",
            preprocessing_version="rank_value_v1",
        )

        metadata = adapter.result_data()["foundation_model"]

        assert decode_foundation_text(metadata["model_family"]) == "single_cell_transformer"
        assert decode_foundation_text(metadata["adapter_mode"]) == "precomputed"
        assert decode_foundation_text(metadata["artifact_id"]) == "geneformer.v1"
        assert decode_foundation_text(metadata["preprocessing_version"]) == "rank_value_v1"
        assert decode_foundation_text(metadata["pooling_strategy"]) == "mean"


class TestScGPTPrecomputedAdapter:
    """Tests for strict scGPT artifact handling."""

    def test_exposes_foundation_model_metadata(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "scgpt_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.ones((2, 3), dtype=np.float32),
            cell_ids=np.array(["cell_a", "cell_b"]),
        )
        adapter = ScGPTPrecomputedAdapter(
            artifact_path=artifact_path,
            artifact_id="scgpt.v1",
            preprocessing_version="gene_vocab_v1",
        )

        metadata = adapter.result_data()["foundation_model"]

        assert decode_foundation_text(metadata["model_family"]) == "single_cell_transformer"
        assert decode_foundation_text(metadata["adapter_mode"]) == "precomputed"
        assert decode_foundation_text(metadata["artifact_id"]) == "scgpt.v1"
        assert decode_foundation_text(metadata["preprocessing_version"]) == "gene_vocab_v1"
        assert decode_foundation_text(metadata["pooling_strategy"]) == "mean"

    def test_exposes_batch_context_metadata(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "scgpt_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.ones((2, 3), dtype=np.float32),
            cell_ids=np.array(["cell_a", "cell_b"]),
        )
        adapter = ScGPTPrecomputedAdapter(
            artifact_path=artifact_path,
            artifact_id="scgpt.v1",
            preprocessing_version="gene_vocab_v1",
            batch_key="batch",
            context_version="obs_batch_v1",
        )

        metadata = adapter.benchmark_metadata()

        assert metadata["foundation_source_name"] == "scgpt_precomputed"
        assert metadata["requires_batch_context"] is True
        assert metadata["batch_key"] == "batch"
        assert metadata["context_version"] == "obs_batch_v1"


class TestDNABERT2PrecomputedAdapter:
    """Tests for the DNABERT-2 precomputed adapter."""

    def test_exposes_foundation_model_metadata(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "dnabert2_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.ones((2, 4), dtype=np.float32),
            sequence_ids=np.array(["seq_a", "seq_b"]),
        )
        adapter = DNABERT2PrecomputedAdapter(artifact_path=artifact_path)

        metadata = adapter.result_data()["foundation_model"]

        assert decode_foundation_text(metadata["model_family"]) == "sequence_transformer"
        assert decode_foundation_text(metadata["adapter_mode"]) == "precomputed"
        assert decode_foundation_text(metadata["artifact_id"]) == "dnabert2.v1"
        assert decode_foundation_text(metadata["preprocessing_version"]) == "kmer6_v1"
        assert decode_foundation_text(metadata["pooling_strategy"]) == "mean"

    def test_loads_aligned_embeddings_from_sequence_ids(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "dnabert2_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.array([[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]], dtype=np.float32),
            sequence_ids=np.array(["seq_c", "seq_a", "seq_b"]),
        )
        adapter = DNABERT2PrecomputedAdapter(artifact_path=artifact_path)

        aligned = adapter.load_aligned_embeddings(
            reference_sequence_ids=["seq_a", "seq_b", "seq_c"],
        )

        np.testing.assert_allclose(
            np.asarray(aligned),
            np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float32),
        )

    def test_load_dataset_embeddings_uses_shared_sequence_contract(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "dnabert2_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.array([[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]], dtype=np.float32),
            sequence_ids=np.array(["seq_c", "seq_a", "seq_b"]),
        )
        adapter = DNABERT2PrecomputedAdapter(artifact_path=artifact_path)

        aligned = adapter.load_dataset_embeddings(
            reference_sequence_ids=["seq_a", "seq_b", "seq_c"],
            one_hot_sequences=np.zeros((3, 8, 4), dtype=np.float32),
        )

        np.testing.assert_allclose(
            np.asarray(aligned),
            np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float32),
        )


class TestNucleotideTransformerPrecomputedAdapter:
    """Tests for the Nucleotide Transformer precomputed adapter."""

    def test_exposes_foundation_model_metadata(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "nucleotide_transformer_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.ones((2, 4), dtype=np.float32),
            sequence_ids=np.array(["seq_a", "seq_b"]),
        )
        adapter = NucleotideTransformerPrecomputedAdapter(artifact_path=artifact_path)

        metadata = adapter.result_data()["foundation_model"]

        assert decode_foundation_text(metadata["model_family"]) == "sequence_transformer"
        assert decode_foundation_text(metadata["adapter_mode"]) == "precomputed"
        assert decode_foundation_text(metadata["artifact_id"]) == "nucleotide_transformer.v1"
        assert decode_foundation_text(metadata["preprocessing_version"]) == "bpe_v1"
        assert decode_foundation_text(metadata["pooling_strategy"]) == "mean"

    def test_load_dataset_embeddings_uses_shared_sequence_contract(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "nucleotide_transformer_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.array([[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]], dtype=np.float32),
            sequence_ids=np.array(["seq_c", "seq_a", "seq_b"]),
        )
        adapter = NucleotideTransformerPrecomputedAdapter(artifact_path=artifact_path)

        aligned = adapter.load_dataset_embeddings(
            reference_sequence_ids=["seq_a", "seq_b", "seq_c"],
            one_hot_sequences=np.zeros((3, 8, 4), dtype=np.float32),
        )

        np.testing.assert_allclose(
            np.asarray(aligned),
            np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float32),
        )


class TestProteinLMPrecomputedAdapter:
    """Tests for the protein-LM precomputed adapter."""

    def test_exposes_foundation_model_metadata(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "protein_lm_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.ones((2, 4), dtype=np.float32),
            sequence_ids=np.array(["protein_a", "protein_b"]),
        )
        adapter = ProteinLMPrecomputedAdapter(artifact_path=artifact_path)

        metadata = adapter.result_data()["foundation_model"]

        assert decode_foundation_text(metadata["model_family"]) == "sequence_transformer"
        assert decode_foundation_text(metadata["adapter_mode"]) == "precomputed"
        assert decode_foundation_text(metadata["artifact_id"]) == "protein_lm.v1"
        assert decode_foundation_text(metadata["preprocessing_version"]) == "protein_tokens_v1"
        assert decode_foundation_text(metadata["pooling_strategy"]) == "mean"

    def test_load_dataset_embeddings_uses_shared_sequence_contract(self, tmp_path: Path) -> None:
        artifact_path = tmp_path / "protein_lm_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.array([[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]], dtype=np.float32),
            sequence_ids=np.array(["protein_c", "protein_a", "protein_b"]),
        )
        adapter = ProteinLMPrecomputedAdapter(artifact_path=artifact_path)

        aligned = adapter.load_dataset_embeddings(
            reference_sequence_ids=["protein_a", "protein_b", "protein_c"],
            one_hot_sequences=np.zeros((3, 8, 20), dtype=np.float32),
        )

        np.testing.assert_allclose(
            np.asarray(aligned),
            np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float32),
        )


class TestSequencePrecomputedAdapter:
    """Tests for the shared sequence adapter benchmark contract."""

    def test_benchmark_metadata_is_stable_for_shared_sequence_contract(
        self,
        tmp_path: Path,
    ) -> None:
        artifact_path = tmp_path / "sequence_embeddings.npz"
        np.savez(
            artifact_path,
            embeddings=np.ones((2, 4), dtype=np.float32),
            sequence_ids=np.array(["seq_a", "seq_b"]),
        )
        adapter = SequencePrecomputedAdapter(
            artifact_path=artifact_path,
            artifact_spec=FoundationArtifactSpec(
                model_family=FoundationModelKind.SEQUENCE_TRANSFORMER,
                artifact_id="sequence.contract.v1",
                preprocessing_version="one_hot_v1",
                adapter_mode=AdapterMode.PRECOMPUTED,
                pooling_strategy=PoolingStrategy.MEAN,
            ),
            source_name="sequence_precomputed",
        )

        assert adapter.benchmark_metadata() == {
            "embedding_source": "external_artifact",
            "foundation_source_name": "sequence_precomputed",
        }
