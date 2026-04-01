"""Tests for deterministic DTI sources and paired-input contracts."""

from __future__ import annotations

import numpy as np
import pytest


class TestDTISourceContract:
    """Tests for the shared DTI source contract."""

    def test_imports_and_contract_keys(self) -> None:
        from diffbio.sources import (
            DTI_DATASET_CONTRACT_KEYS,
            BioSNAPDTISource,
            DTISourceConfig,
            DavisDTISource,
        )

        assert DTISourceConfig is not None
        assert DavisDTISource is not None
        assert BioSNAPDTISource is not None
        assert DTI_DATASET_CONTRACT_KEYS == (
            "pair_ids",
            "protein_ids",
            "protein_sequences",
            "drug_ids",
            "drug_smiles",
            "targets",
            "task_type",
        )

    def test_fallback_sources_load_deterministically(self, tmp_path) -> None:
        from diffbio.sources import BioSNAPDTISource, DTISourceConfig, DavisDTISource

        davis_config = DTISourceConfig(dataset_name="davis", split="train", data_dir=tmp_path)
        biosnap_config = DTISourceConfig(dataset_name="biosnap", split="train", data_dir=tmp_path)

        davis_a = DavisDTISource(davis_config).load()
        davis_b = DavisDTISource(davis_config).load()
        biosnap_a = BioSNAPDTISource(biosnap_config).load()
        biosnap_b = BioSNAPDTISource(biosnap_config).load()

        assert davis_a["pair_ids"] == davis_b["pair_ids"]
        assert davis_a["protein_sequences"] == davis_b["protein_sequences"]
        np.testing.assert_allclose(np.asarray(davis_a["targets"]), np.asarray(davis_b["targets"]))

        assert biosnap_a["pair_ids"] == biosnap_b["pair_ids"]
        assert biosnap_a["drug_smiles"] == biosnap_b["drug_smiles"]
        np.testing.assert_array_equal(
            np.asarray(biosnap_a["targets"]), np.asarray(biosnap_b["targets"])
        )

    def test_split_is_deterministic_and_exhaustive(self, tmp_path) -> None:
        from diffbio.sources import DTISourceConfig, DavisDTISource

        split_names = ("train", "valid", "test")
        split_pair_ids: list[set[str]] = []

        for split_name in split_names:
            data = DavisDTISource(
                DTISourceConfig(
                    dataset_name="davis",
                    split=split_name,
                    data_dir=tmp_path,
                )
            ).load()
            split_pair_ids.append(set(data["pair_ids"]))

        assert split_pair_ids[0].isdisjoint(split_pair_ids[1])
        assert split_pair_ids[0].isdisjoint(split_pair_ids[2])
        assert split_pair_ids[1].isdisjoint(split_pair_ids[2])
        assert sum(len(pair_ids) for pair_ids in split_pair_ids) == 12

    def test_build_paired_batch_preserves_alignment(self, tmp_path) -> None:
        from diffbio.sources import DTISourceConfig, DavisDTISource, build_paired_dti_batch

        data = DavisDTISource(
            DTISourceConfig(dataset_name="davis", split="train", data_dir=tmp_path)
        ).load()
        batch = build_paired_dti_batch(data, indices=np.array([0, 1], dtype=np.int32))

        assert batch["pair_ids"] == data["pair_ids"][:2]
        assert batch["protein_ids"] == data["protein_ids"][:2]
        assert batch["drug_ids"] == data["drug_ids"][:2]
        np.testing.assert_allclose(np.asarray(batch["targets"]), np.asarray(data["targets"])[:2])

    def test_validate_rejects_misaligned_lengths(self) -> None:
        from diffbio.sources.dti import validate_dti_dataset

        bad = {
            "pair_ids": ["pair_0", "pair_1"],
            "protein_ids": ["P0"],
            "protein_sequences": ["MAAA", "MCCC"],
            "drug_ids": ["D0", "D1"],
            "drug_smiles": ["CCO", "CCC"],
            "targets": np.array([1.0, 2.0], dtype=np.float32),
            "task_type": "affinity_regression",
        }

        with pytest.raises(ValueError, match="same length"):
            validate_dti_dataset(bad)
