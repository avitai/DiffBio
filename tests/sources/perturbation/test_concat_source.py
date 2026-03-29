"""Tests for PerturbationConcatSource."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("anndata")

from diffbio.sources.perturbation.concat_source import PerturbationConcatSource
from diffbio.sources.perturbation.perturbation_source import (
    PerturbationAnnDataSource,
    PerturbationSourceConfig,
)
from tests.sources.perturbation.conftest import N_TOTAL_CELLS


class TestPerturbationConcatSource:
    """Tests for PerturbationConcatSource."""

    def test_length_is_sum(self, synthetic_h5ad_pair: tuple[Path, Path]) -> None:
        path1, path2 = synthetic_h5ad_pair
        s1 = PerturbationAnnDataSource(
            PerturbationSourceConfig(file_path=str(path1), output_space="all")
        )
        s2 = PerturbationAnnDataSource(
            PerturbationSourceConfig(file_path=str(path2), output_space="all")
        )
        concat = PerturbationConcatSource(sources=[s1, s2])
        assert len(concat) == N_TOTAL_CELLS * 2

    def test_getitem_first_source(self, synthetic_h5ad_pair: tuple[Path, Path]) -> None:
        path1, path2 = synthetic_h5ad_pair
        s1 = PerturbationAnnDataSource(
            PerturbationSourceConfig(file_path=str(path1), output_space="all")
        )
        s2 = PerturbationAnnDataSource(
            PerturbationSourceConfig(file_path=str(path2), output_space="all")
        )
        concat = PerturbationConcatSource(sources=[s1, s2])
        elem_direct = s1[0]
        elem_concat = concat[0]
        assert elem_direct["pert_name"] == elem_concat["pert_name"]

    def test_getitem_second_source(self, synthetic_h5ad_pair: tuple[Path, Path]) -> None:
        path1, path2 = synthetic_h5ad_pair
        s1 = PerturbationAnnDataSource(
            PerturbationSourceConfig(file_path=str(path1), output_space="all")
        )
        s2 = PerturbationAnnDataSource(
            PerturbationSourceConfig(file_path=str(path2), output_space="all")
        )
        concat = PerturbationConcatSource(sources=[s1, s2])
        offset = len(s1)
        elem_direct = s2[0]
        elem_concat = concat[offset]
        assert elem_direct["pert_name"] == elem_concat["pert_name"]

    def test_get_control_mask_concatenated(self, synthetic_h5ad_pair: tuple[Path, Path]) -> None:
        path1, path2 = synthetic_h5ad_pair
        s1 = PerturbationAnnDataSource(
            PerturbationSourceConfig(file_path=str(path1), output_space="all")
        )
        s2 = PerturbationAnnDataSource(
            PerturbationSourceConfig(file_path=str(path2), output_space="all")
        )
        concat = PerturbationConcatSource(sources=[s1, s2])
        mask = concat.get_control_mask()
        assert mask.shape == (N_TOTAL_CELLS * 2,)
        expected = s1.get_control_mask().sum() + s2.get_control_mask().sum()
        assert mask.sum() == expected

    def test_get_group_codes_concatenated(self, synthetic_h5ad_pair: tuple[Path, Path]) -> None:
        path1, path2 = synthetic_h5ad_pair
        s1 = PerturbationAnnDataSource(
            PerturbationSourceConfig(file_path=str(path1), output_space="all")
        )
        s2 = PerturbationAnnDataSource(
            PerturbationSourceConfig(file_path=str(path2), output_space="all")
        )
        concat = PerturbationConcatSource(sources=[s1, s2])
        codes = concat.get_group_codes()
        assert codes.shape == (N_TOTAL_CELLS * 2,)

    def test_index_out_of_range_raises(self, synthetic_h5ad_pair: tuple[Path, Path]) -> None:
        path1, path2 = synthetic_h5ad_pair
        s1 = PerturbationAnnDataSource(
            PerturbationSourceConfig(file_path=str(path1), output_space="all")
        )
        concat = PerturbationConcatSource(sources=[s1])
        with pytest.raises(IndexError):
            concat[N_TOTAL_CELLS + 1]

    def test_empty_sources_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            PerturbationConcatSource(sources=[])
