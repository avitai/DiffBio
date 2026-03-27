"""Tests for ZeroShotSplitter and FewShotSplitter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("anndata")

from diffbio.sources.perturbation.perturbation_source import (
    PerturbationAnnDataSource,
    PerturbationSourceConfig,
)
from diffbio.splitters.perturbation import (
    FewShotSplitter,
    FewShotSplitterConfig,
    ZeroShotSplitter,
    ZeroShotSplitterConfig,
)
from tests.sources.perturbation.conftest import (
    CONTROL_PERT,
    N_TOTAL_CELLS,
    _build_synthetic_adata,
)


@pytest.fixture()
def synthetic_h5ad_path(tmp_path: Path) -> Path:
    """Write synthetic AnnData to a temp file."""
    rng = np.random.default_rng(42)
    adata = _build_synthetic_adata(rng)
    path = tmp_path / "test_dataset.h5ad"
    adata.write_h5ad(path)
    return path


@pytest.fixture()
def source(synthetic_h5ad_path: Path) -> PerturbationAnnDataSource:
    """Create a perturbation source."""
    return PerturbationAnnDataSource(
        PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path), output_space="all"
        )
    )


class TestZeroShotSplitter:
    """Tests for ZeroShotSplitter."""

    def test_held_out_types_in_test(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ZeroShotSplitterConfig(
            held_out_cell_types=("TypeA",),
            pert_col="perturbation",
            cell_type_col="cell_type",
        )
        splitter = ZeroShotSplitter(config)
        result = splitter.split(source)

        # All TypeA cells should be in test
        for idx in result.test_indices:
            elem = source[int(idx)]
            assert elem["cell_type_name"] == "TypeA"

    def test_remaining_in_train_val(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ZeroShotSplitterConfig(
            held_out_cell_types=("TypeA",),
            pert_col="perturbation",
            cell_type_col="cell_type",
        )
        splitter = ZeroShotSplitter(config)
        result = splitter.split(source)

        # Train + val should only have TypeB and TypeC
        for idx in np.concatenate([result.train_indices, result.valid_indices]):
            elem = source[int(idx)]
            assert elem["cell_type_name"] in {"TypeB", "TypeC"}

    def test_all_indices_covered(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ZeroShotSplitterConfig(
            held_out_cell_types=("TypeA",),
            pert_col="perturbation",
            cell_type_col="cell_type",
        )
        splitter = ZeroShotSplitter(config)
        result = splitter.split(source)

        total = len(result.train_indices) + len(result.valid_indices) + len(result.test_indices)
        assert total == N_TOTAL_CELLS

    def test_no_overlap(self, source: PerturbationAnnDataSource) -> None:
        config = ZeroShotSplitterConfig(
            held_out_cell_types=("TypeA",),
            pert_col="perturbation",
            cell_type_col="cell_type",
        )
        splitter = ZeroShotSplitter(config)
        result = splitter.split(source)

        train_set = set(np.array(result.train_indices))
        valid_set = set(np.array(result.valid_indices))
        test_set = set(np.array(result.test_indices))
        assert len(train_set & valid_set) == 0
        assert len(train_set & test_set) == 0
        assert len(valid_set & test_set) == 0


class TestFewShotSplitter:
    """Tests for FewShotSplitter."""

    def test_held_out_perts_in_test(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = FewShotSplitterConfig(
            held_out_perturbations=("GeneX",),
            pert_col="perturbation",
            cell_type_col="cell_type",
            control_pert=CONTROL_PERT,
        )
        splitter = FewShotSplitter(config)
        result = splitter.split(source)

        # All GeneX cells should be in test
        for idx in result.test_indices:
            elem = source[int(idx)]
            assert elem["pert_name"] == "GeneX"

    def test_controls_in_train(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = FewShotSplitterConfig(
            held_out_perturbations=("GeneX", "GeneY"),
            pert_col="perturbation",
            cell_type_col="cell_type",
            control_pert=CONTROL_PERT,
        )
        splitter = FewShotSplitter(config)
        result = splitter.split(source)

        # Control cells should be in train (not held out)
        ctrl_mask = source.get_control_mask()
        train_set = set(np.array(result.train_indices))
        ctrl_indices = set(np.where(ctrl_mask)[0])
        assert ctrl_indices.issubset(train_set)

    def test_all_indices_covered(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = FewShotSplitterConfig(
            held_out_perturbations=("GeneX",),
            pert_col="perturbation",
            cell_type_col="cell_type",
            control_pert=CONTROL_PERT,
        )
        splitter = FewShotSplitter(config)
        result = splitter.split(source)

        total = len(result.train_indices) + len(result.valid_indices) + len(result.test_indices)
        assert total == N_TOTAL_CELLS

    def test_no_overlap(self, source: PerturbationAnnDataSource) -> None:
        config = FewShotSplitterConfig(
            held_out_perturbations=("GeneX",),
            pert_col="perturbation",
            cell_type_col="cell_type",
            control_pert=CONTROL_PERT,
        )
        splitter = FewShotSplitter(config)
        result = splitter.split(source)

        train_set = set(np.array(result.train_indices))
        valid_set = set(np.array(result.valid_indices))
        test_set = set(np.array(result.test_indices))
        assert len(train_set & valid_set) == 0
        assert len(train_set & test_set) == 0
        assert len(valid_set & test_set) == 0
