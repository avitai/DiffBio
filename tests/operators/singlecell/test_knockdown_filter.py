"""Tests for OnTargetKnockdownFilter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("anndata")

from diffbio.operators.singlecell.knockdown_filter import (
    KnockdownFilterConfig,
    OnTargetKnockdownFilter,
    is_on_target_knockdown,
)
from diffbio.sources.perturbation.perturbation_source import (
    PerturbationAnnDataSource,
    PerturbationSourceConfig,
)
from tests.sources.perturbation.conftest import (
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
    """Create a source with synthetic knockdown data."""
    return PerturbationAnnDataSource(
        PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path), output_space="all"
        )
    )


class TestIsOnTargetKnockdown:
    """Tests for is_on_target_knockdown standalone function."""

    def test_strong_knockdown_detected(self) -> None:
        # Control mean = 5, perturbed mean = 0.5 -> ratio 0.1 < 0.3
        counts = np.array([5.0, 5.0, 5.0, 0.5, 0.5])
        ctrl_mask = np.array([True, True, True, False, False])
        pert_mask = np.array([False, False, False, True, True])
        assert is_on_target_knockdown(counts, pert_mask, ctrl_mask, 0.3) is True

    def test_no_knockdown(self) -> None:
        # Control mean = 5, perturbed mean = 4.5 -> ratio 0.9 > 0.3
        counts = np.array([5.0, 5.0, 5.0, 4.5, 4.5])
        ctrl_mask = np.array([True, True, True, False, False])
        pert_mask = np.array([False, False, False, True, True])
        assert is_on_target_knockdown(counts, pert_mask, ctrl_mask, 0.3) is False

    def test_zero_control_mean_returns_false(self) -> None:
        counts = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
        ctrl_mask = np.array([True, True, True, False, False])
        pert_mask = np.array([False, False, False, True, True])
        assert is_on_target_knockdown(counts, pert_mask, ctrl_mask, 0.3) is False


class TestOnTargetKnockdownFilter:
    """Tests for OnTargetKnockdownFilter."""

    def test_returns_boolean_mask(self, source: PerturbationAnnDataSource) -> None:
        config = KnockdownFilterConfig(
            residual_expression=0.30,
            cell_residual_expression=0.50,
            min_cells=5,
            var_gene_col="gene_name",
        )
        filt = OnTargetKnockdownFilter(config)
        mask = filt.process(source)
        assert mask.dtype == bool
        assert mask.shape == (N_TOTAL_CELLS,)

    def test_controls_always_preserved(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = KnockdownFilterConfig(
            residual_expression=0.30,
            cell_residual_expression=0.50,
            min_cells=5,
            var_gene_col="gene_name",
        )
        filt = OnTargetKnockdownFilter(config)
        mask = filt.process(source)
        ctrl_mask = source.get_control_mask()
        # All control cells should pass
        assert np.all(mask[ctrl_mask])

    def test_strict_filter_removes_more(
        self, source: PerturbationAnnDataSource
    ) -> None:
        lenient = KnockdownFilterConfig(
            residual_expression=0.90,
            cell_residual_expression=0.90,
            min_cells=1,
            var_gene_col="gene_name",
        )
        strict = KnockdownFilterConfig(
            residual_expression=0.10,
            cell_residual_expression=0.10,
            min_cells=1,
            var_gene_col="gene_name",
        )
        mask_lenient = OnTargetKnockdownFilter(lenient).process(source)
        mask_strict = OnTargetKnockdownFilter(strict).process(source)
        assert mask_strict.sum() <= mask_lenient.sum()
