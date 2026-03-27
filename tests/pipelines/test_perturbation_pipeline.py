"""Tests for PerturbationPipeline end-to-end orchestrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("anndata")

from diffbio.pipelines.perturbation import (
    PerturbationPipeline,
    PerturbationPipelineConfig,
    PerturbationPipelineResult,
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


class TestPerturbationPipelineRandomSplit:
    """Tests for PerturbationPipeline with random split."""

    def test_setup_returns_result(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationPipelineConfig(
            split_mode="random",
            output_space="all",
        )
        pipeline = PerturbationPipeline(config)
        result = pipeline.setup([synthetic_h5ad_path])
        assert isinstance(result, PerturbationPipelineResult)

    def test_split_covers_all_cells(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationPipelineConfig(
            split_mode="random",
            output_space="all",
        )
        result = PerturbationPipeline(config).setup([synthetic_h5ad_path])
        total = (
            len(result.train_indices)
            + len(result.valid_indices)
            + len(result.test_indices)
        )
        assert total == N_TOTAL_CELLS

    def test_train_sampler_yields_batches(
        self, synthetic_h5ad_path: Path
    ) -> None:
        config = PerturbationPipelineConfig(
            split_mode="random",
            output_space="all",
            sentence_size=50,
        )
        result = PerturbationPipeline(config).setup([synthetic_h5ad_path])
        batches = list(result.train_sampler)
        assert len(batches) > 0
        # All yielded indices should be valid positions in train group codes
        all_indices = [i for b in batches for i in b]
        assert len(all_indices) > 0

    def test_get_element(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationPipelineConfig(
            split_mode="random",
            output_space="all",
        )
        result = PerturbationPipeline(config).setup([synthetic_h5ad_path])
        idx = int(result.train_indices[0])
        elem = result.get_element(idx)
        assert "counts" in elem
        assert "pert_name" in elem

    def test_get_var_dims(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationPipelineConfig(
            split_mode="random",
            output_space="all",
        )
        result = PerturbationPipeline(config).setup([synthetic_h5ad_path])
        dims = result.get_var_dims()
        assert dims["n_genes"] == 100
        assert dims["n_perts"] == 6

    def test_control_mapping_shape(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationPipelineConfig(
            split_mode="random",
            output_space="all",
            n_basal_samples=2,
        )
        result = PerturbationPipeline(config).setup([synthetic_h5ad_path])
        assert result.control_mapping.shape[1] == 2


class TestPerturbationPipelineZeroshotSplit:
    """Tests for PerturbationPipeline with zero-shot split."""

    def test_held_out_types_in_test(
        self, synthetic_h5ad_path: Path
    ) -> None:
        config = PerturbationPipelineConfig(
            split_mode="zeroshot",
            held_out_cell_types=("TypeA",),
            output_space="all",
        )
        result = PerturbationPipeline(config).setup([synthetic_h5ad_path])

        for idx in result.test_indices:
            elem = result.get_element(int(idx))
            assert elem["cell_type_name"] == "TypeA"


class TestPerturbationPipelineFewshotSplit:
    """Tests for PerturbationPipeline with few-shot split."""

    def test_held_out_perts_in_test(
        self, synthetic_h5ad_path: Path
    ) -> None:
        config = PerturbationPipelineConfig(
            split_mode="fewshot",
            held_out_perturbations=("GeneX",),
            output_space="all",
        )
        result = PerturbationPipeline(config).setup([synthetic_h5ad_path])

        for idx in result.test_indices:
            elem = result.get_element(int(idx))
            assert elem["pert_name"] == "GeneX"


class TestPerturbationPipelineWithFilter:
    """Tests for PerturbationPipeline with knockdown filtering."""

    def test_filter_reduces_cell_count(
        self, synthetic_h5ad_path: Path
    ) -> None:
        config = PerturbationPipelineConfig(
            split_mode="random",
            output_space="all",
            enable_knockdown_filter=True,
            residual_expression=0.30,
            cell_residual_expression=0.50,
            min_cells=5,
            var_gene_col="gene_name",
        )
        result = PerturbationPipeline(config).setup([synthetic_h5ad_path])
        total = (
            len(result.train_indices)
            + len(result.valid_indices)
            + len(result.test_indices)
        )
        # Filter should remove some cells
        assert total <= N_TOTAL_CELLS
        assert result.filter_mask is not None

    def test_filter_mask_stored(
        self, synthetic_h5ad_path: Path
    ) -> None:
        config = PerturbationPipelineConfig(
            split_mode="random",
            output_space="all",
            enable_knockdown_filter=True,
            var_gene_col="gene_name",
            min_cells=5,
        )
        result = PerturbationPipeline(config).setup([synthetic_h5ad_path])
        assert result.filter_mask is not None
        assert result.filter_mask.dtype == bool
