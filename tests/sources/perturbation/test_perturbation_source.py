"""Tests for PerturbationAnnDataSource."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("anndata")

from datarax.core.data_source import DataSourceModule

from diffbio.sources.perturbation.perturbation_source import (
    PerturbationAnnDataSource,
    PerturbationSourceConfig,
)
from tests.sources.perturbation.conftest import (
    ALL_PERTS,
    CELL_TYPES,
    N_GENES,
    N_TOTAL_CELLS,
)


class TestPerturbationSourceConfig:
    """Tests for PerturbationSourceConfig."""

    def test_defaults(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(file_path=str(synthetic_h5ad_path))
        assert config.pert_col == "perturbation"
        assert config.control_pert == "non-targeting"
        assert config.output_space == "gene"

    def test_file_path_required(self) -> None:
        with pytest.raises(ValueError, match="file_path"):
            PerturbationSourceConfig()


class TestPerturbationAnnDataSource:
    """Tests for PerturbationAnnDataSource."""

    def test_isinstance_data_source_module(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        assert isinstance(source, DataSourceModule)

    def test_length(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        assert len(source) == N_TOTAL_CELLS

    def test_getitem_returns_dict(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        element = source[0]
        assert isinstance(element, dict)
        assert "counts" in element
        assert "pert_code" in element
        assert "cell_type_code" in element
        assert "batch_code" in element
        assert "is_control" in element
        assert "pert_emb" in element
        assert "pert_name" in element

    def test_counts_shape_all_mode(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        element = source[0]
        assert element["counts"].shape == (N_GENES,)

    def test_counts_shape_gene_mode(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="gene",
            hvg_col="is_hvg",
        )
        source = PerturbationAnnDataSource(config)
        element = source[0]
        # 20% of 100 genes = 20 HVGs
        assert element["counts"].shape == (20,)

    def test_pert_codes_valid_range(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        codes = source.get_pert_codes()
        assert codes.dtype == np.int32
        assert np.all(codes >= 0)
        assert np.all(codes < len(ALL_PERTS))

    def test_control_mask(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        mask = source.get_control_mask()
        assert mask.dtype == bool
        assert mask.shape == (N_TOTAL_CELLS,)
        # 3 cell types * 50 = 150 control cells
        assert mask.sum() == len(CELL_TYPES) * 50

    def test_cell_type_categories(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        assert set(source.get_cell_type_categories()) == set(CELL_TYPES)

    def test_pert_categories(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        assert set(source.get_pert_categories()) == set(ALL_PERTS)

    def test_onehot_map(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        onehot = source.get_onehot_map()
        assert len(onehot) == len(ALL_PERTS)
        for key, vec in onehot.items():
            assert vec.shape == (len(ALL_PERTS),)
            assert float(vec.sum()) == pytest.approx(1.0)

    def test_group_codes(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        codes = source.get_group_codes()
        assert codes.shape == (N_TOTAL_CELLS,)
        # Number of unique groups = n_cell_types * n_perts
        n_unique = len(np.unique(codes))
        assert n_unique == len(CELL_TYPES) * len(ALL_PERTS)

    def test_pert_emb_is_onehot(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        element = source[0]
        emb = element["pert_emb"]
        assert emb.shape == (len(ALL_PERTS),)
        assert float(emb.sum()) == pytest.approx(1.0)

    def test_is_control_flag(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        # Find a control cell
        mask = source.get_control_mask()
        ctrl_idx = int(np.where(mask)[0][0])
        assert source[ctrl_idx]["is_control"] is True

        # Find a non-control cell
        non_ctrl_idx = int(np.where(~mask)[0][0])
        assert source[non_ctrl_idx]["is_control"] is False

    def test_get_gene_names(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        names = source.get_gene_names()
        assert len(names) == N_GENES

    def test_get_var_dims(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
        )
        source = PerturbationAnnDataSource(config)
        dims = source.get_var_dims()
        assert dims["n_genes"] == N_GENES
        assert dims["n_perts"] == len(ALL_PERTS)

    def test_additional_obs(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
            include_barcodes=True,
        )
        source = PerturbationAnnDataSource(config)
        element = source[0]
        assert "barcode" in element

    def test_embedding_output_space(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="embedding",
            embedding_key="X_hvg",
        )
        source = PerturbationAnnDataSource(config)
        element = source[0]
        # In embedding mode, counts should be empty
        assert element["counts"].shape == (0,)
        # But obsm embedding should be available
        assert "obsm" in element
