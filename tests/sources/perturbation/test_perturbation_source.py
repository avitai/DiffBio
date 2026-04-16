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

    def test_embedding_output_space_requires_embedding_key(self, synthetic_h5ad_path: Path) -> None:
        with pytest.raises(ValueError, match="embedding_key"):
            PerturbationSourceConfig(
                file_path=str(synthetic_h5ad_path),
                output_space="embedding",
            )


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

    def test_additional_obs_passthrough(self, synthetic_adata, tmp_path: Path) -> None:
        synthetic_adata.obs["donor"] = ["donor_a"] * synthetic_adata.n_obs
        path = tmp_path / "with_donor.h5ad"
        synthetic_adata.write_h5ad(path)

        config = PerturbationSourceConfig(
            file_path=str(path),
            output_space="all",
            additional_obs=("donor",),
        )
        source = PerturbationAnnDataSource(config)
        element = source[0]
        assert element["donor"] == "donor_a"

    def test_missing_additional_obs_fails(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
            additional_obs=("donor",),
        )
        with pytest.raises(ValueError, match="additional_obs"):
            PerturbationAnnDataSource(config)

    def test_include_barcodes_requires_barcode_column(self, tmp_path: Path) -> None:
        from tests.sources.perturbation.conftest import _build_synthetic_adata

        adata = _build_synthetic_adata(np.random.default_rng(7), include_barcodes=False)
        path = tmp_path / "without_barcodes.h5ad"
        adata.write_h5ad(path)

        config = PerturbationSourceConfig(
            file_path=str(path),
            output_space="all",
            include_barcodes=True,
        )
        with pytest.raises(ValueError, match="barcode"):
            PerturbationAnnDataSource(config)

    def test_embedding_output_space(self, synthetic_adata, tmp_path: Path) -> None:
        synthetic_adata.obsm["X_alt"] = np.ones_like(synthetic_adata.obsm["X_hvg"])
        path = tmp_path / "with_multiple_embeddings.h5ad"
        synthetic_adata.write_h5ad(path)

        config = PerturbationSourceConfig(
            file_path=str(path),
            output_space="embedding",
            embedding_key="X_hvg",
        )
        source = PerturbationAnnDataSource(config)
        element = source[0]
        # In embedding mode, counts should be empty
        assert element["counts"].shape == (0,)
        # Only the requested embedding should be exposed
        assert set(element["obsm"]) == {"X_hvg"}

    def test_missing_embedding_key_fails(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="embedding",
            embedding_key="X_missing",
        )
        with pytest.raises(ValueError, match="embedding_key"):
            PerturbationAnnDataSource(config)

    def test_external_perturbation_embeddings_override_one_hot(
        self, synthetic_h5ad_path: Path, tmp_path: Path
    ) -> None:
        embedding_dim = 7
        external_embeddings = np.arange(len(ALL_PERTS) * embedding_dim, dtype=np.float32).reshape(
            len(ALL_PERTS), embedding_dim
        )
        embedding_path = tmp_path / "pert_embeddings.npy"
        np.save(embedding_path, external_embeddings)

        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
            perturbation_features_file=str(embedding_path),
        )
        source = PerturbationAnnDataSource(config)
        element = source[0]

        assert element["pert_emb"].shape == (embedding_dim,)
        np.testing.assert_allclose(
            np.asarray(element["pert_emb"]),
            external_embeddings[element["pert_code"]],
            atol=1e-6,
        )

    def test_external_perturbation_embeddings_row_mismatch_fails(
        self, synthetic_h5ad_path: Path, tmp_path: Path
    ) -> None:
        wrong_rows = np.zeros((len(ALL_PERTS) - 1, 7), dtype=np.float32)
        embedding_path = tmp_path / "wrong_pert_embeddings.npy"
        np.save(embedding_path, wrong_rows)

        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
            perturbation_features_file=str(embedding_path),
        )
        with pytest.raises(ValueError, match="perturbation_features_file"):
            PerturbationAnnDataSource(config)

    def test_should_yield_controls_false_skips_controls(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
            should_yield_controls=False,
        )
        source = PerturbationAnnDataSource(config)

        # __len__ should reflect only non-control cells
        n_controls = source.get_control_mask().sum()
        assert len(source) == N_TOTAL_CELLS - n_controls

        # Iteration should never yield a control cell
        for elem in source:
            assert elem["is_control"] is False
            break  # just check the first one

    def test_should_yield_controls_true_includes_controls(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
            should_yield_controls=True,
        )
        source = PerturbationAnnDataSource(config)
        assert len(source) == N_TOTAL_CELLS

    def test_should_yield_controls_false_getitem_remaps(self, synthetic_h5ad_path: Path) -> None:
        config = PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path),
            output_space="all",
            should_yield_controls=False,
        )
        source = PerturbationAnnDataSource(config)

        # Every index in [0, len(source)) should return a non-control
        for i in range(min(20, len(source))):
            assert source[i]["is_control"] is False
