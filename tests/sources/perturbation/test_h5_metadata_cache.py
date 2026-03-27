"""Tests for H5MetadataCache and GlobalH5MetadataCache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.sources.perturbation.conftest import (
    ALL_PERTS,
    BATCHES,
    CELL_TYPES,
    CONTROL_PERT,
    N_TOTAL_CELLS,
)

pytest.importorskip("anndata")
pytest.importorskip("h5py")

from diffbio.sources.perturbation.h5_metadata_cache import (
    GlobalH5MetadataCache,
    H5MetadataCache,
)


class TestH5MetadataCache:
    """Tests for H5MetadataCache."""

    def test_loads_from_h5ad(self, synthetic_h5ad_path: Path) -> None:
        cache = H5MetadataCache(
            h5_path=str(synthetic_h5ad_path),
            pert_col="perturbation",
            cell_type_key="cell_type",
            control_pert=CONTROL_PERT,
            batch_col="batch",
        )
        assert cache.n_cells == N_TOTAL_CELLS

    def test_perturbation_categories(self, synthetic_h5ad_path: Path) -> None:
        cache = H5MetadataCache(
            h5_path=str(synthetic_h5ad_path),
            pert_col="perturbation",
            cell_type_key="cell_type",
            control_pert=CONTROL_PERT,
            batch_col="batch",
        )
        assert set(cache.pert_categories) == set(ALL_PERTS)

    def test_cell_type_categories(self, synthetic_h5ad_path: Path) -> None:
        cache = H5MetadataCache(
            h5_path=str(synthetic_h5ad_path),
            pert_col="perturbation",
            cell_type_key="cell_type",
            control_pert=CONTROL_PERT,
            batch_col="batch",
        )
        assert set(cache.cell_type_categories) == set(CELL_TYPES)

    def test_batch_categories(self, synthetic_h5ad_path: Path) -> None:
        cache = H5MetadataCache(
            h5_path=str(synthetic_h5ad_path),
            pert_col="perturbation",
            cell_type_key="cell_type",
            control_pert=CONTROL_PERT,
            batch_col="batch",
        )
        assert set(cache.batch_categories) == set(BATCHES)

    def test_control_mask(self, synthetic_h5ad_path: Path) -> None:
        cache = H5MetadataCache(
            h5_path=str(synthetic_h5ad_path),
            pert_col="perturbation",
            cell_type_key="cell_type",
            control_pert=CONTROL_PERT,
            batch_col="batch",
        )
        assert cache.control_mask.dtype == bool
        assert cache.control_mask.shape == (N_TOTAL_CELLS,)
        # 3 cell types * 50 cells each = 150 control cells
        assert cache.control_mask.sum() == len(CELL_TYPES) * 50

    def test_pert_codes_valid(self, synthetic_h5ad_path: Path) -> None:
        cache = H5MetadataCache(
            h5_path=str(synthetic_h5ad_path),
            pert_col="perturbation",
            cell_type_key="cell_type",
            control_pert=CONTROL_PERT,
            batch_col="batch",
        )
        assert cache.pert_codes.dtype == np.int32
        assert cache.pert_codes.shape == (N_TOTAL_CELLS,)
        assert np.all(cache.pert_codes >= 0)
        assert np.all(cache.pert_codes < len(ALL_PERTS))

    def test_get_pert_names(self, synthetic_h5ad_path: Path) -> None:
        cache = H5MetadataCache(
            h5_path=str(synthetic_h5ad_path),
            pert_col="perturbation",
            cell_type_key="cell_type",
            control_pert=CONTROL_PERT,
            batch_col="batch",
        )
        names = cache.get_pert_names(cache.pert_codes[:5])
        assert len(names) == 5
        assert all(isinstance(n, str) for n in names)

    def test_invalid_control_pert_raises(self, synthetic_h5ad_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            H5MetadataCache(
                h5_path=str(synthetic_h5ad_path),
                pert_col="perturbation",
                cell_type_key="cell_type",
                control_pert="NONEXISTENT",
                batch_col="batch",
            )


class TestGlobalH5MetadataCache:
    """Tests for GlobalH5MetadataCache singleton."""

    def test_singleton_returns_same_instance(self) -> None:
        cache1 = GlobalH5MetadataCache()
        cache2 = GlobalH5MetadataCache()
        assert cache1 is cache2

    def test_get_cache_returns_cached_instance(
        self, synthetic_h5ad_path: Path
    ) -> None:
        global_cache = GlobalH5MetadataCache()
        c1 = global_cache.get_cache(
            str(synthetic_h5ad_path),
            pert_col="perturbation",
            cell_type_key="cell_type",
            control_pert=CONTROL_PERT,
            batch_col="batch",
        )
        c2 = global_cache.get_cache(
            str(synthetic_h5ad_path),
            pert_col="perturbation",
            cell_type_key="cell_type",
            control_pert=CONTROL_PERT,
            batch_col="batch",
        )
        assert c1 is c2

    def test_different_paths_give_different_caches(
        self, synthetic_h5ad_pair: tuple[Path, Path]
    ) -> None:
        path1, path2 = synthetic_h5ad_pair
        global_cache = GlobalH5MetadataCache()
        c1 = global_cache.get_cache(
            str(path1),
            pert_col="perturbation",
            cell_type_key="cell_type",
            control_pert=CONTROL_PERT,
            batch_col="batch",
        )
        c2 = global_cache.get_cache(
            str(path2),
            pert_col="perturbation",
            cell_type_key="cell_type",
            control_pert=CONTROL_PERT,
            batch_col="batch",
        )
        assert c1 is not c2
