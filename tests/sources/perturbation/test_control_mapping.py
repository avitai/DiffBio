"""Tests for control cell mapping strategies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("anndata")

from diffbio.sources.perturbation.control_mapping import (
    BatchControlMapping,
    ControlMappingConfig,
    RandomControlMapping,
)
from diffbio.sources.perturbation.perturbation_source import (
    PerturbationAnnDataSource,
    PerturbationSourceConfig,
)



@pytest.fixture()
def source(synthetic_h5ad_path: Path) -> PerturbationAnnDataSource:
    """Create a perturbation source."""
    return PerturbationAnnDataSource(
        PerturbationSourceConfig(
            file_path=str(synthetic_h5ad_path), output_space="all"
        )
    )


class TestRandomControlMapping:
    """Tests for RandomControlMapping."""

    def test_mapping_shape(self, source: PerturbationAnnDataSource) -> None:
        config = ControlMappingConfig(n_basal_samples=1, seed=42)
        mapper = RandomControlMapping(config)
        mapping = mapper.build_mapping(source)
        n_pert = (~source.get_control_mask()).sum()
        assert mapping.shape == (n_pert, 1)

    def test_controls_are_valid_indices(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ControlMappingConfig(n_basal_samples=1, seed=42)
        mapper = RandomControlMapping(config)
        mapping = mapper.build_mapping(source)
        ctrl_mask = source.get_control_mask()
        # All mapped indices should be control cells
        for ctrl_idx in mapping.flat:
            assert ctrl_mask[ctrl_idx], f"Index {ctrl_idx} is not a control"

    def test_controls_same_cell_type(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ControlMappingConfig(n_basal_samples=1, seed=42)
        mapper = RandomControlMapping(config)
        mapping = mapper.build_mapping(source)
        ct_codes = source.get_cell_type_codes()
        ctrl_mask = source.get_control_mask()
        pert_indices = np.where(~ctrl_mask)[0]
        for i, pert_idx in enumerate(pert_indices):
            ctrl_idx = mapping[i, 0]
            assert ct_codes[pert_idx] == ct_codes[ctrl_idx]

    def test_n_basal_samples(self, source: PerturbationAnnDataSource) -> None:
        config = ControlMappingConfig(n_basal_samples=3, seed=42)
        mapper = RandomControlMapping(config)
        mapping = mapper.build_mapping(source)
        n_pert = (~source.get_control_mask()).sum()
        assert mapping.shape == (n_pert, 3)

    def test_deterministic(self, source: PerturbationAnnDataSource) -> None:
        config = ControlMappingConfig(n_basal_samples=1, seed=42)
        m1 = RandomControlMapping(config).build_mapping(source)
        m2 = RandomControlMapping(config).build_mapping(source)
        np.testing.assert_array_equal(m1, m2)


class TestBatchControlMapping:
    """Tests for BatchControlMapping."""

    def test_mapping_shape(self, source: PerturbationAnnDataSource) -> None:
        config = ControlMappingConfig(strategy="batch", n_basal_samples=1, seed=42)
        mapper = BatchControlMapping(config)
        mapping = mapper.build_mapping(source)
        n_pert = (~source.get_control_mask()).sum()
        assert mapping.shape == (n_pert, 1)

    def test_controls_are_valid(self, source: PerturbationAnnDataSource) -> None:
        config = ControlMappingConfig(strategy="batch", n_basal_samples=1, seed=42)
        mapper = BatchControlMapping(config)
        mapping = mapper.build_mapping(source)
        ctrl_mask = source.get_control_mask()
        for ctrl_idx in mapping.flat:
            assert ctrl_mask[ctrl_idx]

    def test_controls_same_cell_type(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ControlMappingConfig(strategy="batch", n_basal_samples=1, seed=42)
        mapper = BatchControlMapping(config)
        mapping = mapper.build_mapping(source)
        ct_codes = source.get_cell_type_codes()
        ctrl_mask = source.get_control_mask()
        pert_indices = np.where(~ctrl_mask)[0]
        for i, pert_idx in enumerate(pert_indices):
            ctrl_idx = mapping[i, 0]
            assert ct_codes[pert_idx] == ct_codes[ctrl_idx]

    def test_prefers_same_batch(self, source: PerturbationAnnDataSource) -> None:
        config = ControlMappingConfig(strategy="batch", n_basal_samples=1, seed=42)
        mapper = BatchControlMapping(config)
        mapping = mapper.build_mapping(source)
        batch_codes = source.get_batch_codes()
        ctrl_mask = source.get_control_mask()
        pert_indices = np.where(~ctrl_mask)[0]

        same_batch_count = 0
        for i, pert_idx in enumerate(pert_indices):
            ctrl_idx = mapping[i, 0]
            if batch_codes[pert_idx] == batch_codes[ctrl_idx]:
                same_batch_count += 1

        # Most controls should be from the same batch
        assert same_batch_count > len(pert_indices) * 0.3


class TestMapControlsFlag:
    """Tests for the map_controls flag."""

    def test_map_controls_includes_control_rows(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ControlMappingConfig(
            n_basal_samples=1, seed=42, map_controls=True
        )
        mapper = RandomControlMapping(config)
        mapping = mapper.build_mapping(source)

        # With map_controls=True, mapping should cover ALL cells
        assert mapping.shape[0] == len(source)

    def test_map_controls_false_excludes_controls(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ControlMappingConfig(
            n_basal_samples=1, seed=42, map_controls=False
        )
        mapper = RandomControlMapping(config)
        mapping = mapper.build_mapping(source)

        # Without map_controls, only perturbed cells are mapped
        n_pert = (~source.get_control_mask()).sum()
        assert mapping.shape[0] == n_pert

    def test_map_controls_control_mapped_to_other_control(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ControlMappingConfig(
            n_basal_samples=1, seed=42, map_controls=True
        )
        mapper = RandomControlMapping(config)
        mapping = mapper.build_mapping(source)

        ctrl_mask = source.get_control_mask()
        ctrl_indices = np.where(ctrl_mask)[0]
        # Controls should be mapped to other controls
        for ctrl_idx in ctrl_indices[:10]:
            mapped = mapping[ctrl_idx, 0]
            assert ctrl_mask[mapped]

    def test_map_controls_batch_strategy(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ControlMappingConfig(
            strategy="batch", n_basal_samples=1, seed=42, map_controls=True
        )
        mapper = BatchControlMapping(config)
        mapping = mapper.build_mapping(source)
        assert mapping.shape[0] == len(source)


class TestCachePairsFlag:
    """Tests for the cache_pairs flag."""

    def test_cached_mapping_is_deterministic(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ControlMappingConfig(
            n_basal_samples=1, seed=42, cache_pairs=True
        )
        mapper = RandomControlMapping(config)
        m1 = mapper.build_mapping(source)
        m2 = mapper.build_mapping(source)
        # Cached: exact same object returned
        assert m1 is m2

    def test_uncached_returns_fresh(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ControlMappingConfig(
            n_basal_samples=1, seed=42, cache_pairs=False
        )
        mapper = RandomControlMapping(config)
        m1 = mapper.build_mapping(source)
        m2 = mapper.build_mapping(source)
        # Uncached: same values (same seed) but different objects
        np.testing.assert_array_equal(m1, m2)
        # They should be separate arrays
        assert m1 is not m2

    def test_cache_batch_strategy(
        self, source: PerturbationAnnDataSource
    ) -> None:
        config = ControlMappingConfig(
            strategy="batch", n_basal_samples=1, seed=42, cache_pairs=True
        )
        mapper = BatchControlMapping(config)
        m1 = mapper.build_mapping(source)
        m2 = mapper.build_mapping(source)
        assert m1 is m2
