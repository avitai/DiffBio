"""Integration tests for the full perturbation data workflow.

Tests the complete flow: load H5AD -> knockdown filter -> split -> sample
-> control mapping -> paired output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("anndata")

from diffbio.operators.singlecell.knockdown_filter import (
    KnockdownFilterConfig,
    OnTargetKnockdownFilter,
)
from diffbio.samplers.perturbation_sampler import (
    PerturbationBatchSampler,
    PerturbationSamplerConfig,
)
from diffbio.sources.perturbation.control_mapping import (
    ControlMappingConfig,
    RandomControlMapping,
)
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
from tests.sources.perturbation.conftest import _build_synthetic_adata


@pytest.fixture()
def h5ad_path(tmp_path: Path) -> Path:
    """Create synthetic H5AD file."""
    rng = np.random.default_rng(42)
    adata = _build_synthetic_adata(rng)
    path = tmp_path / "integration_test.h5ad"
    adata.write_h5ad(path)
    return path


class TestFullZeroShotWorkflow:
    """Integration test: load -> filter -> zero-shot split -> sample -> pair."""

    def test_end_to_end(self, h5ad_path: Path) -> None:
        # 1. Load source
        source = PerturbationAnnDataSource(
            PerturbationSourceConfig(
                file_path=str(h5ad_path),
                output_space="all",
            )
        )
        assert len(source) > 0

        # 2. Knockdown filter
        filt = OnTargetKnockdownFilter(
            KnockdownFilterConfig(
                residual_expression=0.50,
                cell_residual_expression=0.70,
                min_cells=5,
                var_gene_col="gene_name",
            )
        )
        mask = filt.process(source)
        assert mask.sum() > 0
        assert mask.sum() <= len(source)

        # 3. Zero-shot split
        splitter = ZeroShotSplitter(
            ZeroShotSplitterConfig(
                held_out_cell_types=("TypeA",),
                seed=42,
            )
        )
        split = splitter.split(source)

        # Verify held-out types in test
        for idx in split.test_indices[:10]:
            assert source[int(idx)]["cell_type_name"] == "TypeA"

        # 4. Control mapping on train set
        mapper = RandomControlMapping(
            ControlMappingConfig(n_basal_samples=1, seed=42)
        )
        mapping = mapper.build_mapping(source)

        # Verify mapping produces valid control indices
        ctrl_mask = source.get_control_mask()
        for ctrl_idx in mapping[:10, 0]:
            assert ctrl_mask[ctrl_idx]

        # 5. Batch sampler on train split
        train_group_codes = source.get_group_codes()[
            np.array(split.train_indices)
        ]
        sampler = PerturbationBatchSampler(
            PerturbationSamplerConfig(sentence_size=25, seed=42),
            train_group_codes,
        )
        batches = list(sampler)
        assert len(batches) > 0

        # 6. Verify a batch produces valid paired elements
        batch = batches[0]
        train_indices = np.array(split.train_indices)
        for local_idx in batch[:5]:
            global_idx = int(train_indices[local_idx])
            elem = source[global_idx]
            assert "counts" in elem
            assert "pert_emb" in elem
            assert "pert_name" in elem
            assert "cell_type_name" in elem


class TestFullFewShotWorkflow:
    """Integration test: load -> few-shot split -> sample."""

    def test_end_to_end(self, h5ad_path: Path) -> None:
        # 1. Load
        source = PerturbationAnnDataSource(
            PerturbationSourceConfig(
                file_path=str(h5ad_path),
                output_space="all",
            )
        )

        # 2. Few-shot split
        splitter = FewShotSplitter(
            FewShotSplitterConfig(
                held_out_perturbations=("GeneX", "GeneY"),
                control_pert="non-targeting",
                seed=42,
            )
        )
        split = splitter.split(source)

        # Verify held-out perts in test
        test_perts = {
            source[int(idx)]["pert_name"] for idx in split.test_indices
        }
        assert test_perts == {"GeneX", "GeneY"}

        # Verify controls in train
        ctrl_mask = source.get_control_mask()
        train_set = set(np.array(split.train_indices))
        for idx in np.where(ctrl_mask)[0]:
            assert idx in train_set

        # 3. Sampler
        train_codes = source.get_group_codes()[np.array(split.train_indices)]
        sampler = PerturbationBatchSampler(
            PerturbationSamplerConfig(sentence_size=25, seed=42),
            train_codes,
        )
        assert len(list(sampler)) > 0

        # Total coverage
        total = (
            len(split.train_indices)
            + len(split.valid_indices)
            + len(split.test_indices)
        )
        assert total == len(source)


class TestMultiDatasetWorkflow:
    """Integration test with multiple H5AD files."""

    def test_concat_source_workflow(self, tmp_path: Path) -> None:
        from diffbio.sources.perturbation.concat_source import (
            PerturbationConcatSource,
        )

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)
        adata1 = _build_synthetic_adata(rng1)
        adata2 = _build_synthetic_adata(rng2)

        path1 = tmp_path / "ds1.h5ad"
        path2 = tmp_path / "ds2.h5ad"
        adata1.write_h5ad(path1)
        adata2.write_h5ad(path2)

        s1 = PerturbationAnnDataSource(
            PerturbationSourceConfig(
                file_path=str(path1), output_space="all"
            )
        )
        s2 = PerturbationAnnDataSource(
            PerturbationSourceConfig(
                file_path=str(path2), output_space="all"
            )
        )
        concat = PerturbationConcatSource(sources=[s1, s2])

        # Basic checks
        assert len(concat) == len(s1) + len(s2)

        # Sampler on concatenated source
        group_codes = concat.get_group_codes()
        sampler = PerturbationBatchSampler(
            PerturbationSamplerConfig(sentence_size=50, seed=42),
            group_codes,
        )
        all_indices = [i for b in sampler for i in b]
        assert set(all_indices) == set(range(len(concat)))
