"""Tests for benchmarks.multiomics.bench_spatial_deconv."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.multiomics.bench_spatial_deconv import (
    SpatialDeconvBenchmark,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Data/spatial")
_SKIP = not (_DATA_DIR / "seqfish_cortex.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="seqFISH dataset not downloaded")
class TestSpatialDeconvBenchmark:
    """Tests for the spatial deconvolution benchmark."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        """Run benchmark once for the entire test class."""
        bench = SpatialDeconvBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        """Verify the full standard benchmark result contract."""
        assert_valid_benchmark_result(
            result,
            expected_name="multiomics/spatial_deconvolution",
            required_metric_keys=[
                "pearson_correlation",
                "rmse",
                "proportion_sum_to_one",
            ],
        )

    def test_pearson_in_range(self, result: BenchmarkResult) -> None:
        """Pearson correlation should be between -1 and 1."""
        value = result.metrics["pearson_correlation"].value
        assert -1.0 <= value <= 1.0

    def test_rmse_non_negative(self, result: BenchmarkResult) -> None:
        """RMSE must be non-negative."""
        value = result.metrics["rmse"].value
        assert value >= 0.0

    def test_proportion_sum_close_to_one(self, result: BenchmarkResult) -> None:
        """Softmax proportions should sum to ~1.0 per spot."""
        value = result.metrics["proportion_sum_to_one"].value
        assert value > 0.99

    def test_operator_tag(self, result: BenchmarkResult) -> None:
        """Operator tag must identify SpatialDeconvolution."""
        assert "SpatialDeconvolution" in result.tags["operator"]

    def test_dataset_tag(self, result: BenchmarkResult) -> None:
        """Dataset tag must reference seqFISH cortex."""
        assert result.tags["dataset"] == "seqfish_cortex"

    def test_has_spot_count_in_metadata(self, result: BenchmarkResult) -> None:
        """Metadata should record the number of spots."""
        info = result.metadata["dataset_info"]
        assert info["n_spots"] > 0

    def test_has_reference_spatial_split(self, result: BenchmarkResult) -> None:
        """Metadata should record reference/spatial split."""
        info = result.metadata["dataset_info"]
        assert info["n_reference"] > 0
        assert info["n_spatial"] > 0
        assert info["n_reference"] > info["n_spatial"]

    def test_records_multiomics_provenance_and_contract(self, result: BenchmarkResult) -> None:
        """Benchmark metadata should expose canonical multi-omics provenance."""
        assert result.metadata["dataset_provenance"] == {
            "dataset_name": "seqfish_cortex",
            "source_type": "curated_spatial_transcriptomics",
            "modalities": ["rna", "spatial"],
            "curation_status": "download_required_local_h5ad",
            "biological_validation": "published_benchmark_dataset",
            "promotion_eligible": True,
            "source_path": str(_DATA_DIR / "seqfish_cortex.h5ad"),
        }
        assert result.metadata["modality_contract"] == {
            "modalities": ["rna", "spatial"],
            "primary_modality": "rna",
            "spatial_key": "spatial",
            "label_key": "celltype_mapped_refined",
            "count_key": "X",
        }
        assert result.metadata["multiomics_scope"] == {
            "promoted_task": "spatial_deconvolution",
            "stable_scope": "benchmark_backed",
            "scope_exclusions": [
                "imported_multiomics_foundation_checkpoint_loading",
                "stable_metabolomics_benchmark_promotion",
            ],
        }

    def test_records_multiomics_artifact_metadata(self, result: BenchmarkResult) -> None:
        """Operator output artifacts should use the shared artifact metadata contract."""
        assert result.metadata["multiomics_artifact"] == {
            "artifact_id": "diffbio.spatial_deconvolution.proportions",
            "artifact_type": "operator_output",
            "modalities": ["rna", "spatial"],
            "embedding_source": "in_process_operator",
            "foundation_source_name": "SpatialDeconvolution",
            "promotion_eligible": True,
        }


def test_multiomics_docs_distinguish_verified_and_planned_scope() -> None:
    """Docs should separate benchmark-backed multi-omics from planned metabolomics."""
    multiomics_doc = Path("docs/user-guide/operators/multiomics.md").read_text(encoding="utf-8")
    metabolomics_doc = Path("docs/user-guide/operators/metabolomics.md").read_text(encoding="utf-8")

    assert "Multi-omics benchmark scope" in multiomics_doc
    assert "seqFISH spatial deconvolution" in multiomics_doc
    assert "not stable imported multi-omics foundation-model support" in multiomics_doc
    assert "Metabolomics expansion scope" in metabolomics_doc
    assert "not yet benchmark-promoted" in metabolomics_doc
