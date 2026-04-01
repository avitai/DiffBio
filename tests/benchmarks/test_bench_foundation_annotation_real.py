"""Real-data smoke tests for the foundation annotation benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibrax.core.result import BenchmarkResult

from benchmarks.singlecell.bench_foundation_annotation import (
    SingleCellFoundationAnnotationBenchmark,
)
from tests.benchmarks.conftest import assert_valid_benchmark_result

_DATA_DIR = Path("/media/mahdi/ssd23/Data/scib")
_SKIP = not (_DATA_DIR / "Immune_ALL_human.h5ad").exists()


@pytest.mark.skipif(_SKIP, reason="Dataset not downloaded")
class TestSingleCellFoundationAnnotationBenchmarkRealData:
    """Smoke tests for the quick real-data benchmark path."""

    @pytest.fixture(scope="class")
    def result(self) -> BenchmarkResult:
        bench = SingleCellFoundationAnnotationBenchmark(quick=True)
        return bench.run()

    def test_standard_contract(self, result: BenchmarkResult) -> None:
        assert_valid_benchmark_result(
            result,
            expected_name="singlecell/foundation_annotation",
            required_metric_keys=["accuracy", "macro_f1", "train_loss"],
        )

    def test_metadata_declares_suite_contract(self, result: BenchmarkResult) -> None:
        assert result.tags["task"] == "cell_annotation"
        assert result.metadata["baseline_families"] == [
            "diffbio_native",
            "geneformer_precomputed",
            "scgpt_precomputed",
        ]
        assert (
            result.metadata["suite_scenarios"]["batch_correction"] == "singlecell/batch_correction"
        )
        assert "cell_ids" in result.metadata["dataset_contract_keys"]
