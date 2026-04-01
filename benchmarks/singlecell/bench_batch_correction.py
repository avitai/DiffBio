#!/usr/bin/env python3
"""Batch correction benchmark: DifferentiableHarmony on immune_human.

Evaluates DiffBio's DifferentiableHarmony operator on the scib
immune human integration benchmark (33,506 cells, 10 batches,
16 cell types) using the full scib-metrics suite.

Results are compared against published baselines from Luecken et al.
2022: scVI, Harmony (R), Scanorama, BBKNN, and Unintegrated.

Usage:
    python benchmarks/singlecell/bench_batch_correction.py
    python benchmarks/singlecell/bench_batch_correction.py --quick
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.scib import INTEGRATION_BASELINES
from benchmarks._metrics.scib_bridge import evaluate_integration
from diffbio.operators.foundation_models import SingleCellPrecomputedAdapter
from diffbio.operators.singlecell import (
    BatchCorrectionConfig,
    DifferentiableHarmony,
)
from diffbio.sources.immune_human import (
    ImmuneHumanConfig,
    ImmuneHumanSource,
)

logger = logging.getLogger(__name__)

_CONFIG = DiffBioBenchmarkConfig(
    name="singlecell/batch_correction",
    domain="singlecell",
    quick_subsample=2000,
)


class _SingleCellSource(Protocol):
    """Protocol for benchmark data loading."""

    def load(self) -> dict[str, Any]:
        """Load the dataset payload for the benchmark."""
        ...


class BatchCorrectionBenchmark(DiffBioBenchmark):
    """Evaluate DifferentiableHarmony on immune_human dataset."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = "/media/mahdi/ssd23/Data/scib",
        source_factory: Callable[[int | None], _SingleCellSource] | None = None,
        embedding_adapter: SingleCellPrecomputedAdapter | None = None,
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)
        self._source_factory = source_factory or self._default_source_factory
        self.embedding_adapter = embedding_adapter

    def _default_source_factory(self, subsample: int | None) -> ImmuneHumanSource:
        """Create the default immune_human source."""
        return ImmuneHumanSource(ImmuneHumanConfig(data_dir=self.data_dir, subsample=subsample))

    def _run_core(self) -> dict[str, Any]:
        """Load immune_human, run Harmony, compute scib-metrics."""
        subsample = self.config.quick_subsample if self.quick else None

        # 1. Load dataset
        logger.info("Loading immune_human dataset...")
        source = self._source_factory(subsample)
        data = source.load()
        n_cells = data["n_cells"]
        n_batches = data["n_batches"]
        if self.embedding_adapter is None:
            embeddings = data["embeddings"]
            embedding_source = "dataset_embeddings"
            result_data: dict[str, Any] = {}
        else:
            embeddings = self.embedding_adapter.load_aligned_embeddings(
                reference_cell_ids=data["cell_ids"],
            )
            embedding_source = "external_artifact"
            result_data = self.embedding_adapter.result_data()
        batch_labels = data["batch_labels"]
        cell_type_labels = data["cell_type_labels"]

        logger.info(
            "  %d cells, %d genes, %d batches, %d types",
            n_cells,
            data["n_genes"],
            n_batches,
            data["n_types"],
        )

        # 2. Create and run operator
        n_features = embeddings.shape[1]
        op_config = BatchCorrectionConfig(
            n_clusters=min(20, n_cells // 10),
            n_features=n_features,
            n_batches=n_batches,
            temperature=1.0,
        )
        rngs = nnx.Rngs(42)
        operator = DifferentiableHarmony(op_config, rngs=rngs)

        input_data = {
            "embeddings": embeddings,
            "batch_labels": jnp.array(batch_labels),
        }
        result, _, _ = operator.apply(input_data, {}, None)
        corrected = result["corrected_embeddings"]

        # 3. Compute scib-metrics
        logger.info("Computing scib-metrics...")
        quality = evaluate_integration(
            corrected_embeddings=np.asarray(corrected),
            labels=np.asarray(cell_type_labels),
            batch=np.asarray(batch_labels),
        )
        for key, value in sorted(quality.items()):
            logger.info("  %s: %.4f", key, value)

        # Loss function for gradient check
        def loss_fn(model: DifferentiableHarmony, d: dict[str, Any]) -> jnp.ndarray:
            res, _, _ = model.apply(d, {}, None)
            return jnp.sum(res["corrected_embeddings"])

        baselines = INTEGRATION_BASELINES.get("immune_human", {})

        return {
            "metrics": quality,
            "operator": operator,
            "input_data": input_data,
            "loss_fn": loss_fn,
            "n_items": n_cells,
            "iterate_fn": lambda: operator.apply(input_data, {}, None),
            "baselines": baselines,
            "dataset_info": {
                "name": "immune_human",
                "n_cells": n_cells,
                "n_genes": data["n_genes"],
                "n_batches": n_batches,
                "n_types": data["n_types"],
            },
            "operator_config": {
                "n_clusters": op_config.n_clusters,
                "n_features": n_features,
                "temperature": op_config.temperature,
            },
            "operator_name": "DifferentiableHarmony",
            "dataset_name": "immune_human",
            "result_data": result_data,
            "benchmark_metadata": {
                "embedding_source": embedding_source,
                **(
                    self.embedding_adapter.benchmark_metadata()
                    if self.embedding_adapter is not None
                    else {}
                ),
            },
        }


def main() -> None:
    """CLI entry point."""
    DiffBioBenchmark.cli_main(
        BatchCorrectionBenchmark,
        _CONFIG,
        data_dir="/media/mahdi/ssd23/Data/scib",
    )


if __name__ == "__main__":
    main()
