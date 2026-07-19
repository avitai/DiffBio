#!/usr/bin/env python3
"""Frozen classic-preprocessing single-cell annotation benchmark on immune_human.

Establishes the frozen, non-differentiable preprocessing baseline (count
normalization, log1p, highly-variable-gene selection, scaling, PCA) that the
jointly optimized differentiable preprocessing stack is measured against for
cell-type annotation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks.singlecell._foundation import SingleCellSource
from benchmarks.singlecell.frozen_annotation_baseline import (
    frozen_preprocess,
    score_annotation,
    train_annotation_probe,
)
from diffbio.operators.foundation_models import LinearEmbeddingProbe
from diffbio.sources.immune_human import ImmuneHumanConfig, ImmuneHumanSource

logger = logging.getLogger(__name__)

_CONFIG = DiffBioBenchmarkConfig(
    name="singlecell/frozen_annotation",
    domain="singlecell",
    quick_subsample=2000,
    n_iterations_quick=10,
    n_iterations_full=50,
)
_TRAIN_FRACTION = 0.8
_TRAIN_STEPS_QUICK = 75
_TRAIN_STEPS_FULL = 250
_LEARNING_RATE = 1e-2
_SPLIT_SEED = 42
_DEFAULT_N_TOP_GENES = 2000
_DEFAULT_N_COMPONENTS = 50
_DEFAULT_DATA_DIR = "/media/mahdi/ssd23/Data/scib"


class FrozenAnnotationBaselineBenchmark(DiffBioBenchmark):
    """Evaluate the frozen classic preprocessing stack on cell annotation."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig = _CONFIG,
        *,
        quick: bool = False,
        data_dir: str = _DEFAULT_DATA_DIR,
        source_factory: Callable[[int | None], SingleCellSource] | None = None,
        n_top_genes: int = _DEFAULT_N_TOP_GENES,
        n_components: int = _DEFAULT_N_COMPONENTS,
    ) -> None:
        super().__init__(config, quick=quick, data_dir=data_dir)
        self._source_factory = source_factory or self._default_source_factory
        self.n_top_genes = n_top_genes
        self.n_components = n_components

    def _default_source_factory(self, subsample: int | None) -> ImmuneHumanSource:
        """Create the default immune_human source."""
        return ImmuneHumanSource(ImmuneHumanConfig(data_dir=self.data_dir, subsample=subsample))

    def _run_core(self) -> dict[str, Any]:
        """Load counts, run frozen preprocessing, and score cell annotation."""
        subsample = self.config.quick_subsample if self.quick else None
        n_train_steps = _TRAIN_STEPS_QUICK if self.quick else _TRAIN_STEPS_FULL

        logger.info("Loading immune_human single-cell dataset...")
        source: SingleCellSource = self._source_factory(subsample)
        data = source.load()

        counts = np.asarray(data["counts"], dtype=np.float32)
        labels = np.asarray(data["cell_type_labels"], dtype=np.int32)
        n_types = int(data["n_types"])

        logger.info("Running frozen preprocessing (HVG + PCA)...")
        features = frozen_preprocess(
            counts, n_top_genes=self.n_top_genes, n_components=self.n_components
        )
        outcome = train_annotation_probe(
            features,
            labels,
            n_classes=n_types,
            seed=_SPLIT_SEED,
            n_train_steps=n_train_steps,
            learning_rate=_LEARNING_RATE,
            train_fraction=_TRAIN_FRACTION,
        )
        metrics = score_annotation(outcome)

        train_labels = outcome.train_labels

        def loss_fn(model: LinearEmbeddingProbe, batch_data: dict[str, Any]) -> jnp.ndarray:
            batch_result, _, _ = model.apply(batch_data, {}, None)
            log_probs = jax.nn.log_softmax(batch_result["logits"], axis=-1)
            return -jnp.mean(log_probs[jnp.arange(train_labels.shape[0]), train_labels])

        return {
            "metrics": metrics,
            "operator": outcome.probe,
            "input_data": {"embeddings": outcome.train_features},
            "loss_fn": loss_fn,
            "n_items": int(outcome.test_features.shape[0]),
            "iterate_fn": lambda: outcome.probe.apply(
                {"embeddings": outcome.test_features}, {}, None
            ),
            "task_name": "cell_annotation",
            "operator_name": "LinearEmbeddingProbe",
            "dataset_name": "immune_human",
            "dataset_info": {
                "n_cells": int(data["n_cells"]),
                "n_genes": int(data["n_genes"]),
                "n_types": n_types,
                "n_components": int(features.shape[1]),
            },
            "operator_config": {
                "n_top_genes": self.n_top_genes,
                "n_components": self.n_components,
                "learning_rate": _LEARNING_RATE,
                "n_train_steps": n_train_steps,
                "train_fraction": _TRAIN_FRACTION,
            },
        }


def main() -> None:
    """Run the frozen annotation baseline benchmark from the command line."""
    logging.basicConfig(level=logging.INFO)
    result = FrozenAnnotationBaselineBenchmark(quick=True).run()
    logger.info("macro_f1=%.4f", result.metrics["macro_f1"].value)


if __name__ == "__main__":
    main()
