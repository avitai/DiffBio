#!/usr/bin/env python3
"""Gate 1: the frozen JointPreprocessingPipeline reproduces the baseline annotation F1.

The differentiable ``JointPreprocessingPipeline`` composed of LearnableNormalization,
SoftHVG, DifferentiableScaler, and DifferentiablePCA is configured to mimic the
standard scanpy stack. Trained conventionally (only the probe head), it must
reproduce the frozen scanpy baseline's cell-annotation macro-F1 within noise on
one atlas -- the parity floor that earns the right to claim any joint gain. This
module reuses the ticket-01 harness (``annotation_baseline`` -> calibrax metrics)
to score both feature sets under identical seeds.
"""

from __future__ import annotations

import logging
import os

import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks.singlecell.frozen_annotation_baseline import (
    annotation_baseline,
    frozen_preprocess,
)
from diffbio.pipelines.joint_preprocessing import (
    JointPreprocessingPipeline,
    JointPreprocessingPipelineConfig,
)
from diffbio.sources.immune_human import ImmuneHumanConfig, ImmuneHumanSource

logger = logging.getLogger(__name__)

_SPLIT_SEED = 42
_DEFAULT_N_TOP_GENES = 2000
_DEFAULT_N_COMPONENTS = 50
_DEFAULT_TRAIN_STEPS = 200
_DEFAULT_DATA_DIR = os.environ.get("DIFFBIO_SCIB_DATA_DIR", "/media/mahdi/ssd23/Data/scib")


def pipeline_frozen_features(
    counts: np.ndarray,
    *,
    n_classes: int,
    n_top_genes: int,
    n_components: int,
    seed: int = 0,
) -> np.ndarray:
    """Run the frozen ``JointPreprocessingPipeline`` and return its PCA embeddings.

    Args:
        counts: Raw ``(n_cells, n_genes)`` count matrix.
        n_classes: Number of annotation classes (for the pipeline's probe head).
        n_top_genes: Number of highly-variable genes to keep.
        n_components: Number of principal components to return.
        seed: Seed for the (unused) probe initialization; the preprocessing is
            deterministic.

    Returns:
        The ``(n_cells, n_components)`` PCA feature matrix, matching the frozen
        baseline's preprocessing output.
    """
    counts_array = np.asarray(counts, dtype=np.float32)
    config = JointPreprocessingPipelineConfig(
        n_genes=int(counts_array.shape[1]),
        n_classes=n_classes,
        n_top_genes=n_top_genes,
        n_components=n_components,
    )
    pipeline = JointPreprocessingPipeline(config, rngs=nnx.Rngs(seed))
    output, _, _ = pipeline.apply({"counts": jnp.asarray(counts_array)}, {}, None)
    return np.asarray(output["embeddings"], dtype=np.float32)


def gate1_annotation_comparison(
    counts: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int,
    n_top_genes: int = _DEFAULT_N_TOP_GENES,
    n_components: int = _DEFAULT_N_COMPONENTS,
    seed: int = _SPLIT_SEED,
    n_train_steps: int = _DEFAULT_TRAIN_STEPS,
) -> dict[str, float]:
    """Score the frozen pipeline against the baseline on annotation macro-F1.

    Both feature sets are scored with the same probe training (identical split,
    initialization, and optimizer seed), so the macro-F1 gap isolates the
    preprocessing difference -- which should be within noise (Gate 1).

    Args:
        counts: Raw ``(n_cells, n_genes)`` count matrix.
        labels: ``(n_cells,)`` integer cell-type labels.
        n_classes: Number of distinct cell-type classes.
        n_top_genes: Number of highly-variable genes to keep.
        n_components: Number of principal components.
        seed: Seed for the split, probe initialization, and optimizer.
        n_train_steps: Number of full-batch probe training steps.

    Returns:
        A mapping with ``baseline_macro_f1``, ``pipeline_macro_f1``, and the
        absolute ``macro_f1_gap`` between them.
    """
    counts_array = np.asarray(counts, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int32)

    baseline_features = frozen_preprocess(
        counts_array, n_top_genes=n_top_genes, n_components=n_components
    )
    pipeline_features = pipeline_frozen_features(
        counts_array,
        n_classes=n_classes,
        n_top_genes=n_top_genes,
        n_components=n_components,
        seed=seed,
    )

    baseline_metrics = annotation_baseline(
        baseline_features, labels_array, n_classes=n_classes, seed=seed, n_train_steps=n_train_steps
    )
    pipeline_metrics = annotation_baseline(
        pipeline_features, labels_array, n_classes=n_classes, seed=seed, n_train_steps=n_train_steps
    )
    return {
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "pipeline_macro_f1": pipeline_metrics["macro_f1"],
        "macro_f1_gap": abs(baseline_metrics["macro_f1"] - pipeline_metrics["macro_f1"]),
    }


def main() -> None:
    """Record Gate 1 on the immune_human atlas from the command line."""
    logging.basicConfig(level=logging.INFO)
    source = ImmuneHumanSource(ImmuneHumanConfig(data_dir=_DEFAULT_DATA_DIR, subsample=2000))
    data = source.load()
    comparison = gate1_annotation_comparison(
        np.asarray(data["counts"], dtype=np.float32),
        np.asarray(data["cell_type_labels"], dtype=np.int32),
        n_classes=int(data["n_types"]),
    )
    logger.info(
        "Gate 1 | baseline macro_f1=%.4f pipeline macro_f1=%.4f gap=%.4f",
        comparison["baseline_macro_f1"],
        comparison["pipeline_macro_f1"],
        comparison["macro_f1_gap"],
    )


if __name__ == "__main__":
    main()
