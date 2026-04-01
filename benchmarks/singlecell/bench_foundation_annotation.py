#!/usr/bin/env python3
"""Single-cell foundation annotation benchmark on immune_human.

This benchmark evaluates cell-type transfer quality from single-cell embedding
representations using a lightweight differentiable probe. It is the Wave 3.1
scaffold benchmark for comparing native DiffBio embeddings with imported
foundation-model embeddings under a shared contract.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from calibrax.core.result import BenchmarkResult

from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks._baselines.singlecell_foundation import (
    SINGLECELL_FOUNDATION_BASELINE_FAMILIES,
)
from benchmarks.singlecell._foundation import (
    compute_annotation_metrics,
    SINGLECELL_FOUNDATION_DATASET_CONTRACT_KEYS,
    SINGLECELL_FOUNDATION_SUITE_SCENARIOS,
    stratified_cell_annotation_split,
)
from diffbio.operators.foundation_models import (
    EmbeddingProbeConfig,
    LinearEmbeddingProbe,
    SingleCellPrecomputedAdapter,
)
from diffbio.sources.immune_human import ImmuneHumanConfig, ImmuneHumanSource

logger = logging.getLogger(__name__)

_CONFIG = DiffBioBenchmarkConfig(
    name="singlecell/foundation_annotation",
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


class _SingleCellSource(Protocol):
    """Protocol for benchmark data sources."""

    def load(self) -> dict[str, Any]:
        """Load the dataset payload for the benchmark."""
        ...


def _create_probe_train_step() -> Any:
    """Create a JIT-compiled training step for the embedding probe."""

    @nnx.jit
    def train_step(
        probe: LinearEmbeddingProbe,
        opt: nnx.Optimizer,
        embeddings: jax.Array,
        labels: jax.Array,
    ) -> jax.Array:
        def loss_fn(model_inner: LinearEmbeddingProbe) -> jax.Array:
            result, _, _ = model_inner.apply({"embeddings": embeddings}, {}, None)
            log_probs = jax.nn.log_softmax(result["logits"], axis=-1)
            return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])

        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(probe)
        opt.update(probe, grads)
        return loss

    return train_step


class SingleCellFoundationAnnotationBenchmark(DiffBioBenchmark):
    """Evaluate single-cell embeddings on cell annotation transfer."""

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
        return ImmuneHumanSource(
            ImmuneHumanConfig(
                data_dir=self.data_dir,
                subsample=subsample,
            )
        )

    def _run_core(self) -> dict[str, Any]:
        """Load embeddings, train a probe, and evaluate cell annotation."""
        subsample = self.config.quick_subsample if self.quick else None
        n_train_steps = _TRAIN_STEPS_QUICK if self.quick else _TRAIN_STEPS_FULL

        logger.info("Loading immune_human single-cell dataset...")
        source: _SingleCellSource = self._source_factory(subsample)
        data = source.load()

        labels = np.asarray(data["cell_type_labels"], dtype=np.int32)
        train_indices, test_indices = stratified_cell_annotation_split(
            labels,
            train_fraction=_TRAIN_FRACTION,
            seed=_SPLIT_SEED,
        )

        if self.embedding_adapter is None:
            embeddings = jnp.asarray(data["embeddings"], dtype=jnp.float32)
            embedding_source = "dataset_embeddings"
            result_data: dict[str, Any] = {}
        else:
            embeddings = self.embedding_adapter.load_aligned_embeddings(
                reference_cell_ids=data["cell_ids"],
                require_cell_ids=True,
            )
            embedding_source = "external_artifact"
            result_data = self.embedding_adapter.result_data()

        train_embeddings = embeddings[train_indices]
        test_embeddings = embeddings[test_indices]
        train_labels = jnp.asarray(labels[train_indices], dtype=jnp.int32)
        test_labels = np.asarray(labels[test_indices], dtype=np.int32)

        probe_config = EmbeddingProbeConfig(
            input_dim=int(train_embeddings.shape[1]),
            n_classes=int(data["n_types"]),
        )
        probe = LinearEmbeddingProbe(probe_config, rngs=nnx.Rngs(42))
        optimizer = nnx.Optimizer(probe, optax.adam(_LEARNING_RATE), wrt=nnx.Param)
        train_step = _create_probe_train_step()

        logger.info("Training embedding probe (%d steps)...", n_train_steps)
        train_loss = jnp.array(0.0, dtype=jnp.float32)
        for step in range(n_train_steps):
            train_loss = train_step(probe, optimizer, train_embeddings, train_labels)
            if (step + 1) % max(1, n_train_steps // 5) == 0:
                logger.info("  step %d/%d loss=%.4f", step + 1, n_train_steps, float(train_loss))

        result, _, _ = probe.apply({"embeddings": test_embeddings}, {}, None)
        predicted_labels = np.asarray(result["predicted_labels"], dtype=np.int32)
        quality = compute_annotation_metrics(test_labels, predicted_labels)
        quality["train_loss"] = float(train_loss)

        def loss_fn(model: LinearEmbeddingProbe, batch_data: dict[str, Any]) -> jnp.ndarray:
            batch_result, _, _ = model.apply(batch_data, {}, None)
            logits = batch_result["logits"]
            batch_labels = train_labels
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.mean(log_probs[jnp.arange(batch_labels.shape[0]), batch_labels])

        return {
            "metrics": quality,
            "operator": probe,
            "input_data": {"embeddings": train_embeddings},
            "loss_fn": loss_fn,
            "n_items": int(test_embeddings.shape[0]),
            "iterate_fn": lambda: probe.apply({"embeddings": test_embeddings}, {}, None),
            "task_name": "cell_annotation",
            "result_data": result_data,
            "dataset_info": {
                "n_cells": int(data["n_cells"]),
                "n_train_cells": int(train_indices.shape[0]),
                "n_test_cells": int(test_indices.shape[0]),
                "n_types": int(data["n_types"]),
                "embedding_dim": int(embeddings.shape[1]),
            },
            "operator_config": {
                "input_dim": int(train_embeddings.shape[1]),
                "n_classes": int(data["n_types"]),
                "learning_rate": _LEARNING_RATE,
                "n_train_steps": n_train_steps,
                "train_fraction": _TRAIN_FRACTION,
            },
            "operator_name": "LinearEmbeddingProbe",
            "dataset_name": "immune_human",
            "benchmark_metadata": {
                "baseline_families": list(SINGLECELL_FOUNDATION_BASELINE_FAMILIES),
                "suite_scenarios": dict(SINGLECELL_FOUNDATION_SUITE_SCENARIOS),
                "dataset_contract_keys": list(SINGLECELL_FOUNDATION_DATASET_CONTRACT_KEYS),
                "embedding_source": embedding_source,
                "split_seed": _SPLIT_SEED,
                "alignment_required": self.embedding_adapter is not None,
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
        SingleCellFoundationAnnotationBenchmark,
        _CONFIG,
        data_dir="/media/mahdi/ssd23/Data/scib",
    )


def run_foundation_annotation_suite(
    *,
    quick: bool = False,
    data_dir: str = "/media/mahdi/ssd23/Data/scib",
    source_factory: Callable[[int | None], _SingleCellSource] | None = None,
    adapters: dict[str, SingleCellPrecomputedAdapter] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run the native and imported annotation benchmarks under one harness."""
    results = {
        "diffbio_native": SingleCellFoundationAnnotationBenchmark(
            quick=quick,
            data_dir=data_dir,
            source_factory=source_factory,
        ).run()
    }

    for baseline_name in SINGLECELL_FOUNDATION_BASELINE_FAMILIES:
        if baseline_name == "diffbio_native":
            continue
        if adapters is None or baseline_name not in adapters:
            continue

        results[baseline_name] = SingleCellFoundationAnnotationBenchmark(
            quick=quick,
            data_dir=data_dir,
            source_factory=source_factory,
            embedding_adapter=adapters[baseline_name],
        ).run()

    return results


def build_foundation_annotation_report(
    results: dict[str, BenchmarkResult],
) -> dict[str, Any]:
    """Build a deterministic comparison report for annotation benchmark runs."""
    model_order = [name for name in SINGLECELL_FOUNDATION_BASELINE_FAMILIES if name in results]
    models: dict[str, Any] = {}
    stable_metric_keys = ("accuracy", "macro_f1", "train_loss")
    stable_tag_keys = (
        "dataset",
        "task",
        "model_family",
        "adapter_mode",
        "artifact_id",
        "preprocessing_version",
    )
    stable_metadata_keys = ("embedding_source", "foundation_source_name")

    for model_name in model_order:
        result = results[model_name]
        models[model_name] = {
            "metrics": {
                key: float(result.metrics[key].value)
                for key in stable_metric_keys
                if key in result.metrics
            },
            "tags": {
                key: result.tags[key]
                for key in stable_tag_keys
                if key in result.tags
            },
            "metadata": {
                key: result.metadata[key]
                for key in stable_metadata_keys
                if key in result.metadata
            },
        }

    dataset = next(iter(results.values())).tags["dataset"]
    task = next(iter(results.values())).tags["task"]

    return {
        "benchmark": _CONFIG.name,
        "dataset": dataset,
        "task": task,
        "model_order": model_order,
        "models": models,
    }


if __name__ == "__main__":
    main()
