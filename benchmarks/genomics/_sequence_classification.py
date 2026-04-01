"""Shared benchmark scaffold for genomics foundation-model tasks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from benchmarks._classification import create_embedding_probe_train_step
from benchmarks._base import DiffBioBenchmark, DiffBioBenchmarkConfig
from benchmarks.genomics._foundation import (
    GENOMICS_FOUNDATION_DATASET_CONTRACT_KEYS,
    GENOMICS_FOUNDATION_SUITE_SCENARIOS,
    compute_sequence_classification_metrics,
    stratified_sequence_classification_split,
)
from diffbio.operators.foundation_models import (
    EmbeddingProbeConfig,
    LinearEmbeddingProbe,
    SequencePrecomputedAdapter,
    TransformerSequenceEncoder,
    TransformerSequenceEncoderConfig,
)
from diffbio.sequences.dna import encode_dna_string

logger = logging.getLogger(__name__)

_TRAIN_FRACTION = 0.8
_TRAIN_STEPS_QUICK = 60
_TRAIN_STEPS_FULL = 200
_LEARNING_RATE = 1e-2
_SPLIT_SEED = 42
_NATIVE_HIDDEN_DIM = 32

_TASK_CLASS_MOTIFS = {
    "promoter": ("TATAAA", "CGCGCG", "AATTAA"),
    "tfbs": ("GGGCGG", "CACGTG", "TTCCGG"),
    "splice_site": ("CAGGTA", "TTTCAG", "AAGGTT"),
}
_TASK_FILLERS = {
    "promoter": ("A", "C", "G"),
    "tfbs": ("T", "A", "C"),
    "splice_site": ("G", "T", "A"),
}


class _SequenceSource(Protocol):
    """Protocol for genomics benchmark data sources."""

    def load(self) -> dict[str, Any]:
        """Load the dataset payload for the benchmark."""
        ...


@dataclass(frozen=True, slots=True)
class SequenceTaskSpec:
    """Benchmark identity and default scaffold settings for one task."""

    benchmark_name: str
    task_name: str
    dataset_name: str = "synthetic_genomics"
    quick_samples_per_class: int = 8
    full_samples_per_class: int = 24
    sequence_length: int = 24


class _SyntheticSequenceClassificationSource:
    """Deterministic synthetic source used for benchmark scaffolding."""

    def __init__(self, task_name: str, samples_per_class: int, sequence_length: int) -> None:
        self._task_name = task_name
        self._samples_per_class = samples_per_class
        self._sequence_length = sequence_length

    def load(self) -> dict[str, Any]:
        return build_synthetic_sequence_classification_dataset(
            task_name=self._task_name,
            samples_per_class=self._samples_per_class,
            sequence_length=self._sequence_length,
        )


def build_synthetic_sequence_classification_dataset(
    *,
    task_name: str,
    samples_per_class: int,
    sequence_length: int,
) -> dict[str, Any]:
    """Build a deterministic synthetic dataset for a genomics task."""
    motifs = _TASK_CLASS_MOTIFS[task_name]
    fillers = _TASK_FILLERS[task_name]

    sequences: list[str] = []
    labels: list[int] = []
    sequence_ids: list[str] = []
    motif_length = len(motifs[0])

    for label, (motif, filler) in enumerate(zip(motifs, fillers, strict=True)):
        prefix_length = (sequence_length - motif_length) // 2
        suffix_length = sequence_length - prefix_length - motif_length
        for sample_index in range(samples_per_class):
            rotated_prefix = (filler * (prefix_length + sample_index + 1))[:prefix_length]
            rotated_suffix = (filler * (suffix_length + label + 2))[:suffix_length]
            sequence = f"{rotated_prefix}{motif}{rotated_suffix}"
            sequences.append(sequence)
            labels.append(label)
            sequence_ids.append(f"{task_name}_{label}_{sample_index}")

    one_hot_sequences = jnp.asarray(
        np.stack(
            [np.asarray(encode_dna_string(sequence), dtype=np.float32) for sequence in sequences]
        ),
        dtype=jnp.float32,
    )

    return {
        "sequence_ids": sequence_ids,
        "sequences": sequences,
        "one_hot_sequences": one_hot_sequences,
        "labels": np.asarray(labels, dtype=np.int32),
    }


def _validate_sequence_dataset(data: dict[str, Any]) -> None:
    """Validate the shared genomics dataset contract."""
    missing_keys = [key for key in GENOMICS_FOUNDATION_DATASET_CONTRACT_KEYS if key not in data]
    if missing_keys:
        raise ValueError(f"Sequence benchmark data is missing required keys: {missing_keys}")

    sequence_ids = tuple(str(item) for item in data["sequence_ids"])
    sequences = tuple(str(item) for item in data["sequences"])
    one_hot_sequences = jnp.asarray(data["one_hot_sequences"], dtype=jnp.float32)
    labels = np.asarray(data["labels"], dtype=np.int32)

    n_items = len(sequence_ids)
    if (
        len(sequences) != n_items
        or one_hot_sequences.shape[0] != n_items
        or labels.shape[0] != n_items
    ):
        raise ValueError("Sequence benchmark data keys must all have the same leading dimension.")
    if len(set(sequence_ids)) != n_items:
        raise ValueError("Sequence benchmark sequence_ids must be unique.")
    if one_hot_sequences.ndim != 3 or one_hot_sequences.shape[-1] != 4:
        raise ValueError("one_hot_sequences must have shape (n_sequences, sequence_length, 4).")
    if labels.ndim != 1:
        raise ValueError("Sequence benchmark labels must be rank-1.")
    if len({sequence.shape[0] for sequence in one_hot_sequences}) != 1:
        raise ValueError("Sequence benchmark one_hot_sequences must share one sequence length.")


class SequenceClassificationBenchmark(DiffBioBenchmark):
    """Evaluate sequence embeddings on a genomics classification task."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig,
        *,
        task_spec: SequenceTaskSpec,
        quick: bool = False,
        source_factory: Callable[[int | None], _SequenceSource] | None = None,
        embedding_adapter: SequencePrecomputedAdapter | None = None,
    ) -> None:
        super().__init__(config, quick=quick, data_dir="")
        self.task_spec = task_spec
        self.embedding_adapter = embedding_adapter
        self._source_factory = source_factory or self._default_source_factory

    def _default_source_factory(
        self,
        subsample: int | None,
    ) -> _SyntheticSequenceClassificationSource:
        """Create the default deterministic synthetic source."""
        del subsample
        samples_per_class = (
            self.task_spec.quick_samples_per_class
            if self.quick
            else self.task_spec.full_samples_per_class
        )
        return _SyntheticSequenceClassificationSource(
            self.task_spec.task_name,
            samples_per_class=samples_per_class,
            sequence_length=self.task_spec.sequence_length,
        )

    def _native_embeddings(
        self,
        one_hot_sequences: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        """Encode DNA sequences with the native DiffBio sequence encoder."""
        encoder = TransformerSequenceEncoder(
            TransformerSequenceEncoderConfig(
                hidden_dim=_NATIVE_HIDDEN_DIM,
                num_layers=1,
                num_heads=4,
                intermediate_dim=2 * _NATIVE_HIDDEN_DIM,
                max_length=int(one_hot_sequences.shape[1]),
                dropout_rate=0.0,
                pooling="mean",
            ),
            rngs=nnx.Rngs(42),
        )
        result, _, _ = encoder.apply({"sequence": one_hot_sequences}, {}, None)
        return jnp.asarray(result["embeddings"], dtype=jnp.float32), {
            "foundation_model": result["foundation_model"]
        }

    def _run_core(self) -> dict[str, Any]:
        """Load sequences, build embeddings, train a probe, and evaluate."""
        subsample = self.config.quick_subsample if self.quick else None
        n_train_steps = _TRAIN_STEPS_QUICK if self.quick else _TRAIN_STEPS_FULL

        logger.info("Loading %s sequence benchmark dataset...", self.task_spec.task_name)
        source = self._source_factory(subsample)
        data = source.load()
        _validate_sequence_dataset(data)

        labels = np.asarray(data["labels"], dtype=np.int32)
        train_indices, test_indices = stratified_sequence_classification_split(
            labels,
            train_fraction=_TRAIN_FRACTION,
            seed=_SPLIT_SEED,
        )

        if self.embedding_adapter is None:
            embeddings, result_data = self._native_embeddings(
                jnp.asarray(data["one_hot_sequences"], dtype=jnp.float32)
            )
            embedding_source = "native_sequence_encoder"
        else:
            embeddings = self.embedding_adapter.load_aligned_embeddings(
                reference_sequence_ids=data["sequence_ids"],
                require_sequence_ids=True,
            )
            result_data = self.embedding_adapter.result_data()
            embedding_source = "external_artifact"

        train_embeddings = embeddings[train_indices]
        test_embeddings = embeddings[test_indices]
        train_labels = jnp.asarray(labels[train_indices], dtype=jnp.int32)
        test_labels = np.asarray(labels[test_indices], dtype=np.int32)
        n_classes = int(np.max(labels)) + 1

        probe = LinearEmbeddingProbe(
            EmbeddingProbeConfig(
                input_dim=int(train_embeddings.shape[1]),
                n_classes=n_classes,
            ),
            rngs=nnx.Rngs(42),
        )
        optimizer = nnx.Optimizer(probe, optax.adam(_LEARNING_RATE), wrt=nnx.Param)
        train_step = create_embedding_probe_train_step()

        logger.info("Training sequence embedding probe (%d steps)...", n_train_steps)
        train_loss = jnp.array(0.0, dtype=jnp.float32)
        for step in range(n_train_steps):
            train_loss = train_step(probe, optimizer, train_embeddings, train_labels)
            if (step + 1) % max(1, n_train_steps // 4) == 0:
                logger.info("  step %d/%d loss=%.4f", step + 1, n_train_steps, float(train_loss))

        result, _, _ = probe.apply({"embeddings": test_embeddings}, {}, None)
        predicted_labels = np.asarray(result["predicted_labels"], dtype=np.int32)
        quality = compute_sequence_classification_metrics(test_labels, predicted_labels)
        quality["train_loss"] = float(train_loss)

        def loss_fn(model: LinearEmbeddingProbe, batch_data: dict[str, Any]) -> jnp.ndarray:
            batch_result, _, _ = model.apply(batch_data, {}, None)
            log_probs = jax.nn.log_softmax(batch_result["logits"], axis=-1)
            return -jnp.mean(log_probs[jnp.arange(train_labels.shape[0]), train_labels])

        return {
            "metrics": quality,
            "operator": probe,
            "input_data": {"embeddings": train_embeddings},
            "loss_fn": loss_fn,
            "n_items": int(test_embeddings.shape[0]),
            "iterate_fn": lambda: probe.apply({"embeddings": test_embeddings}, {}, None),
            "task_name": self.task_spec.task_name,
            "result_data": result_data,
            "dataset_info": {
                "n_sequences": len(data["sequence_ids"]),
                "n_train_sequences": int(train_indices.shape[0]),
                "n_test_sequences": int(test_indices.shape[0]),
                "sequence_length": int(data["one_hot_sequences"].shape[1]),
                "n_classes": n_classes,
            },
            "operator_config": {
                "input_dim": int(train_embeddings.shape[1]),
                "n_classes": n_classes,
                "learning_rate": _LEARNING_RATE,
                "n_train_steps": n_train_steps,
                "train_fraction": _TRAIN_FRACTION,
            },
            "operator_name": "LinearEmbeddingProbe",
            "dataset_name": self.task_spec.dataset_name,
            "benchmark_metadata": {
                "suite_scenarios": dict(GENOMICS_FOUNDATION_SUITE_SCENARIOS),
                "dataset_contract_keys": list(GENOMICS_FOUNDATION_DATASET_CONTRACT_KEYS),
                "embedding_source": embedding_source,
                **(
                    self.embedding_adapter.benchmark_metadata()
                    if self.embedding_adapter is not None
                    else {}
                ),
            },
        }
