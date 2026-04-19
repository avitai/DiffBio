"""Shared contextual epigenomics benchmark harness and ablation helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import jax.numpy as jnp
import numpy as np
from datarax.core.config import OperatorConfig
from flax import nnx

from benchmarks._base import (
    DiffBioBenchmark,
    DiffBioBenchmarkConfig,
    build_benchmark_comparison_key,
)
from benchmarks._optimizers import (
    BENCHMARK_OPTIMIZER_SUBSTRATE,
    create_benchmark_optimizer,
)
from diffbio.operators.epigenomics.contextual import (
    ContextualEpigenomicsConfig,
    ContextualEpigenomicsOperator,
    compute_chromatin_guidance_loss,
    compute_contextual_epigenomics_loss,
)
from diffbio.sources import (
    CONTEXTUAL_EPIGENOMICS_DATASET_CONTRACT_KEYS,
    build_synthetic_contextual_epigenomics_dataset as build_contextual_dataset,
    validate_contextual_epigenomics_dataset,
)

_DATASET_NAME = "synthetic_contextual_epigenomics"
_DEFAULT_CONTEXTUAL_VARIANT = "tf_plus_chromatin"
_TRAIN_STEPS_QUICK = 20
_TRAIN_STEPS_FULL = 60
_LEARNING_RATE = 1e-2
_OPTIMIZER_TYPE = "adam"
CONTEXTUAL_TRAINING_SUBSTRATE = {
    **BENCHMARK_OPTIMIZER_SUBSTRATE,
    "optimizer_type": _OPTIMIZER_TYPE,
}
CONTEXTUAL_ABLATION_COMPARISON_AXES = (
    "dataset",
    "task",
    "contextual_variant",
)


class _ContextualSource(Protocol):
    """Minimal source protocol for contextual epigenomics benchmarks."""

    def load(self) -> dict[str, Any]:
        """Load the benchmark payload."""
        ...


@dataclass(frozen=True, slots=True)
class ContextualEpigenomicsTaskSpec:
    """Specification for one contextual epigenomics benchmark task."""

    task_name: str
    target_semantics: Literal["binary_peak_mask", "chromatin_state_id"]
    quick_examples: int = 6
    full_examples: int = 18
    sequence_length: int = 24
    num_tf_features: int = 3
    num_output_classes: int = 1

    @property
    def primary_metric(self) -> str:
        """Return the primary downstream metric for the task."""
        if self.target_semantics == "binary_peak_mask":
            return "f1"
        return "accuracy"


@dataclass(frozen=True, slots=True)
class ContextualAblationSpec:
    """Configuration for one contextual epigenomics ablation variant."""

    name: str
    use_tf_context: bool
    use_chromatin_guidance: bool
    chromatin_guidance_weight: float = 0.0


CONTEXTUAL_ABLATION_ORDER = (
    "sequence_only",
    "tf_context",
    "tf_plus_chromatin",
)
CONTEXTUAL_ABLATION_SPECS = {
    "sequence_only": ContextualAblationSpec(
        name="sequence_only",
        use_tf_context=False,
        use_chromatin_guidance=False,
    ),
    "tf_context": ContextualAblationSpec(
        name="tf_context",
        use_tf_context=True,
        use_chromatin_guidance=False,
    ),
    "tf_plus_chromatin": ContextualAblationSpec(
        name="tf_plus_chromatin",
        use_tf_context=True,
        use_chromatin_guidance=True,
        chromatin_guidance_weight=0.5,
    ),
}


@dataclass(frozen=True)
class ContextualEpigenomicsBenchmarkConfig(OperatorConfig):
    """Benchmark-owned model configuration for contextual epigenomics."""

    hidden_dim: int = 24
    num_layers: int = 1
    num_heads: int = 2
    intermediate_dim: int = 48
    dropout_rate: float = 0.0


class _SyntheticContextualEpigenomicsSource:
    """Deterministic synthetic source for contextual epigenomics benchmarks."""

    def __init__(self, task_spec: ContextualEpigenomicsTaskSpec, n_examples: int) -> None:
        self._task_spec = task_spec
        self._n_examples = n_examples

    def load(self) -> dict[str, Any]:
        return build_synthetic_contextual_epigenomics_dataset(
            task_spec=self._task_spec,
            n_examples=self._n_examples,
        )


def build_synthetic_contextual_epigenomics_dataset(
    *,
    task_spec: ContextualEpigenomicsTaskSpec,
    n_examples: int,
) -> dict[str, Any]:
    """Build a deterministic synthetic contextual epigenomics dataset."""
    return build_contextual_dataset(
        n_examples=n_examples,
        sequence_length=task_spec.sequence_length,
        num_tf_features=task_spec.num_tf_features,
        target_semantics=task_spec.target_semantics,
        num_output_classes=task_spec.num_output_classes,
    )


def compute_contextual_peak_metrics(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
) -> dict[str, float]:
    """Compute position-wise binary metrics for contextual peak calling."""
    predicted = np.asarray(logits >= 0.0)
    truth = np.asarray(targets > 0)

    true_positives = float(np.sum(predicted & truth))
    predicted_positives = float(np.sum(predicted))
    actual_positives = float(np.sum(truth))

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_chromatin_state_accuracy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
) -> dict[str, float]:
    """Compute simple position-wise accuracy for chromatin-state prediction."""
    predicted = np.asarray(jnp.argmax(logits, axis=-1), dtype=np.int32)
    truth = np.asarray(targets, dtype=np.int32)
    return {"accuracy": float(np.mean(predicted == truth))}


class ContextualEpigenomicsBenchmark(DiffBioBenchmark):
    """Benchmark harness for contextual epigenomics tasks and ablations."""

    def __init__(
        self,
        config: DiffBioBenchmarkConfig,
        *,
        task_spec: ContextualEpigenomicsTaskSpec,
        quick: bool = False,
        variant_name: str = _DEFAULT_CONTEXTUAL_VARIANT,
        source_factory: Callable[[int | None], _ContextualSource] | None = None,
    ) -> None:
        super().__init__(config, quick=quick, data_dir="")
        self.task_spec = task_spec
        self.variant_name = variant_name
        self.ablation = CONTEXTUAL_ABLATION_SPECS[variant_name]
        self._source_factory = source_factory or self._default_source_factory

    def _default_source_factory(
        self,
        subsample: int | None,
    ) -> _SyntheticContextualEpigenomicsSource:
        del subsample
        n_examples = self.task_spec.quick_examples if self.quick else self.task_spec.full_examples
        return _SyntheticContextualEpigenomicsSource(self.task_spec, n_examples)

    def _run_core(self) -> dict[str, Any]:
        """Run one contextual epigenomics benchmark variant."""
        source = self._source_factory(self.config.quick_subsample if self.quick else None)
        data = source.load()
        validate_contextual_epigenomics_dataset(
            data,
            target_semantics=self.task_spec.target_semantics,
            num_output_classes=self.task_spec.num_output_classes,
        )

        model_config = _build_model_config(
            data=data,
            task_spec=self.task_spec,
            ablation=self.ablation,
        )
        operator = ContextualEpigenomicsOperator(model_config, rngs=nnx.Rngs(42))
        optimizer = nnx.Optimizer(
            operator,
            create_benchmark_optimizer(
                optimizer_type=_OPTIMIZER_TYPE,
                learning_rate=_LEARNING_RATE,
            ),
            wrt=nnx.Param,
        )

        def loss_fn(
            model: ContextualEpigenomicsOperator, batch_data: dict[str, Any]
        ) -> jnp.ndarray:
            del batch_data
            return compute_contextual_epigenomics_loss(model, data)["total"]

        initial_losses = compute_contextual_epigenomics_loss(operator, data)
        n_train_steps = _TRAIN_STEPS_QUICK if self.quick else _TRAIN_STEPS_FULL
        for _ in range(n_train_steps):
            loss, grads = nnx.value_and_grad(lambda model: loss_fn(model, data))(operator)
            optimizer.update(operator, grads)

        final_losses = compute_contextual_epigenomics_loss(operator, data)
        result_data, _, _ = operator.apply(data, {}, None)
        logits = jnp.asarray(result_data["logits"])
        targets = jnp.asarray(data["targets"])

        if self.task_spec.target_semantics == "binary_peak_mask":
            metrics = compute_contextual_peak_metrics(logits, targets)
        else:
            metrics = compute_chromatin_state_accuracy(logits, targets)
        metrics["chromatin_consistency"] = compute_chromatin_consistency(
            token_embeddings=jnp.asarray(result_data["token_embeddings"], dtype=jnp.float32),
            chromatin_contacts=jnp.asarray(data["chromatin_contacts"], dtype=jnp.float32),
            sequence_mask=_get_sequence_mask(data),
        )
        contextual_tags = {
            "dataset": _DATASET_NAME,
            "task": self.task_spec.task_name,
            "contextual_variant": self.variant_name,
        }

        return {
            "metrics": metrics,
            "operator": operator,
            "input_data": data,
            "loss_fn": loss_fn,
            "n_items": int(jnp.asarray(data["sequence"]).shape[0]),
            "iterate_fn": lambda: operator.apply(data, {}, None),
            "dataset_info": {
                "n_examples": int(jnp.asarray(data["sequence"]).shape[0]),
                "sequence_length": int(jnp.asarray(data["sequence"]).shape[1]),
                "num_tf_features": int(jnp.asarray(data["tf_context"]).shape[1]),
            },
            "operator_config": {
                "hidden_dim": model_config.hidden_dim,
                "num_layers": model_config.num_layers,
                "num_heads": model_config.num_heads,
                "intermediate_dim": model_config.intermediate_dim,
                "num_outputs": model_config.num_outputs,
                "use_tf_context": model_config.use_tf_context,
                "use_chromatin_guidance": model_config.use_chromatin_guidance,
                "chromatin_guidance_weight": model_config.chromatin_guidance_weight,
                "n_train_steps": n_train_steps,
                "learning_rate": _LEARNING_RATE,
            },
            "operator_name": "ContextualEpigenomicsOperator",
            "dataset_name": _DATASET_NAME,
            "task_name": self.task_spec.task_name,
            "result_data": result_data,
            "benchmark_tags": {
                "contextual_variant": self.variant_name,
            },
            "benchmark_metadata": {
                "comparison_axes": list(CONTEXTUAL_ABLATION_COMPARISON_AXES),
                "comparison_key": build_benchmark_comparison_key(
                    comparison_axes=CONTEXTUAL_ABLATION_COMPARISON_AXES,
                    tags=contextual_tags,
                ),
                "contextual_contract": {
                    "required_keys": list(CONTEXTUAL_EPIGENOMICS_DATASET_CONTRACT_KEYS),
                    "target_semantics": self.task_spec.target_semantics,
                    "num_output_classes": self.task_spec.num_output_classes,
                },
                "ablation": {
                    "name": self.ablation.name,
                    "use_tf_context": self.ablation.use_tf_context,
                    "use_chromatin_guidance": self.ablation.use_chromatin_guidance,
                    "chromatin_guidance_weight": self.ablation.chromatin_guidance_weight,
                },
                "training": {
                    **CONTEXTUAL_TRAINING_SUBSTRATE,
                    "n_steps": n_train_steps,
                    "learning_rate": _LEARNING_RATE,
                    "initial_total_loss": float(initial_losses["total"]),
                    "final_total_loss": float(final_losses["total"]),
                    "initial_supervised_loss": float(initial_losses["supervised"]),
                    "final_supervised_loss": float(final_losses["supervised"]),
                    "final_chromatin_guidance": float(final_losses["chromatin_guidance"]),
                },
            },
        }


def _build_model_config(
    *,
    data: dict[str, Any],
    task_spec: ContextualEpigenomicsTaskSpec,
    ablation: ContextualAblationSpec,
) -> ContextualEpigenomicsConfig:
    """Build the operator config for one benchmark run."""
    benchmark_config = ContextualEpigenomicsBenchmarkConfig()
    return ContextualEpigenomicsConfig(
        hidden_dim=benchmark_config.hidden_dim,
        num_layers=benchmark_config.num_layers,
        num_heads=benchmark_config.num_heads,
        intermediate_dim=benchmark_config.intermediate_dim,
        max_length=int(jnp.asarray(data["sequence"]).shape[1]),
        num_tf_features=int(jnp.asarray(data["tf_context"]).shape[1]),
        num_outputs=task_spec.num_output_classes,
        dropout_rate=benchmark_config.dropout_rate,
        use_tf_context=ablation.use_tf_context,
        use_chromatin_guidance=ablation.use_chromatin_guidance,
        chromatin_guidance_weight=ablation.chromatin_guidance_weight,
    )


def _get_sequence_mask(data: dict[str, Any]) -> jnp.ndarray:
    """Return the sequence mask for consistency evaluation."""
    sequence = jnp.asarray(data["sequence"], dtype=jnp.float32)
    if "sequence_mask" in data:
        return jnp.asarray(data["sequence_mask"], dtype=jnp.float32)
    return jnp.ones(sequence.shape[:2], dtype=jnp.float32)


def compute_chromatin_consistency(
    *,
    token_embeddings: jnp.ndarray,
    chromatin_contacts: jnp.ndarray,
    sequence_mask: jnp.ndarray,
) -> float:
    """Convert the chromatin-guidance loss into a bounded higher-is-better metric."""
    guidance_loss = compute_chromatin_guidance_loss(
        token_embeddings=token_embeddings,
        chromatin_contacts=chromatin_contacts,
        sequence_mask=sequence_mask,
    )
    return float(1.0 / (1.0 + guidance_loss))
