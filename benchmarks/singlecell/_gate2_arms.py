"""Gate-2 arm runners: frozen genuine PCA vs learnable PCA-initialized projection.

Both arms share the frozen preprocessing statistics (fitted once on the training
split) and differ only in the dimensionality-reduction degree of freedom they train:

- ``run_frozen_pca_arm`` keeps the PCA eigenvectors fixed and trains only the probe
  over the frozen PCA embedding -- the "learn-the-dimension" arm (the number of
  components is the swept parameter).
- ``run_learnable_projection_arm`` trains a :class:`LearnableProjection` initialized
  from the PCA loadings (residual delta starts at zero) jointly with the probe -- the
  "learn-the-directions" arm.

Feeding the learnable arm the mean-centered scaled features means its zero-delta init
reproduces the frozen PCA embedding exactly, so it starts at the PCA baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import balanced_accuracy, f1_score
from flax import nnx

from benchmarks.singlecell.frozen_annotation_baseline import fit_frozen_preprocess
from diffbio.operators.foundation_models import LinearEmbeddingProbe
from diffbio.operators.foundation_models.embedding_probe import EmbeddingProbeConfig
from diffbio.operators.normalization.learnable_projection import (
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.operators.normalization.soft_pca import (
    SoftComponentSelection,
    SoftComponentSelectionConfig,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Held-out metrics for one Gate-2 arm.

    Attributes:
        macro_f1: Macro-averaged F1 over all classes.
        balanced_accuracy: Balanced accuracy over all classes.
        rare_macro_f1: Mean F1 over the rare classes (NaN if none were given).
    """

    macro_f1: float
    balanced_accuracy: float
    rare_macro_f1: float


@dataclass(frozen=True, slots=True)
class SoftDimensionResult:
    """Held-out metrics plus the learned effective dimensionality of the soft-k arm.

    Attributes:
        metrics: The arm's held-out :class:`ArmResult`.
        effective_dimension: Soft component count learned by the coverage gate.
        coverage: Learned cumulative-variance coverage target in ``(0, 1)``.
    """

    metrics: ArmResult
    effective_dimension: float
    coverage: float


class _ProjectionProbe(nnx.Module):
    """A learnable projection composed with an annotation probe."""

    def __init__(self, projection: LearnableProjection, probe: LinearEmbeddingProbe) -> None:
        """Store the projection and probe submodules.

        Args:
            projection: The learnable dimensionality-reduction stage.
            probe: The annotation classifier head.
        """
        self.projection = projection
        self.probe = probe


class _SoftDimensionProbe(nnx.Module):
    """A soft component selector composed with an annotation probe."""

    def __init__(self, selector: SoftComponentSelection, probe: LinearEmbeddingProbe) -> None:
        """Store the selector and probe submodules.

        Args:
            selector: The soft component-selection (learnable dimension) stage.
            probe: The annotation classifier head.
        """
        self.selector = selector
        self.probe = probe


def per_class_f1(predictions: np.ndarray, targets: np.ndarray, n_classes: int) -> np.ndarray:
    """Return the per-class F1 scores.

    Args:
        predictions: ``(n,)`` predicted integer labels.
        targets: ``(n,)`` true integer labels.
        n_classes: Number of classes.

    Returns:
        A ``(n_classes,)`` float32 array of per-class F1 scores.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    scores = np.zeros(n_classes, dtype=np.float32)
    for class_index in range(n_classes):
        true_positive = int(np.sum((predictions == class_index) & (targets == class_index)))
        false_positive = int(np.sum((predictions == class_index) & (targets != class_index)))
        false_negative = int(np.sum((predictions != class_index) & (targets == class_index)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores[class_index] = (2 * true_positive / denominator) if denominator > 0 else 0.0
    return scores


def _probe_forward(model: LinearEmbeddingProbe, features: jnp.ndarray) -> jnp.ndarray:
    """Return the probe logits for pre-projected embeddings."""
    return model.apply({"embeddings": features}, {}, None)[0]["logits"]


def _project_probe_forward(model: _ProjectionProbe, features: jnp.ndarray) -> jnp.ndarray:
    """Project the scaled features then return the probe logits."""
    projected = model.projection.apply({"features": features}, {}, None)[0]["projection"]
    return model.probe.apply({"embeddings": projected}, {}, None)[0]["logits"]


def _soft_probe_forward(model: _SoftDimensionProbe, features: jnp.ndarray) -> jnp.ndarray:
    """Soft-gate the PCA components then return the probe logits."""
    gated = model.selector.apply({"projection": features}, {}, None)[0]["projection"]
    return model.probe.apply({"embeddings": gated}, {}, None)[0]["logits"]


def _evaluate(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
    rare_classes: np.ndarray,
) -> ArmResult:
    """Score predictions with macro-F1, balanced accuracy, and rare-class F1."""
    predictions_j = jnp.asarray(predictions)
    targets_j = jnp.asarray(targets)
    per_class = per_class_f1(predictions, targets, n_classes)
    rare_macro_f1 = (
        float(np.mean(per_class[rare_classes])) if rare_classes.size > 0 else float("nan")
    )
    return ArmResult(
        macro_f1=float(f1_score(predictions_j, targets_j, average="macro")),
        balanced_accuracy=float(balanced_accuracy(predictions_j, targets_j)),
        rare_macro_f1=rare_macro_f1,
    )


def _embedding_probe(
    input_dim: int, n_classes: int, hidden_dim: int | None, seed: int
) -> LinearEmbeddingProbe:
    """Build an annotation probe head."""
    return LinearEmbeddingProbe(
        EmbeddingProbeConfig(input_dim=input_dim, n_classes=n_classes, hidden_dim=hidden_dim),
        rngs=nnx.Rngs(seed),
    )


def run_frozen_pca_arm(
    train_counts: np.ndarray,
    train_labels: np.ndarray,
    test_counts: np.ndarray,
    test_labels: np.ndarray,
    *,
    n_classes: int,
    n_top_genes: int,
    n_components: int,
    hidden_dim: int | None,
    rare_classes: np.ndarray,
    config: MiniBatchConfig,
    hvg_method: str = "dispersion",
    probe_seed: int = 0,
) -> ArmResult:
    """Train and score the frozen-PCA arm (eigenvectors fixed, probe trained).

    Args:
        train_counts: ``(n_train, n_genes)`` training counts.
        train_labels: ``(n_train,)`` training labels.
        test_counts: ``(n_test, n_genes)`` held-out counts.
        test_labels: ``(n_test,)`` held-out labels.
        n_classes: Number of cell-type classes.
        n_top_genes: Highly-variable genes retained by the frozen transform.
        n_components: Number of principal components (the swept dimensionality knob).
        hidden_dim: Probe hidden width (``None`` for a linear head).
        rare_classes: Indices of the rare classes for the breakdown metric.
        config: Mini-batch training configuration.
        probe_seed: Seed for the probe initialization.

    Returns:
        The arm's held-out :class:`ArmResult`.
    """
    transform = fit_frozen_preprocess(
        train_counts,
        n_top_genes=n_top_genes,
        n_components=n_components,
        hvg_method=hvg_method,
        labels=train_labels,
    )
    train_features = transform.transform(train_counts)
    test_features = transform.transform(test_counts)

    probe = _embedding_probe(train_features.shape[1], n_classes, hidden_dim, probe_seed)
    train_minibatch(
        probe, _probe_forward, train_features, train_labels, n_classes=n_classes, config=config
    )

    predictions = np.asarray(jnp.argmax(_probe_forward(probe, jnp.asarray(test_features)), axis=-1))
    return _evaluate(predictions, test_labels, n_classes, rare_classes)


def run_learnable_projection_arm(
    train_counts: np.ndarray,
    train_labels: np.ndarray,
    test_counts: np.ndarray,
    test_labels: np.ndarray,
    *,
    n_classes: int,
    n_top_genes: int,
    n_components: int,
    hidden_dim: int | None,
    rare_classes: np.ndarray,
    config: MiniBatchConfig,
    hvg_method: str = "dispersion",
    probe_seed: int = 0,
) -> ArmResult:
    """Train and score the learnable-projection arm (directions trained jointly).

    Args:
        train_counts: ``(n_train, n_genes)`` training counts.
        train_labels: ``(n_train,)`` training labels.
        test_counts: ``(n_test, n_genes)`` held-out counts.
        test_labels: ``(n_test,)`` held-out labels.
        n_classes: Number of cell-type classes.
        n_top_genes: Highly-variable genes retained by the frozen transform.
        n_components: Projection output dimension.
        hidden_dim: Probe hidden width (``None`` for a linear head).
        rare_classes: Indices of the rare classes for the breakdown metric.
        config: Mini-batch training configuration.
        probe_seed: Seed for the probe (and projection) initialization.

    Returns:
        The arm's held-out :class:`ArmResult`.
    """
    transform = fit_frozen_preprocess(
        train_counts,
        n_top_genes=n_top_genes,
        n_components=n_components,
        hvg_method=hvg_method,
        labels=train_labels,
    )
    # Mean-center so the zero-delta projection reproduces the frozen PCA embedding.
    train_features = transform.scaled(train_counts) - transform.pca_mean
    test_features = transform.scaled(test_counts) - transform.pca_mean

    n_hvg, n_output = transform.loadings.shape
    projection = LearnableProjection(
        LearnableProjectionConfig(n_genes=n_hvg, n_components=n_output),
        init_loadings=transform.loadings,
        rngs=nnx.Rngs(probe_seed),
    )
    probe = _embedding_probe(n_output, n_classes, hidden_dim, probe_seed)
    model = _ProjectionProbe(projection, probe)
    train_minibatch(
        model,
        _project_probe_forward,
        train_features,
        train_labels,
        n_classes=n_classes,
        config=config,
    )

    predictions = np.asarray(
        jnp.argmax(_project_probe_forward(model, jnp.asarray(test_features)), axis=-1)
    )
    return _evaluate(predictions, test_labels, n_classes, rare_classes)


def run_soft_dimension_arm(
    train_counts: np.ndarray,
    train_labels: np.ndarray,
    test_counts: np.ndarray,
    test_labels: np.ndarray,
    *,
    n_classes: int,
    n_top_genes: int,
    n_components: int,
    hidden_dim: int | None,
    rare_classes: np.ndarray,
    config: MiniBatchConfig,
    hvg_method: str = "dispersion",
    init_coverage: float = 0.9,
    temperature: float = 0.05,
    probe_seed: int = 0,
) -> SoftDimensionResult:
    """Train and score the soft-k arm (PCA fixed; the dimensionality is learned).

    The PCA directions stay frozen; a :class:`SoftComponentSelection` gate learns the
    cumulative-variance coverage (hence the effective number of components) jointly with
    the probe.

    Args:
        train_counts: ``(n_train, n_genes)`` training counts.
        train_labels: ``(n_train,)`` training labels.
        test_counts: ``(n_test, n_genes)`` held-out counts.
        test_labels: ``(n_test,)`` held-out labels.
        n_classes: Number of cell-type classes.
        n_top_genes: Highly-variable genes retained by the frozen transform.
        n_components: Number of principal components exposed to the gate.
        hidden_dim: Probe hidden width (``None`` for a linear head).
        rare_classes: Indices of the rare classes for the breakdown metric.
        config: Mini-batch training configuration.
        init_coverage: Initial cumulative-variance coverage target.
        temperature: Soft-gate sharpness.
        probe_seed: Seed for the probe and selector initialization.

    Returns:
        The arm's :class:`SoftDimensionResult` (metrics plus the learned dimension).
    """
    transform = fit_frozen_preprocess(
        train_counts,
        n_top_genes=n_top_genes,
        n_components=n_components,
        hvg_method=hvg_method,
        labels=train_labels,
    )
    train_features = transform.transform(train_counts)
    test_features = transform.transform(test_counts)

    n_output = train_features.shape[1]
    selector = SoftComponentSelection(
        SoftComponentSelectionConfig(
            n_components=n_output, init_coverage=init_coverage, temperature=temperature
        ),
        eigenvalues=transform.eigenvalues,
        rngs=nnx.Rngs(probe_seed),
    )
    probe = _embedding_probe(n_output, n_classes, hidden_dim, probe_seed)
    model = _SoftDimensionProbe(selector, probe)
    train_minibatch(
        model, _soft_probe_forward, train_features, train_labels, n_classes=n_classes, config=config
    )

    predictions = np.asarray(
        jnp.argmax(_soft_probe_forward(model, jnp.asarray(test_features)), axis=-1)
    )
    metrics = _evaluate(predictions, test_labels, n_classes, rare_classes)
    coverage = float(jax.nn.sigmoid(model.selector.raw_coverage[...]))
    return SoftDimensionResult(
        metrics=metrics,
        effective_dimension=float(model.selector.effective_dimension()),
        coverage=coverage,
    )
