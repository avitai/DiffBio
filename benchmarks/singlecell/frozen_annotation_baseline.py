"""Frozen classic-preprocessing cell-annotation baseline.

This module wires the standard, non-differentiable single-cell preprocessing stack
(count normalization -> log1p -> highly-variable-gene selection -> scaling -> PCA)
to the shared linear annotation probe, producing the frozen baseline that the
jointly-optimized differentiable stack is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import balanced_accuracy, f1_score
from flax import nnx

from benchmarks._classification import (
    create_embedding_probe_train_step,
    stratified_label_split,
)
from benchmarks._optimizers import create_benchmark_optimizer
from diffbio.reductions import fit_pca_reduction
from diffbio.operators.foundation_models import LinearEmbeddingProbe
from diffbio.operators.foundation_models.embedding_probe import EmbeddingProbeConfig

_TARGET_SUM = 1.0e4
_SCALE_CLIP = 10.0


@dataclass(frozen=True)
class AnnotationProbeResult:
    """A trained annotation probe and its held-out evaluation split."""

    probe: LinearEmbeddingProbe
    train_features: jax.Array
    test_features: jax.Array
    train_labels: jax.Array
    test_labels: jax.Array
    predicted_labels: jax.Array


@dataclass(frozen=True)
class FrozenTransform:
    """The fitted parameters of the classic frozen preprocessing stack.

    Fitting once (on the training split) and applying the same transform to any
    split keeps the preprocessing statistics out of the gradient path and free of
    test leakage. The stored loadings also anchor the learnable-projection arm.

    Attributes:
        hvg_indices: ``(n_hvg,)`` indices of the retained highly-variable genes.
        mean: ``(n_hvg,)`` per-gene mean used for z-scoring the log-normed data.
        std: ``(n_hvg,)`` per-gene standard deviation (zeros replaced by one).
        pca_mean: ``(n_hvg,)`` mean removed by PCA before projection.
        loadings: ``(n_hvg, k)`` principal-component loadings (``components_.T``).
        eigenvalues: ``(k,)`` explained variance per component, descending.
    """

    hvg_indices: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    pca_mean: np.ndarray
    loadings: np.ndarray
    eigenvalues: np.ndarray

    def scaled(self, counts: np.ndarray) -> np.ndarray:
        """Return the clipped z-scored HVG features (the pre-PCA representation).

        Args:
            counts: Raw ``(n_cells, n_genes)`` count matrix.

        Returns:
            The ``(n_cells, n_hvg)`` clipped, standardized features that feed the
            projection -- the input the learnable-projection arm operates on.
        """
        logged = _log_normalize(counts)
        selected = logged[:, self.hvg_indices]
        return np.clip((selected - self.mean) / self.std, -_SCALE_CLIP, _SCALE_CLIP)

    def transform(self, counts: np.ndarray) -> np.ndarray:
        """Project ``counts`` onto the fitted principal components.

        Args:
            counts: Raw ``(n_cells, n_genes)`` count matrix.

        Returns:
            A ``(n_cells, k)`` float32 PCA feature matrix, matching
            ``PCA.fit_transform`` of the frozen stack on the fitted data.
        """
        projected = (self.scaled(counts) - self.pca_mean) @ self.loadings
        return np.asarray(projected, dtype=np.float32)


def _log_normalize(counts: np.ndarray) -> np.ndarray:
    """Library-size-normalize to ``_TARGET_SUM`` then ``log1p`` (the classic stack)."""
    if counts.ndim != 2:
        raise ValueError("counts must be a rank-2 (n_cells, n_genes) array.")
    library = counts.sum(axis=1, keepdims=True)
    library = np.where(library == 0.0, 1.0, library)
    return np.log1p(counts / library * _TARGET_SUM)


def supervised_hvg_indices(logged: np.ndarray, labels: np.ndarray, n_top: int) -> np.ndarray:
    """Return the indices of the most class-discriminative genes (Wilcoxon AUC).

    Label-aware gene selection: for each class (one-vs-rest) the per-gene rank-sum
    (Wilcoxon/Mann-Whitney) AUC measures how well the gene separates that class, and
    its distance from 0.5 is the effect size. Taking the strongest genes per class
    yields a supervised highly-variable-gene set -- a stronger frozen baseline than
    unsupervised dispersion, since it uses the annotation labels.

    Args:
        logged: ``(n_cells, n_genes)`` log-normalized expression.
        labels: ``(n_cells,)`` integer class labels.
        n_top: Number of genes to return.

    Returns:
        A sorted ``int64`` array of at most ``n_top`` gene indices.
    """
    from scipy.sparse import csr_matrix  # noqa: PLC0415
    from scipy.stats import rankdata  # noqa: PLC0415

    n_cells, n_genes = logged.shape
    classes = np.unique(labels)
    ranks = rankdata(logged, axis=0)
    one_hot = csr_matrix(
        (np.ones(n_cells), (np.searchsorted(classes, labels), np.arange(n_cells))),
        shape=(classes.size, n_cells),
    )
    class_sizes = np.asarray(one_hot.sum(axis=1)).ravel()
    rank_sums = one_hot @ ranks
    rest_sizes = n_cells - class_sizes
    auc = (rank_sums - (class_sizes * (class_sizes + 1) / 2)[:, None]) / (
        class_sizes[:, None] * np.where(rest_sizes == 0, 1.0, rest_sizes)[:, None]
    )
    effect = np.abs(auc - 0.5)

    per_class = max(1, n_top // classes.size)
    selected: set[int] = set()
    for class_index in range(classes.size):
        selected.update(int(gene) for gene in np.argsort(effect[class_index])[::-1][:per_class])
    if len(selected) < min(n_top, n_genes):
        for gene in np.argsort(effect.max(axis=0))[::-1]:
            selected.add(int(gene))
            if len(selected) >= min(n_top, n_genes):
                break
    return np.sort(np.fromiter(selected, dtype=np.int64))[:n_top]


def fit_frozen_preprocess(
    counts: np.ndarray,
    *,
    n_top_genes: int,
    n_components: int,
    hvg_method: str = "dispersion",
    labels: np.ndarray | None = None,
) -> FrozenTransform:
    """Fit the classic frozen preprocessing stack and return its parameters.

    Fits highly-variable-gene selection, per-gene standardization, and an exact PCA
    on ``counts`` (intended to be the training split), returning a
    :class:`FrozenTransform` that applies the identical transform to any split.

    Args:
        counts: Raw ``(n_cells, n_genes)`` count matrix to fit on.
        n_top_genes: Number of highly-variable genes to retain.
        n_components: Number of principal components to keep.
        hvg_method: ``"dispersion"`` (unsupervised) or ``"supervised"`` (Wilcoxon).
        labels: ``(n_cells,)`` labels, required when ``hvg_method`` is ``"supervised"``.

    Returns:
        The fitted :class:`FrozenTransform`.

    Raises:
        ValueError: If ``hvg_method`` is unknown, or ``"supervised"`` without labels.
    """
    logged = _log_normalize(counts)
    retained_genes = min(n_top_genes, logged.shape[1])
    if hvg_method == "dispersion":
        hvg_indices = _select_highly_variable_genes(logged, retained_genes)
    elif hvg_method == "supervised":
        if labels is None:
            raise ValueError("supervised HVG selection requires labels")
        hvg_indices = supervised_hvg_indices(logged, labels, retained_genes)
    else:
        raise ValueError(f"unknown hvg_method {hvg_method!r}; use 'dispersion' or 'supervised'")

    # Standardization + exact PCA are shared with every other modality's frozen
    # reduction, so delegate them to the modality-agnostic fit_pca_reduction.
    reduction = fit_pca_reduction(logged[:, hvg_indices], n_components)
    return FrozenTransform(
        hvg_indices=np.asarray(hvg_indices, dtype=np.int64),
        mean=reduction.mean,
        std=reduction.std,
        pca_mean=reduction.pca_mean,
        loadings=reduction.loadings,
        eigenvalues=reduction.eigenvalues,
    )


def frozen_preprocess(
    counts: np.ndarray,
    *,
    n_top_genes: int,
    n_components: int,
) -> np.ndarray:
    """Run the standard frozen preprocessing stack and return PCA features.

    Fits the transform on ``counts`` and applies it to the same data (the classic
    transductive baseline), delegating to :func:`fit_frozen_preprocess`.

    Args:
        counts: Raw ``(n_cells, n_genes)`` count matrix.
        n_top_genes: Number of highly-variable genes to retain.
        n_components: Number of principal components to return.

    Returns:
        A ``(n_cells, k)`` float32 feature matrix where ``k`` is
        ``min(n_components, n_top_genes, n_genes, n_cells)``.
    """
    transform = fit_frozen_preprocess(counts, n_top_genes=n_top_genes, n_components=n_components)
    return transform.transform(counts)


def _select_highly_variable_genes(logged: np.ndarray, n_top_genes: int) -> np.ndarray:
    """Return the indices of genes with the highest normalized dispersion.

    Normalized dispersion (variance / mean) is the Seurat highly-variable-gene
    measure, and is more faithful to the standard pipeline than raw variance.
    """
    gene_mean = logged.mean(axis=0)
    gene_variance = logged.var(axis=0)
    safe_mean = np.where(gene_mean == 0.0, 1.0, gene_mean)
    normalized_dispersion = gene_variance / safe_mean
    ranked = np.argsort(normalized_dispersion)[::-1][:n_top_genes]
    return np.sort(ranked)


def train_annotation_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int,
    seed: int = 42,
    n_train_steps: int = 200,
    learning_rate: float = 1.0e-2,
    train_fraction: float = 0.8,
) -> AnnotationProbeResult:
    """Train a linear probe on frozen features and predict the held-out split.

    Args:
        features: ``(n_cells, n_features)`` frozen feature matrix.
        labels: ``(n_cells,)`` integer cell-type labels.
        n_classes: Number of distinct cell-type classes.
        seed: Seed for the split, probe initialization, and optimizer.
        n_train_steps: Number of full-batch probe training steps.
        learning_rate: Probe optimizer learning rate.
        train_fraction: Fraction of cells assigned to the training split.

    Returns:
        The trained probe together with its train/test split and predictions.
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must share the same number of cells.")

    train_indices, test_indices = stratified_label_split(
        labels,
        train_fraction=train_fraction,
        seed=seed,
        minimum_count_name="cells",
    )

    train_features = jnp.asarray(features[train_indices], dtype=jnp.float32)
    test_features = jnp.asarray(features[test_indices], dtype=jnp.float32)
    train_labels = jnp.asarray(labels[train_indices], dtype=jnp.int32)
    test_labels = jnp.asarray(labels[test_indices], dtype=jnp.int32)

    probe = LinearEmbeddingProbe(
        EmbeddingProbeConfig(input_dim=int(features.shape[1]), n_classes=n_classes),
        rngs=nnx.Rngs(seed),
    )
    optimizer = nnx.Optimizer(
        probe,
        create_benchmark_optimizer(learning_rate=learning_rate),
        wrt=nnx.Param,
    )
    train_step = create_embedding_probe_train_step()

    for _ in range(n_train_steps):
        train_step(probe, optimizer, train_features, train_labels)

    result, _, _ = probe.apply({"embeddings": test_features}, {}, None)
    predicted_labels = jnp.asarray(result["predicted_labels"], dtype=jnp.int32)

    return AnnotationProbeResult(
        probe=probe,
        train_features=train_features,
        test_features=test_features,
        train_labels=train_labels,
        test_labels=test_labels,
        predicted_labels=predicted_labels,
    )


def score_annotation(outcome: AnnotationProbeResult) -> dict[str, float]:
    """Score a trained annotation probe with calibrax macro-F1 and balanced accuracy."""
    return {
        "macro_f1": float(f1_score(outcome.predicted_labels, outcome.test_labels, average="macro")),
        "balanced_accuracy": float(
            balanced_accuracy(outcome.predicted_labels, outcome.test_labels)
        ),
    }


def annotation_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int,
    seed: int = 42,
    n_train_steps: int = 200,
    learning_rate: float = 1.0e-2,
    train_fraction: float = 0.8,
) -> dict[str, float]:
    """Train a linear probe on frozen features and score cell annotation.

    Args:
        features: ``(n_cells, n_features)`` frozen feature matrix.
        labels: ``(n_cells,)`` integer cell-type labels.
        n_classes: Number of distinct cell-type classes.
        seed: Seed for the split, probe initialization, and optimizer.
        n_train_steps: Number of full-batch probe training steps.
        learning_rate: Probe optimizer learning rate.
        train_fraction: Fraction of cells assigned to the training split.

    Returns:
        A mapping with ``macro_f1`` and ``balanced_accuracy`` on the held-out split.
    """
    outcome = train_annotation_probe(
        features,
        labels,
        n_classes=n_classes,
        seed=seed,
        n_train_steps=n_train_steps,
        learning_rate=learning_rate,
        train_fraction=train_fraction,
    )
    return score_annotation(outcome)
