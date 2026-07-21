"""Case study: frozen scanpy HVG vs a jointly-trained differentiable gene-selection gate.

The scRNA gate-2 study trains the PCA *projection* with highly-variable-gene (HVG)
selection frozen. This study makes the *gene-selection gate itself* the trained degree of
freedom, holding the projection and everything downstream fixed:

- Frozen arm: rank the candidate genes by normalized dispersion (the scanpy ``seurat``
  measure), hard-select the top k, feed the masked z-scored genes through a fixed PCA, and
  train a probe on the PCA embedding.
- Joint arm: the same, but the top-k mask comes from a learnable per-gene gate
  (``base_dispersion * softplus(gene_weights)``, ``gene_weights`` initialized at 0), trained
  jointly with the probe. At init the softplus is a positive constant, so the ranking -- and
  therefore the selected genes -- match the frozen arm exactly (init-at-frozen); any gain
  comes from the gate re-selecting genes the task cares about that dispersion under-ranks.

The PCA is fit once on the full candidate pool (all genes in the cached matrix), so its
loadings cover every gene and the gate has real leverage: changing which genes are unmasked
changes the dense PCA embedding the probe sees. Both arms share that frozen PCA and probe
initialization, so the only moving part is *which* k genes feed it -- the selection effect in
isolation. The hypothesis predicts gains concentrate at aggressive small k, where
dispersion-vs-task-relevance disagreements matter most.

Reuses ``diffbio.operators.singlecell.soft_hvg`` (dispersion + straight-through top-k),
``diffbio.reductions`` (PCA), ``_gate2_arms`` (probe), ``train_minibatch``, and calibrax
metrics + ``paired_significance_test``.

Run: ``python -m benchmarks.singlecell.hvg_gate`` (needs the cached Tabula Sapiens atlas), or
``--smoke`` to exercise the pipeline on synthetic data with no staged inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial

import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import balanced_accuracy, f1_score
from calibrax.statistics.significance import paired_significance_test
from flax import nnx

from benchmarks._classification import stratified_label_split
from benchmarks.singlecell._gate2_arms import _embedding_probe, _probe_forward
from diffbio.core import soft_ops
from diffbio.operators.singlecell.soft_hvg import gene_dispersion
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch
from diffbio.reductions import fit_pca_reduction

_DATA_PATH = os.environ.get("DIFFBIO_TS_CACHE", "/mnt/ssd2/Data/tabula_sapiens/ts_cached.npz")
OUT = "benchmarks/results/singlecell/hvg_gate.json"
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
K_VALUES = (100, 200, 500, 1000, 2000)
_TARGET_SUM = 1.0e4
_N_COMPONENTS = 50
_SOFTNESS = 0.1
_HIDDEN = 128
_TRAIN_FRACTION = 0.8


def log_normalize(counts: np.ndarray) -> np.ndarray:
    """Depth-normalize to a fixed library size and ``log1p`` (the frozen scanpy step)."""
    library = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(counts / library * _TARGET_SUM).astype(np.float32)


class _HVGGateProbe(nnx.Module):
    """A learnable per-gene selection gate composed with an annotation probe."""

    def __init__(self, n_genes: int, probe: nnx.Module) -> None:
        """Store the zero-initialized gate weights and the probe.

        Args:
            n_genes: Number of candidate genes (the gate width).
            probe: The annotation classifier head.
        """
        self.gene_weights = nnx.Param(jnp.zeros(n_genes, dtype=jnp.float32))
        self.probe = probe


def _gate_mask(gene_weights: jnp.ndarray, base_dispersion: jnp.ndarray, k: int) -> jnp.ndarray:
    """Straight-through top-k mask from dispersion modulated by the learnable gate."""
    score = base_dispersion * nnx.softplus(gene_weights)
    return soft_ops.top_k_mask_st(score, k, softness=_SOFTNESS)


def _project(
    scaled: jnp.ndarray, mask: jnp.ndarray, pca_mean: jnp.ndarray, loadings: jnp.ndarray
) -> jnp.ndarray:
    """Gate the scaled genes by ``mask`` and project through the frozen PCA loadings."""
    return (scaled * mask[None, :] - pca_mean) @ loadings


def _gate_probe_forward(
    model: _HVGGateProbe,
    scaled: jnp.ndarray,
    *,
    base_dispersion: jnp.ndarray,
    pca_mean: jnp.ndarray,
    loadings: jnp.ndarray,
    k: int,
) -> jnp.ndarray:
    """Gate genes by the learnable mask, project, and return probe logits."""
    mask = _gate_mask(model.gene_weights[...], base_dispersion, k)
    embedding = _project(scaled, mask, pca_mean, loadings)
    return model.probe.apply({"embeddings": embedding}, {}, None)[0]["logits"]


def macro_f1(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(f1_score(jnp.asarray(predictions), jnp.asarray(targets), average="macro"))


def balanced_acc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the balanced accuracy of predictions against ground truth."""
    return float(balanced_accuracy(jnp.asarray(predictions), jnp.asarray(targets)))


def run_k(
    scaled_train: jnp.ndarray,
    scaled_test: jnp.ndarray,
    base_dispersion: jnp.ndarray,
    pca_mean: jnp.ndarray,
    loadings: jnp.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    *,
    k: int,
    n_classes: int,
    seeds: tuple[int, ...],
    n_epochs: int,
    batch_size: int,
) -> dict[str, object]:
    """Run the frozen-vs-joint gate comparison at one selection size ``k`` across seeds."""
    n_components = int(loadings.shape[1])
    frozen_mask = (soft_ops.top_k_mask(base_dispersion, k, mode="hard") > 0.5).astype(jnp.float32)
    xf_train = _project(scaled_train, frozen_mask, pca_mean, loadings)
    xf_test = _project(scaled_test, frozen_mask, pca_mean, loadings)

    frozen_f1: list[float] = []
    joint_f1: list[float] = []
    frozen_bal: list[float] = []
    joint_bal: list[float] = []
    for seed in seeds:
        config = MiniBatchConfig(
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=1.0e-2,
            weight_decay=0.0,
            seed=seed,
        )
        probe_frozen = _embedding_probe(n_components, n_classes, _HIDDEN, seed)
        train_minibatch(
            probe_frozen, _probe_forward, xf_train, train_labels, n_classes=n_classes, config=config
        )
        pred_frozen = np.asarray(jnp.argmax(_probe_forward(probe_frozen, xf_test), -1))
        frozen_f1.append(macro_f1(pred_frozen, test_labels))
        frozen_bal.append(balanced_acc(pred_frozen, test_labels))

        n_genes = int(scaled_train.shape[1])
        model = _HVGGateProbe(n_genes, _embedding_probe(n_components, n_classes, _HIDDEN, seed))
        forward = partial(
            _gate_probe_forward,
            base_dispersion=base_dispersion,
            pca_mean=pca_mean,
            loadings=loadings,
            k=k,
        )
        train_minibatch(
            model, forward, scaled_train, train_labels, n_classes=n_classes, config=config
        )
        pred_joint = np.asarray(jnp.argmax(forward(model, scaled_test), -1))
        joint_f1.append(macro_f1(pred_joint, test_labels))
        joint_bal.append(balanced_acc(pred_joint, test_labels))
        print(
            f"  k={k:4d} [seed {seed}] frozen {frozen_f1[-1]:.4f}  joint {joint_f1[-1]:.4f}  "
            f"gain {100 * (joint_f1[-1] - frozen_f1[-1]):+.2f}pp",
            flush=True,
        )

    significance = paired_significance_test(frozen_f1, joint_f1)
    return {
        "k": k,
        "frozen_macro_f1": [float(np.mean(frozen_f1)), float(np.std(frozen_f1))],
        "joint_macro_f1": [float(np.mean(joint_f1)), float(np.std(joint_f1))],
        "gain_pp": float(100 * (np.mean(joint_f1) - np.mean(frozen_f1))),
        "frozen_balanced_acc": [float(np.mean(frozen_bal)), float(np.std(frozen_bal))],
        "joint_balanced_acc": [float(np.mean(joint_bal)), float(np.std(joint_bal))],
        "paired_p_value": float(significance.p_value),
        "paired_significant": bool(significance.significant),
    }


def run_dataset(
    counts: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    *,
    seeds: tuple[int, ...] = SEEDS,
    k_values: tuple[int, ...] = K_VALUES,
    n_components: int = _N_COMPONENTS,
    n_epochs: int = 40,
    batch_size: int = 4096,
) -> dict[str, object]:
    """Split, fit frozen stats once, and sweep ``k`` for the frozen-vs-joint gate study."""
    train_idx, test_idx = stratified_label_split(
        labels, train_fraction=_TRAIN_FRACTION, seed=0, minimum_count_name="cells"
    )
    logged = log_normalize(counts)
    reduction = fit_pca_reduction(logged[train_idx], n_components)
    scaled_train = jnp.asarray(reduction.scaled(logged[train_idx]))
    scaled_test = jnp.asarray(reduction.scaled(logged[test_idx]))
    pca_mean = jnp.asarray(reduction.pca_mean)
    loadings = jnp.asarray(reduction.loadings)
    base_dispersion = gene_dispersion(logged[train_idx])
    y_train, y_test = labels[train_idx], labels[test_idx]
    print(
        f"  cells {counts.shape} | train {len(train_idx)} test {len(test_idx)} "
        f"| {n_classes} types {counts.shape[1]} genes | PCA {n_components}",
        flush=True,
    )
    return {
        "n_cells": int(counts.shape[0]),
        "n_genes": int(counts.shape[1]),
        "n_types": n_classes,
        "n_components": n_components,
        "by_k": [
            run_k(
                scaled_train,
                scaled_test,
                base_dispersion,
                pca_mean,
                loadings,
                y_train,
                y_test,
                k=k,
                n_classes=n_classes,
                seeds=seeds,
                n_epochs=n_epochs,
                batch_size=batch_size,
            )
            for k in k_values
        ],
    }


def _synthetic_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """Build data where the class-informative genes are NOT the highest-dispersion ones."""
    rng = np.random.default_rng(seed)
    n_cells, n_genes, n_types = 3000, 200, 6
    n_informative = 20
    labels = rng.integers(0, n_types, size=n_cells)
    counts = rng.gamma(2.0, 20.0, size=(n_cells, n_genes)).astype(np.float32)
    # Class signal lives in genes [0:20]; heavy noise variance in genes [180:200] gives
    # those high dispersion without carrying class information -- the dispersion trap.
    informative = rng.uniform(30.0, 60.0, size=(n_types, n_informative))
    for cell in range(n_cells):
        counts[cell, :n_informative] += informative[labels[cell]]
    counts[:, -20:] += rng.gamma(2.0, 200.0, size=(n_cells, 20)).astype(np.float32)
    return counts, labels.astype(np.int32), n_types


def _smoke() -> None:
    """Exercise the pipeline on synthetic data and assert init-at-frozen equality."""
    counts, labels, n_types = _synthetic_dataset()
    logged = log_normalize(counts)
    reduction = fit_pca_reduction(logged, 10)
    scaled = jnp.asarray(reduction.scaled(logged))
    pca_mean = jnp.asarray(reduction.pca_mean)
    loadings = jnp.asarray(reduction.loadings)
    base_dispersion = gene_dispersion(logged)
    k = 30

    # Init-at-frozen: with gene_weights=0 the gate embedding equals the frozen top-k embedding.
    frozen_mask = (soft_ops.top_k_mask(base_dispersion, k, mode="hard") > 0.5).astype(jnp.float32)
    gate_mask = _gate_mask(jnp.zeros(counts.shape[1]), base_dispersion, k)
    np.testing.assert_allclose(
        np.asarray(_project(scaled, frozen_mask, pca_mean, loadings)),
        np.asarray(_project(scaled, gate_mask, pca_mean, loadings)),
        atol=1e-5,
        err_msg="init-at-frozen violated: gate embedding != frozen embedding at init",
    )
    print("init-at-frozen OK: gate embedding == frozen embedding at init", flush=True)

    result = run_dataset(
        counts,
        labels,
        n_types,
        seeds=(0, 1),
        k_values=(10, 30),
        n_components=10,
        n_epochs=25,
        batch_size=512,
    )
    print(json.dumps(result, indent=2), flush=True)
    print("HVG GATE SMOKE DONE", flush=True)


def main() -> None:
    """Run the HVG-gate study on the cached atlas (or the synthetic smoke run)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="run on synthetic data, no staged inputs"
    )
    args = parser.parse_args()

    if args.smoke:
        _smoke()
        return

    with np.load(_DATA_PATH) as data:
        counts = np.asarray(data["counts"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int32)
        n_classes = int(data["n_types"])

    print(
        "=== HVG gate: frozen dispersion top-k vs learnable gate (Tabula Sapiens) ===", flush=True
    )
    result = run_dataset(counts, labels, n_classes)
    for row in result["by_k"]:  # type: ignore[attr-defined]
        print(
            f"k={row['k']:4d}: frozen {row['frozen_macro_f1'][0]:.4f} "
            f"joint {row['joint_macro_f1'][0]:.4f} gain {row['gain_pp']:+.2f}pp "
            f"p={row['paired_p_value']:.3f}",
            flush=True,
        )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"HVG GATE DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
