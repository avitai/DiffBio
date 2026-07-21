"""STG arm of the HVG-gate study: stochastic-gate L0 selection vs frozen dispersion HVG.

Companion to ``hvg_gate.py``. The frozen arm is identical (dispersion top-k into a fixed
PCA); the joint arm here is a :class:`StochasticGateSelector` initialized from the frozen
top-k mask (init-at-frozen) and trained with the task loss plus an L0 sparsity penalty
(via ``train_minibatch``'s ``aux_loss_fn``). Training injects gate noise; evaluation uses the
deterministic ``clip(mu, 0, 1)`` gate. This tests the STG mechanism against the same frozen
baseline the ``SoftHVG`` top-k arm beats, at the k values where that gain is significant.

Run: ``python -m benchmarks.singlecell.hvg_gate_stg`` (needs the cached Tabula Sapiens atlas).
"""

from __future__ import annotations

import json
import os
from functools import partial

import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmarks._classification import stratified_label_split
from benchmarks.singlecell._gate2_arms import _embedding_probe, _probe_forward
from benchmarks.singlecell.hvg_gate import (
    _HIDDEN,
    _N_COMPONENTS,
    _TRAIN_FRACTION,
    balanced_acc,
    log_normalize,
    macro_f1,
)
from diffbio.core import soft_ops
from diffbio.operators.singlecell.soft_hvg import gene_dispersion
from diffbio.operators.singlecell.stochastic_gate_selector import (
    StochasticGateSelector,
    StochasticGateSelectorConfig,
    l0_penalty,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch
from diffbio.reductions import fit_pca_reduction

_DATA_PATH = os.environ.get("DIFFBIO_TS_CACHE", "/mnt/ssd2/Data/tabula_sapiens/ts_cached.npz")
OUT = "benchmarks/results/singlecell/hvg_gate_stg.json"
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
K_VALUES = (200, 500)
_SIGMA = 0.5
_L0_LAMBDA = 1.0e-3


class _STGGateProbe(nnx.Module):
    """A stochastic-gate selector composed with an annotation probe."""

    def __init__(self, selector: StochasticGateSelector, probe: nnx.Module) -> None:
        """Store the STG selector and probe submodules."""
        self.selector = selector
        self.probe = probe


def _stg_train_forward(
    model: _STGGateProbe, scaled: jnp.ndarray, *, pca_mean: jnp.ndarray, loadings: jnp.ndarray
) -> jnp.ndarray:
    """Noise-injected forward: gate genes stochastically, project, and return logits."""
    gated = model.selector.apply({"features": scaled}, {}, None)[0]["features"]
    embedding = (gated - pca_mean) @ loadings
    return model.probe.apply({"embeddings": embedding}, {}, None)[0]["logits"]


def _stg_eval_forward(
    model: _STGGateProbe, scaled: jnp.ndarray, *, pca_mean: jnp.ndarray, loadings: jnp.ndarray
) -> jnp.ndarray:
    """Deterministic forward: gate by clip(mu, 0, 1), project, and return logits."""
    gate = jnp.clip(model.selector.mu[...], 0.0, 1.0)
    embedding = (scaled * gate[None, :] - pca_mean) @ loadings
    return model.probe.apply({"embeddings": embedding}, {}, None)[0]["logits"]


def _stg_l0_aux(model: _STGGateProbe, *, l0_lambda: float, sigma: float) -> jnp.ndarray:
    """L0 sparsity penalty on the gate means for train_minibatch's aux hook."""
    return l0_lambda * l0_penalty(model.selector.mu[...], sigma)


def run_k_stg(
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
    l0_lambda: float = _L0_LAMBDA,
) -> dict[str, object]:
    """Compare frozen dispersion top-k against the STG L0 gate at one ``k`` across seeds."""
    n_genes = int(scaled_train.shape[1])
    n_components = int(loadings.shape[1])
    frozen_mask = (soft_ops.top_k_mask(base_dispersion, k, mode="hard") > 0.5).astype(jnp.float32)
    xf_train = (scaled_train * frozen_mask[None, :] - pca_mean) @ loadings
    xf_test = (scaled_test * frozen_mask[None, :] - pca_mean) @ loadings

    frozen_f1: list[float] = []
    stg_f1: list[float] = []
    stg_bal: list[float] = []
    open_genes: list[float] = []
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

        selector = StochasticGateSelector(
            StochasticGateSelectorConfig(
                n_genes=n_genes, sigma=_SIGMA, stochastic=True, stream_name="gate_noise"
            ),
            init_gate=frozen_mask,
            rngs=nnx.Rngs(seed, gate_noise=seed),
        )
        model = _STGGateProbe(selector, _embedding_probe(n_components, n_classes, _HIDDEN, seed))
        train_minibatch(
            model,
            partial(_stg_train_forward, pca_mean=pca_mean, loadings=loadings),
            scaled_train,
            train_labels,
            n_classes=n_classes,
            config=config,
            aux_loss_fn=partial(_stg_l0_aux, l0_lambda=l0_lambda, sigma=_SIGMA),
        )
        pred_stg = np.asarray(
            jnp.argmax(
                _stg_eval_forward(model, scaled_test, pca_mean=pca_mean, loadings=loadings), -1
            )
        )
        stg_f1.append(macro_f1(pred_stg, test_labels))
        stg_bal.append(balanced_acc(pred_stg, test_labels))
        open_genes.append(float(jnp.sum(jnp.clip(model.selector.mu[...], 0.0, 1.0) > 0.5)))
        print(
            f"  k={k:4d} [seed {seed}] frozen {frozen_f1[-1]:.4f}  stg {stg_f1[-1]:.4f}  "
            f"gain {100 * (stg_f1[-1] - frozen_f1[-1]):+.2f}pp  open~{open_genes[-1]:.0f}",
            flush=True,
        )

    from calibrax.statistics.significance import paired_significance_test

    significance = paired_significance_test(frozen_f1, stg_f1)
    return {
        "k": k,
        "frozen_macro_f1": [float(np.mean(frozen_f1)), float(np.std(frozen_f1))],
        "stg_macro_f1": [float(np.mean(stg_f1)), float(np.std(stg_f1))],
        "gain_pp": float(100 * (np.mean(stg_f1) - np.mean(frozen_f1))),
        "open_genes_mean": float(np.mean(open_genes)),
        "paired_p_value": float(significance.p_value),
        "paired_significant": bool(significance.significant),
    }


def run_dataset_stg(
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
    """Fit frozen stats once and sweep ``k`` for the frozen-vs-STG comparison."""
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
        "n_types": n_classes,
        "l0_lambda": _L0_LAMBDA,
        "by_k": [
            run_k_stg(
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


def main() -> None:
    """Run the frozen-vs-STG HVG-gate comparison on the cached Tabula Sapiens atlas."""
    with np.load(_DATA_PATH) as data:
        counts = np.asarray(data["counts"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int32)
        n_classes = int(data["n_types"])

    print("=== HVG gate STG: frozen dispersion top-k vs stochastic-gate L0 ===", flush=True)
    result = run_dataset_stg(counts, labels, n_classes)
    for row in result["by_k"]:  # type: ignore[attr-defined]
        print(
            f"k={row['k']:4d}: frozen {row['frozen_macro_f1'][0]:.4f} "
            f"stg {row['stg_macro_f1'][0]:.4f} gain {row['gain_pp']:+.2f}pp "
            f"open~{row['open_genes_mean']:.0f} p={row['paired_p_value']:.3f}",
            flush=True,
        )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"HVG GATE STG DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
