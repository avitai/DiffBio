"""Budget-matched control for the STG gene-selection result.

The headline STG study (``hvg_gate_stg``) compared an unconstrained stochastic gate against
frozen dispersion top-k at a *nominal* k, but the STG gate self-selects its panel size
(~90-116 genes) rather than honouring k -- so that comparison is not budget-matched. This
control removes the confound: after training STG, it counts the genes the gate actually keeps
open and evaluates frozen dispersion selection at *that same count*, plus the overlap between
the two gene sets. If STG still wins at matched budget with low overlap, the gain is genuine
selection quality, not a budget artifact.
"""

from __future__ import annotations

import json
import os

import jax.numpy as jnp
import numpy as np
from calibrax.statistics.significance import paired_significance_test
from flax import nnx

from benchmarks._classification import stratified_label_split
from benchmarks.singlecell._gate2_arms import _embedding_probe, _probe_forward
from benchmarks.singlecell.hvg_gate import (
    _HIDDEN,
    _N_COMPONENTS,
    _TRAIN_FRACTION,
    log_normalize,
    macro_f1,
)
from benchmarks.singlecell.hvg_gate_stg import (
    _SIGMA,
    _STGGateProbe,
    _stg_eval_forward,
    _stg_l0_aux,
    _stg_train_forward,
)
from diffbio.core import soft_ops
from diffbio.operators.singlecell.soft_hvg import gene_dispersion
from diffbio.operators.singlecell.stochastic_gate_selector import (
    StochasticGateSelector,
    StochasticGateSelectorConfig,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch
from diffbio.reductions import fit_pca_reduction

_DATA_PATH = os.environ.get("DIFFBIO_TS_CACHE", "/mnt/ssd2/Data/tabula_sapiens/ts_cached.npz")
OUT = "benchmarks/results/singlecell/hvg_gate_stg_matched.json"
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
_K_INIT = 200
_L0_LAMBDA = 1.0e-3


def _top_k_indices(base_dispersion: jnp.ndarray, k: int) -> np.ndarray:
    """Return the indices of the top-k dispersion genes."""
    return np.asarray(jnp.argsort(base_dispersion)[::-1][:k])


def run_matched(
    counts: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    *,
    seeds: tuple[int, ...] = SEEDS,
    k_init: int = _K_INIT,
    n_epochs: int = 40,
    batch_size: int = 4096,
) -> dict[str, object]:
    """Compare STG against frozen dispersion at STG's own selected gene count."""
    train_idx, test_idx = stratified_label_split(
        labels, train_fraction=_TRAIN_FRACTION, seed=0, minimum_count_name="cells"
    )
    logged = log_normalize(counts)
    reduction = fit_pca_reduction(logged[train_idx], _N_COMPONENTS)
    scaled_train = jnp.asarray(reduction.scaled(logged[train_idx]))
    scaled_test = jnp.asarray(reduction.scaled(logged[test_idx]))
    pca_mean = jnp.asarray(reduction.pca_mean)
    loadings = jnp.asarray(reduction.loadings)
    base_dispersion = gene_dispersion(logged[train_idx])
    y_train, y_test = labels[train_idx], labels[test_idx]
    n_genes = int(counts.shape[1])
    init_mask = (soft_ops.top_k_mask(base_dispersion, k_init, mode="hard") > 0.5).astype(
        jnp.float32
    )

    stg_f1: list[float] = []
    frozen_f1: list[float] = []
    open_counts: list[int] = []
    overlaps: list[float] = []
    for seed in seeds:
        config = MiniBatchConfig(
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=1.0e-2,
            weight_decay=0.0,
            seed=seed,
        )
        selector = StochasticGateSelector(
            StochasticGateSelectorConfig(
                n_genes=n_genes, sigma=_SIGMA, stochastic=True, stream_name="gate_noise"
            ),
            init_gate=init_mask,
            rngs=nnx.Rngs(seed, gate_noise=seed),
        )
        model = _STGGateProbe(selector, _embedding_probe(_N_COMPONENTS, n_classes, _HIDDEN, seed))
        train_minibatch(
            model,
            lambda m, x: _stg_train_forward(m, x, pca_mean=pca_mean, loadings=loadings),
            scaled_train,
            y_train,
            n_classes=n_classes,
            config=config,
            aux_loss_fn=lambda m: _stg_l0_aux(m, l0_lambda=_L0_LAMBDA, sigma=_SIGMA),
        )
        stg_pred = np.asarray(
            jnp.argmax(
                _stg_eval_forward(model, scaled_test, pca_mean=pca_mean, loadings=loadings), -1
            )
        )
        stg_f1.append(macro_f1(stg_pred, y_test))

        open_gene_mask = np.asarray(jnp.clip(model.selector.mu[...], 0.0, 1.0) > 0.5)
        open_idx = np.flatnonzero(open_gene_mask)
        k_matched = max(1, int(len(open_idx)))
        open_counts.append(k_matched)

        # Frozen dispersion at the SAME gene count STG chose.
        disp_idx = _top_k_indices(base_dispersion, k_matched)
        frozen_mask = jnp.asarray(np.isin(np.arange(n_genes), disp_idx).astype(np.float32))
        xf_train = (scaled_train * frozen_mask[None, :] - pca_mean) @ loadings
        xf_test = (scaled_test * frozen_mask[None, :] - pca_mean) @ loadings
        probe_frozen = _embedding_probe(_N_COMPONENTS, n_classes, _HIDDEN, seed)
        train_minibatch(
            probe_frozen, _probe_forward, xf_train, y_train, n_classes=n_classes, config=config
        )
        frozen_pred = np.asarray(jnp.argmax(_probe_forward(probe_frozen, xf_test), -1))
        frozen_f1.append(macro_f1(frozen_pred, y_test))

        overlaps.append(float(len(np.intersect1d(open_idx, disp_idx)) / k_matched))
        gain = 100 * (stg_f1[-1] - frozen_f1[-1])
        print(
            f"  [seed {seed}] STG {stg_f1[-1]:.4f} ({k_matched} genes)  "
            f"frozen@{k_matched} {frozen_f1[-1]:.4f}  gain {gain:+.2f}pp  "
            f"overlap {overlaps[-1]:.2f}",
            flush=True,
        )

    significance = paired_significance_test(frozen_f1, stg_f1)
    return {
        "k_init": k_init,
        "stg_macro_f1": [float(np.mean(stg_f1)), float(np.std(stg_f1))],
        "frozen_matched_macro_f1": [float(np.mean(frozen_f1)), float(np.std(frozen_f1))],
        "gain_pp": float(100 * (np.mean(stg_f1) - np.mean(frozen_f1))),
        "open_genes_mean": float(np.mean(open_counts)),
        "gene_overlap_mean": float(np.mean(overlaps)),
        "paired_p_value": float(significance.p_value),
        "paired_significant": bool(significance.significant),
    }


def main() -> None:
    """Run the budget-matched STG control on the cached Tabula Sapiens atlas."""
    with np.load(_DATA_PATH) as data:
        counts = np.asarray(data["counts"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int32)
        n_classes = int(data["n_types"])

    print("=== STG budget-matched control: STG vs frozen at STG's own gene count ===", flush=True)
    result = run_matched(counts, labels, n_classes)
    print(
        f"STG {result['stg_macro_f1'][0]:.4f} ({result['open_genes_mean']:.0f} genes)  "  # type: ignore[index]
        f"frozen@matched {result['frozen_matched_macro_f1'][0]:.4f}  "
        f"gain {result['gain_pp']:+.2f}pp  overlap {result['gene_overlap_mean']:.2f}  "
        f"p={result['paired_p_value']:.3f}",
        flush=True,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"STG MATCHED DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
