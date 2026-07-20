"""Case study A: scATAC cell-type annotation, frozen TF-IDF+LSI vs learnable projection.

New modality (chromatin accessibility), SAME machinery/test as the scRNA moat: a frozen
lossy k-bottleneck (Signac TF-IDF + truncated SVD, depth component dropped) vs a learnable
projection initialized from those LSI loadings and jointly trained with the classifier,
swept over the reduction dimension k. Real CATLAS (GSE184462) cells; stratified split.

This is the correctly-designed hypothesis test the variant case study lacked: TF-IDF+SVD
is a genuine compression bottleneck that discards accessibility signal at small k, so the
hypothesis predicts a joint gain concentrated at aggressive k (the gain-vs-k signature).
"""

import os

import json

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from calibrax.metrics.functional.classification import balanced_accuracy, f1_score
from flax import nnx

from benchmarks.singlecell._gate2_arms import (
    _embedding_probe,
    _probe_forward,
    _project_probe_forward,
    _ProjectionProbe,
)
from diffbio.operators.normalization.learnable_projection import (
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch
from diffbio.reductions import fit_tfidf_reduction


_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
COUNTS = f"{_DATA}/catlas/atac_counts.npz"
META = f"{_DATA}/catlas/atac_meta.npz"
OUT = "benchmarks/results/crossmodality/gate_atac.json"
SEEDS = (0, 1, 2)
K_VALUES = [5, 10, 20, 50]
MAX_COMPONENTS = 50


def stratified_split(
    labels: np.ndarray, frac_test: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean train/test masks with each cell type split by ``frac_test``."""
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * frac_test))
        test_mask[idx[:n_test]] = True
    return ~test_mask, test_mask


def macro(pred: np.ndarray, true: np.ndarray) -> float:
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(f1_score(jnp.asarray(pred), jnp.asarray(true), average="macro"))


def main() -> None:
    """Fit the frozen LSI once, then sweep k for frozen vs learnable projection."""
    counts = sp.load_npz(COUNTS).tocsr()
    meta = np.load(META)  # own build output; string arrays save as unicode, no pickle needed
    labels = meta["labels"].astype(np.int32)
    n_types = len(meta["type_names"])
    train_mask, test_mask = stratified_split(labels, frac_test=0.3, seed=0)
    tr_counts, te_counts = counts[train_mask], counts[test_mask]
    tr_y, te_y = labels[train_mask], labels[test_mask]
    print(
        f"cells {counts.shape} | train {tr_counts.shape[0]} test {te_counts.shape[0]} "
        f"| {n_types} cell types",
        flush=True,
    )

    # Frozen LSI is deterministic -> fit once; densify the TF-IDF for the learnable arm once.
    reduction = fit_tfidf_reduction(tr_counts, MAX_COMPONENTS)
    tr_scaled = reduction.scaled(tr_counts)
    te_scaled = reduction.scaled(te_counts)
    tr_dense = jnp.asarray(np.asarray(tr_scaled.todense(), dtype=np.float32))
    te_dense = jnp.asarray(np.asarray(te_scaled.todense(), dtype=np.float32))
    n_features = reduction.loadings.shape[0]
    print(f"LSI fit; TF-IDF densified train {tr_dense.shape}", flush=True)

    frozen = {k: [] for k in K_VALUES}
    joint = {k: [] for k in K_VALUES}
    bal = {"frozen": [], "joint": []}
    for seed in SEEDS:
        config = MiniBatchConfig(
            batch_size=2048, n_epochs=60, learning_rate=1e-2, weight_decay=5e-2, seed=seed
        )
        for k in K_VALUES:
            loadings_k = reduction.loadings[:, :k]
            xtr_f = jnp.asarray(np.asarray(tr_scaled @ loadings_k, dtype=np.float32))
            xte_f = jnp.asarray(np.asarray(te_scaled @ loadings_k, dtype=np.float32))
            probe_f = _embedding_probe(k, n_types, 128, seed)
            train_minibatch(probe_f, _probe_forward, xtr_f, tr_y, n_classes=n_types, config=config)
            pred_f = np.asarray(jnp.argmax(_probe_forward(probe_f, xte_f), -1))
            frozen[k].append(macro(pred_f, te_y))

            proj = LearnableProjection(
                LearnableProjectionConfig(n_genes=n_features, n_components=k),
                init_loadings=loadings_k,
                rngs=nnx.Rngs(seed),
            )
            model = _ProjectionProbe(proj, _embedding_probe(k, n_types, 128, seed))
            train_minibatch(
                model, _project_probe_forward, tr_dense, tr_y, n_classes=n_types, config=config
            )
            pred_j = np.asarray(jnp.argmax(_project_probe_forward(model, te_dense), -1))
            joint[k].append(macro(pred_j, te_y))
            if k == 10:
                bal["frozen"].append(
                    float(balanced_accuracy(jnp.asarray(pred_f), jnp.asarray(te_y)))
                )
                bal["joint"].append(
                    float(balanced_accuracy(jnp.asarray(pred_j), jnp.asarray(te_y)))
                )
            print(
                f"  [seed {seed}] k={k:2d} frozen {frozen[k][-1]:.4f} joint {joint[k][-1]:.4f}",
                flush=True,
            )

    print("=== scATAC cell-type annotation (TF-IDF+LSI vs learnable) SUMMARY ===", flush=True)
    for k in K_VALUES:
        fm, jm = np.mean(frozen[k]), np.mean(joint[k])
        print(
            f"k={k:2d}: frozen {fm:.4f}+/-{np.std(frozen[k]):.4f}  "
            f"joint {jm:.4f}+/-{np.std(joint[k]):.4f}  gain {100 * (jm - fm):+.1f}pp",
            flush=True,
        )
    print(
        f"balanced-acc @k=10: frozen {np.mean(bal['frozen']):.4f} "
        f"joint {np.mean(bal['joint']):.4f}",
        flush=True,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(
            {
                "k_values": K_VALUES,
                "frozen": {
                    str(k): [float(np.mean(v)), float(np.std(v))] for k, v in frozen.items()
                },
                "joint": {str(k): [float(np.mean(v)), float(np.std(v))] for k, v in joint.items()},
            },
            handle,
            indent=2,
        )
    print("ATAC GATE DONE", flush=True)


if __name__ == "__main__":
    main()
