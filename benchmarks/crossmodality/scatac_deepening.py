"""Deepen case study A (scATAC): a domain baseline and a data-scaling curve.

Two additions matching the depth of the single-cell annotation result:

  1. Domain baseline -- the standard Signac scATAC annotation pipeline: TF-IDF + LSI
     (depth component dropped) at the dimensionality practitioners use (50), followed by
     the two label-transfer classifiers the field actually uses (multinomial logistic
     regression and kNN in LSI space). This checks the frozen arm we compare against is a
     faithful, competitive representation of standard practice, not a strawman.

  2. Scaling curve -- the joint-minus-frozen gain at aggressive reduction (k=5) as the
     training set grows, to test whether (as for scRNA) the advantage grows with data and
     crosses from negative on small data to strongly positive at scale.

Real CATLAS (GSE184462) cells; the frozen LSI is fit on the training split only.
"""

import os

import json

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from calibrax.metrics.functional.classification import f1_score
from flax import nnx
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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
OUT = "benchmarks/results/crossmodality/gate_atac_deepen.json"
SEEDS = (0, 1, 2)
K = 5
DOMAIN_DIM = 50
TRAIN_SIZES = [5000, 12000, 30000, 60000, None]  # None = full training split


def stratified_split(labels, frac_test, seed):
    """Boolean train/test masks with each cell type split by ``frac_test`` (matches gate_atac)."""
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        test_mask[idx[: max(1, int(len(idx) * frac_test))]] = True
    return ~test_mask, test_mask


def macro(pred, true):
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(
        f1_score(jnp.asarray(np.asarray(pred)), jnp.asarray(np.asarray(true)), average="macro")
    )


def frozen_vs_joint(tr_counts, tr_y, te_scaled, te_y, n_types, seed):
    """Train frozen-PCA-probe and joint-projection-probe arms at k=K; return (frozen, joint) F1."""
    reduction = fit_tfidf_reduction(tr_counts, max(K, DOMAIN_DIM))
    tr_scaled = reduction.scaled(tr_counts)
    loadings_k = reduction.loadings[:, :K]
    config = MiniBatchConfig(
        batch_size=2048, n_epochs=60, learning_rate=1e-2, weight_decay=5e-2, seed=seed
    )

    xtr_f = jnp.asarray(np.asarray(tr_scaled @ loadings_k, np.float32))
    xte_f = jnp.asarray(np.asarray(te_scaled @ loadings_k, np.float32))
    probe_f = _embedding_probe(K, n_types, 128, seed)
    train_minibatch(probe_f, _probe_forward, xtr_f, tr_y, n_classes=n_types, config=config)
    frozen = macro(jnp.argmax(_probe_forward(probe_f, xte_f), -1), te_y)

    tr_dense = jnp.asarray(np.asarray(tr_scaled.todense(), np.float32))
    te_dense = jnp.asarray(np.asarray(te_scaled.todense(), np.float32))
    proj = LearnableProjection(
        LearnableProjectionConfig(n_genes=reduction.loadings.shape[0], n_components=K),
        init_loadings=loadings_k,
        rngs=nnx.Rngs(seed),
    )
    model = _ProjectionProbe(proj, _embedding_probe(K, n_types, 128, seed))
    train_minibatch(model, _project_probe_forward, tr_dense, tr_y, n_classes=n_types, config=config)
    joint = macro(jnp.argmax(_project_probe_forward(model, te_dense), -1), te_y)
    return frozen, joint


def main() -> None:
    """Run the domain baseline and the scaling curve, saving both to JSON."""
    counts = sp.load_npz(COUNTS).tocsr()
    meta = np.load(META)
    labels = meta["labels"].astype(np.int32)
    n_types = len(meta["type_names"])
    train_mask, test_mask = stratified_split(labels, 0.3, 0)
    tr_all, te_counts = counts[train_mask], counts[test_mask]
    tr_y_all, te_y = labels[train_mask], labels[test_mask]
    print(f"train {tr_all.shape[0]} test {te_counts.shape[0]} | {n_types} types", flush=True)

    # --- 1. Domain baseline: standard LSI + label-transfer classifiers -------------
    reduction = fit_tfidf_reduction(tr_all, DOMAIN_DIM)
    xtr = reduction.transform(tr_all)
    xte = reduction.transform(te_counts)
    logreg = LogisticRegression(max_iter=2000, C=1.0)
    logreg.fit(xtr, tr_y_all)
    f1_logreg = macro(logreg.predict(xte), te_y)
    knn = KNeighborsClassifier(n_neighbors=30, n_jobs=-1)
    knn.fit(xtr, tr_y_all)
    f1_knn = macro(knn.predict(xte), te_y)
    print(f"[domain] LSI-{DOMAIN_DIM} + LogReg  macro-F1 {f1_logreg:.4f}", flush=True)
    print(f"[domain] LSI-{DOMAIN_DIM} + kNN(30) macro-F1 {f1_knn:.4f}", flush=True)

    # --- 2. Scaling curve: gain@k=5 vs training-set size ---------------------------
    scaling = []
    for size in TRAIN_SIZES:
        if size is None or size >= tr_all.shape[0]:
            idx = np.arange(tr_all.shape[0])
            size_label = tr_all.shape[0]
        else:
            idx = np.random.default_rng(0).permutation(tr_all.shape[0])[:size]
            size_label = size
        tr_counts, tr_y = tr_all[idx], tr_y_all[idx]
        # Each size refits its own frozen LSI (train-only) but is evaluated on the same test.
        size_reduction = fit_tfidf_reduction(tr_counts, DOMAIN_DIM)
        te_scaled_size = size_reduction.scaled(te_counts)
        frozen_s, joint_s = [], []
        for seed in SEEDS:
            f, j = frozen_vs_joint(tr_counts, tr_y, te_scaled_size, te_y, n_types, seed)
            frozen_s.append(f)
            joint_s.append(j)
        fm, jm = float(np.mean(frozen_s)), float(np.mean(joint_s))
        scaling.append(
            {"n_train": int(size_label), "frozen": fm, "joint": jm, "gain_pp": 100 * (jm - fm)}
        )
        print(
            f"[scaling] n_train={size_label:>7} frozen {fm:.4f} joint {jm:.4f} "
            f"gain {100 * (jm - fm):+.1f}pp",
            flush=True,
        )

    result = {
        "domain_baseline": {"lsi_logreg": f1_logreg, "lsi_knn": f1_knn, "dim": DOMAIN_DIM},
        "scaling": scaling,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print("ATAC DEEPEN DONE", flush=True)


if __name__ == "__main__":
    main()
