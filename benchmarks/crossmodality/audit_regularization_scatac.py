"""Control the joint-arm overfitting revealed by the leak-free scATAC audit.

The leak-free (held-out-donor) run showed the joint projection memorizing the training
donors (train F1 ~= 1.0, test ~= 0.5). This sweep, on the same held-out-donor split and
train-only bins, tests whether regularizing the learnable projection shrinks the
train-test gap while keeping the held-out gain at k=5: stronger weight decay, early
stopping, and the orthonormal (Stiefel) projection, which has fewer effective degrees of
freedom. Data prep (fragment re-import + donor split + train-only bins + LSI) is cached so
the sweep is cheap to iterate.
"""

import os

import json

import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import f1_score
from flax import nnx

import benchmarks.crossmodality.audit_leakfree_scatac as base
from benchmarks.singlecell._gate2_arms import (
    _embedding_probe,
    _probe_forward,
    _project_probe_forward,
    _ProjectionProbe,
)
from diffbio.operators.normalization.learnable_orthogonal_projection import (
    LearnableOrthogonalProjection,
    LearnableOrthogonalProjectionConfig,
)
from diffbio.operators.normalization.learnable_projection import (
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch
from diffbio.reductions import fit_tfidf_reduction

_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
CACHE = f"{_DATA}/catlas/atac_leakfree_prep.npz"
OUT = "benchmarks/results/crossmodality/audit_atac_regularized.json"
K = 5
SEEDS = (0, 1, 2)


def macro(pred, true):
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(
        f1_score(jnp.asarray(np.asarray(pred)), jnp.asarray(np.asarray(true)), average="macro")
    )


def prepare():
    """Load (or build+cache) the leak-free train/test LSI features and labels."""
    try:
        d = np.load(CACHE, allow_pickle=False)
        return (
            d["tr_dense"],
            d["te_dense"],
            d["tr_scaled_data"],
            d["tr_y"],
            d["te_y"],
            d["loadings"],
            int(d["n_types"]),
        )
    except FileNotFoundError:
        pass
    counts, labels, donors = base.load_matrix()
    test_mask = np.isin(donors, list(base.TEST_DONORS))
    import collections

    tr_ct = collections.Counter(labels[~test_mask])
    te_ct = collections.Counter(labels[test_mask])
    keep_types = {
        t for t in tr_ct if tr_ct[t] >= base.MIN_TRAIN and te_ct.get(t, 0) >= base.MIN_TEST
    }
    keep = np.isin(labels, list(keep_types))
    counts, labels, test_mask = counts[keep], labels[keep], test_mask[keep]
    type_names, label_ids = np.unique(labels, return_inverse=True)
    tr_counts, te_counts = counts[~test_mask], counts[test_mask]
    tr_y, te_y = label_ids[~test_mask], label_ids[test_mask]
    accessibility = np.asarray((tr_counts > 0).sum(0)).ravel()
    top = np.sort(np.argsort(accessibility)[::-1][: base.TOP_BINS])
    tr_counts, te_counts = tr_counts[:, top].tocsr(), te_counts[:, top].tocsr()
    reduction = fit_tfidf_reduction(tr_counts, K)
    tr_scaled = np.asarray(reduction.scaled(tr_counts).todense(), np.float32)
    te_scaled = np.asarray(reduction.scaled(te_counts).todense(), np.float32)
    loadings = np.asarray(reduction.loadings[:, :K], np.float32)
    np.savez(
        CACHE,
        tr_dense=tr_scaled,
        te_dense=te_scaled,
        tr_scaled_data=np.zeros(1),
        tr_y=tr_y,
        te_y=te_y,
        loadings=loadings,
        n_types=len(type_names),
    )
    return tr_scaled, te_scaled, np.zeros(1), tr_y, te_y, loadings, len(type_names)


def main() -> None:
    """Sweep regularization strategies and report train/test macro-F1 at k=5."""
    tr_dense, te_dense, _, tr_y, te_y, loadings, n_types = prepare()
    tr_dense_j, te_dense_j = jnp.asarray(tr_dense), jnp.asarray(te_dense)
    n_features = loadings.shape[0]
    loadings_k = jnp.asarray(loadings)
    xtr_f = jnp.asarray(tr_dense @ loadings)
    xte_f = jnp.asarray(te_dense @ loadings)
    print(f"train {tr_dense.shape} test {te_dense.shape} | {n_types} types", flush=True)

    configs = {
        "frozen": dict(kind="frozen", wd=5e-2, ep=60),
        "joint_wd5e-2": dict(kind="joint", wd=5e-2, ep=60),
        "joint_wd5e-1": dict(kind="joint", wd=5e-1, ep=60),
        "joint_earlystop15": dict(kind="joint", wd=5e-2, ep=15),
        "joint_stiefel_wd5e-1": dict(kind="stiefel", wd=5e-1, ep=60),
    }
    results = {name: {"train": [], "test": []} for name in configs}
    for seed in SEEDS:
        for name, cfg in configs.items():
            mbc = MiniBatchConfig(
                batch_size=2048,
                n_epochs=cfg["ep"],
                learning_rate=1e-2,
                weight_decay=cfg["wd"],
                seed=seed,
            )
            if cfg["kind"] == "frozen":
                probe = _embedding_probe(K, n_types, 128, seed)
                train_minibatch(probe, _probe_forward, xtr_f, tr_y, n_classes=n_types, config=mbc)
                tr = macro(jnp.argmax(_probe_forward(probe, xtr_f), -1), tr_y)
                te = macro(jnp.argmax(_probe_forward(probe, xte_f), -1), te_y)
            else:
                if cfg["kind"] == "stiefel":
                    proj = LearnableOrthogonalProjection(
                        LearnableOrthogonalProjectionConfig(n_features=n_features, n_components=K),
                        init_loadings=loadings_k,
                        rngs=nnx.Rngs(seed),
                    )
                else:
                    proj = LearnableProjection(
                        LearnableProjectionConfig(n_genes=n_features, n_components=K),
                        init_loadings=loadings_k,
                        rngs=nnx.Rngs(seed),
                    )
                model = _ProjectionProbe(proj, _embedding_probe(K, n_types, 128, seed))
                train_minibatch(
                    model, _project_probe_forward, tr_dense_j, tr_y, n_classes=n_types, config=mbc
                )
                tr = macro(jnp.argmax(_project_probe_forward(model, tr_dense_j), -1), tr_y)
                te = macro(jnp.argmax(_project_probe_forward(model, te_dense_j), -1), te_y)
            results[name]["train"].append(tr)
            results[name]["test"].append(te)
        print(
            f"  [seed {seed}] "
            + " ".join(
                f"{n} tr{results[n]['train'][-1]:.2f}/te{results[n]['test'][-1]:.2f}"
                for n in configs
            ),
            flush=True,
        )

    print("=== scATAC regularization sweep (leak-free, k=5) SUMMARY ===", flush=True)
    frozen_te = np.mean(results["frozen"]["test"])
    summary = {}
    for name in configs:
        tr, te = np.mean(results[name]["train"]), np.mean(results[name]["test"])
        summary[name] = {
            "train": float(tr),
            "test": float(te),
            "gap_pp": 100 * (tr - te),
            "gain_vs_frozen_pp": 100 * (te - frozen_te),
        }
        print(
            f"{name:>22}: train {tr:.3f} test {te:.3f} (gap {100 * (tr - te):.1f}pp"
            f", gain vs frozen {100 * (te - frozen_te):+.1f}pp)",
            flush=True,
        )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("REGULARIZED DONE", flush=True)


if __name__ == "__main__":
    main()
