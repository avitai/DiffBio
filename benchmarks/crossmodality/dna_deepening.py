"""Deepen case study B (DNA regulatory-element classification): domain baseline + scaling.

  1. Domain baseline -- the classic non-deep genomic-sequence classifier: the canonical
     k-mer spectrum fed to linear models (multinomial logistic regression and a linear
     SVM). This checks the frozen arm is a faithful representation of standard practice.
  2. Scaling curve -- the joint-minus-frozen gain at aggressive reduction (k=5) as the
     training set grows, testing the same crossover seen for scRNA and scATAC.

Real Genomic Benchmarks human_ocr_ensembl (open-chromatin regions); PCA fit on train only.
"""

import json
import os

import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import f1_score
from flax import nnx
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
os.environ.setdefault("HF_HOME", f"{_DATA}/huggingface")
from datasets import load_dataset  # noqa: E402

from benchmarks.singlecell._gate2_arms import (  # noqa: E402
    _embedding_probe,
    _probe_forward,
    _project_probe_forward,
    _ProjectionProbe,
)
from diffbio.operators.normalization.learnable_projection import (  # noqa: E402
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch  # noqa: E402
from diffbio.reductions import fit_pca_reduction  # noqa: E402
from diffbio.sequences.kmer import kmer_featurize  # noqa: E402


OUT = "benchmarks/results/crossmodality/gate_seq_deepen.json"
DATASET = "human_ocr_ensembl"
KMER = 6
K = 5
SEEDS = (0, 1, 2)
TRAIN_SIZES = [8000, 20000, 50000, 100000, None]


def macro(pred, true):
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(
        f1_score(jnp.asarray(np.asarray(pred)), jnp.asarray(np.asarray(true)), average="macro")
    )


def frozen_vs_joint(x_tr, y_tr, x_te, y_te, n_classes, seed):
    """Frozen-PCA-probe vs joint-projection-probe at k=K; return (frozen, joint) macro-F1."""
    reduction = fit_pca_reduction(x_tr, K)
    tr_c = reduction.scaled(x_tr) - reduction.pca_mean
    te_c = reduction.scaled(x_te) - reduction.pca_mean
    loadings_k = reduction.loadings[:, :K]
    config = MiniBatchConfig(
        batch_size=1024, n_epochs=60, learning_rate=1e-2, weight_decay=5e-2, seed=seed
    )

    xtr_f = jnp.asarray(tr_c @ loadings_k)
    xte_f = jnp.asarray(te_c @ loadings_k)
    probe = _embedding_probe(K, n_classes, 128, seed)
    train_minibatch(probe, _probe_forward, xtr_f, y_tr, n_classes=n_classes, config=config)
    frozen = macro(jnp.argmax(_probe_forward(probe, xte_f), -1), y_te)

    proj = LearnableProjection(
        LearnableProjectionConfig(n_genes=x_tr.shape[1], n_components=K),
        init_loadings=loadings_k,
        rngs=nnx.Rngs(seed),
    )
    model = _ProjectionProbe(proj, _embedding_probe(K, n_classes, 128, seed))
    train_minibatch(
        model, _project_probe_forward, jnp.asarray(tr_c), y_tr, n_classes=n_classes, config=config
    )
    joint = macro(jnp.argmax(_project_probe_forward(model, jnp.asarray(te_c)), -1), y_te)
    return frozen, joint


def main() -> None:
    """Load, featurize, run the domain baseline and the scaling curve."""
    ds = load_dataset(f"katarinagresova/Genomic_Benchmarks_{DATASET}")  # nosec B615
    train_seq, test_seq = list(ds["train"]["seq"]), list(ds["test"]["seq"])
    y_tr = np.asarray(ds["train"]["label"], np.int32)
    y_te = np.asarray(ds["test"]["label"], np.int32)
    n_classes = int(max(y_tr.max(), y_te.max())) + 1
    x_tr = kmer_featurize(train_seq, KMER, canonical=True)
    x_te = kmer_featurize(test_seq, KMER, canonical=True)
    print(f"[{DATASET}] train {x_tr.shape} test {x_te.shape} | {n_classes} classes", flush=True)

    # --- 1. Domain baseline: k-mer spectrum + linear models ------------------------
    logreg = LogisticRegression(max_iter=2000, C=1.0)
    logreg.fit(x_tr, y_tr)
    f1_logreg = macro(logreg.predict(x_te), y_te)
    svm = LinearSVC(C=1.0, max_iter=5000)
    svm.fit(x_tr, y_tr)
    f1_svm = macro(svm.predict(x_te), y_te)
    print(f"[domain] 6-mer + LogReg   macro-F1 {f1_logreg:.4f}", flush=True)
    print(f"[domain] 6-mer + LinSVM   macro-F1 {f1_svm:.4f}", flush=True)

    # --- 2. Scaling curve: gain@k=5 vs training-set size ---------------------------
    scaling = []
    for size in TRAIN_SIZES:
        if size is None or size >= x_tr.shape[0]:
            idx = np.arange(x_tr.shape[0])
            n = x_tr.shape[0]
        else:
            idx = np.random.default_rng(0).permutation(x_tr.shape[0])[:size]
            n = size
        frozen_s, joint_s = [], []
        for seed in SEEDS:
            f, j = frozen_vs_joint(x_tr[idx], y_tr[idx], x_te, y_te, n_classes, seed)
            frozen_s.append(f)
            joint_s.append(j)
        fm, jm = float(np.mean(frozen_s)), float(np.mean(joint_s))
        scaling.append({"n_train": int(n), "frozen": fm, "joint": jm, "gain_pp": 100 * (jm - fm)})
        print(
            f"[scaling] n_train={n:>7} frozen {fm:.4f} joint {jm:.4f} "
            f"gain {100 * (jm - fm):+.1f}pp",
            flush=True,
        )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(
            {
                "domain_baseline": {"kmer_logreg": f1_logreg, "kmer_linsvm": f1_svm},
                "scaling": scaling,
            },
            handle,
            indent=2,
        )
    print("SEQ DEEPEN DONE", flush=True)


if __name__ == "__main__":
    main()
