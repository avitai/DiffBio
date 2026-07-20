"""Case study D: VCC perturbation-identity classification, frozen PCA vs learnable projection.

Target-masked (target-gene columns dropped), split by batch (held-out batches), so the
task is to recover perturbation identity from the downstream transcriptional signature.
Reuses the single-cell frozen transform (VCC is scRNA-seq counts) and the learnable
projection, sweeping the reduction dimension k.
"""

import os

import json

import jax.numpy as jnp
import numpy as np
from calibrax.metrics.functional.classification import balanced_accuracy, f1_score
from flax import nnx

from benchmarks.singlecell._gate2_arms import (
    _embedding_probe,
    _probe_forward,
    _project_probe_forward,
    _ProjectionProbe,
)
from benchmarks.singlecell.frozen_annotation_baseline import fit_frozen_preprocess
from diffbio.operators.normalization.learnable_projection import (
    LearnableProjection,
    LearnableProjectionConfig,
)
from diffbio.pipelines.minibatch_training import MiniBatchConfig, train_minibatch


_DATA = os.environ.get("DIFFBIO_DATA_ROOT", "/mnt/ssd2/Data")
d = np.load(f"{_DATA}/Virtual-Cell-Challenge/vcc_masked.npz")
counts = np.asarray(d["counts"], np.float32)
labels = np.asarray(d["labels"], np.int32)
batch = np.asarray(d["batch"], np.int32)
n_types = int(d["n_types"])
SEEDS = (0, 1, 2)
K_VALUES = [5, 10, 20, 50]

# Held-out-batch split: every batch contains all perturbations, so this keeps all classes.
test_batches = set(range(0, int(d["n_batches"]), 5))
test_mask = np.isin(batch, list(test_batches))
tr_counts, te_counts = counts[~test_mask], counts[test_mask]
tr_y, te_y = labels[~test_mask], labels[test_mask]
print(f"train {tr_counts.shape} test {te_counts.shape} | {n_types} perturbations", flush=True)


def macro(pred, true):
    """Return the macro-averaged F1 of predictions against ground truth."""
    return float(f1_score(jnp.asarray(pred), jnp.asarray(true), average="macro"))


frozen = {k: [] for k in K_VALUES}
joint = {k: [] for k in K_VALUES}
bal = {"frozen": [], "joint": []}
for seed in SEEDS:
    transform = fit_frozen_preprocess(tr_counts, n_top_genes=2000, n_components=50)
    tr_c = transform.scaled(tr_counts) - transform.pca_mean
    te_c = transform.scaled(te_counts) - transform.pca_mean
    n_features = transform.loadings.shape[0]
    cfg = MiniBatchConfig(
        batch_size=4096, n_epochs=100, learning_rate=1e-2, weight_decay=5e-2, seed=seed
    )
    for k in K_VALUES:
        loadings_k = transform.loadings[:, :k]
        xtr_f = jnp.asarray(tr_c @ loadings_k)
        xte_f = jnp.asarray(te_c @ loadings_k)
        probe_f = _embedding_probe(k, n_types, 128, seed)
        train_minibatch(probe_f, _probe_forward, xtr_f, tr_y, n_classes=n_types, config=cfg)
        pred_f = np.asarray(jnp.argmax(_probe_forward(probe_f, xte_f), -1))
        frozen[k].append(macro(pred_f, te_y))
        proj = LearnableProjection(
            LearnableProjectionConfig(n_genes=n_features, n_components=k),
            init_loadings=loadings_k,
            rngs=nnx.Rngs(seed),
        )
        model = _ProjectionProbe(proj, _embedding_probe(k, n_types, 128, seed))
        train_minibatch(
            model, _project_probe_forward, jnp.asarray(tr_c), tr_y, n_classes=n_types, config=cfg
        )
        pred_j = np.asarray(jnp.argmax(_project_probe_forward(model, jnp.asarray(te_c)), -1))
        joint[k].append(macro(pred_j, te_y))
        if k == 10:
            bal["frozen"].append(float(balanced_accuracy(jnp.asarray(pred_f), jnp.asarray(te_y))))
            bal["joint"].append(float(balanced_accuracy(jnp.asarray(pred_j), jnp.asarray(te_y))))
        print(
            f"  [seed {seed}] k={k:2d} frozen {frozen[k][-1]:.4f} joint {joint[k][-1]:.4f}",
            flush=True,
        )

print("=== VCC perturbation-ID (target-masked, batch-split) SUMMARY ===", flush=True)
for k in K_VALUES:
    fm, jm = np.mean(frozen[k]), np.mean(joint[k])
    print(
        f"k={k:2d}: frozen {fm:.4f}±{np.std(frozen[k]):.4f}  "
        f"joint {jm:.4f}±{np.std(joint[k]):.4f}  gain {100 * (jm - fm):+.1f}pp",
        flush=True,
    )
print(
    f"balanced-acc @k=10: frozen {np.mean(bal['frozen']):.4f}  joint {np.mean(bal['joint']):.4f}",
    flush=True,
)
with open("benchmarks/results/crossmodality/gate_vcc.json", "w") as h:
    json.dump(
        {
            "k_values": K_VALUES,
            "frozen": {str(k): [float(np.mean(v)), float(np.std(v))] for k, v in frozen.items()},
            "joint": {str(k): [float(np.mean(v)), float(np.std(v))] for k, v in joint.items()},
        },
        h,
        indent=2,
    )
print("VCC DONE", flush=True)
